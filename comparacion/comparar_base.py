
import numpy as np
import json
import torch
import torch.nn as nn
import pybullet as p
from collections import deque
from mamba_ssm import Mamba
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

import os as _os
_RESULTS        = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results")
STATS_PATH      = _os.path.join(_RESULTS, "stats_normalizacion.json")
WINDOW_SIZE          = 50
DURACION_SEG         = 20
DURACION_FALLA_SEG   = 15
SIM_FREQ             = 240
CTRL_FREQ            = 48
Z_CRUCERO            = 1.2
Z_SUELO              = 0.1
N_INTERMEDIOS        = 5
HOVER_RPM            = 14300
UMBRAL_WAYPOINT      = 0.25
MIN_RPM              = 9440
MAX_RPM              = 21700
COLOR_PID            = (1.0, 0.3, 0.2)
COLOR_LSTM           = (0.2, 0.4, 1.0)
COLOR_MAMBA          = (0.2, 0.8, 0.3)

# ── Constantes de falla ───────────────────────────────────────
MOTOR_FALLA   = 0
T_FALLA_SEG   = 3.0
T_FALLA_MIN   = 1.0   # rango de aleatorización del instante de falla (entrenamiento RL)
T_FALLA_MAX   = 6.0   # deja >=9s de los 15s de DURACION_FALLA_SEG para observar recuperación
ANGULO_CRASH  = np.radians(35)
ESCENARIOS    = [0.90, 0.85, 0.80, 0.75, 0.70]   # pérdida 10%, 15%, 20%, 25%, 30%

# ── Umbrales de aterrizaje controlado ────────────────────────
ANGULO_LAND   = np.radians(20)   # más estricto que crash (35°)
VEL_VERT_LAND = 0.4              # m/s descenso máximo aceptable
ANG_VEL_LAND  = 1.5              # rad/s suma velocidades angulares


def es_caida(obs, pos, paso):
    if pos[2] >= 0.05 or paso <= 10:
        return False
    rpy = obs[0][7:10]
    vel = obs[0][10:13]
    actitud_critica = abs(rpy[0]) > ANGULO_CRASH or abs(rpy[1]) > ANGULO_CRASH
    cayendo         = vel[2] < -0.5
    return actitud_critica or cayendo


def es_aterrizaje(obs, pos, paso):
    """Toca suelo de forma controlada: actitud ≤20°, vz > -0.4 m/s, ω baja."""
    if pos[2] >= 0.05 or paso <= 10:
        return False
    rpy     = obs[0][7:10]
    vel     = obs[0][10:13]   # velocidad lineal [vx, vy, vz]
    ang_vel = obs[0][13:16]   # velocidad angular [wx, wy, wz]
    inclinacion_ok  = abs(rpy[0]) < ANGULO_LAND and abs(rpy[1]) < ANGULO_LAND
    vel_vertical_ok = vel[2] > -VEL_VERT_LAND
    ang_vel_ok      = float(np.sum(np.abs(ang_vel))) < ANG_VEL_LAND
    return inclinacion_ok and vel_vertical_ok and ang_vel_ok


def aplicar_falla(action, paso, severidad):
    if paso >= int(T_FALLA_SEG * CTRL_FREQ):
        action = action.copy()
        action[0, MOTOR_FALLA] *= severidad
    return action


def marcar_falla(pos, client):
    x, y, z = pos
    d = 0.15
    p.addUserDebugLine([x-d, y, z], [x+d, y, z], [1,0,0], lineWidth=3, lifeTime=0, physicsClientId=client)
    p.addUserDebugLine([x, y-d, z], [x, y+d, z], [1,0,0], lineWidth=3, lifeTime=0, physicsClientId=client)
    p.addUserDebugLine([x, y, z-d], [x, y, z+d], [1,0,0], lineWidth=3, lifeTime=0, physicsClientId=client)
    p.addUserDebugText(f"FALLA M{MOTOR_FALLA}",
                       [x, y, z+0.25], textColorRGB=[1,0,0],
                       textSize=1.2, lifeTime=0, physicsClientId=client)

INPUT_COLS = [
    'pos_x', 'pos_y', 'pos_z',
    'vel_x', 'vel_y', 'vel_z',
    'roll',  'pitch', 'yaw',
    'ang_x', 'ang_y', 'ang_z',
    'target_x', 'target_y', 'target_z',
    'err_x', 'err_y', 'err_z',
]
OUTPUT_COLS = ['motor_0', 'motor_1', 'motor_2', 'motor_3']

TRAYECTORIAS = [
    {"A": [-1.5, -1.5, Z_SUELO], "B": [ 1.5,  1.5, Z_SUELO]},
    {"A": [ 0.0, -1.5, Z_SUELO], "B": [ 0.0,  1.5, Z_SUELO]},
    {"A": [-1.5,  0.0, Z_SUELO], "B": [ 1.5,  0.0, Z_SUELO]},
    {"A": [ 1.0, -1.0, Z_SUELO], "B": [-1.0,  1.0, Z_SUELO]},
    {"A": [-1.0,  1.0, Z_SUELO], "B": [ 1.0, -1.0, Z_SUELO]},
]

# ── Modelos ──────────────────────────────────────────────────
class LSTMDrone(nn.Module):
    def __init__(self, input_size=18, hidden_size=128,
                 num_layers=2, output_size=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(64, output_size))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class MambaDrone(nn.Module):
    def __init__(self, input_size=18, d_model=128, n_layers=2, output_size=4):
        super().__init__()
        self.input_proj   = nn.Linear(input_size, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fc    = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(64, output_size))
    def forward(self, x):
        x = self.input_proj(x)
        for mamba, norm in zip(self.mamba_layers, self.norms):
            x = norm(x + mamba(x))
        return self.fc(x[:, -1, :])


# ── Funciones compartidas ───────────────────────────────────
def cargar_stats(path):
    with open(path) as f:
        data = json.load(f)
    return {k: (v[0], v[1]) for k, v in data.items()}


def normalizar_estado(obs, wp_actual, stats):
    estado = obs[0]
    pos = estado[0:3]; rpy = estado[7:10]
    vel = estado[10:13]; ang = estado[13:16]
    error = wp_actual - pos
    valores = {
        'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
        'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
        'roll':  rpy[0], 'pitch': rpy[1], 'yaw':   rpy[2],
        'ang_x': ang[0], 'ang_y': ang[1], 'ang_z': ang[2],
        'target_x': wp_actual[0], 'target_y': wp_actual[1], 'target_z': wp_actual[2],
        'err_x': error[0], 'err_y': error[1], 'err_z': error[2],
    }
    return np.array([(valores[c] - stats[c][0]) / stats[c][1]
                     for c in INPUT_COLS], dtype=np.float32)


def desnormalizar_accion(pred, stats):
    return np.array([pred[i] * stats[OUTPUT_COLS[i]][1] + stats[OUTPUT_COLS[i]][0]
                     for i in range(4)], dtype=np.float32)


def generar_waypoints(punto_A, punto_B):
    A      = np.array([punto_A[0], punto_A[1], Z_CRUCERO])
    B      = np.array([punto_B[0], punto_B[1], Z_CRUCERO])
    B_suelo = np.array([punto_B[0], punto_B[1], Z_SUELO])
    wps = []
    for i in range(1, N_INTERMEDIOS + 1):
        t = i / (N_INTERMEDIOS + 1)
        wp = A + t * (B - A)
        wp[2] = Z_CRUCERO + 0.1 * np.sin(t * np.pi)
        wps.append(wp)
    return [A] + wps + [B] + [B_suelo]


def nueva_env(punto_A, gui=True):
    return CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array(punto_A).reshape(1, 3),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB, pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=gui, obstacles=False, user_debug_gui=False,
    )


def redibujar(posiciones, color, client):
    for k in range(1, len(posiciones)):
        p.addUserDebugLine(posiciones[k-1], posiciones[k], list(color),
                           lineWidth=2, lifeTime=0, physicsClientId=client)


def _metricas(pos_hist, punto_B, waypoints):
    pos      = np.array(pos_hist)
    longitud = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    min_dists = [float(np.min(np.linalg.norm(pos - np.array(wp), axis=1)))
                 for wp in waypoints]
    error_fin = float(np.linalg.norm(pos[-1] - np.array(punto_B)))
    return longitud, min_dists, error_fin


def volar_pid(env, punto_B, waypoints):
    obs, _ = env.reset()
    ctrl   = DSLPIDControl(drone_model=DroneModel.CF2X)
    action = np.zeros((1, 4))
    pos_hist = []; rpy_hist = []; rpm_hist = []
    wp_idx = 0; llego = False

    for _ in range(int(DURACION_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        rpy_hist.append(obs[0][7:10].tolist())

        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0], target_pos=ta, target_rpy=np.zeros(3),
        )
        rpm_hist.append(action[0].tolist())

        if len(pos_hist) > 1:
            p.addUserDebugLine(pos_hist[-2], pos_hist[-1], list(COLOR_PID),
                               lineWidth=2, lifeTime=0, physicsClientId=env.CLIENT)
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    longitud, min_dists, error = _metricas(pos_hist, punto_B, waypoints)
    return {"posiciones": pos_hist, "orientaciones": rpy_hist, "rpms": rpm_hist,
            "llego": llego, "error_final": error, "pasos": len(pos_hist),
            "longitud": longitud, "min_dist_por_wp": min_dists,
            "media_min_dist": float(np.mean(min_dists))}


def volar_modelo(env, punto_B, waypoints, model, stats, device, color):
    obs, _ = env.reset()
    ventana = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)] * WINDOW_SIZE,
                    maxlen=WINDOW_SIZE)
    action = np.ones((1, 4)) * HOVER_RPM
    pos_hist = []; rpy_hist = []; rpm_hist = []
    wp_idx = 0; llego = False

    for _ in range(int(DURACION_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        rpy_hist.append(obs[0][7:10].tolist())

        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        ventana.append(normalizar_estado(obs, ta, stats))

        x = torch.tensor(np.array(ventana), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]
        action = np.clip(desnormalizar_accion(pred, stats), 9440, 21700).reshape(1, 4)
        rpm_hist.append(action[0].tolist())

        if len(pos_hist) > 1:
            p.addUserDebugLine(pos_hist[-2], pos_hist[-1], list(color),
                               lineWidth=2, lifeTime=0, physicsClientId=env.CLIENT)

        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    longitud, min_dists, error = _metricas(pos_hist, punto_B, waypoints)
    return {"posiciones": pos_hist, "orientaciones": rpy_hist, "rpms": rpm_hist,
            "llego": llego, "error_final": error, "pasos": len(pos_hist),
            "longitud": longitud, "min_dist_por_wp": min_dists,
            "media_min_dist": float(np.mean(min_dists))}


def imprimir_resumen(resultados, nombre_modelo, key):
    n = len(resultados)
    W = 80
    print("\n" + "=" * W)
    print(f"  {'Tray':>4} | {'PID':^34} | {nombre_modelo:^34}")
    print(f"  {'':>4} | {'✓':^3} {'err_f':>6} {'long':>6} {'min_wp':>6} {'p':>4} "
          f"| {'✓':^3} {'err_f':>6} {'long':>6} {'min_wp':>6} {'p':>4}")
    print("-" * W)
    for r in resultados:
        pid = r['pid'];  mdl = r[key]
        print(f"    {r['trayectoria']:>2}   "
              f"  {'✓' if pid['llego'] else '✗'} "
              f"{pid['error_final']:>6.3f} {pid['longitud']:>6.2f} {pid['media_min_dist']:>6.3f} {pid['pasos']:>4}   "
              f"  {'✓' if mdl['llego'] else '✗'} "
              f"{mdl['error_final']:>6.3f} {mdl['longitud']:>6.2f} {mdl['media_min_dist']:>6.3f} {mdl['pasos']:>4}")
    print("-" * W)
    for nombre, k in [("PID", "pid"), (nombre_modelo, key)]:
        ok  = sum(r[k]['llego'] for r in resultados)
        err = np.mean([r[k]['error_final']    for r in resultados])
        lon = np.mean([r[k]['longitud']       for r in resultados])
        mmd = np.mean([r[k]['media_min_dist'] for r in resultados])
        print(f"  {nombre:6s}: {ok}/{n} llegaron | "
              f"err_final={err:.3f} m | long={lon:.2f} m | min_wp={mmd:.3f} m")
    print("=" * W)


def guardar(resultados, output_file):
    with open(output_file, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"Guardado en {output_file}")


# ── Funciones de vuelo con falla ─────────────────────────────────────────────

def fuera_zona_segura(pos, rpy):
    """True si el dron está fuera de la zona segura de operación post-falla."""
    altitud_ok  = pos[2] >= Z_CRUCERO * 0.6
    actitud_ok  = abs(rpy[0]) < ANGULO_LAND and abs(rpy[1]) < ANGULO_LAND
    return not (altitud_ok and actitud_ok)


def _resultado_falla(pos_hist, rpy_hist, rpm_hist, punto_B, waypoints,
                     llego, caida_paso, wp_idx, aterrizo=False, estado_falla=None,
                     pasos_fuera_zona=0, t_falla_paso=None):
    total_wp   = len(waypoints)
    pasos_post = len(pos_hist) - (t_falla_paso or 0)
    result = {
        "posiciones":           pos_hist,
        "orientaciones":        rpy_hist,
        "rpms":                 rpm_hist,
        "llego":                llego,
        "cayo":                 caida_paso is not None,
        "aterrizo":             aterrizo,
        "t_caida_seg":          caida_paso / CTRL_FREQ if caida_paso else None,
        "t_vuelo_seg":          len(pos_hist) / CTRL_FREQ,
        "error_final":          float(np.linalg.norm(np.array(pos_hist[-1]) - np.array(punto_B))),
        "pasos":                len(pos_hist),
        "wp_idx_alcanzado":     wp_idx,
        "total_waypoints":      total_wp,
        "porcentaje_mision":    round(100 * wp_idx / total_wp, 1),
        "pasos_fuera_zona":     pasos_fuera_zona,
        "pct_fuera_zona":       round(100 * pasos_fuera_zona / max(pasos_post, 1), 1),
        "t_fuera_zona_seg":     round(pasos_fuera_zona / CTRL_FREQ, 2),
    }
    if estado_falla is not None:
        result["estado_falla"] = estado_falla
    return result


def volar_pid_falla(env, punto_B, waypoints, estado_falla, color):
    severidad = estado_falla["severidad"]
    obs, _ = env.reset()
    ctrl   = DSLPIDControl(drone_model=DroneModel.CF2X)
    action = np.zeros((1, 4))
    pos_hist = []; rpy_hist = []; rpm_hist = []
    wp_idx = 0; llego = False; caida_paso = None; aterrizo = False
    pasos_fuera = 0; t_falla_paso = int(T_FALLA_SEG * CTRL_FREQ)

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        rpy = obs[0][7:10]
        pos_hist.append(pos.tolist())
        rpy_hist.append(rpy.tolist())

        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0], target_pos=ta, target_rpy=np.zeros(3),
        )
        action = aplicar_falla(action, paso, severidad)
        rpm_hist.append(action[0].tolist())

        if paso == t_falla_paso:
            marcar_falla(pos, env.CLIENT)
        if paso >= t_falla_paso and fuera_zona_segura(pos, rpy):
            pasos_fuera += 1

        if len(pos_hist) > 1:
            p.addUserDebugLine(pos_hist[-2], pos_hist[-1], list(color),
                               lineWidth=2, lifeTime=0, physicsClientId=env.CLIENT)
        if es_caida(obs, pos, paso):
            caida_paso = paso; break
        if es_aterrizaje(obs, pos, paso):
            aterrizo = True; break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    return _resultado_falla(pos_hist, rpy_hist, rpm_hist, punto_B, waypoints,
                            llego, caida_paso, wp_idx, aterrizo, estado_falla,
                            pasos_fuera, t_falla_paso)


def volar_modelo_falla(env, punto_B, waypoints, model, stats, device, estado_falla, color):
    severidad = estado_falla["severidad"]
    obs, _ = env.reset()
    ventana = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)] * WINDOW_SIZE,
                    maxlen=WINDOW_SIZE)
    action = np.ones((1, 4)) * HOVER_RPM
    pos_hist = []; rpy_hist = []; rpm_hist = []
    wp_idx = 0; llego = False; caida_paso = None; aterrizo = False
    pasos_fuera = 0; t_falla_paso = int(T_FALLA_SEG * CTRL_FREQ)

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        rpy = obs[0][7:10]
        pos_hist.append(pos.tolist())
        rpy_hist.append(rpy.tolist())

        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        ventana.append(normalizar_estado(obs, ta, stats))

        x = torch.tensor(np.array(ventana), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]
        action = np.clip(desnormalizar_accion(pred, stats), MIN_RPM, MAX_RPM).reshape(1, 4)
        action = aplicar_falla(action, paso, severidad)
        rpm_hist.append(action[0].tolist())

        if paso == t_falla_paso:
            marcar_falla(pos, env.CLIENT)
        if paso >= t_falla_paso and fuera_zona_segura(pos, rpy):
            pasos_fuera += 1

        if len(pos_hist) > 1:
            p.addUserDebugLine(pos_hist[-2], pos_hist[-1], list(color),
                               lineWidth=2, lifeTime=0, physicsClientId=env.CLIENT)
        if es_caida(obs, pos, paso):
            caida_paso = paso; break
        if es_aterrizaje(obs, pos, paso):
            aterrizo = True; break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    return _resultado_falla(pos_hist, rpy_hist, rpm_hist, punto_B, waypoints,
                            llego, caida_paso, wp_idx, aterrizo, estado_falla,
                            pasos_fuera, t_falla_paso)
