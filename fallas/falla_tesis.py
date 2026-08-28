"""
Evaluación con falla de motor sobre las 5 trayectorias de la tesis.
Corre PID, LSTM, Mamba y Mamba+PPO con pérdida del 5%, 10% y 15%.

Uso (WSL):
    cd /mnt/d/TesisI/gym-pybullet-drones
    /mnt/d/venv_mamba/bin/python3 fallas/falla_tesis.py

Salida: results/falla_tesis.json
"""
import argparse
import sys, json
import numpy as np
import torch
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from stable_baselines3 import PPO

from comparar_base import (
    cargar_stats, generar_waypoints,
    normalizar_estado, desnormalizar_accion,
    LSTMDrone, MambaDrone,
    STATS_PATH, INPUT_COLS,
    WINDOW_SIZE, HOVER_RPM, CTRL_FREQ, SIM_FREQ,
    UMBRAL_WAYPOINT, DURACION_FALLA_SEG,
    MIN_RPM, MAX_RPM, MOTOR_FALLA, T_FALLA_SEG,
    es_caida, es_aterrizaje, TRAYECTORIAS,
)

LSTM_MODEL_PATH  = str(_ROOT / "results" / "modelo_lstm.pth")
MAMBA_MODEL_PATH = str(_ROOT / "results" / "modelo_mamba.pth")
PPO_PATH_MANUAL  = str(_ROOT / "results" / "ppo_mamba_5" / "best_model.zip")
PPO_PATH_IRL     = str(_ROOT / "results" / "ppo_mamba_irl_10" / "best_model.zip")
OUTPUT_FILE      = str(_ROOT / "results" / "falla_tesis.json")
DETECCION_MODEL_PATH = str(_ROOT / "results" / "modelo_deteccion_mamba.pth")
STATS_DETECCION_PATH = str(_ROOT / "results" / "stats_deteccion.json")

SEVERIDADES = {
    "5pct":  0.95,
    "10pct": 0.90,
    "15pct": 0.85,
}
DELTA_MAX = 1500
Z_SUELO   = 0.1

TIPOS = [
    "Diagonal larga",
    "Recto norte",
    "Recto este",
    "Diagonal inversa",
    "Diagonal sur",
]


def nueva_env(punto_A):
    return CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array(punto_A).reshape(1, 3),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB, pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=False, obstacles=False, user_debug_gui=False,
    )


def resultado(pos_hist, llego, aterrizo, cayo, t_caida):
    return {
        "llego":    llego,
        "aterrizo": aterrizo,
        "cayo":     cayo,
        "resultado": "llego" if llego else ("aterrizo" if aterrizo else "cayo"),
        "t_caida_seg": round(t_caida / CTRL_FREQ, 2) if cayo else None,
        "error_final": float(np.linalg.norm(np.array(pos_hist[-1]))),
        "posiciones": pos_hist,
    }


def volar_pid(env, waypoints, punto_B, severidad):
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset()
    action = np.zeros((1, 4))
    pos_hist = []; wp_idx = 0; llego = False; aterrizo = False; cayo = False; t_caida = 0

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0], target_pos=ta, target_rpy=np.zeros(3),
        )
        if paso >= int(T_FALLA_SEG * CTRL_FREQ):
            action[0, MOTOR_FALLA] *= severidad
        if es_caida(obs, pos, paso):    cayo = True;     t_caida = paso; break
        if es_aterrizaje(obs, pos, paso): aterrizo = True; break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    r = resultado(pos_hist, llego, aterrizo, cayo, t_caida)
    r["error_final"] = float(np.linalg.norm(np.array(pos_hist[-1]) - punto_B))
    return r


def volar_modelo(env, waypoints, punto_B, model, stats, device, severidad):
    obs, _ = env.reset()
    ventana = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)] * WINDOW_SIZE,
                    maxlen=WINDOW_SIZE)
    action = np.ones((1, 4)) * HOVER_RPM
    pos_hist = []; wp_idx = 0; llego = False; aterrizo = False; cayo = False; t_caida = 0

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        ventana.append(normalizar_estado(obs, ta, stats))
        x = torch.tensor(np.array(ventana), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]
        rpm = np.clip(desnormalizar_accion(pred, stats), MIN_RPM, MAX_RPM)
        if paso >= int(T_FALLA_SEG * CTRL_FREQ):
            rpm[MOTOR_FALLA] *= severidad
        action = rpm.reshape(1, 4)
        if es_caida(obs, pos, paso):      cayo = True;     t_caida = paso; break
        if es_aterrizaje(obs, pos, paso): aterrizo = True; break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    r = resultado(pos_hist, llego, aterrizo, cayo, t_caida)
    r["error_final"] = float(np.linalg.norm(np.array(pos_hist[-1]) - punto_B))
    return r


def _estado_crudo(obs, wp_actual):
    """Igual que normalizar_estado pero sin normalizar — para el detector,
    que usa su propia normalización (stats_deteccion.json)."""
    estado = obs[0]
    pos = estado[0:3]; rpy = estado[7:10]; vel = estado[10:13]; ang = estado[13:16]
    error = wp_actual - pos
    valores = {
        'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
        'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
        'roll':  rpy[0], 'pitch': rpy[1], 'yaw':   rpy[2],
        'ang_x': ang[0], 'ang_y': ang[1], 'ang_z': ang[2],
        'target_x': wp_actual[0], 'target_y': wp_actual[1], 'target_z': wp_actual[2],
        'err_x': error[0], 'err_y': error[1], 'err_z': error[2],
    }
    return np.array([valores[c] for c in INPUT_COLS], dtype=np.float32)


def volar_ppo(env, waypoints, punto_B, mamba_model, ppo, stats, device, severidad,
             detector=None, stats_deteccion=None):
    obs, _ = env.reset()
    ventana = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)] * WINDOW_SIZE,
                    maxlen=WINDOW_SIZE)
    ventana_det = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)] * WINDOW_SIZE,
                        maxlen=WINDOW_SIZE) if detector is not None else None
    action = np.ones((1, 4)) * HOVER_RPM
    pos_hist = []; wp_idx = 0; llego = False; aterrizo = False; cayo = False; t_caida = 0
    pred_cache = np.zeros(4, dtype=np.float32)

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        ventana.append(normalizar_estado(obs, ta, stats))
        if detector is not None:
            crudo = _estado_crudo(obs, ta)
            ventana_det.append(np.array([
                (crudo[i] - stats_deteccion[c][0]) / stats_deteccion[c][1]
                for i, c in enumerate(INPUT_COLS)
            ], dtype=np.float32))
        x = torch.tensor(np.array(ventana), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = mamba_model(x).cpu().numpy()[0]
        pred_cache = pred
        base_rpm = np.clip(desnormalizar_accion(pred, stats), MIN_RPM, MAX_RPM)

        if paso >= int(T_FALLA_SEG * CTRL_FREQ):
            if detector is not None:
                x_det = torch.tensor(np.array(ventana_det), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit_activa, pred_pct = detector(x_det)
                falla_activa = float(torch.sigmoid(logit_activa).item() > 0.5)
                fault_pct    = float(np.clip(pred_pct.item(), 0.0, 1.0)) if falla_activa else 0.0
                fault_info   = np.array([falla_activa, fault_pct], dtype=np.float32)
            else:
                fault_pct  = 1.0 - severidad
                fault_info = np.array([1.0, fault_pct], dtype=np.float32)
            obs_ppo    = np.concatenate([ventana[-1], pred_cache, fault_info])
            delta, _   = ppo.predict(obs_ppo, deterministic=True)
            rpm        = np.clip(base_rpm + delta * DELTA_MAX, MIN_RPM, MAX_RPM)
            rpm[MOTOR_FALLA] *= severidad
            action = np.clip(rpm, MIN_RPM, MAX_RPM).reshape(1, 4)
        else:
            action = base_rpm.reshape(1, 4)

        if es_caida(obs, pos, paso):      cayo = True;     t_caida = paso; break
        if es_aterrizaje(obs, pos, paso): aterrizo = True; break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    r = resultado(pos_hist, llego, aterrizo, cayo, t_caida)
    r["error_final"] = float(np.linalg.norm(np.array(pos_hist[-1]) - punto_B))
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault-info", type=str, default="oracle",
                        choices=["oracle", "detected"],
                        help="oracle (default, comportamiento sin cambios): PPO recibe "
                             "falla_activa/fault_pct como verdad simulada. "
                             "detected: los recibe del modelo de detección Mamba entrenado.")
    parser.add_argument("--ppo", type=str, default="manual",
                        choices=["manual", "irl"],
                        help="manual (default, comportamiento sin cambios): ppo_mamba_5 "
                             "(recompensa manual). irl: ppo_mamba_irl_10 (recompensa "
                             "aprendida vía Apprenticeship Learning).")
    args = parser.parse_args()

    ppo_path = PPO_PATH_IRL if args.ppo == "irl" else PPO_PATH_MANUAL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluación con falla — 5 trayectorias × 3 severidades "
          f"| ppo={args.ppo} | fault_info={args.fault_info} | device={device}")
    print("=" * 65)

    stats = cargar_stats(STATS_PATH)

    lstm_model = LSTMDrone()
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device,
                                          weights_only=True))
    lstm_model.to(device); lstm_model.eval()

    mamba_model = MambaDrone()
    mamba_model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device,
                                           weights_only=True))
    mamba_model.to(device); mamba_model.eval()

    ppo = PPO.load(ppo_path)

    detector, stats_deteccion = None, None
    sufijo = "" if args.ppo == "manual" else "_irl"
    if args.fault_info == "detected":
        from entrenar_deteccion import MambaDetector
        with open(STATS_DETECCION_PATH) as f:
            stats_deteccion = json.load(f)
        detector = MambaDetector()
        detector.load_state_dict(torch.load(DETECCION_MODEL_PATH, map_location=device))
        detector.to(device); detector.eval()
        sufijo += "_detected"
    output_file = str(_ROOT / "results" / f"falla_tesis{sufijo}.json") if sufijo else OUTPUT_FILE

    resultados = []

    for i, (tray, tipo) in enumerate(zip(TRAYECTORIAS, TIPOS)):
        punto_A   = np.array(tray["A"])
        punto_B   = np.array(tray["B"])
        waypoints = generar_waypoints(punto_A, punto_B)

        print(f"\n── T{i+1}: {tipo}  A={punto_A[:2]} → B={punto_B[:2]}")

        res_tray = {
            "trayectoria": i + 1,
            "tipo":    tipo,
            "punto_A": punto_A.tolist(),
            "punto_B": punto_B.tolist(),
            "escenarios": [],
        }

        for nombre, sev in SEVERIDADES.items():
            pct = round((1 - sev) * 100)
            print(f"\n  [{pct}% pérdida]")

            env = nueva_env(punto_A)

            res_pid   = volar_pid(env, waypoints, punto_B, sev)
            print(f"    PID      : {res_pid['resultado']:8s}  err={res_pid['error_final']:.3f}m")

            env.INIT_XYZS = punto_A.reshape(1, 3)
            res_lstm  = volar_modelo(env, waypoints, punto_B, lstm_model,  stats, device, sev)
            print(f"    LSTM     : {res_lstm['resultado']:8s}  err={res_lstm['error_final']:.3f}m")

            env.INIT_XYZS = punto_A.reshape(1, 3)
            res_mamba = volar_modelo(env, waypoints, punto_B, mamba_model, stats, device, sev)
            print(f"    Mamba    : {res_mamba['resultado']:8s}  err={res_mamba['error_final']:.3f}m")

            env.INIT_XYZS = punto_A.reshape(1, 3)
            res_ppo   = volar_ppo(env, waypoints, punto_B, mamba_model, ppo, stats, device, sev,
                                  detector=detector, stats_deteccion=stats_deteccion)
            print(f"    Mamba+PPO: {res_ppo['resultado']:8s}  err={res_ppo['error_final']:.3f}m")

            env.close()

            res_tray["escenarios"].append({
                "perdida_pct": pct,
                "severidad":   sev,
                "pid":         res_pid,
                "lstm":        res_lstm,
                "mamba":       res_mamba,
                "mamba_ppo":   res_ppo,
            })

        resultados.append(res_tray)

    with open(output_file, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\nGuardado en {output_file}")


if __name__ == "__main__":
    main()
