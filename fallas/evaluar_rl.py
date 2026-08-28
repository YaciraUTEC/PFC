"""
Evaluación del compensador PPO sobre las 5 trayectorias bajo falla de motor.
Compara: PID | LSTM | Mamba | Mamba+PPO
"""

import sys
import json
import numpy as np
import torch
import pybullet as p
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import (
    cargar_stats, generar_waypoints, nueva_env, redibujar,
    normalizar_estado, desnormalizar_accion,
    LSTMDrone, MambaDrone,
    STATS_PATH, TRAYECTORIAS, INPUT_COLS,
    WINDOW_SIZE, HOVER_RPM, CTRL_FREQ, UMBRAL_WAYPOINT, DURACION_FALLA_SEG,
    MIN_RPM, MAX_RPM,
    COLOR_PID, COLOR_LSTM, COLOR_MAMBA,
    MOTOR_FALLA, T_FALLA_SEG, ESCENARIOS,
    es_caida, es_aterrizaje, marcar_falla, _resultado_falla,
    volar_pid_falla, volar_modelo_falla,
)
from stable_baselines3 import PPO

LSTM_MODEL_PATH      = str(_ROOT / "results" / "modelo_lstm.pth")
MAMBA_MODEL_PATH     = str(_ROOT / "results" / "modelo_mamba.pth")
PPO_MAMBA_MODEL_PATH = str(_ROOT / "results" / "ppo_mamba_irl_10" / "best_model.zip")
PPO_LSTM_MODEL_PATH  = str(_ROOT / "results" / "ppo_lstm"  / "best_model.zip")
OUTPUT_MAMBA = str(_ROOT / "results" / "evaluacion_mamba_ppo.json")
OUTPUT_LSTM  = str(_ROOT / "results" / "evaluacion_lstm_ppo.json")

COLOR_PPO_MAMBA = (0.9, 0.6, 0.0)
COLOR_PPO_LSTM  = (0.8, 0.2, 0.8)
DELTA_MAX   = 1500
ESCENARIOS  = [0.95, 0.90, 0.85, 0.80,0.70]


def volar_modelo_ppo(env, punto_B, waypoints, base_model, ppo, stats, device, severidad, color):
    obs, _ = env.reset()
    ventana = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)]* WINDOW_SIZE,
                    maxlen=WINDOW_SIZE)
    action = np.ones((1, 4)) * HOVER_RPM
    pos_hist = []; rpy_hist = []; rpm_hist = []
    wp_idx = 0; llego = False; caida_paso = None; aterrizo = False
    mamba_pred_cache = np.zeros(4, dtype=np.float32)

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        rpy_hist.append(obs[0][7:10].tolist())

        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        ventana.append(normalizar_estado(obs, ta, stats))

        x = torch.tensor(np.array(ventana), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = base_model(x).cpu().numpy()[0]
        mamba_pred_cache = pred
        base_rpm = np.clip(desnormalizar_accion(pred, stats), MIN_RPM, MAX_RPM)

        # PPO genera delta solo post-falla
        if paso >= int(T_FALLA_SEG * CTRL_FREQ):
            fault_pct  = 1.0 - severidad
            fault_info = np.array([1.0, fault_pct], dtype=np.float32)   # falla_activa=1 (igual que training)
            obs_ppo    = np.concatenate([ventana[-1], mamba_pred_cache, fault_info])
            delta_norm, _ = ppo.predict(obs_ppo, deterministic=True)
            cmd_rpm       = np.clip(base_rpm + delta_norm * DELTA_MAX, MIN_RPM, MAX_RPM)
            cmd_rpm[MOTOR_FALLA] *= severidad
            action_final  = np.clip(cmd_rpm, MIN_RPM, MAX_RPM)
        else:
            action_final = base_rpm

        action = action_final.reshape(1, 4)
        rpm_hist.append(action[0].tolist())

        if paso == int(T_FALLA_SEG * CTRL_FREQ):
            marcar_falla(pos, env.CLIENT)

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
                            llego, caida_paso, wp_idx, aterrizo,
                            {"motor": MOTOR_FALLA, "severidad": severidad})


def _imprimir(nombre, res):
    if res["llego"]:       estado = "✓ llegó"
    elif res["cayo"]:      estado = "✗ cayó"
    elif res["aterrizo"]:  estado = "⬇ aterrizó"
    else:                  estado = "✗ no llegó"
    t_info = (f"caída en t={res['t_caida_seg']:.1f}s"
              if res["cayo"] else f"vuelo={res['t_vuelo_seg']:.1f}s")
    wps     = f"wp {res['wp_idx_alcanzado']}/{res['total_waypoints']} ({res['porcentaje_mision']:.0f}%)"
    fuera   = f"fuera_zona={res['t_fuera_zona_seg']:.1f}s ({res['pct_fuera_zona']:.0f}%)"
    print(f"    {nombre:<12}: {estado} | err={res['error_final']:.3f} m | {wps} | {t_info} | {fuera}")


def _tabla_resumen(resultados):
    modelos = [("pid", "PID"), ("lstm", "LSTM"), ("mamba", "Mamba"), ("mamba_ppo", "Mamba+PPO")]
    n = len(resultados)
    top = "╔" + "═"*14 + "╦" + "═"*8 + "╦" + "═"*12 + "╦" + "═"*10 + "╦" + "═"*9 + "╦" + "═"*10 + "╦" + "═"*12 + "╗"
    sep = "╠" + "═"*14 + "╬" + "═"*8 + "╬" + "═"*12 + "╬" + "═"*10 + "╬" + "═"*9 + "╬" + "═"*10 + "╬" + "═"*12 + "╣"
    bot = "╚" + "═"*14 + "╩" + "═"*8 + "╩" + "═"*12 + "╩" + "═"*10 + "╩" + "═"*9 + "╩" + "═"*10 + "╩" + "═"*12 + "╝"

    def fila(c):
        return f"║{c[0]:^14}║{c[1]:^8}║{c[2]:^12}║{c[3]:^10}║{c[4]:^9}║{c[5]:^10}║{c[6]:^12}║"

    print("\n" + top)
    print(fila(["Modelo", "Llegaron", "Error prom", "% Misión", "Caídas", "Aterrizó", "Fuera zona"]))
    print(sep)
    for key, lbl in modelos:
        llegan    = sum(r[key]["llego"]              for r in resultados)
        caen      = sum(r[key]["cayo"]               for r in resultados)
        aterrizan = sum(r[key]["aterrizo"]           for r in resultados)
        err_avg   = np.mean([r[key]["error_final"]           for r in resultados])
        mision    = np.mean([r[key]["porcentaje_mision"]     for r in resultados])
        fuera_seg = np.mean([r[key]["t_fuera_zona_seg"]      for r in resultados])
        print(fila([lbl, f"{llegan}/{n}", f"{err_avg:.3f} m", f"{mision:.1f}%",
                    str(caen), str(aterrizan), f"{fuera_seg:.1f}s"]))
    print(bot)
    print(f"  Motor {MOTOR_FALLA} | falla en t={T_FALLA_SEG}s\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando Motor {MOTOR_FALLA} | pérdidas: {[f'{round((1-s)*100)}%' for s in ESCENARIOS]}")
    print("Rojo=PID | Azul=LSTM | Verde=Mamba | Naranja=Mamba+PPO")
    print("=" * 65)

    stats = cargar_stats(STATS_PATH)

    lstm_model = LSTMDrone()
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True))
    lstm_model.to(device); lstm_model.eval()

    mamba_model = MambaDrone()
    mamba_model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device, weights_only=True))
    mamba_model.to(device); mamba_model.eval()

    ppo_mamba = PPO.load(PPO_MAMBA_MODEL_PATH)
    print(f"PPO Mamba cargado: {PPO_MAMBA_MODEL_PATH}\n")

    primer_A = np.array(TRAYECTORIAS[0]["A"])
    env = nueva_env(primer_A)

    resultados_escenarios = []

    for severidad in ESCENARIOS:
        perdida = round((1 - severidad) * 100)
        ef = {"motor": MOTOR_FALLA, "severidad": severidad}
        print(f"\n{'═'*65}")
        print(f"  Pérdida {perdida}% (potencia residual {round(severidad*100)}%)")
        print(f"{'═'*65}")

        resultados = []
        for i, tray in enumerate(TRAYECTORIAS):
            punto_A   = np.array(tray["A"])
            punto_B   = np.array(tray["B"])
            waypoints = generar_waypoints(punto_A, punto_B)
            env.INIT_XYZS = punto_A.reshape(1, 3)

            print(f"\n── Trayectoria {i+1}/{len(TRAYECTORIAS)} "
                  f"A={np.round(punto_A[:2],1)} → B={np.round(punto_B[:2],1)}")

            res_pid = volar_pid_falla(env, punto_B, waypoints, ef, COLOR_PID)
            _imprimir("PID", res_pid)
            input("    Enter para LSTM...")

            env.INIT_XYZS = punto_A.reshape(1, 3)
            res_lstm = volar_modelo_falla(env, punto_B, waypoints, lstm_model, stats, device, ef, COLOR_LSTM)
            redibujar(res_pid["posiciones"], COLOR_PID, env.CLIENT)
            _imprimir("LSTM", res_lstm)
            input("    Enter para Mamba...")

            env.INIT_XYZS = punto_A.reshape(1, 3)
            res_mamba = volar_modelo_falla(env, punto_B, waypoints, mamba_model, stats, device, ef, COLOR_MAMBA)
            redibujar(res_pid["posiciones"],  COLOR_PID,  env.CLIENT)
            redibujar(res_lstm["posiciones"], COLOR_LSTM, env.CLIENT)
            _imprimir("Mamba", res_mamba)
            input("    Enter para Mamba+PPO...")

            env.INIT_XYZS = punto_A.reshape(1, 3)
            res_mamba_ppo = volar_modelo_ppo(env, punto_B, waypoints, mamba_model, ppo_mamba,
                                             stats, device, severidad, COLOR_PPO_MAMBA)
            redibujar(res_pid["posiciones"],   COLOR_PID,   env.CLIENT)
            redibujar(res_lstm["posiciones"],  COLOR_LSTM,  env.CLIENT)
            redibujar(res_mamba["posiciones"], COLOR_MAMBA, env.CLIENT)
            _imprimir("Mamba+PPO", res_mamba_ppo)

            resultados.append({
                "tray_idx": i + 1,
                "punto_A": punto_A.tolist(), "punto_B": punto_B.tolist(),
                "pid": res_pid, "lstm": res_lstm, "mamba": res_mamba,
                "mamba_ppo": res_mamba_ppo,
            })

            if i < len(TRAYECTORIAS) - 1:
                input("    Enter para siguiente trayectoria...")

        _tabla_resumen(resultados)
        resultados_escenarios.append({"perdida_pct": perdida, "trayectorias": resultados})

        if severidad != ESCENARIOS[-1]:
            input(f"\n  Enter para siguiente escenario ({round((1-ESCENARIOS[ESCENARIOS.index(severidad)+1])*100)}%)...")

    meta = {"escenarios": ESCENARIOS, "motor": MOTOR_FALLA}
    with open(OUTPUT_MAMBA, "w") as f:
        json.dump({**meta, "escenarios": resultados_escenarios}, f, indent=2)
    print(f"\nGuardado en {OUTPUT_MAMBA}")
    env.close()


if __name__ == "__main__":
    main()
