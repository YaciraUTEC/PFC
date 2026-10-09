"""
Evaluación de PID+PPO(IRL) vs PID solo en compensación de fallas.

Evalúa en 5 trayectorias con severidades 5%, 10%, 15%, 20%, 30%, 40%.

Uso:
    python fallas/evaluar_pid_ppo.py                    # Sin GUI
    python fallas/evaluar_pid_ppo.py --gui              # Con visualización
    python fallas/evaluar_pid_ppo.py --severidad 10     # Solo 10% de pérdida
"""

import argparse
import sys
import json
import time
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
    STATS_PATH, INPUT_COLS,
    WINDOW_SIZE, HOVER_RPM, CTRL_FREQ, SIM_FREQ,
    UMBRAL_WAYPOINT, DURACION_FALLA_SEG,
    MIN_RPM, MAX_RPM, MOTOR_FALLA, T_FALLA_SEG,
    es_caida, es_aterrizaje, TRAYECTORIAS,
)

from irl_features import cargar_escalas, FEATURE_NAMES, N_FEATURES

PPO_PATH_IRL     = str(_ROOT / "results" / "ppo_compensador_mamba_irl_10.zip")
OUTPUT_FILE      = str(_ROOT / "results" / "evaluar_pid_ppo.json")
DELTA_MAX = 3000
Z_SUELO   = 0.1

SEVERIDADES_DEFAULT = {
    "5pct":  0.95,
    "10pct": 0.90,
    "15pct": 0.85,
    "20pct": 0.80,
    "30pct": 0.70,
    "40pct": 0.60,
}

TIPOS_TRAYECTORIA = [
    "Diagonal larga",
    "Recto norte",
    "Recto este",
    "Diagonal inversa",
    "Diagonal sur",
]


def nueva_env(punto_A, gui=False):
    return CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array(punto_A).reshape(1, 3),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB, pyb_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
        gui=gui, obstacles=False, user_debug_gui=False,
    )


def _tiempo_real(env):
    if env.GUI:
        time.sleep(env.CTRL_TIMESTEP)


def volar_pid(env, waypoints, punto_B, severidad, gui=False):
    """Vuelo con PID solo (sin PPO)."""
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset()
    action = np.zeros((1, 4))
    pos_hist = []
    wp_idx = 0
    llego = False
    aterrizo = False
    cayo = False
    t_caida = 0

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        _tiempo_real(env)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.copy())
        ta = waypoints[min(wp_idx, len(waypoints) - 1)]

        # PID genera RPM base
        base_rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=1.0 / CTRL_FREQ,
            state=obs[0],
            target_pos=ta,
            target_rpy=np.zeros(3),
        )
        base_rpm = np.clip(base_rpm, MIN_RPM, MAX_RPM)

        # Aplicar falla
        rpm = base_rpm.copy()
        if paso >= int(T_FALLA_SEG * CTRL_FREQ):
            rpm[MOTOR_FALLA] *= severidad
        action = np.clip(rpm, MIN_RPM, MAX_RPM).reshape(1, 4)

        if es_caida(obs, pos, paso):
            cayo = True
            t_caida = paso
            break
        if es_aterrizaje(obs, pos, paso):
            aterrizo = True
            break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                llego = True
                break
        if term or trunc:
            break

    return {
        "llego": llego,
        "aterrizo": aterrizo,
        "cayo": cayo,
        "resultado": "llego" if llego else ("aterrizo" if aterrizo else "cayo"),
        "t_caida_seg": round(t_caida / CTRL_FREQ, 2) if cayo else None,
        "error_final": float(np.linalg.norm(pos_hist[-1] - punto_B)) if pos_hist else None,
        "duracion_pasos": len(pos_hist),
    }


def volar_pid_ppo(env, waypoints, punto_B, severidad, ppo_model, gui=False):
    """Vuelo con PID + PPO residual."""
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset()
    pos_hist = []
    wp_idx = 0
    llego = False
    aterrizo = False
    cayo = False
    t_caida = 0
    t_falla = int(T_FALLA_SEG * CTRL_FREQ)

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.copy())
        ta = waypoints[min(wp_idx, len(waypoints) - 1)]

        # PID genera RPM base
        base_rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=1.0 / CTRL_FREQ,
            state=obs[0],
            target_pos=ta,
            target_rpy=np.zeros(3),
        )
        base_rpm = np.clip(base_rpm, MIN_RPM, MAX_RPM)

        # PPO decide delta solo después de la falla
        rpm = base_rpm.copy()
        if paso >= t_falla:
            fault_pct = 1.0 - severidad
            fault_info = np.array([1.0, fault_pct], dtype=np.float32)
            estado_norm = normalizar_estado(obs, ta, cargar_stats(STATS_PATH))
            model_pred = np.zeros(4, dtype=np.float32)  # Sin predicción neuronal
            obs_ppo = np.concatenate([estado_norm, model_pred, fault_info]).astype(np.float32)

            delta_norm, _ = ppo_model.predict(obs_ppo, deterministic=True)
            delta = delta_norm * DELTA_MAX
            rpm = np.clip(base_rpm + delta, MIN_RPM, MAX_RPM)
            rpm[MOTOR_FALLA] *= severidad
        else:
            rpm[MOTOR_FALLA] *= severidad

        action = np.clip(rpm, MIN_RPM, MAX_RPM).reshape(1, 4)
        obs, _, term, trunc, _ = env.step(action)
        _tiempo_real(env)

        if es_caida(obs, pos, paso):
            cayo = True
            t_caida = paso
            break
        if es_aterrizaje(obs, pos, paso):
            aterrizo = True
            break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                llego = True
                break
        if term or trunc:
            break

    return {
        "llego": llego,
        "aterrizo": aterrizo,
        "cayo": cayo,
        "resultado": "llego" if llego else ("aterrizo" if aterrizo else "cayo"),
        "t_caida_seg": round(t_caida / CTRL_FREQ, 2) if cayo else None,
        "error_final": float(np.linalg.norm(pos_hist[-1] - punto_B)) if pos_hist else None,
        "duracion_pasos": len(pos_hist),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true",
                        help="Visualiza la simulación (más lento)")
    parser.add_argument("--severidad", type=int, default=None,
                        help="Evalúa solo esta severidad en %% (ej: 10). "
                             "Default: todas (5, 10, 15, 20, 30, 40)")
    args = parser.parse_args()

    if not Path(PPO_PATH_IRL).exists():
        print(f"Error: No encontrado {PPO_PATH_IRL}")
        sys.exit(1)

    ppo_model = PPO.load(PPO_PATH_IRL)
    stats = cargar_stats(STATS_PATH)

    severidades = SEVERIDADES_DEFAULT
    if args.severidad is not None:
        key = f"{args.severidad}pct"
        severidades = {key: round(1.0 - args.severidad / 100, 4)}

    print(f"\nEvaluación PID vs PID+PPO(IRL)")
    print(f"  Severidades: {', '.join(severidades.keys())}")
    print(f"  Trayectorias: {len(TRAYECTORIAS)}")
    print(f"  Modelo PPO: {PPO_PATH_IRL}")
    print(f"  GUI: {args.gui}")
    print("=" * 70)

    resultados = {
        "fecha": str(Path(PPO_PATH_IRL).stat().st_mtime),
        "modelo_ppo": str(PPO_PATH_IRL),
        "severidades": list(severidades.keys()),
        "trayectorias": TIPOS_TRAYECTORIA,
        "datos": {},
    }

    env = None
    try:
        for sev_nombre, severidad in severidades.items():
            print(f"\nSeveridad {sev_nombre} (pérdida {100*(1-severidad):.0f}%)")
            print("-" * 70)
            resultados["datos"][sev_nombre] = {}

            for traj_idx, traj in enumerate(TRAYECTORIAS):
                traj_nombre = TIPOS_TRAYECTORIA[traj_idx]
                punto_A = traj["A"]
                punto_B = traj["B"]
                waypoints = generar_waypoints(punto_A, punto_B)

                if env is None:
                    env = nueva_env(punto_A, gui=args.gui)
                else:
                    env.INIT_XYZS = np.array(punto_A).reshape(1, 3)

                # Vuelo PID
                res_pid = volar_pid(env, waypoints, punto_B, severidad, gui=args.gui)

                # Vuelo PID+PPO
                res_ppo = volar_pid_ppo(env, waypoints, punto_B, severidad, ppo_model, gui=args.gui)

                # Comparación
                recupera_pid = res_pid["llego"] or res_pid["aterrizo"]
                recupera_ppo = res_ppo["llego"] or res_ppo["aterrizo"]
                mejora = "✓" if recupera_ppo and not recupera_pid else ("✗" if not recupera_ppo and recupera_pid else "~")

                print(f"  {traj_nombre:20s} | "
                      f"PID: {res_pid['resultado']:8s} | "
                      f"PID+PPO: {res_ppo['resultado']:8s} | "
                      f"{mejora}")

                resultados["datos"][sev_nombre][traj_nombre] = {
                    "pid": res_pid,
                    "pid_ppo": res_ppo,
                }

        # Resumen
        print("\n" + "=" * 70)
        print("RESUMEN")
        print("=" * 70)
        for sev_nombre in severidades.keys():
            pid_llegó = sum(1 for t in resultados["datos"][sev_nombre].values()
                           if t["pid"]["llego"])
            ppo_llegó = sum(1 for t in resultados["datos"][sev_nombre].values()
                           if t["pid_ppo"]["llego"])
            pid_recupera = sum(1 for t in resultados["datos"][sev_nombre].values()
                              if t["pid"]["llego"] or t["pid"]["aterrizo"])
            ppo_recupera = sum(1 for t in resultados["datos"][sev_nombre].values()
                              if t["pid_ppo"]["llego"] or t["pid_ppo"]["aterrizo"])

            print(f"{sev_nombre:8s} | PID: {pid_recupera}/5 recupera, {pid_llegó}/5 llega | "
                  f"PID+PPO: {ppo_recupera}/5 recupera, {ppo_llegó}/5 llega")

        # Guardar JSON
        with open(OUTPUT_FILE, "w") as f:
            json.dump(resultados, f, indent=2)
        print(f"\nResultados guardados en {OUTPUT_FILE}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
