"""
Genera episodios PID con falla leve de motor (2%-10%, instante fijo 3 s) y conserva
solo los que llegan a punto_B. Se detiene al reunir --objetivo episodios exitosos.

Niveles en rotación: 2, 4, 6, 8, 10 % de pérdida (progresivo, intento i usa niveles[i % 5]).

Uso (desde /mnt/d/TesisI):
    /mnt/d/venv_mamba/bin/python3 -m fallas.generar_datos_pid_falla_leve
    /mnt/d/venv_mamba/bin/python3 -m fallas.generar_datos_pid_falla_leve --objetivo 5  # prueba

Salida: results/datos_pid_falla_leve.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402
from comparar_base import (  # noqa: E402
    nueva_env, generar_waypoints,
    INPUT_COLS, UMBRAL_WAYPOINT, CTRL_FREQ, DURACION_FALLA_SEG,
    Z_SUELO, MOTOR_FALLA, MIN_RPM, MAX_RPM,
    es_caida, es_aterrizaje,
)

XY_LIM   = 2.0
MIN_DIST = 0.8
T_FALLA_SEG = 3.0
NIVELES_PCT = [2, 4, 6, 8, 10]

# Reglas de fase y distancia idénticas a datos_CF2X_800ep.csv (verificadas contra ese archivo)
Z_OBJETIVO_ATERRIZAJE = 0.5
Z_FASE_CRUCERO = 1.1003

OUTPUT_FILE = _ROOT / "results" / "datos_pid_falla_leve.csv"
MOTOR_COLS  = ["motor_0", "motor_1", "motor_2", "motor_3"]
COLUMNAS    = INPUT_COLS + MOTOR_COLS + ["fase", "episodio", "paso", "distancia_objetivo",
                                         "falla_activa", "fault_pct"]


def fase_de(pos, wp_actual):
    if wp_actual[2] < Z_OBJETIVO_ATERRIZAJE:
        return 2
    if pos[2] < Z_FASE_CRUCERO:
        return 0
    return 1


def volar_episodio(punto_A, punto_B, waypoints, fault_pct, episodio):
    severidad = 1.0 - fault_pct / 100.0
    t_falla_paso = int(T_FALLA_SEG * CTRL_FREQ)
    env  = nueva_env(punto_A, gui=False)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset()
    action = np.zeros((1, 4))
    wp_idx = 0
    llego = False
    filas = []

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        estado = obs[0]
        pos = estado[0:3]; rpy = estado[7:10]
        vel = estado[10:13]; ang = estado[13:16]

        wp_actual = waypoints[min(wp_idx, len(waypoints) - 1)]
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=estado, target_pos=wp_actual, target_rpy=np.zeros(3),
        )
        rpm = np.clip(rpm, MIN_RPM, MAX_RPM)
        cmd_pid = rpm.copy()

        falla_activa = int(paso >= t_falla_paso)
        if falla_activa:
            rpm[MOTOR_FALLA] *= severidad
        action[0, :] = np.clip(rpm, MIN_RPM, MAX_RPM)

        error = punto_B - pos
        filas.append({
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
            "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "ang_x": ang[0], "ang_y": ang[1], "ang_z": ang[2],
            "target_x": wp_actual[0], "target_y": wp_actual[1], "target_z": wp_actual[2],
            "err_x": error[0], "err_y": error[1], "err_z": error[2],
            "motor_0": cmd_pid[0], "motor_1": cmd_pid[1],
            "motor_2": cmd_pid[2], "motor_3": cmd_pid[3],
            "fase": fase_de(pos, wp_actual),
            "episodio": episodio,
            "paso": paso,
            "distancia_objetivo": float(np.linalg.norm(pos - wp_actual)),
            "falla_activa": falla_activa,
            "fault_pct": float(fault_pct) if falla_activa else 0.0,
        })

        if es_caida(obs, pos, paso) or es_aterrizaje(obs, pos, paso):
            break
        if np.linalg.norm(wp_actual - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                llego = np.linalg.norm(punto_B - pos) < UMBRAL_WAYPOINT
                break
        if term or trunc:
            break

    env.close()
    return llego, filas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objetivo", type=int, default=800,
                        help="Episodios exitosos a reunir (default 800)")
    parser.add_argument("--max-intentos", type=int, default=5000)
    args = parser.parse_args()

    print(f"Reuniendo {args.objetivo} episodios PID exitosos con falla "
          f"{NIVELES_PCT}% a los {T_FALLA_SEG}s")
    print(f"Salida: {OUTPUT_FILE}")
    print("-" * 60)

    todas_las_filas = []
    exitos = 0
    intentos_por_nivel = {n: [0, 0] for n in NIVELES_PCT}

    for intento in range(args.max_intentos):
        if exitos >= args.objetivo:
            break
        nivel = NIVELES_PCT[intento % len(NIVELES_PCT)]

        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            b_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
                break
        punto_A = np.array([a_xy[0], a_xy[1], Z_SUELO])
        punto_B = np.array([b_xy[0], b_xy[1], Z_SUELO])
        waypoints = generar_waypoints(punto_A, punto_B)

        llego, filas = volar_episodio(punto_A, punto_B, waypoints, nivel, exitos)
        intentos_por_nivel[nivel][0] += 1
        if llego:
            intentos_por_nivel[nivel][1] += 1
            todas_las_filas.extend(filas)
            exitos += 1

        print(f"  Intento {intento+1:4d} | falla={nivel:2d}% | "
              f"{'llegó ✓' if llego else 'no llegó'} | exitos={exitos}/{args.objetivo}")

    df = pd.DataFrame(todas_las_filas, columns=COLUMNAS)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "-" * 60)
    print(f"Guardado {len(df)} filas, {exitos} episodios exitosos en {OUTPUT_FILE}")
    for nivel, (total, ok) in intentos_por_nivel.items():
        pct = 100 * ok / total if total else 0.0
        print(f"  {nivel:2d}%: {ok}/{total} llegaron ({pct:.0f}%)")
    if exitos < args.objetivo:
        print(f"Aviso: se alcanzó --max-intentos sin reunir {args.objetivo} exitosos.")


if __name__ == "__main__":
    main()
