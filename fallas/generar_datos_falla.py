"""
Genera un dataset etiquetado de vuelos PID con falla de motor inyectada, para
entrenar el modelo de detección de falla (entrenar_deteccion.py).

A diferencia de datos/generar_datos_CF2X.py (vuelo nominal, sin falla), aquí
cada episodio tiene una falla de motor con severidad e instante aleatorios:
- severidad: 2%-80% de pérdida de efectividad (mismo rango que el currículo
  de entrenamiento RL en entrenar_rl.py)
- instante:  aleatorio en [T_FALLA_MIN, T_FALLA_MAX] (comparar_base.py)

Cada fila queda etiquetada con falla_activa (0/1) y fault_pct (0 si inactiva).
Los valores de estado se guardan crudos (sin normalizar), igual que
datos/datos_CF2X_800ep.csv — entrenar_deteccion.py normaliza con su propio
split train/val, siguiendo el mismo patrón que Modelos_deep/entrenar.py.

Uso (venv_mamba):
    cd /mnt/d/TesisI/gym-pybullet-drones
    /mnt/d/venv_mamba/bin/python3 fallas/generar_datos_falla.py
    /mnt/d/venv_mamba/bin/python3 fallas/generar_datos_falla.py --episodios 20  # prueba rápida

Salida: results/datos_falla_deteccion.csv
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
    Z_SUELO, MOTOR_FALLA, T_FALLA_MIN, T_FALLA_MAX,
    es_caida, es_aterrizaje,
)

XY_LIM   = 2.0
MIN_DIST = 0.8
MIN_FAULT_PCT, MAX_FAULT_PCT = 0.02, 0.80

OUTPUT_FILE = _ROOT / "results" / "datos_falla_deteccion.csv"
META_COLS   = ["falla_activa", "fault_pct", "episodio", "paso"]
COLUMNAS    = INPUT_COLS + META_COLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios", type=int, default=600)
    args = parser.parse_args()

    print(f"Generando {args.episodios} episodios con falla aleatoria (severidad e instante)...")
    print(f"Salida: {OUTPUT_FILE}")
    print("-" * 60)

    todos_los_datos = []
    for ep in range(args.episodios):
        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            b_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
                break
        punto_A = np.array([a_xy[0], a_xy[1], Z_SUELO])
        punto_B = np.array([b_xy[0], b_xy[1], Z_SUELO])
        waypoints = generar_waypoints(punto_A, punto_B)

        fault_pct   = float(np.random.uniform(MIN_FAULT_PCT, MAX_FAULT_PCT))
        severidad   = 1.0 - fault_pct
        t_falla_seg = float(np.random.uniform(T_FALLA_MIN, T_FALLA_MAX))
        t_falla_paso = int(t_falla_seg * CTRL_FREQ)

        env  = nueva_env(punto_A, gui=False)
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        obs, _ = env.reset()
        action = np.zeros((1, 4))
        wp_idx = 0
        datos_episodio = []

        for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
            obs, _, term, trunc, _ = env.step(action)
            estado = obs[0]
            pos = estado[0:3]; rpy = estado[7:10]
            vel = estado[10:13]; ang = estado[13:16]

            wp_actual = waypoints[min(wp_idx, len(waypoints) - 1)]
            action[0, :], _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=estado, target_pos=wp_actual, target_rpy=np.zeros(3),
            )

            falla_activa = int(paso >= t_falla_paso)
            if falla_activa:
                action[0, MOTOR_FALLA] *= severidad

            error = wp_actual - pos
            datos_episodio.append({
                "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                "vel_x": vel[0], "vel_y": vel[1], "vel_z": vel[2],
                "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
                "ang_x": ang[0], "ang_y": ang[1], "ang_z": ang[2],
                "target_x": wp_actual[0], "target_y": wp_actual[1], "target_z": wp_actual[2],
                "err_x": error[0], "err_y": error[1], "err_z": error[2],
                "falla_activa": falla_activa,
                "fault_pct": fault_pct if falla_activa else 0.0,
                "episodio": ep,
                "paso": paso,
            })

            if es_caida(obs, pos, paso) or es_aterrizaje(obs, pos, paso):
                break
            if np.linalg.norm(wp_actual - pos) < UMBRAL_WAYPOINT:
                wp_idx += 1
                if wp_idx >= len(waypoints):
                    break
            if term or trunc:
                break

        env.close()
        todos_los_datos.extend(datos_episodio)
        print(f"  Ep {ep+1:3d}/{args.episodios} | falla={fault_pct*100:4.1f}% @ {t_falla_seg:.1f}s "
              f"| pasos={len(datos_episodio)}")

    df = pd.DataFrame(todos_los_datos, columns=COLUMNAS)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "-" * 60)
    print(f"Guardado {len(df)} filas, {args.episodios} episodios en {OUTPUT_FILE}")
    print(f"Filas con falla activa: {int(df['falla_activa'].sum())} "
          f"({100*df['falla_activa'].mean():.1f}%)")


if __name__ == "__main__":
    main()
