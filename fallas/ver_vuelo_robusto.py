
"""
Vuelo en vivo del controlador RobustDSLPIDControl con falla de motor, en una ventana de PyBullet.
Todos los episodios se hacen en una sola ventana: el entorno se crea una vez y se reinicia en cada vuelo.

Modos:
  viaje        : punto A -> B aleatorio, como en generar_datos_pid_falla_leve.py. Llegar a B = éxito.
  estacionario : el dron mantiene una altura fija (HOVER_Z) sobre su punto A, sin avanzar.
                 Éxito = no caer durante toda la duración.

Por episodio imprime: causa de término, instante de la falla, inclinación, velocidad horizontal y
altura al fallar, inclinación máxima y desviación máxima de yaw.

Uso (desde /mnt/d/TesisI):
    /mnt/d/venv_mamba/bin/python3 fallas/ver_vuelo_robusto.py --nivel 60 --episodios 10
    /mnt/d/venv_mamba/bin/python3 fallas/ver_vuelo_robusto.py --nivel 60 --modo estacionario --t-falla 5 --episodios 10
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from gym_pybullet_drones.control.RobustDSLPIDControl import RobustDSLPIDControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402
from comparar_base import (  # noqa: E402
    nueva_env, generar_waypoints,
    UMBRAL_WAYPOINT, CTRL_FREQ, DURACION_FALLA_SEG,
    Z_SUELO, MOTOR_FALLA, MIN_RPM, MAX_RPM,
    es_caida, es_aterrizaje,
)

XY_LIM = 2.0
MIN_DIST = 0.8
T_FALLA_SEG = 3.0
HOVER_Z = 1.25  # m, altura de crucero donde ocurrieron las caídas


def escenario(rng):
    while True:
        a_xy = rng.uniform(-XY_LIM, XY_LIM, size=2)
        b_xy = rng.uniform(-XY_LIM, XY_LIM, size=2)
        if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
            break
    punto_A = np.array([a_xy[0], a_xy[1], Z_SUELO])
    punto_B = np.array([b_xy[0], b_xy[1], Z_SUELO])
    return punto_A, punto_B, generar_waypoints(punto_A, punto_B)


def volar(env, punto_A, punto_B, waypoints, fault_pct, fault_aware, t_falla_seg, estacionario):
    severidad = 1.0 - fault_pct / 100.0
    t_falla_paso = int(t_falla_seg * CTRL_FREQ)
    env.INIT_XYZS = punto_A.reshape(1, 3)
    ctrl = RobustDSLPIDControl(drone_model=DroneModel.CF2X, fault_aware=fault_aware)
    obs, _ = env.reset()
    action = np.zeros((1, 4))
    wp_idx = 0
    llego = False
    causa = "tiempo"
    paso = 0
    pos_prev = obs[0][0:3].copy()
    yaw0 = obs[0][9]
    dist_recorrida = 0.0
    max_inclinacion = 0.0
    max_desv_yaw = 0.0
    incl_falla = 0.0
    vel_falla = 0.0
    alt_falla = 0.0

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        estado = obs[0]
        pos = estado[0:3]
        dist_recorrida += np.linalg.norm(pos - pos_prev)
        pos_prev = pos.copy()
        max_inclinacion = max(max_inclinacion, abs(estado[7]), abs(estado[8]))
        max_desv_yaw = max(max_desv_yaw, abs(estado[9] - yaw0))
        if paso == t_falla_paso:
            incl_falla = np.degrees(max(abs(estado[7]), abs(estado[8])))
            vel_falla = np.linalg.norm(estado[10:12])
            alt_falla = pos[2]

        wp_actual = waypoints[min(wp_idx, len(waypoints) - 1)]
        factors = np.ones(4)
        if paso >= t_falla_paso:
            factors[MOTOR_FALLA] = severidad
        ctrl.set_motor_effectiveness(factors)
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=estado, target_pos=wp_actual, target_rpy=np.zeros(3),
        )
        rpm = np.clip(rpm, MIN_RPM, MAX_RPM)
        if paso >= t_falla_paso:
            rpm[MOTOR_FALLA] *= severidad
        action[0, :] = np.clip(rpm, MIN_RPM, MAX_RPM)
        time.sleep(1.0 / CTRL_FREQ)

        if es_caida(obs, pos, paso):
            causa = "caida"
            break
        if es_aterrizaje(obs, pos, paso):
            causa = "aterrizaje"
            break
        if not estacionario and np.linalg.norm(wp_actual - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                llego = np.linalg.norm(punto_B - pos) < UMBRAL_WAYPOINT
                causa = "llego" if llego else "aterrizaje"
                break
        if term or trunc:
            causa = "trunc"
            break

    if estacionario and causa == "tiempo":
        llego = True  # sobrevivió: se mantuvo en el aire toda la duración
    return {
        "llego": llego,
        "causa": causa,
        "pasos": paso + 1,
        "dist_recorrida": dist_recorrida,
        "dist_AB": np.linalg.norm(punto_B[:2] - punto_A[:2]),
        "incl_al_fallar_deg": incl_falla,
        "vel_xy_al_fallar": vel_falla,
        "altura_al_fallar": alt_falla,
        "max_inclinacion_deg": np.degrees(max_inclinacion),
        "max_desv_yaw_deg": np.degrees(max_desv_yaw),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nivel", type=float, default=60.0,
                        help="Pérdida de RPM del motor en %% (ej: 2 a 10)")
    parser.add_argument("--modo", type=str, default="viaje", choices=["viaje", "estacionario"],
                        help="viaje: A -> B. estacionario: mantener altura fija sobre A, sin avanzar")
    parser.add_argument("--t-falla", type=float, default=T_FALLA_SEG,
                        help="Instante de la falla en segundos")
    parser.add_argument("--episodios", type=int, default=1,
                        help="Vuelos seguidos en la misma ventana")
    parser.add_argument("--semilla", type=int, default=0,
                        help="Semilla de los escenarios (mismo número = mismos puntos A, B y falla)")
    parser.add_argument("--fault-aware", action="store_true",
                        help="El controlador conoce la falla (información privilegiada)")
    args = parser.parse_args()

    estacionario = args.modo == "estacionario"
    rng = np.random.default_rng(args.semilla)
    escenarios = [escenario(rng) for _ in range(args.episodios)]

    print(f"Robusto | modo={args.modo} | falla {args.nivel:.1f}% a los {args.t_falla}s | "
          f"fault_aware={args.fault_aware} | episodios={args.episodios} | semilla={args.semilla}")
    env = nueva_env(escenarios[0][0], gui=True)
    causas = Counter()
    llegaron = 0
    etiqueta_ok = "sobrevivió" if estacionario else "llegó"
    for ep, (punto_A, punto_B, waypoints) in enumerate(escenarios, start=1):
        if estacionario:
            waypoints = [np.array([punto_A[0], punto_A[1], HOVER_Z])]
        r = volar(env, punto_A, punto_B, waypoints, args.nivel, args.fault_aware, args.t_falla, estacionario)
        llegaron += r["llego"]
        causas[r["causa"]] += 1
        print(f"  Episodio {ep:2d}/{args.episodios} | {etiqueta_ok if r['llego'] else 'no ' + etiqueta_ok:<12} "
              f"| causa={r['causa']:<10} | pasos={r['pasos']:4d} "
              f"| recorrido={r['dist_recorrida']:.2f} m (A-B {r['dist_AB']:.2f} m) "
              f"| incl. al fallar={r['incl_al_fallar_deg']:5.1f}° "
              f"| vel. xy al fallar={r['vel_xy_al_fallar']:4.2f} m/s | altura al fallar={r['altura_al_fallar']:4.2f} m "
              f"| incl. máx={r['max_inclinacion_deg']:5.1f}° | desv. yaw máx={r['max_desv_yaw_deg']:5.1f}°")
    env.close()
    print(f"Resumen: {etiqueta_ok} {llegaron}/{args.episodios} | causas: {dict(causas)}")


if __name__ == "__main__":
    main()
