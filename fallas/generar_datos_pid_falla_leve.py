
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

#from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  # noqa: E402
from gym_pybullet_drones.control.RobustDSLPIDControl import RobustDSLPIDControl 
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402
from comparar_base import (  # noqa: E402
    nueva_env, generar_waypoints,
    INPUT_COLS, UMBRAL_WAYPOINT, CTRL_FREQ, DURACION_FALLA_SEG,
    Z_SUELO, MOTOR_FALLA, MIN_RPM, MAX_RPM,
    es_caida, es_aterrizaje,
)

XY_LIM   = 2.0
MIN_DIST = 0.8
NIVELES_2A20 = [2, 5, 10, 15, 20]
NIVELES_2A50 = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
NIVELES_SIN_FALLA = [0]

# Reglas de fase y distancia idénticas a datos_CF2X_800ep.csv (verificadas contra ese archivo)
Z_OBJETIVO_ATERRIZAJE = 0.5
Z_FASE_CRUCERO = 1.1003

OUTPUT_FILE_2A20 = _ROOT / "results" / "datos_robust_pid_falla_2a20.csv"
OUTPUT_FILE_2A50 = _ROOT / "results" / "datos_robust_pid_falla_2a50.csv"
OUTPUT_FILE_NOMINAL = _ROOT / "results" / "datos_robust_pid_nominal.csv"
MOTOR_COLS  = ["motor_0", "motor_1", "motor_2", "motor_3"]
COLUMNAS    = INPUT_COLS + MOTOR_COLS + ["fase", "episodio", "paso", "distancia_objetivo",
                                         "falla_activa", "fault_pct"]


def fase_de(pos, wp_actual):
    if wp_actual[2] < Z_OBJETIVO_ATERRIZAJE:
        return 2
    if pos[2] < Z_FASE_CRUCERO:
        return 0
    return 1


def volar_episodio(punto_A, punto_B, waypoints, fault_pct, episodio, gui=False):
    """La falla se activa a la mitad del tramo de crucero: exige estar en fase
    de crucero (fase_de()==1, altura real de crucero, no por tiempo fijo) Y
    haber avanzado ya la mitad de los waypoints de crucero -- asi nunca ocurre
    en despegue, justo al entrar a crucero, ni en aterrizaje."""
    severidad = 1.0 - fault_pct / 100.0
    # Waypoints de crucero: todos menos el ultimo (que es el aterrizaje en B).
    wp_medio_crucero = max(1, (len(waypoints) - 2) // 2)
    env  = nueva_env(punto_A, gui=gui)
    ctrl = RobustDSLPIDControl(drone_model=DroneModel.CF2X, fault_aware=True)
    obs, _ = env.reset()
    estado = obs[0]
    wp_idx = 0
    llego = False
    falla_activada = False
    paso_falla = None
    filas = []

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        wp_actual = waypoints[min(wp_idx, len(waypoints) - 1)]
        en_crucero = fase_de(estado[0:3], wp_actual) == 1
        if not falla_activada and en_crucero and wp_idx >= wp_medio_crucero:
            falla_activada = True
            paso_falla = paso
        factors = np.ones(4)
        if falla_activada:
            factors[MOTOR_FALLA] = severidad
        ctrl.set_motor_effectiveness(factors)
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=estado, target_pos=wp_actual, target_rpy=np.zeros(3),
        )
        rpm = np.clip(rpm, MIN_RPM, MAX_RPM)
        cmd_pid = rpm.copy()

        falla_activa = int(fault_pct > 0 and falla_activada)
        action = cmd_pid.copy()
        if falla_activa:
            action[MOTOR_FALLA] *= severidad
        action = np.clip(action, MIN_RPM, MAX_RPM)

        # Aplicar la accion AHORA y loguear el estado resultante (en sincronia con cmd_pid)
        obs, _, term, trunc, _ = env.step(action.reshape(1, 4))
        if gui:
            time.sleep(1.0 / CTRL_FREQ)
        estado = obs[0]
        pos = estado[0:3]; rpy = estado[7:10]
        vel = estado[10:13]; ang = estado[13:16]

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

        if es_caida(obs, pos, paso, wp_idx, len(waypoints)) or es_aterrizaje(obs, pos, paso):
            break
        if np.linalg.norm(wp_actual - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                llego_a_B = np.linalg.norm(punto_B - pos) < UMBRAL_WAYPOINT
                llego = llego_a_B and falla_activada  # exigir >=1 paso con la falla activa
                break
        if term or trunc:
            break

    env.close()
    pasos_post_falla = max(0, paso + 1 - paso_falla) if paso_falla is not None else 0
    return llego, filas, pasos_post_falla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objetivo", type=int, default=800,
                        help="Episodios exitosos a reunir (default 800)")
    parser.add_argument("--max-intentos", type=int, default=5000)
    parser.add_argument("--sin-falla", action="store_true",
                        help="Vuelos nominales sin falla (fault_pct=0), salida separada")
    parser.add_argument("--episodio-inicio", type=int, default=None,
                        help="Primer id de episodio (default 0, o 800 con --sin-falla)")
    parser.add_argument("--niveles-falla", type=str, default="2a50", choices=["2a20", "2a50"],
                        help="Rango de falla: 2a20 (2,5,10,15,20%%) o 2a50 (2..50%%, default)")
    parser.add_argument("--sufijo-salida", type=str, default=None,
                        help="Si se da, el archivo de salida se renombra a "
                             "<nombre_original>_<sufijo>.csv en vez de sobreescribir el default "
                             "(util para pruebas sin perder el dataset actual)")
    parser.add_argument("--gui", action="store_true",
                        help="Muestra cada vuelo en PyBullet, a velocidad real (usar con "
                             "--objetivo y --max-intentos chicos, es mucho mas lento)")
    parser.add_argument("--nivel", type=int, default=None,
                        help="Solo este nivel de falla en %% (ej: 20), ignora --niveles-falla")
    args = parser.parse_args()
    inicio = args.episodio_inicio if args.episodio_inicio is not None else (800 if args.sin_falla else 0)

    if args.sin_falla:
        niveles = NIVELES_SIN_FALLA
        output_file = OUTPUT_FILE_NOMINAL
        print(f"Reuniendo {args.objetivo} episodios PID exitosos SIN falla")
    elif args.nivel is not None:
        niveles = [args.nivel]
        output_file = OUTPUT_FILE_2A20 if args.niveles_falla == "2a20" else OUTPUT_FILE_2A50
        print(f"Reuniendo {args.objetivo} episodios PID exitosos con falla fija {args.nivel}% "
              f"a la mitad del crucero")
    else:
        niveles = NIVELES_2A20 if args.niveles_falla == "2a20" else NIVELES_2A50
        output_file = OUTPUT_FILE_2A20 if args.niveles_falla == "2a20" else OUTPUT_FILE_2A50
        print(f"Reuniendo {args.objetivo} episodios PID exitosos con falla "
              f"{niveles}% a la mitad del crucero")
    if args.sufijo_salida:
        output_file = output_file.with_name(f"{output_file.stem}_{args.sufijo_salida}{output_file.suffix}")
    print(f"Salida: {output_file}")
    print("-" * 60)

    todas_las_filas = []
    exitos = 0
    intentos_por_nivel = {n: [0, 0] for n in niveles}
    # Todo "llego" ya exige >=1 paso con la falla activa (ver volar_episodio),
    # asi que esto solo registra cuanto tiempo post-falla tuvo cada exito.
    tiempos_post_falla_por_nivel = {n: [] for n in niveles}

    for intento in range(args.max_intentos):
        if exitos >= args.objetivo:
            break
        nivel = niveles[intento % len(niveles)]

        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            b_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
                break
        punto_A = np.array([a_xy[0], a_xy[1], Z_SUELO])
        punto_B = np.array([b_xy[0], b_xy[1], Z_SUELO])
        waypoints = generar_waypoints(punto_A, punto_B)

        llego, filas, pasos_post_falla = volar_episodio(
            punto_A, punto_B, waypoints, nivel, inicio + exitos, gui=args.gui)
        intentos_por_nivel[nivel][0] += 1
        if llego:
            intentos_por_nivel[nivel][1] += 1
            todas_las_filas.extend(filas)
            exitos += 1
            tiempos_post_falla_por_nivel[nivel].append(pasos_post_falla / CTRL_FREQ)

        print(f"  Intento {intento+1:4d} | falla={nivel:2d}% | "
              f"{'llegó ✓' if llego else 'no llegó'} | "
              f"exitos={exitos}/{args.objetivo}")

    df = pd.DataFrame(todas_las_filas, columns=COLUMNAS)
    df.to_csv(output_file, index=False)

    print("\n" + "-" * 60)
    print(f"Guardado {len(df)} filas, {exitos} episodios exitosos en {output_file}")
    for nivel, (total, ok) in intentos_por_nivel.items():
        pct = 100 * ok / total if total else 0.0
        tiempos = tiempos_post_falla_por_nivel[nivel]
        t_medio = np.mean(tiempos) if tiempos else 0.0
        print(f"  {nivel:2d}%: {ok}/{total} llegaron ({pct:.0f}%) | "
              f"tiempo medio post-falla: {t_medio:.2f}s")
    if exitos < args.objetivo:
        print(f"Aviso: se alcanzó --max-intentos sin reunir {args.objetivo} exitosos.")


if __name__ == "__main__":
    main()
