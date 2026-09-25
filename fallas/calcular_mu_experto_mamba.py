"""
Calcula µ_experto a partir de 800 episodios de vuelo de Mamba SIN falla.

Similar a calcular_mu_experto.py pero usa el modelo Mamba entrenado en lugar
de leer de un CSV de PID. Esto sirve para comparar resultados de IRL cuando
el experto es Mamba vs cuando es PID.
"""
import sys
from pathlib import Path
from collections import deque
import numpy as np
import torch
import json

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import (  # noqa: E402
    cargar_stats, normalizar_estado, desnormalizar_accion,
    generar_waypoints, nueva_env,
    STATS_PATH, Z_SUELO, UMBRAL_WAYPOINT, DURACION_SEG,
    CTRL_FREQ, HOVER_RPM, MIN_RPM, MAX_RPM, INPUT_COLS,
    es_caida, es_aterrizaje,
    MambaDrone,
)
from irl_features import (  # noqa: E402
    phi, distancia_z, cargar_escalas, N_FEATURES, FEATURE_NAMES,
    horizonte_efectivo,
)

MAMBA_MODEL_PATH = _ROOT / "results" / "modelo_mamba.pth"
OUT_PATH = _ROOT / "results" / "mu_experto_mamba.npy"
ESCALA_PATH = _ROOT / "results" / "mu_escala_mamba.npy"

GAMMA = 0.99
EPS_ESCALA = 0.01
N_EPISODIOS = 800
XY_LIM = 2.0
ANGULO_CRASH = np.radians(35)


def es_caida_simple(rpy, vel, pos, paso):
    if pos[2] >= 0.05 or paso <= 10:
        return False
    actitud_critica = abs(rpy[0]) > ANGULO_CRASH or abs(rpy[1]) > ANGULO_CRASH
    cayendo = vel[2] < -0.5
    return actitud_critica or cayendo


def main():
    print(f"Calculando µ_experto de Mamba ({N_EPISODIOS} episodios sin falla)...")

    if not MAMBA_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo Mamba no encontrado en {MAMBA_MODEL_PATH}")

    stats = cargar_stats(STATS_PATH)
    escalas = cargar_escalas(stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo Mamba
    model = MambaDrone()
    model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Modelo Mamba cargado desde {MAMBA_MODEL_PATH}")

    retornos = []
    env = None

    for ep_idx in range(N_EPISODIOS):
        # Generar trayectoria A->B aleatoria
        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            dist_deseada = np.random.uniform(1.0, 2 * XY_LIM * (2 ** 0.5))
            angulo = np.random.uniform(0, 2 * np.pi)
            b_xy = a_xy + dist_deseada * np.array([np.cos(angulo), np.sin(angulo)])
            if np.all(np.abs(b_xy) <= XY_LIM):
                break

        punto_A = np.array([a_xy[0], a_xy[1], Z_SUELO])
        punto_B = np.array([b_xy[0], b_xy[1], Z_SUELO])
        waypoints = generar_waypoints(punto_A, punto_B)

        if env is None:
            env = nueva_env(punto_A, gui=False)
        else:
            env.INIT_XYZS = punto_A.reshape(1, 3)

        obs_raw, _ = env.reset()
        ventana = deque(
            [np.zeros(len(INPUT_COLS), dtype=np.float32)] * 50, maxlen=50
        )

        wp_idx = 0
        paso = 0
        rpm_anterior = np.ones(4, dtype=np.float64) * HOVER_RPM
        vel_z_anterior = float(obs_raw[0][12])
        dist_prev_z = distancia_z(obs_raw[0][0:3], punto_B, escalas)

        acumulado = np.zeros(N_FEATURES, dtype=np.float64)
        max_pasos = int(DURACION_SEG * CTRL_FREQ)

        done = False
        while not done and paso < max_pasos:
            pos = obs_raw[0][0:3].copy()
            rpy = obs_raw[0][7:10]
            ang_vel = obs_raw[0][13:16]
            vel = obs_raw[0][10:13]

            ta = waypoints[min(wp_idx, len(waypoints) - 1)]
            ventana.append(normalizar_estado(obs_raw, ta, stats))

            # Mamba genera acción
            x = torch.tensor(
                np.array(ventana), dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x).cpu().numpy()[0]
            rpm = np.clip(desnormalizar_accion(pred, stats), MIN_RPM, MAX_RPM)

            # Paso en simulación
            es_caida_ahora = es_caida_simple(rpy, vel, pos, paso)
            vec, dist_prev_z = phi(
                pos, rpy, ang_vel, vel, ta, punto_B, rpm,
                rpm_anterior, vel_z_anterior, dist_prev_z, escalas,
                es_caida_ahora=es_caida_ahora,
            )
            acumulado += (GAMMA ** paso) * vec

            obs_raw, _, term, trunc, _ = env.step(rpm.reshape(1, 4))
            paso += 1
            rpm_anterior = rpm.astype(np.float64)
            vel_z_anterior = float(vel[2])

            # Waypoint alcanzado
            if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
                wp_idx += 1
                if wp_idx >= len(waypoints):
                    done = True

            # Caída o timeout
            if es_caida_ahora or term or trunc:
                done = True

        acumulado = acumulado / horizonte_efectivo(paso, GAMMA)
        retornos.append(acumulado)

        if (ep_idx + 1) % 100 == 0:
            print(f"  Episodio {ep_idx + 1}/{N_EPISODIOS} completado ({paso} pasos)")

    if env is not None:
        env.close()

    retornos = np.array(retornos)
    mu_experto = np.mean(retornos, axis=0)
    np.save(OUT_PATH, mu_experto)

    mu_escala = np.maximum(np.std(retornos, axis=0), EPS_ESCALA)
    np.save(ESCALA_PATH, mu_escala)

    print("\nµ_experto (expectativas de características de Mamba SIN FALLA):")
    for name, val, esc in zip(FEATURE_NAMES, mu_experto, mu_escala):
        print(f"  {name:24s} {val:+9.4f}   escala={esc:.4f}")
    print(f"\nGuardado en {OUT_PATH}")
    print(f"Escala guardada en {ESCALA_PATH}")


if __name__ == "__main__":
    main()
