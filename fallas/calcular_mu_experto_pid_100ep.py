"""
Calcula µ_experto del PID usando solo los primeros 100 episodios del CSV.
Permite comparación justa: PID (100 ep) vs Mamba (100 ep).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from irl_features import (  # noqa: E402
    phi, distancia_z, cargar_escalas, N_FEATURES, FEATURE_NAMES,
    horizonte_efectivo, HOVER_RPM,
)
from comparar_base import cargar_stats, STATS_PATH  # noqa: E402

CSV_PATH = _ROOT / "results" / "datos_CF2X_800ep.csv"
OUT_PATH = _ROOT / "results" / "mu_experto_pid_100ep.npy"
ESCALA_PATH = _ROOT / "results" / "mu_escala_pid_100ep.npy"

GAMMA = 0.99
EPS_ESCALA = 0.01
N_EPISODIOS = 100
ANGULO_CRASH = np.radians(35)


def es_caida_simple(rpy, vel, pos, paso):
    if pos[2] >= 0.05 or paso <= 10:
        return False
    actitud_critica = abs(rpy[0]) > ANGULO_CRASH or abs(rpy[1]) > ANGULO_CRASH
    cayendo = vel[2] < -0.5
    return actitud_critica or cayendo


def cargar_stats_fn(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return {k: (v[0], v[1]) for k, v in data.items()}


def main():
    print(f"Calculando µ_experto del PID (primeros {N_EPISODIOS} episodios)...")

    stats = cargar_stats_fn(STATS_PATH)
    escalas = cargar_escalas(stats)

    print(f"Cargando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    episodios = sorted(df["episodio"].unique())[:N_EPISODIOS]
    print(f"  Usando episodios: 0 a {N_EPISODIOS - 1}")

    retornos = []
    for ep_idx, ep in enumerate(episodios):
        ep_df = df[df["episodio"] == ep].reset_index(drop=True)

        pos = ep_df[["pos_x", "pos_y", "pos_z"]].values
        rpy = ep_df[["roll", "pitch", "yaw"]].values
        ang_vel = ep_df[["ang_x", "ang_y", "ang_z"]].values
        vel = ep_df[["vel_x", "vel_y", "vel_z"]].values
        err = ep_df[["err_x", "err_y", "err_z"]].values
        rpm = ep_df[["motor_0", "motor_1", "motor_2", "motor_3"]].values
        wp_actual = pos + err
        punto_final = wp_actual[-1]

        dist_prev_z = distancia_z(pos[0], punto_final, escalas)

        acumulado = np.zeros(N_FEATURES, dtype=np.float64)
        for t in range(len(ep_df)):
            rpm_anterior = rpm[t - 1] if t > 0 else np.full(4, HOVER_RPM, dtype=np.float64)
            vel_z_anterior = vel[t - 1, 2] if t > 0 else vel[t, 2]
            es_caida_ahora = es_caida_simple(rpy[t], vel[t], pos[t], t)
            vec, dist_prev_z = phi(
                pos[t], rpy[t], ang_vel[t], vel[t], wp_actual[t], punto_final, rpm[t],
                rpm_anterior, vel_z_anterior, dist_prev_z, escalas,
                es_caida_ahora=es_caida_ahora,
            )
            acumulado += (GAMMA ** t) * vec

        acumulado = acumulado / horizonte_efectivo(len(ep_df), GAMMA)
        retornos.append(acumulado)

        if (ep_idx + 1) % 20 == 0:
            print(f"  Episodio {ep_idx + 1}/{N_EPISODIOS} procesado")

    retornos = np.array(retornos)
    mu_experto = np.mean(retornos, axis=0)
    np.save(OUT_PATH, mu_experto)

    mu_escala = np.maximum(np.std(retornos, axis=0), EPS_ESCALA)
    np.save(ESCALA_PATH, mu_escala)

    print("\nµ_experto (PID, 100 episodios):")
    for name, val, esc in zip(FEATURE_NAMES, mu_experto, mu_escala):
        print(f"  {name:24s} {val:+9.4f}   escala={esc:.4f}")
    print(f"\nGuardado en {OUT_PATH}")
    print(f"Escala guardada en {ESCALA_PATH}")


if __name__ == "__main__":
    main()
