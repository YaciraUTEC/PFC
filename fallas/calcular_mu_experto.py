"""
Calcula las expectativas de características del experto (PID) para
Apprenticeship Learning / Feature Expectation Matching (Abbeel & Ng, 2004),
sobre los 800 episodios de results/datos_CF2X_800ep.csv.

No simula nada: reconstruye phi(s,a) directamente de las columnas ya
registradas durante la generación de datos (pos, rpy, ang_vel, vel_z,
err -> waypoint actual, motor_0..3).

Uso (venv_mamba):
    cd /mnt/d/TesisI/gym-pybullet-drones
    /mnt/d/venv_mamba/bin/python3 fallas/calcular_mu_experto.py

Salida: results/mu_experto.npy (vector de 8 features, promedio sobre episodios
del retorno descontado de phi bajo la política del PID).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import cargar_stats, STATS_PATH  # noqa: E402
from irl_features import (  # noqa: E402
    phi, distancia_z, cargar_escalas, N_FEATURES, FEATURE_NAMES,
)

CSV_PATH    = _ROOT / "results" / "datos_CF2X_800ep.csv"
OUT_PATH    = _ROOT / "results" / "mu_experto.npy"
ESCALA_PATH = _ROOT / "results" / "mu_escala.npy"
GAMMA       = 0.99  # igual que gamma en entrenar_rl.py (PPO)

# Escala a nivel de mu (no de phi): al acumular phi con descuento sobre un
# episodio completo, features que oscilan alrededor de 0 (ej. progreso) se
# cancelan y quedan chicas, mientras que features siempre negativas (ej.
# estabilidad_altura) se acumulan sin cancelación y quedan grandes — aunque
# cada una ya esté escalada por su propia std a nivel de un solo paso
# (irl_features.cargar_escalas). Sin esto, el margen y la proyección quedan
# dominados por las features de mayor magnitud acumulada, casi ignorando
# balance_motores/progreso. EPS_ESCALA evita dividir por un valor casi cero
# si alguna componente de mu_experto queda muy chica.
EPS_ESCALA = 0.01


def main():
    stats   = cargar_stats(STATS_PATH)
    escalas = cargar_escalas(stats)

    print(f"Cargando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    episodios = df["episodio"].unique()
    print(f"  {len(df):,} filas | {len(episodios)} episodios")

    retornos = []
    for ep in episodios:
        ep_df = df[df["episodio"] == ep].reset_index(drop=True)
        pos     = ep_df[["pos_x", "pos_y", "pos_z"]].values
        rpy     = ep_df[["roll", "pitch", "yaw"]].values
        ang_vel = ep_df[["ang_x", "ang_y", "ang_z"]].values
        vel_z   = ep_df["vel_z"].values
        err     = ep_df[["err_x", "err_y", "err_z"]].values
        rpm     = ep_df[["motor_0", "motor_1", "motor_2", "motor_3"]].values
        wp_actual = pos + err  # wp_actual = pos + (wp_actual - pos)

        # progreso = 0 en el primer paso del episodio
        dist_prev_z = distancia_z(pos[0], wp_actual[0], escalas)

        acumulado = np.zeros(N_FEATURES, dtype=np.float64)
        for t in range(len(ep_df)):
            vec, dist_prev_z = phi(
                pos[t], rpy[t], ang_vel[t], vel_z[t],
                wp_actual[t], rpm[t], dist_prev_z, escalas,
            )
            acumulado += (GAMMA ** t) * vec
        retornos.append(acumulado)

    mu_experto = np.mean(retornos, axis=0)
    np.save(OUT_PATH, mu_experto)

    mu_escala = np.maximum(np.abs(mu_experto), EPS_ESCALA)
    np.save(ESCALA_PATH, mu_escala)

    print("\nmu_experto (expectativas de características del PID):")
    for name, val, esc in zip(FEATURE_NAMES, mu_experto, mu_escala):
        print(f"  {name:24s} {val:+9.4f}   escala={esc:.4f}")
    print(f"\nGuardado en {OUT_PATH}")
    print(f"Escala guardada en {ESCALA_PATH} (usada por entrenar_irl_apprenticeship.py "
          f"para que las 8 features pesen comparablemente en el margen)")


if __name__ == "__main__":
    main()
