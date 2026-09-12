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

Salida: results/mu_experto.npy (vector de 7 features, promedio sobre episodios
del retorno descontado de phi bajo la política del PID).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

STATS_PATH = _ROOT / "results" / "stats_normalizacion.json"


def cargar_stats(path):
    with open(path) as f:
        data = json.load(f)
    return {k: (v[0], v[1]) for k, v in data.items()}


from irl_features import (  # noqa: E402
    phi, distancia_z, cargar_escalas, N_FEATURES, FEATURE_NAMES, HOVER_RPM,
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
# (irl_features.cargar_escalas).
#
# mu_escala = desviación estándar de esa feature ENTRE LOS 800 EPISODIOS (no
# el propio valor de mu_experto). Dividir por la propia magnitud de mu_experto
# (versión anterior) colapsaba mu_experto_r a exactamente ±1 en todas las
# componentes siempre -- borraba toda la información de qué tan fuerte o débil
# es cada feature en el comportamiento del experto, tratando una señal grande
# y consistente igual que una chica y ruidosa. Escalar por la variabilidad
# entre episodios preserva esa diferencia: una feature con promedio grande y
# poca variación entre episodios (señal fuerte y confiable del PID) queda con
# |mu_r| grande; una con promedio chico o muy variable entre episodios queda
# con |mu_r| chico. EPS_ESCALA evita dividir por casi cero si alguna
# componente varía muy poco entre episodios.
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
        vel     = ep_df[["vel_x", "vel_y", "vel_z"]].values
        err     = ep_df[["err_x", "err_y", "err_z"]].values
        rpm     = ep_df[["motor_0", "motor_1", "motor_2", "motor_3"]].values
        wp_actual   = pos + err  # wp_actual = pos + (wp_actual - pos), waypoint local por fila
        punto_final = wp_actual[-1]  # último target del episodio = destino real (B_suelo)

        # progreso = 0 en el primer paso del episodio (respecto al destino final)
        dist_prev_z = distancia_z(pos[0], punto_final, escalas)

        acumulado = np.zeros(N_FEATURES, dtype=np.float64)
        for t in range(len(ep_df)):
            rpm_anterior   = rpm[t - 1] if t > 0 else np.full(4, HOVER_RPM, dtype=np.float64)
            vel_z_anterior = vel[t - 1, 2] if t > 0 else vel[t, 2]
            vec, dist_prev_z = phi(
                pos[t], rpy[t], ang_vel[t], vel[t], wp_actual[t], punto_final, rpm[t],
                rpm_anterior, vel_z_anterior, dist_prev_z, escalas,
            )
            acumulado += (GAMMA ** t) * vec
        retornos.append(acumulado)

    retornos   = np.array(retornos)  # (n_episodios, N_FEATURES)
    mu_experto = np.mean(retornos, axis=0)
    np.save(OUT_PATH, mu_experto)

    mu_escala = np.maximum(np.std(retornos, axis=0), EPS_ESCALA)
    np.save(ESCALA_PATH, mu_escala)

    print("\nmu_experto (expectativas de características del PID):")
    for name, val, esc in zip(FEATURE_NAMES, mu_experto, mu_escala):
        print(f"  {name:24s} {val:+9.4f}   escala={esc:.4f}")
    print(f"\nGuardado en {OUT_PATH}")
    print(f"Escala guardada en {ESCALA_PATH} (usada por entrenar_irl_apprenticeship.py "
          f"para que las 7 features pesen comparablemente en el margen)")


if __name__ == "__main__":
    main()
