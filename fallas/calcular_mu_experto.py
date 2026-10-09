
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


ANGULO_CRASH = np.radians(35)  # igual que comparar_base.ANGULO_CRASH
Z_CRUCERO = 1.2  # altura de crucero nominal


def detectar_fase_csv(pos, punto_final):
    """Detecta fase basado en altura y proximidad al destino"""
    altura = pos[2]
    distancia_xy = np.linalg.norm(pos[:2] - punto_final[:2])

    if altura < 0.5:
        return 0  # despegue
    elif altura < Z_CRUCERO * 0.5 or (altura < 1.0 and distancia_xy < 0.5):
        return 2  # aterrizaje
    else:
        return 1  # crucero


def es_caida_phase_aware(rpy, vel, pos, paso, fase):
    """Detección de crash con thresholds por fase"""
    if pos[2] >= 0.05 or paso <= 10:
        return False

    # Thresholds por fase
    if fase == 0:  # DESPEGUE
        angulo_max = np.radians(25)
        vel_max = -0.3
    elif fase == 2:  # ATERRIZAJE
        angulo_max = np.radians(20)
        vel_max = -0.4
    else:  # CRUCERO
        angulo_max = ANGULO_CRASH  # 35°
        vel_max = -0.5

    actitud_critica = abs(rpy[0]) > angulo_max or abs(rpy[1]) > angulo_max
    cayendo = vel[2] < vel_max
    return actitud_critica or cayendo


from irl_features import (  # noqa: E402
    phi, distancia_z, cargar_escalas, N_FEATURES, FEATURE_NAMES, HOVER_RPM,
    horizonte_efectivo,
)

CSV_PATH    = _ROOT / "results" / "datos_CF2X_800ep.csv"
OUT_PATH    = _ROOT / "results" / "mu_experto.npy"
ESCALA_PATH = _ROOT / "results" / "mu_escala.npy"
GAMMA       = 0.99  # igual que gamma en entrenar_rl.py (PPO)
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
            fase = detectar_fase_csv(pos[t], punto_final)
            es_caida_ahora = es_caida_phase_aware(rpy[t], vel[t], pos[t], t, fase)
            vec, dist_prev_z = phi(
                pos[t], rpy[t], ang_vel[t], vel[t], wp_actual[t], punto_final, rpm[t],
                rpm_anterior, vel_z_anterior, dist_prev_z, escalas,
                es_caida_ahora=es_caida_ahora,
            )
            acumulado += (GAMMA ** t) * vec
        acumulado = acumulado / horizonte_efectivo(len(ep_df), GAMMA)
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
