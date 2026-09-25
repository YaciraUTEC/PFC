"""
Histograma de los valores crudos (sin acotar) de las 6 features de phi() que
se limitan con max(valor, LIMITE), calculados sobre TODOS los pasos de los
800 episodios del PID (no solo el promedio por episodio, como mu_experto).
Sirve para calibrar LIMITE_FEATURE / LIMITE_FEATURE_RUTA con datos reales en
vez de una convención arbitraria -- el objetivo es que ningún paso real del
PID quede saturado (aplastado al mismo valor), porque eso hace que el experto
y una candidata parezcan idénticos en esa dimensión sin serlo (ver diagnóstico
de prueba_08 en results/pruebas_irl/).

Uso: python fallas/histograma_limites_features.py
Salida: results/histograma_limites_features.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from irl_features import (  # noqa: E402
    cargar_escalas, distancia_z, HOVER_RPM, DT, G,
    LIMITE_FEATURE, LIMITE_FEATURE_RUTA,
)

STATS_PATH = _ROOT / "results" / "stats_normalizacion.json"
CSV_PATH = _ROOT / "results" / "datos_CF2X_800ep.csv"
OUT_PATH = _ROOT / "results" / "histograma_limites_features.png"


def cargar_stats(path):
    with open(path) as f:
        data = json.load(f)
    return {k: (v[0], v[1]) for k, v in data.items()}


def main():
    stats = cargar_stats(STATS_PATH)
    escalas = cargar_escalas(stats)

    print(f"Cargando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    episodios = df["episodio"].unique()
    print(f"  {len(df):,} filas | {len(episodios)} episodios")

    nombres = [
        "proximidad_objetivo", "estabilidad_altura", "estabilidad_angular_rp",
        "velocidad", "oscilacion", "aceleracion_vertical",
    ]
    valores = {n: [] for n in nombres}

    for ep in episodios:
        ep_df = df[df["episodio"] == ep].reset_index(drop=True)
        pos = ep_df[["pos_x", "pos_y", "pos_z"]].values
        ang_vel = ep_df[["ang_x", "ang_y", "ang_z"]].values
        vel = ep_df[["vel_x", "vel_y", "vel_z"]].values
        err = ep_df[["err_x", "err_y", "err_z"]].values
        rpm = ep_df[["motor_0", "motor_1", "motor_2", "motor_3"]].values
        wp_actual = pos + err
        punto_final = wp_actual[-1]

        for t in range(len(ep_df)):
            rpm_anterior = rpm[t - 1] if t > 0 else np.full(4, HOVER_RPM, dtype=np.float64)
            vel_z_anterior = vel[t - 1, 2] if t > 0 else vel[t, 2]

            dist_actual_z = distancia_z(pos[t], punto_final, escalas)
            ang_rp_n = np.asarray(ang_vel[t][:2]) / escalas["ang_rp"]
            vel_n = np.asarray(vel[t]) / escalas["vel"]
            rpm_delta_n = (np.asarray(rpm[t]) - np.asarray(rpm_anterior)) / escalas["motor"]
            accel_vertical = (float(vel[t, 2]) - float(vel_z_anterior)) / DT

            valores["proximidad_objetivo"].append(-dist_actual_z)
            valores["estabilidad_altura"].append(
                -abs((float(wp_actual[t, 2]) - float(pos[t, 2])) / np.mean(escalas["err"]))
            )
            valores["estabilidad_angular_rp"].append(-float(np.linalg.norm(ang_rp_n)))
            valores["velocidad"].append(-float(np.linalg.norm(vel_n)))
            valores["oscilacion"].append(-float(np.mean(np.abs(rpm_delta_n))))
            valores["aceleracion_vertical"].append(-abs(accel_vertical / G))

    limites = {
        "proximidad_objetivo": LIMITE_FEATURE_RUTA,
        "estabilidad_altura": LIMITE_FEATURE_RUTA,
        "estabilidad_angular_rp": LIMITE_FEATURE,
        "velocidad": LIMITE_FEATURE,
        "oscilacion": LIMITE_FEATURE,
        "aceleracion_vertical": LIMITE_FEATURE,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, nombre in zip(axes.flat, nombres):
        v = np.array(valores[nombre])
        lim = limites[nombre]
        pct_saturado = 100.0 * np.mean(v <= lim)
        ax.hist(v, bins=80, color="steelblue", edgecolor="none")
        ax.axvline(lim, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"límite = {lim:.1f}\n({pct_saturado:.1f}% del PID lo supera)")
        ax.set_title(nombre)
        ax.set_xlabel("valor crudo (sin acotar)")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Distribución de valores crudos de φ sobre los 800 episodios del PID\n"
                 "(cada paso de cada episodio) vs. los pisos LIMITE_FEATURE / LIMITE_FEATURE_RUTA",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=130)
    print(f"\nGuardado en {OUT_PATH}")

    print("\n% de pasos del PID que superan (serían acotados por) el límite:")
    for nombre in nombres:
        v = np.array(valores[nombre])
        lim = limites[nombre]
        print(f"  {nombre:24s} límite={lim:6.1f}  min={v.min():9.3f}  "
              f"media={v.mean():8.3f}  % acotado={100.0*np.mean(v<=lim):5.2f}%")


if __name__ == "__main__":
    main()
