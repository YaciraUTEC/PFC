"""
Versión simplificada: usa el mismo NominalFlightEnv pero sin falla,
para generar µ_experto de Mamba.
"""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from nominal_flight_env import NominalFlightEnv  # noqa: E402
from irl_features import FEATURE_NAMES, N_FEATURES, horizonte_efectivo, cargar_escalas  # noqa: E402
from comparar_base import cargar_stats, STATS_PATH  # noqa: E402

OUT_PATH = _ROOT / "results" / "mu_experto_mamba.npy"
ESCALA_PATH = _ROOT / "results" / "mu_escala_mamba.npy"
N_EPISODIOS = 800
EPS_ESCALA = 0.01
GAMMA = 0.99


def main():
    print(f"Calculando µ_experto de Mamba ({N_EPISODIOS} episodios SIN FALLA)...")

    stats = cargar_stats(STATS_PATH)
    escalas = cargar_escalas(stats)

    # Crear env con Mamba, sin inyectar falla
    env = NominalFlightEnv(gui=False, modelo="mamba", t_falla_min=999, t_falla_max=999)

    retornos = []

    for ep_idx in range(N_EPISODIOS):
        obs, _ = env.reset()
        done = False
        acumulado = np.zeros(N_FEATURES, dtype=np.float64)
        t = 0
        info = {}

        # Rollout con modelo=None equivale a Mamba base (sin PPO delta)
        # pero NominalFlightEnv siempre usa Mamba, así que simplemente
        # dejamos que corra sin entrenar PPO (como evaluación de expert)
        while not done and t < 10000:
            # Acción nula para que NominalFlightEnv use solo Mamba
            accion = np.zeros(4, dtype=np.float32)
            obs, _, done, trunc, info = env.step(accion)
            acumulado += (GAMMA ** t) * info["phi"]
            t += 1
            if trunc:
                break

        acumulado = acumulado / horizonte_efectivo(t, GAMMA)
        retornos.append(acumulado)

        if (ep_idx + 1) % 10 == 0:
            print(f"  Episodio {ep_idx + 1}/{N_EPISODIOS} completado ({t} pasos)")

    env.close()

    retornos = np.array(retornos)
    mu_experto = np.mean(retornos, axis=0)
    np.save(OUT_PATH, mu_experto)

    mu_escala = np.maximum(np.std(retornos, axis=0), EPS_ESCALA)
    np.save(ESCALA_PATH, mu_escala)

    print("\nµ_experto (Mamba SIN FALLA, sin PPO delta):")
    for name, val, esc in zip(FEATURE_NAMES, mu_experto, mu_escala):
        print(f"  {name:24s} {val:+9.4f}   escala={esc:.4f}")
    print(f"\nGuardado en {OUT_PATH}")
    print(f"Escala guardada en {ESCALA_PATH}")


if __name__ == "__main__":
    main()
