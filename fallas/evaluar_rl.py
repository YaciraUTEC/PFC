"""
Evaluar el modelo PPO entrenado con pesos IRL.
Carga ppo_compensador_mamba_irl_10.zip y lo evalúa con falla a 3.0 segundos.
"""

import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from fault_env_residual_irl import FaultResidualEnvIRL

MODEL_PATH = str(_ROOT / "results" / "ppo_compensador_mamba_irl_10")
N_EPISODIOS_POR_SEVERIDAD = 10


def evaluar():
    print("=" * 70)
    print("EVALUACIÓN: PPO entrenado con pesos IRL (Mamba)")
    print("=" * 70)
    print(f"Falla: 3.0 segundos (fijo)")
    print()

    # Cargar modelo
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✓ Modelo cargado\n")
    except FileNotFoundError:
        print(f"✗ Modelo no encontrado: {MODEL_PATH}.zip")
        return

    # Ambiente con falla a 3.0s (fijo)
    env = FaultResidualEnvIRL(gui=True, modelo="mamba", t_falla_min=3.0, t_falla_max=3.0)

    # Evaluar en severidades variadas
    severidades = np.linspace(0.02, 0.80, 5)
    resultados = {}

    for sev in severidades:
        env.set_max_fault(sev)
        print(f"Severidad: {sev*100:.0f}%")

        outcomes = {"llego": 0, "aterrizo": 0, "cayo": 0, "tiempo": 0}
        duraciones = []

        for ep in range(N_EPISODIOS_POR_SEVERIDAD):
            obs, _ = env.reset()
            done = False
            paso = 0

            while not done:
                accion, _ = model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = env.step(accion)
                paso += 1
                if trunc:
                    break

            outcome = info.get("outcome", "tiempo")
            outcomes[outcome] += 1
            duraciones.append(paso)

        n = N_EPISODIOS_POR_SEVERIDAD
        recuperacion = 100 * (outcomes["llego"] + outcomes["aterrizo"]) / n
        crash = 100 * outcomes["cayo"] / n
        duracion = np.mean(duraciones)

        print(f"  Recuperación: {recuperacion:5.1f}% | Crash: {crash:5.1f}% | Duración: {duracion:.0f} pasos\n")
        resultados[sev] = recuperacion

    env.close()

    # Resumen
    print("=" * 70)
    recuperacion_promedio = np.mean(list(resultados.values()))
    print(f"Recuperación PROMEDIO: {recuperacion_promedio:.1f}%")
    if recuperacion_promedio > 70:
        print("✓ FUNCIONA BIEN (>70%)")
    elif recuperacion_promedio > 50:
        print("⚠ FUNCIONA MODERADAMENTE (50-70%)")
    else:
        print("✗ NECESITA MEJORA (<50%)")


if __name__ == "__main__":
    evaluar()
