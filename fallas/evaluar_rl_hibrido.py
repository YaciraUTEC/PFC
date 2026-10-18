"""
Evalúa el compensador híbrido (Mamba + PPO) frente a Mamba sin compensar,
con fallas fijas de 2, 4, 6, 8 y 10 % a los 3 s.

Uso (desde /mnt/d/TesisI):
    PYTHONPATH=/mnt/d/TesisI:/mnt/d/TesisI/fallas /mnt/d/venv_mamba/bin/python3 fallas/evaluar_rl_hibrido.py --episodios 20
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_ROOT = Path(__file__).parent.parent
os.environ["IRL_WEIGHTS_PATH"] = str(_ROOT / "results" / "irl_weights_1600.json")
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from fault_env_residual_irl_mamba import FaultResidualEnvIRL, DIST_ATERRIZO_B  # noqa: E402
from comparar_base import redibujar  # noqa: E402

MODELOS = {
    "hibrido": str(_ROOT / "results" / "ppo_compensador_mamba_hibrido_10"),
    "irl": str(_ROOT / "results" / "ppo_compensador_mamba_irl_10"),
}
NIVELES = [0.02, 0.04, 0.06, 0.08, 0.10]


def correr_episodio(env, model, semilla):
    np.random.seed(semilla)
    obs, _ = env.reset()
    done = False
    pasos = 0
    info = {}
    posiciones = [env.obs_raw[0][0:3].copy()]
    while not done:
        if model is None:
            accion = np.zeros(4, dtype=np.float32)
        else:
            accion, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(accion)
        posiciones.append(env.obs_raw[0][0:3].copy())
        if env.gui:
            time.sleep(1.0 / 48)
        pasos += 1
    outcome = info.get("outcome", "tiempo")
    if outcome == "aterrizo":
        dist_B = np.linalg.norm(env.punto_B[:2] - env.obs_raw[0][0:2])
        if dist_B >= DIST_ATERRIZO_B:
            outcome = "aterrizo_lejos"
    return outcome, pasos, posiciones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios", type=int, default=20, help="Episodios por nivel y por controlador")
    parser.add_argument("--compensador", type=str, default="hibrido", choices=list(MODELOS),
                        help="hibrido: recompensa manual + IRL. irl: solo los pesos del IRL")
    parser.add_argument("--gui", action="store_true", help="Muestra el vuelo en PyBullet, a velocidad real")
    parser.add_argument("--nivel", type=float, default=None,
                        help="Solo este nivel de falla en %% (ej: 6). Por defecto, todos")
    args = parser.parse_args()

    model = PPO.load(MODELOS[args.compensador])
    env = FaultResidualEnvIRL(gui=args.gui, modelo="mamba", t_falla_min=3.0, t_falla_max=3.0,
                              modo_reward=args.compensador, alpha=0.05)
    niveles = [args.nivel / 100] if args.nivel is not None else NIVELES

    print(f"Evaluación: {args.episodios} episodios por nivel | falla a 3.0 s")
    print("=" * 92)
    print(f"{'falla':>6}  {'controlador':<22} {'llego':>6} {'aterr.B':>8} {'aterr.lej':>10} {'cayo':>6} {'tiempo':>7} {'recup.%':>8} {'pasos':>7}")
    print("-" * 92)

    for nivel in niveles:
        env.falla_fija = nivel
        controladores = [("Mamba sin compensar", None, (1.0, 0.0, 0.0)),
                         (f"Compensador {args.compensador}", model, (0.0, 0.0, 1.0))]
        outcomes = {nombre: Counter() for nombre, _, _ in controladores}
        pasos = {nombre: [] for nombre, _, _ in controladores}
        for semilla in range(args.episodios):
            for nombre, m, color in controladores:
                outcome, n_pasos, posiciones = correr_episodio(env, m, semilla)
                outcomes[nombre][outcome] += 1
                pasos[nombre].append(n_pasos)
                if args.gui:
                    redibujar(posiciones, color, env._env.CLIENT)
        for nombre, _, _ in controladores:
            oc = outcomes[nombre]
            n = args.episodios
            recup = 100 * (oc["llego"] + oc["aterrizo"]) / n
            print(f"{nivel*100:5.0f}%  {nombre:<22} {oc['llego']:>6} {oc['aterrizo']:>8} "
                  f"{oc['aterrizo_lejos']:>10} {oc['cayo']:>6} {oc['tiempo']:>7} "
                  f"{recup:>7.1f}% {np.mean(pasos[nombre]):>7.0f}")

    env.close()


if __name__ == "__main__":
    main()
