
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

_ROOT = Path(__file__).parent.parent
# IRL_WEIGHTS_PATH is set from --pesos in main(), before constructing an environment.
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from fault_env_residual_irl_mamba import FaultResidualEnvIRL as FaultResidualEnvIRLMamba, DIST_ATERRIZO_B  # noqa: E402
from fault_env_residual_irl import FaultResidualEnvIRL as FaultResidualEnvIRLPid  # noqa: E402
from comparar_base import redibujar  # noqa: E402

MODELOS = {
    "hibrido": str(_ROOT / "results" / "ppo_compensador_mamba_hibrido_10"),
    "irl": str(_ROOT / "results" / "ppo_compensador_mamba_irl_11"),
    "irl_2a20": str(_ROOT / "results" / "ppo_compensador_mamba_irl_2a20_12"),
    "irl_mejor": str(_ROOT / "results" / "ppo_compensador_mamba_irl_10"),
    "irl_robust": str(_ROOT / "results" / "ppo_compensador_mamba_irl_13"),
    "irl_pid": str(_ROOT / "results" / "ppo_compensador_pid_irl_1"),
    "irl_pid_best": str(_ROOT / "results" / "ppo_pid_irl_1" / "best_model"),
    "irl_mamba_pidw": str(_ROOT / "results" / "ppo_compensador_mamba_irl_14"),
    "irl_mamba_pidw_best": str(_ROOT / "results" / "ppo_mamba_irl_14" / "best_model"),
    "irl_mamba_fix2": str(_ROOT / "results" / "ppo_compensador_mamba_irl_15"),
    "irl_mamba_fix2_best": str(_ROOT / "results" / "ppo_mamba_irl_15" / "best_model"),
    "irl_mamba_2a20_16": str(_ROOT / "results" / "ppo_compensador_mamba_irl_2a20_16"),
    "irl_mamba_2a20_16_best": str(_ROOT / "results" / "ppo_mamba_irl_2a20_16" / "best_model"),
    "irl_mamba_2a20_17": str(_ROOT / "results" / "ppo_compensador_mamba_irl_2a20_17"),
    "irl_mamba_2a20_17_best": str(_ROOT / "results" / "ppo_mamba_irl_2a20_17" / "best_model"),
    "irl_mamba_2a20_18": str(_ROOT / "results" / "ppo_compensador_mamba_irl_2a20_18"),
    "irl_mamba_2a20_18_best": str(_ROOT / "results" / "ppo_mamba_irl_2a20_18" / "best_model"),
    "irl_mamba_2a20_19": str(_ROOT / "results" / "ppo_compensador_mamba_irl_2a20_19"),
    "irl_mamba_2a20_19_best": str(_ROOT / "results" / "ppo_mamba_irl_2a20_19" / "best_model"),
    "irl_pid_2": str(_ROOT / "results" / "ppo_compensador_pid_irl_2"),
    "irl_pid_2_best": str(_ROOT / "results" / "ppo_pid_irl_2" / "best_model"),
    "irl_pid_2a20_14": str(_ROOT / "results" / "ppo_compensador_pid_irl_2a20_14"),
    "irl_pid_2a20_15": str(_ROOT / "results" / "ppo_compensador_pid_irl_2a20_15"),
    "irl_pid_2a20_16": str(_ROOT / "results" / "ppo_compensador_pid_irl_2a20_16"),
}
NIVELES = [0.02, 0.04, 0.06, 0.08, 0.10]
NIVELES_2A20 = [0.02, 0.05, 0.10, 0.15, 0.20]


def correr_episodio(env, model, semilla):
    np.random.seed(semilla)
    obs, _ = env.reset(seed=semilla)
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
    if info.get("t_falla_seg") is None:
        outcome = "sin_exposicion_" + outcome
    return outcome, pasos, posiciones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios", type=int, default=20, help="Episodios por nivel y por controlador")
    parser.add_argument("--compensador", type=str, default="hibrido", choices=list(MODELOS),
                        help="hibrido: recompensa manual + IRL. irl: solo los pesos del IRL")
    parser.add_argument("--gui", action="store_true", help="Muestra el vuelo en PyBullet, a velocidad real")
    parser.add_argument("--nivel", type=float, default=None,
                        help="Solo este nivel de falla en %% (ej: 6). Por defecto, todos")
    parser.add_argument("--base", type=str, default="mamba", choices=["mamba", "pid", "robust"],
                        help="mamba: compara contra Mamba sin compensar. "
                             "pid: compara contra PID normal sin compensar (usa un compensador "
                             "entrenado con --modelo pid, ej. irl_pid). "
                             "robust: compara contra PID robusto sin compensar (controlador='robust', "
                             "fault_aware=False)")
    parser.add_argument("--pesos", required=True, help="Exact reward JSON used for this checkpoint")
    parser.add_argument("--modelo-path", default=None, help="Optional exact PPO checkpoint path")
    args = parser.parse_args()
    weights_path = Path(args.pesos).resolve()
    if not weights_path.is_file():
        parser.error(f"No existe el archivo de recompensa: {weights_path}")
    for module in (sys.modules[FaultResidualEnvIRLMamba.__module__],
                   sys.modules[FaultResidualEnvIRLPid.__module__]):
        module.IRL_WEIGHTS_PATH = str(weights_path)


    model = PPO.load(args.modelo_path or MODELOS[args.compensador])
    modo_reward = "irl" if args.compensador.startswith("irl") else "hibrido"
    niveles_default = NIVELES_2A20 if args.compensador in (
        "irl_2a20", "irl_mamba_2a20_16", "irl_mamba_2a20_16_best",
        "irl_mamba_2a20_17", "irl_mamba_2a20_17_best",
        "irl_mamba_2a20_18", "irl_mamba_2a20_18_best",
        "irl_mamba_2a20_19", "irl_mamba_2a20_19_best",
        "irl_pid_2a20_14", "irl_pid_2a20_15", "irl_pid_2a20_16",
    ) else NIVELES
    # niveles_falla_pct tiene que coincidir con el maximo usado al entrenar el
    # compensador: desde el ajuste de escalado por severidad, step() usa
    # self.niveles[-1] para normalizar cuanto puede empujar el residual.
    if args.base == "mamba":
        env = FaultResidualEnvIRLMamba(gui=args.gui, modelo="mamba", t_falla_min=3.0, t_falla_max=3.0,
                                       modo_reward=modo_reward, alpha=0.05,
                                       niveles_falla_pct=niveles_default)
        nombre_base = "Mamba sin compensar"
    elif args.base == "pid":
        env = FaultResidualEnvIRLPid(gui=args.gui, modelo="pid", t_falla_min=3.0, t_falla_max=3.0,
                                     modo_reward=modo_reward, alpha=0.05,
                                     controlador="pid", fault_aware=False,
                                     niveles_falla_pct=niveles_default)
        nombre_base = "PID normal sin compensar"
    else:
        env = FaultResidualEnvIRLPid(gui=args.gui, modelo="pid", t_falla_min=3.0, t_falla_max=3.0,
                                     modo_reward=modo_reward, alpha=0.05,
                                     controlador="robust", fault_aware=False,
                                     niveles_falla_pct=niveles_default)
        nombre_base = "PID robusto sin compensar"
    niveles = [args.nivel / 100] if args.nivel is not None else niveles_default

    print(f"Evaluación: {args.episodios} intentos por nivel | falla a mitad del crucero; sin exposición se cuenta por separado")
    print("=" * 92)
    print(f"{'falla':>6}  {'controlador':<22} {'llego':>6} {'aterr.B':>8} {'aterr.lej':>10} {'cayo':>6} {'tiempo':>7} {'sin.exp':>8} {'recup.%':>8} {'pasos':>7}")
    print("-" * 92)

    for nivel in niveles:
        env.falla_fija = nivel
        controladores = [(nombre_base, None, (1.0, 0.0, 0.0)),
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
            sin_exposicion = sum(v for k, v in oc.items() if k.startswith("sin_exposicion_"))
            recup = 100 * (oc["llego"] + oc["aterrizo"]) / n
            print(f"{nivel*100:5.0f}%  {nombre:<22} {oc['llego']:>6} {oc['aterrizo']:>8} "
                  f"{oc['aterrizo_lejos']:>10} {oc['cayo']:>6} {oc['tiempo']:>7} "
                  f"{sin_exposicion:>8} {recup:>7.1f}% {np.mean(pasos[nombre]):>7.0f}")

    env.close()


if __name__ == "__main__":
    main()
