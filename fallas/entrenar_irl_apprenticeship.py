
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from irl_features import FEATURE_NAMES, N_FEATURES, horizonte_efectivo  # noqa: E402
from nominal_flight_env import (  # noqa: E402
    NominalFlightEnv, MIN_FAULT_PCT, MAX_FAULT_PCT, MIN_DIST, MAX_DIST_AB,
)

GAMMA = 0.99  # igual que gamma en entrenar_rl.py (PPO)


def techo_falla(iteracion, total_iteraciones):
    """
    Currículo del techo de severidad entre iteraciones (no dentro de una
    iteración, a diferencia de entrenar_rl.py, porque aquí cada iteración es
    un entrenamiento corto e independiente, no un único entrenamiento largo).
    iteracion=0 -> MIN_FAULT_PCT (usado para mu^(0), la política sin corrección).
    iteracion=total_iteraciones -> MAX_FAULT_PCT.
    """
    progreso = iteracion / total_iteraciones
    return MIN_FAULT_PCT + progreso * (MAX_FAULT_PCT - MIN_FAULT_PCT)


def distancia_max(iteracion, total_iteraciones):
    """
    Currículo de la distancia A->B, mismo mecanismo que techo_falla().
    Empezar con rutas cortas reduce el riesgo de que la política candidata
    se caiga antes de completar el episodio -- episodios más cortos acumulan
    menos costo en las features "siempre negativas" de phi (proximidad_objetivo,
    estabilidad_altura, oscilacion), haciendo que la candidata parezca mejor
    que el experto en esas dimensiones sin serlo realmente (ver
    irl_convergencia.csv de corridas anteriores, donde esas 3 quedaron en 0.0
    de forma consistente en las 30 iteraciones).
    """
    progreso = iteracion / total_iteraciones
    return MIN_DIST + progreso * (MAX_DIST_AB - MIN_DIST)

MU_EXPERTO_PATH = _ROOT / "results" / "mu_experto.npy"
ESCALA_PATH     = _ROOT / "results" / "mu_escala.npy"
WEIGHTS_PATH    = _ROOT / "results" / "irl_weights.json"
CONVERGENCIA_PATH = _ROOT / "results" / "irl_convergencia.csv"
SEARCH_LOG_DIR  = _ROOT / "results" / "irl_search_logs"


def rollout_mu(env, model, n_episodios, gamma):
    """Devuelve (mu, duracion_media_pasos). Cada episodio se normaliza por su
    horizonte descontado efectivo (ver irl_features.horizonte_efectivo) antes
    de promediar entre episodios, para que uno que termina antes (por una
    caída, o por una ruta corta del currículo de distancia) no parezca
    "mejor que el experto" solo por haber acumulado menos costo total.
    duracion_media_pasos se guarda además para diagnóstico (ver
    distancia_max())."""
    retornos   = []
    duraciones = []
    for _ in range(n_episodios):
        obs, _ = env.reset()
        done = False
        acumulado = np.zeros(N_FEATURES, dtype=np.float64)
        t = 0
        while not done:
            if model is None:
                accion = np.zeros(4, dtype=np.float32)
            else:
                accion, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(accion)
            acumulado += (gamma ** t) * info["phi"]
            t += 1
            if trunc:
                break
        retornos.append(acumulado / horizonte_efectivo(t, gamma))
        duraciones.append(t)
    return np.mean(retornos, axis=0), float(np.mean(duraciones))


def proyectar(mu_bar_prev, mu_i, mu_experto):
    """Actualización de proyección del algoritmo de Abbeel & Ng."""
    a = mu_i - mu_bar_prev
    b = mu_experto - mu_bar_prev
    denom = np.dot(a, a)
    if denom < 1e-12:
        return mu_bar_prev
    coef = np.dot(a, b) / denom
    return mu_bar_prev + coef * a


def restringir_w(w_busqueda, w_anterior):
    """
    Proyecta w_busqueda al ortante no-negativo (recorte a 0 de componentes
    negativas). Cada feature de phi() ya esta orientada como "mayor = mas
    parecido al experto", asi que un peso negativo invertiria esa direccion
    (fue justo lo que causo que el compensador se volviera erratico con
    esfuerzo_motores en la version anterior de 8 features).

    Si el recorte colapsa el vector casi a cero -- las 7 componentes querian
    ser negativas a la vez, algo posible sobre todo en las primeras
    iteraciones cuando la politica candidata es casi aleatoria -- se descarta
    esta actualizacion y se mantiene w_anterior, para no entrenar a PPO con
    una recompensa identicamente nula.
    """
    w_no_neg = np.clip(w_busqueda, 0, None)
    norma = np.linalg.norm(w_no_neg)
    if norma < 1e-6:
        print("  [aviso] recorte a w>=0 colapso el vector a ~0 - se mantiene el w anterior")
        return w_anterior.copy()
    return w_no_neg / norma


def entrenar_politica(w, timesteps, n_envs, seed, iteracion, max_fault_pct, max_dist_ab):
    def make_env():
        env = NominalFlightEnv(gui=False)
        env.set_reward_weights(w)
        env.set_max_fault(max_fault_pct)
        env.set_max_distance(max_dist_ab)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs)
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=10,
        gamma=GAMMA, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
        seed=seed, verbose=0,
        tensorboard_log=str(SEARCH_LOG_DIR),
    )
    model.learn(total_timesteps=timesteps, tb_log_name=f"iter{iteracion}", progress_bar=True)
    train_env.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteraciones", type=int, default=15)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--timesteps-por-iter", type=int, default=500)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodios", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true",
                        help="corrida corta para validar que el pipeline no rompe")
    args = parser.parse_args()

    if args.smoke_test:
        args.iteraciones = 2
        args.timesteps_por_iter = 4_000
        args.eval_episodios = 2
        args.n_envs = 2

    if not MU_EXPERTO_PATH.exists() or not ESCALA_PATH.exists():
        raise FileNotFoundError(
            f"Falta {MU_EXPERTO_PATH} o {ESCALA_PATH}. Corre primero calcular_mu_experto.py"
        )
    mu_experto = np.load(MU_EXPERTO_PATH)
    mu_escala  = np.load(ESCALA_PATH)
    print("mu_experto:", dict(zip(FEATURE_NAMES, mu_experto.round(4))))
    print("mu_escala (para que las 7 features pesen comparable en el margen):",
          dict(zip(FEATURE_NAMES, mu_escala.round(4))))

    
    mu_experto_r = mu_experto / mu_escala

    eval_env = NominalFlightEnv(gui=False)

    techo_0    = techo_falla(0, args.iteraciones)      # = MIN_FAULT_PCT
    distancia_0 = distancia_max(0, args.iteraciones)   # = MIN_DIST
    eval_env.set_max_fault(techo_0)
    eval_env.set_max_distance(distancia_0)
    print(f"\nCalculando mu^(0) (política baseline: Mamba sin corrección, "
          f"techo_falla={techo_0*100:.1f}%, distancia_max={distancia_0:.2f}m)...")
    mu_0, duracion_0 = rollout_mu(eval_env, None, args.eval_episodios, GAMMA)
    mu_0_r = mu_0 / mu_escala
    print(f"mu^(0):", dict(zip(FEATURE_NAMES, mu_0.round(4))),
          f"| duración media = {duracion_0:.1f} pasos")

    
    w_uniforme = np.ones(N_FEATURES, dtype=np.float64) / np.sqrt(N_FEATURES)
    w_busqueda = mu_experto_r - mu_0_r
    w_reward   = restringir_w(w_busqueda, w_uniforme)
    mu_bar_r   = mu_0_r.copy()

    convergencia = []
    for i in range(1, args.iteraciones + 1):
        techo     = techo_falla(i, args.iteraciones)
        distancia = distancia_max(i, args.iteraciones)
        w_reward_usado = w_reward.copy()
        print(f"\n{'='*60}\nIteración {i}/{args.iteraciones}  techo_falla={techo*100:.1f}%  "
              f"distancia_max={distancia:.2f}m  w_reward_usado={w_reward_usado.round(3)}")
        model = entrenar_politica(w_reward_usado, args.timesteps_por_iter, args.n_envs,
                                  seed=i, iteracion=i, max_fault_pct=techo,
                                  max_dist_ab=distancia)

        eval_env.set_max_fault(techo)         # evaluar con el mismo techo con que se entrenó
        eval_env.set_max_distance(distancia)  # y la misma distancia máxima
        mu_i, duracion_i = rollout_mu(eval_env, model, args.eval_episodios, GAMMA)
        mu_i_r = mu_i / mu_escala
        mu_bar_r = proyectar(mu_bar_r, mu_i_r, mu_experto_r)
        t_i = float(np.linalg.norm(mu_experto_r - mu_bar_r))
        w_busqueda = mu_experto_r - mu_bar_r
        w_reward   = restringir_w(w_busqueda, w_reward)

        print(f"mu^({i}) (crudo):", dict(zip(FEATURE_NAMES, mu_i.round(4))),
              f"| duración media = {duracion_i:.1f} pasos")
        print(f"margen t^({i}) (espacio rescalado) = {t_i:.4f}")
        convergencia.append({
            "iteracion": i, "margen": t_i, "duracion_media_pasos": duracion_i,
            "w_reward_usado": w_reward_usado.tolist(),
            "w_reward_siguiente_propuesto": w_reward.tolist(),
        })

        model.save(str(_ROOT / "results" / f"ppo_irl_iter{i}"))

        if t_i < args.eps:
            print(f"\nConvergió (margen {t_i:.4f} < eps {args.eps}) en {i} iteraciones.")
            break

    mejor = min(convergencia, key=lambda c: c["margen"])
    print(f"\nMejor iteración: {mejor['iteracion']} (margen={mejor['margen']:.4f}) "
          f"— se guarda su w_reward_usado, no un w extrapolado sin entrenar.")

    with open(WEIGHTS_PATH, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "weights": mejor["w_reward_usado"],
            "margen": mejor["margen"],
            "iteracion_elegida": mejor["iteracion"],
            "iteraciones_totales": len(convergencia),
            "escala_mu_aplicada": True,
        }, f, indent=2)
    print(f"Pesos guardados en {WEIGHTS_PATH}")

    with open(CONVERGENCIA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iteracion", "margen", "duracion_media_pasos",
            "w_reward_usado", "w_reward_siguiente_propuesto",
        ])
        writer.writeheader()
        writer.writerows(convergencia)
    print(f"Convergencia guardada en {CONVERGENCIA_PATH}")

    eval_env.close()


if __name__ == "__main__":
    main()
