"""
Entrenamiento PPO para compensación residual de falla de motor.

Uso:
    python fallas/entrenar_rl.py                          # Mamba+PPO
    python fallas/entrenar_rl.py --modelo lstm            # LSTM+PPO
    python fallas/entrenar_rl.py --timesteps 500000
"""

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from fault_env_residual import FaultResidualEnv, DURACION_SEG as _DURACION_SEG

T_FALLA_TRAIN  = 3.0   # igual que T_FALLA_SEG en comparar_base.py (solo --reward manual)
MIN_FAULT_PCT  = 0.02
MAX_FAULT_PCT  = 0.80
CURRICULUM_FRAC = 0.75


class ProgresionFallaCallback(BaseCallback):
    """Sube el techo de falla linealmente de MIN_FAULT_PCT a MAX_FAULT_PCT."""

    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        denominador = self.total_timesteps * CURRICULUM_FRAC
        progress    = min(self.num_timesteps / denominador, 1.0)
        current_max = MIN_FAULT_PCT + progress * (MAX_FAULT_PCT - MIN_FAULT_PCT)
        self.training_env.env_method("set_max_fault", current_max)
        return True


class MetricasCallback(BaseCallback):
    """
    Imprime cada `log_freq` episodios un resumen de si PPO está aprendiendo.
    Guarda en CSV las trayectorias donde PPO llegó exitosamente.
    """

    def __init__(self, log_freq=200, csv_path=None, total_timesteps=1_000_000, falla_fija=None):
        super().__init__()
        self.falla_fija      = falla_fija
        self.log_freq        = log_freq
        self.csv_path        = csv_path
        self.total_timesteps = total_timesteps
        self._outcomes       = []
        self._misiones       = []
        self._deltas         = []
        self._ep_count       = 0
        self._csv_file       = None
        self._csv_writer     = None

    def _on_training_start(self):
        if self.csv_path:
            self._csv_file   = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "ep", "step", "outcome", "mision_pct",
                "severidad", "perdida_pct", "delta_medio",
                "A_x", "A_y", "B_x", "B_y",
            ])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "outcome" not in info:
                continue
            self._outcomes.append(info["outcome"])
            self._misiones.append(info["mision_pct"])
            self._deltas.append(info["delta_medio"])
            self._ep_count += 1

            # Guardar episodios exitosos en CSV
            if self._csv_writer and info["outcome"] in ("llego", "aterrizo"):
                A = info.get("punto_A", [0, 0, 0])
                B = info.get("punto_B", [0, 0, 0])
                sev = info["severidad"]
                self._csv_writer.writerow([
                    self._ep_count, self.num_timesteps,
                    info["outcome"], f"{info['mision_pct']:.1f}",
                    f"{sev:.3f}", f"{round((1-sev)*100)}",
                    f"{info['delta_medio']:.4f}",
                    f"{A[0]:.3f}", f"{A[1]:.3f}",
                    f"{B[0]:.3f}", f"{B[1]:.3f}",
                ])
                self._csv_file.flush()

            if self._ep_count % self.log_freq == 0:
                n         = len(self._outcomes)
                caen      = self._outcomes.count("cayo")
                llegan    = self._outcomes.count("llego")
                aterrizan = self._outcomes.count("aterrizo")
                recuperan = llegan + aterrizan
                if self.falla_fija is not None:
                    txt_falla = f"falla_fija={self.falla_fija*100:.0f}%"
                else:
                    denominador = self.total_timesteps * CURRICULUM_FRAC
                    progress    = min(self.num_timesteps / denominador, 1.0)
                    falla_max   = MIN_FAULT_PCT + progress * (MAX_FAULT_PCT - MIN_FAULT_PCT)
                    txt_falla   = f"falla_max={falla_max*100:.0f}%"

                print(
                    f"\n  [ep {self._ep_count:>5} | step {self.num_timesteps:>7}]"
                    f"  {txt_falla}"
                    f" | crash={100*caen/n:.0f}%"
                    f" | llegó={100*llegan/n:.0f}%"
                    f" | aterrizó={100*aterrizan/n:.0f}%"
                    f" | recuperación={100*recuperan/n:.0f}%"
                    f" | misión={sum(self._misiones)/n:.0f}%"
                    f" | δ_medio={sum(self._deltas)/n:.3f}"
                )
                self._outcomes.clear()
                self._misiones.clear()
                self._deltas.clear()
        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()

OUTPUT_DIR  = str(_ROOT / "results")
MODEL_PATH  = str(_ROOT / "results" / "ppo_compensador")
LOG_DIR     = str(_ROOT / "results" / "ppo_logs")


REWARDS_ENV_IRL = ("irl", "manual2", "hibrido")   # usan fault_env_residual_irl.py


def _env_irl(modelo, reward, t_falla_min, t_falla_max, rutas, falla_fija, alpha):
    from fault_env_residual_irl import FaultResidualEnvIRL
    return FaultResidualEnvIRL(
        gui=False, modelo=modelo,
        t_falla_min=t_falla_min, t_falla_max=t_falla_max,
        rutas=rutas, falla_fija=falla_fija,
        modo_reward=reward, alpha=alpha,
    )


def make_env(modelo="mamba", reward="manual", t_falla_min=None, t_falla_max=None,
             rutas="aleatorias", falla_fija=None, alpha=0.05):
    def _init():
        if reward in REWARDS_ENV_IRL:
            return _env_irl(modelo, reward, t_falla_min, t_falla_max, rutas, falla_fija, alpha)
        return FaultResidualEnv(gui=False, t_falla_override=T_FALLA_TRAIN, modelo=modelo)
    return _init


def train(timesteps: int, modelo: str, reward: str, t_falla_min: float, t_falla_max: float,
          rutas: str = "aleatorias", falla_fija: float = None, alpha: float = 0.05):
    sufijo      = modelo if reward == "manual" else f"{modelo}_{reward}"
    if reward in REWARDS_ENV_IRL and rutas == "tesis":
        sufijo += "_tesis"   # no sobrescribe el modelo entrenado con rutas aleatorias
    if reward in REWARDS_ENV_IRL and falla_fija is not None:
        sufijo += f"_f{round(falla_fija * 100)}"   # ej. _f10 para 10% fijo
    best_dir    = str(_ROOT / "results" / f"ppo_{sufijo}_10")
    model_path  = str(_ROOT / "results" / f"ppo_compensador_{sufijo}_10")
    log_dir     = str(_ROOT / "results" / f"ppo_logs_{sufijo}_10")
    ckpt_prefix = f"ppo_{sufijo}_10_ckpt"

    txt_falla = (f"falla fija {falla_fija * 100:.0f}%" if falla_fija is not None
                 else f"progresión falla 2%→{MAX_FAULT_PCT * 100:.0f}%")
    print(
        f"Entrenando PPO+{modelo.upper()} (reward={reward}) — {timesteps} pasos "
        f"| {txt_falla} | rutas={rutas}" + (f" | alpha={alpha}" if reward == "hibrido" else "")
    )
    print(f"Guardando en: {model_path}.zip\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(best_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    N_ENVS    = 4
    train_env = make_vec_env(
        make_env(modelo=modelo, reward=reward,
                t_falla_min=t_falla_min, t_falla_max=t_falla_max, rutas=rutas,
                falla_fija=falla_fija, alpha=alpha),
        n_envs=N_ENVS,
    )
    if reward in REWARDS_ENV_IRL:
        eval_env = Monitor(_env_irl(modelo, reward, t_falla_min, t_falla_max,
                                    rutas, falla_fija, alpha))
    else:
        eval_env = Monitor(FaultResidualEnv(gui=False, t_falla_override=T_FALLA_TRAIN, modelo=modelo))
    eval_env.unwrapped.set_max_fault(0.20)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,        # 4 envs × 512 = 2048 pasos por update (igual que antes)
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        target_kl=0.02,
        tensorboard_log=log_dir,
        verbose=0,
    )
    print("Entrenando desde cero (sin warm-start)")

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=OUTPUT_DIR,
        name_prefix=ckpt_prefix,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=log_dir,
        eval_freq=25_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    csv_exitosos  = str(_ROOT / "results" / f"recuperaciones_{sufijo}_10.csv")
    metricas_cb   = MetricasCallback(log_freq=200, csv_path=csv_exitosos, total_timesteps=timesteps,
                                     falla_fija=falla_fija)
    callbacks     = [checkpoint_cb, eval_cb, metricas_cb]
    if falla_fija is None:   # con falla fija no hay currículo de severidad
        callbacks.append(ProgresionFallaCallback(total_timesteps=timesteps))

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(model_path)
    print(f"\nModelo guardado en {model_path}.zip")
    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)  # igual que _5
    parser.add_argument("--modelo", type=str, default="mamba", choices=["mamba", "lstm"])
    parser.add_argument("--reward", type=str, default="manual",
                        choices=["manual", "irl", "manual2", "hibrido"],
                        help="manual: fault_env_residual.py (sin cambios). "
                             "irl: R = w.phi. "
                             "manual2: recompensa manual corregida (vivo, sin premio por "
                             "aterrizar lejos). "
                             "hibrido: manual2 + alpha * w+ . (phi - mu_experto_mamba). "
                             "irl/manual2/hibrido usan fault_env_residual_irl.py")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="solo con --reward hibrido: peso de la parte IRL")
    parser.add_argument("--pesos", type=str, default="irl_weights.json",
                        help="archivo de pesos a usar cuando --reward irl (ej: irl_weights_2.json)")
    parser.add_argument("--t-falla-min", type=float, default=None,
                        help="solo con --reward irl/manual2/hibrido; default T_FALLA_MIN de comparar_base.py")
    parser.add_argument("--t-falla-max", type=float, default=None,
                        help="solo con --reward irl/manual2/hibrido; default T_FALLA_MAX de comparar_base.py")
    parser.add_argument("--rutas", type=str, default="aleatorias", choices=["aleatorias", "tesis"],
                        help="solo con --reward irl/manual2/hibrido. aleatorias: A,B al azar en cada episodio. "
                             "tesis: las 5 TRAYECTORIAS que evalúa falla_tesis.py "
                             "(guarda como ppo_compensador_<modelo>_irl_tesis_10.zip)")
    parser.add_argument("--falla-fija", type=float, default=None,
                        help="solo con --reward irl/manual2/hibrido. Pérdida fija del motor en todos los "
                             "episodios, como fracción (ej: 0.10 = 10%%). Sin currículo de "
                             "severidad. Default: aleatoria 2%%→80%% con currículo.")
    args = parser.parse_args()

    # Pasar el archivo de pesos al entorno si se usa IRL
    if args.reward in REWARDS_ENV_IRL:
        import os
        pesos_path = str(_ROOT / "results" / args.pesos)
        if not os.path.exists(pesos_path):
            raise FileNotFoundError(f"Archivo de pesos no encontrado: {pesos_path}")
        # Establecer variable de entorno para que FaultResidualEnvIRL la lea
        os.environ["IRL_WEIGHTS_PATH"] = pesos_path

    train(args.timesteps, args.modelo, args.reward, args.t_falla_min, args.t_falla_max,
          rutas=args.rutas, falla_fija=args.falla_fija, alpha=args.alpha)


if __name__ == "__main__":
    main()
