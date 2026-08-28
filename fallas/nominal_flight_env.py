"""
Entorno de vuelo nominal (sin falla) para Apprenticeship Learning.

Misma arquitectura que el compensador final (fault_env_residual_irl.py):
Mamba (o LSTM) da la RPM base, PPO aprende un delta residual acotado sobre
esa base. La diferencia con fault_env_residual_irl.py es que aquí NUNCA hay
falla — es el escenario nominal, porque no existe ninguna demostración del
PID con falla contra la cual comparar.

Se cambió deliberadamente de "RPM completa desde cero" (versión anterior) a
"Mamba + delta" para que la política candidata usada en la búsqueda de w
tenga la MISMA arquitectura que la política que finalmente usará esos pesos
— evita que el candidato tenga que reaprender a volar por completo en cada
iteración (Mamba ya vuela bien via BC), y evita que w se valide en una
arquitectura distinta a la que realmente lo va a usar.

Es el "generador" del algoritmo de proyección (entrenar_irl_apprenticeship.py):
en cada iteración se entrena una política PPO nueva sobre este entorno bajo la
recompensa candidata R(s,a) = w . phi(s,a) (ver set_reward_weights), y se miden
sus expectativas de características reales haciendo rollout de esa política.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import (  # noqa: E402
    cargar_stats, normalizar_estado, desnormalizar_accion,
    generar_waypoints, nueva_env,
    STATS_PATH, Z_SUELO, UMBRAL_WAYPOINT, DURACION_SEG,
    CTRL_FREQ, HOVER_RPM, MIN_RPM, MAX_RPM, INPUT_COLS,
    es_caida, es_aterrizaje,
    MambaDrone, LSTMDrone,
)
from irl_features import phi, distancia_z, cargar_escalas, N_FEATURES  # noqa: E402

XY_LIM   = 2.0   # rango de vuelo igual al del dataset de Mamba / fault_env_residual.py
MIN_DIST = 0.8
DELTA_MAX = 1500  # mismo rango que el compensador final (fault_env_residual_irl.py)

MAMBA_MODEL_PATH = str(_ROOT / "results" / "modelo_mamba.pth")
LSTM_MODEL_PATH  = str(_ROOT / "results" / "modelo_lstm.pth")


class NominalFlightEnv(gym.Env):

    def __init__(self, gui=False, modelo="mamba"):
        super().__init__()
        self.gui       = gui
        self.max_pasos = int(DURACION_SEG * CTRL_FREQ)
        self.stats     = cargar_stats(STATS_PATH)
        self.escalas   = cargar_escalas(self.stats)
        self.w         = np.zeros(N_FEATURES, dtype=np.float64)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if modelo == "lstm":
            self.base_model = LSTMDrone()
            model_path = LSTM_MODEL_PATH
        else:
            self.base_model = MambaDrone()
            model_path = MAMBA_MODEL_PATH
        self.base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.base_model.to(self.device)
        self.base_model.eval()

        self._env = None
        # Observacion: 18 estado + 4 pred cruda de Mamba + 2 "fault info" (siempre
        # [0,0] aqui, nunca hay falla) -- mismo formato que fault_env_residual_irl.py
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        # Accion: delta residual sobre la RPM base de Mamba, no RPM completa.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def set_reward_weights(self, w):
        """Pesos w (8,) usados para R(s,a) = w . phi(s,a) en step()."""
        self.w = np.asarray(w, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            b_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
                break
        self.punto_A   = np.array([a_xy[0], a_xy[1], Z_SUELO])
        self.punto_B   = np.array([b_xy[0], b_xy[1], Z_SUELO])
        self.waypoints = generar_waypoints(self.punto_A, self.punto_B)

        if self._env is None:
            self._env = nueva_env(self.punto_A, gui=self.gui)
        else:
            self._env.INIT_XYZS = self.punto_A.reshape(1, 3)
        obs_raw, _ = self._env.reset()

        self.ventana = deque(
            [np.zeros(len(INPUT_COLS), dtype=np.float32)] * 50, maxlen=50
        )
        self.wp_idx     = 0
        self.paso       = 0
        self.action_rpm = np.ones((1, 4)) * HOVER_RPM
        self.model_pred = np.zeros(4, dtype=np.float32)

        ta = self.waypoints[0]
        self.dist_prev_z = distancia_z(obs_raw[0][0:3], ta, self.escalas)

        return self._build_obs(obs_raw, ta), {}

    def step(self, ppo_delta_norm):
        obs_raw, _, term, trunc, _ = self._env.step(self.action_rpm)
        pos     = obs_raw[0][0:3].copy()
        rpy     = obs_raw[0][7:10]
        ang_vel = obs_raw[0][13:16]
        vel_z   = obs_raw[0][12]
        self.paso += 1

        ta = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
        self.ventana.append(normalizar_estado(obs_raw, ta, self.stats))

        # Mamba genera la RPM base
        x = torch.tensor(
            np.array(self.ventana), dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.base_model(x).cpu().numpy()[0]
        self.model_pred = pred
        base_rpm = np.clip(desnormalizar_accion(pred, self.stats), MIN_RPM, MAX_RPM)

        # PPO aprende el delta residual sobre esa base (sin falla, siempre activo aqui)
        delta = np.asarray(ppo_delta_norm, dtype=np.float64) * DELTA_MAX
        rpm   = np.clip(base_rpm + delta, MIN_RPM, MAX_RPM)
        self.action_rpm = rpm.reshape(1, 4)

        vec, self.dist_prev_z = phi(
            pos, rpy, ang_vel, vel_z, ta, rpm, self.dist_prev_z, self.escalas
        )
        reward = float(np.dot(self.w, vec))

        done = False
        if es_caida(obs_raw, pos, self.paso):
            done = True
        elif es_aterrizaje(obs_raw, pos, self.paso):
            done = True
        elif np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            self.wp_idx += 1
            if self.wp_idx >= len(self.waypoints):
                done = True
        if self.paso >= self.max_pasos or term or trunc:
            done = True

        # Recalcular ta por si wp_idx acaba de avanzar
        ta_obs = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]

        info = {"phi": vec}
        return self._build_obs(obs_raw, ta_obs), reward, done, False, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _build_obs(self, obs_raw, ta):
        estado_norm = normalizar_estado(obs_raw, ta, self.stats)
        fault_info  = np.array([0.0, 0.0], dtype=np.float32)  # nunca hay falla aqui
        return np.concatenate([estado_norm, self.model_pred, fault_info]).astype(np.float32)
