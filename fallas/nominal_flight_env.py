
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
    MOTOR_FALLA, T_FALLA_MIN, T_FALLA_MAX,
    es_caida, es_aterrizaje,
)
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402
from irl_features import phi, distancia_z, cargar_escalas, N_FEATURES  # noqa: E402

XY_LIM   = 2.0   # rango de vuelo igual al del dataset de Mamba / fault_env_residual.py
MIN_DIST = 0.8
MAX_DIST_AB = 2 * XY_LIM * (2 ** 0.5)  # diagonal de la caja de vuelo (~5.66 m)
DELTA_MAX = 3000  # aumentado para dar más autoridad al delta residual de PPO


MIN_FAULT_PCT = 0.02  # igual que entrenar_rl.py
MAX_FAULT_PCT = 0.40  # severidad máxima: 40% de pérdida


class NominalFlightEnv(gym.Env):

    def __init__(self, gui=False, t_falla_min=None, t_falla_max=None):
        super().__init__()
        self.gui       = gui
        self.max_pasos = int(DURACION_SEG * CTRL_FREQ)
        self.stats     = cargar_stats(STATS_PATH)
        self.escalas   = cargar_escalas(self.stats)
        self.w         = np.zeros(N_FEATURES, dtype=np.float64)
        self.t_falla_min   = T_FALLA_MIN if t_falla_min is None else t_falla_min
        self.t_falla_max   = T_FALLA_MAX if t_falla_max is None else t_falla_max
        self.max_fault_pct = MAX_FAULT_PCT  # ver set_max_fault() -- currículo externo lo ajusta
        self.max_dist_ab   = MAX_DIST_AB    # ver set_max_distance() -- currículo externo lo ajusta

        self.pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)

        self._env = None
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        # Accion: delta residual sobre la RPM base de Mamba, no RPM completa.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def set_reward_weights(self, w):
        """Pesos w (7,) usados para R(s,a) = w . phi(s,a) en step()."""
        self.w = np.asarray(w, dtype=np.float64)

    def set_max_fault(self, pct):
        """Techo superior de severidad para el currículo (ver entrenar_irl_apprenticeship.py)."""
        self.max_fault_pct = float(np.clip(pct, MIN_FAULT_PCT, MAX_FAULT_PCT))

    def set_max_distance(self, dist):
        """Techo superior de distancia A->B para el currículo (ver entrenar_irl_apprenticeship.py)."""
        self.max_dist_ab = float(np.clip(dist, MIN_DIST, MAX_DIST_AB))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        fault_pct      = np.random.uniform(MIN_FAULT_PCT, self.max_fault_pct)
        self.severidad = 1.0 - fault_pct
        t_falla_seg    = np.random.uniform(self.t_falla_min, self.t_falla_max)
        self.t_falla   = int(t_falla_seg * CTRL_FREQ)

        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            dist_deseada = np.random.uniform(MIN_DIST, self.max_dist_ab)
            angulo = np.random.uniform(0, 2 * np.pi)
            b_xy = a_xy + dist_deseada * np.array([np.cos(angulo), np.sin(angulo)])
            if np.all(np.abs(b_xy) <= XY_LIM):
                break
        self.punto_A   = np.array([a_xy[0], a_xy[1], Z_SUELO])
        self.punto_B   = np.array([b_xy[0], b_xy[1], Z_SUELO])
        self.waypoints = generar_waypoints(self.punto_A, self.punto_B)

        if self._env is None:
            self._env = nueva_env(self.punto_A, gui=self.gui)
        else:
            self._env.INIT_XYZS = self.punto_A.reshape(1, 3)
        obs_raw, _ = self._env.reset()

        self.wp_idx     = 0
        self.paso       = 0
        self.action_rpm = np.ones((1, 4)) * HOVER_RPM

        self.dist_prev_z    = distancia_z(obs_raw[0][0:3], self.punto_B, self.escalas)
        self.rpm_anterior   = np.ones(4, dtype=np.float64) * HOVER_RPM
        self.vel_z_anterior = float(obs_raw[0][12])
        self.falla_activa   = False

        return self._build_obs(obs_raw, self.waypoints[0]), {}

    def step(self, ppo_delta_norm):
        obs_raw, _, term, trunc, _ = self._env.step(self.action_rpm)
        pos     = obs_raw[0][0:3].copy()
        rpy     = obs_raw[0][7:10]
        ang_vel = obs_raw[0][13:16]
        vel     = obs_raw[0][10:13]
        self.paso += 1
        es_caida_ahora = es_caida(obs_raw, pos, self.paso, self.wp_idx, len(self.waypoints))

        ta = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]

        # PID genera la RPM base
        base_rpm, _, _ = self.pid_controller.computeControlFromState(
            control_timestep=1.0/CTRL_FREQ,
            state=obs_raw[0],
            target_pos=ta,
            target_rpy=np.zeros(3),
        )
        base_rpm = np.clip(base_rpm, MIN_RPM, MAX_RPM)

        # PPO aprende el delta residual sobre esa base
        delta        = np.asarray(ppo_delta_norm, dtype=np.float64) * DELTA_MAX
        rpm_comandado = np.clip(base_rpm + delta, MIN_RPM, MAX_RPM)

        # Falla activa desde t_falla en adelante (igual que fault_env_residual_irl.py)
        falla_activa = self.paso >= self.t_falla
        rpm_final = rpm_comandado.copy()
        if falla_activa:
            rpm_final[MOTOR_FALLA] *= self.severidad
        self.action_rpm = rpm_final.reshape(1, 4)

        # La recompensa se calcula SIEMPRE (antes y después de la falla), no
        # solo post-falla como en fault_env_residual_irl.py -- mu_i necesita
        # el episodio completo para ser comparable con mu_experto.
        vec, self.dist_prev_z = phi(
            pos, rpy, ang_vel, vel, ta, self.punto_B, rpm_final,
            self.rpm_anterior, self.vel_z_anterior, self.dist_prev_z, self.escalas,
            es_caida_ahora=es_caida_ahora, delta_residual=delta,
        )
        reward = float(np.dot(self.w, vec))
        self.rpm_anterior   = rpm_final.astype(np.float64)
        self.vel_z_anterior = float(vel[2])
        self.falla_activa   = falla_activa

        done    = False
        outcome = None
        if es_caida_ahora:
            done, outcome = True, "cayo"
        elif es_aterrizaje(obs_raw, pos, self.paso):
            done, outcome = True, "aterrizo"
        elif np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            self.wp_idx += 1
            if self.wp_idx >= len(self.waypoints):
                done, outcome = True, "llego"
        if (self.paso >= self.max_pasos or term or trunc) and not done:
            done, outcome = True, "tiempo"

        # Recalcular ta por si wp_idx acaba de avanzar
        ta_obs = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]

        info = {"phi": vec}
        if done:
            info["outcome"] = outcome
        return self._build_obs(obs_raw, ta_obs), reward, done, False, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _build_obs(self, obs_raw, ta):
        estado_norm = normalizar_estado(obs_raw, ta, self.stats)
        # Oraculo, misma convencion que fault_env_residual_irl.py: la politica
        # ve si hay falla activa y su porcentaje real (no la infiere aqui).
        fault_pct  = 1.0 - self.severidad if self.falla_activa else 0.0
        fault_info = np.array([float(self.falla_activa), fault_pct], dtype=np.float32)
        # Con PID, no hay predicción neuronal - usamos ceros para que el shape sea consistente
        model_pred = np.zeros(4, dtype=np.float32)
        return np.concatenate([estado_norm, model_pred, fault_info]).astype(np.float32)
