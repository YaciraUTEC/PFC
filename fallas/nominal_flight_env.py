
import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import (
    cargar_stats, normalizar_estado, desnormalizar_accion,
    generar_waypoints, nueva_env,
    STATS_PATH, Z_SUELO, UMBRAL_WAYPOINT, DURACION_SEG,
    CTRL_FREQ, HOVER_RPM, MIN_RPM, MAX_RPM, INPUT_COLS,
    MOTOR_FALLA,
    es_caida, es_aterrizaje, _fase_de_waypoint,
)
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  # noqa: E402
from gym_pybullet_drones.control.RobustDSLPIDControl import RobustDSLPIDControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402
from irl_features import phi, distancia_z, cargar_escalas, N_FEATURES  # noqa: E402

XY_LIM   = 2.0   # rango de vuelo igual al del dataset de Mamba / fault_env_residual.py
MIN_DIST = 0.8
DELTA_MAX = 3000  # aumentado para dar más autoridad al delta residual de PPO


NIVELES_FALLA_PCT = [0.02, 0.04, 0.06, 0.08, 0.10]  # mismos niveles que datos_pid_falla_leve.csv
NIVELES_FALLA_2A20 = [0.02, 0.05, 0.10, 0.15, 0.20]  # mismos niveles que datos_robust_pid_falla_2a20.csv
MIN_FAULT_PCT = NIVELES_FALLA_PCT[0]
MAX_FAULT_PCT = NIVELES_FALLA_PCT[-1]
PROB_NOMINAL = 0.5  # probabilidad de episodio sin falla (0%), igual a la composicion
                     # del CSV combinado (falla + nominal) usado para mu_experto


class NominalFlightEnv(gym.Env):

    def __init__(self, gui=False, t_falla_min=None, t_falla_max=None,
                 controlador="pid", fault_aware=False, niveles_falla_pct=None):
        super().__init__()
        # t_falla_min/t_falla_max quedan aceptados por compatibilidad con los
        # llamadores existentes pero ya NO se usan: la falla se activa por
        # progreso de ruta (mitad del tramo de crucero), igual que
        # generar_datos_pid_falla_leve.py -- ver _idx_falla() en reset()/step().
        self.gui       = gui
        self.max_pasos = int(DURACION_SEG * CTRL_FREQ)
        self.stats     = cargar_stats(STATS_PATH)
        self.escalas   = cargar_escalas(self.stats)
        self.w         = np.zeros(N_FEATURES, dtype=np.float64)
        self.niveles       = list(NIVELES_FALLA_PCT if niveles_falla_pct is None else niveles_falla_pct)
        self.max_fault_pct = self.niveles[-1]  # ver set_max_fault() -- currículo externo lo ajusta

        if fault_aware and controlador != "robust":
            raise ValueError("fault_aware=True requiere controlador='robust'")
        self.fault_aware = fault_aware
        if controlador == "robust":
            self.pid_controller = RobustDSLPIDControl(drone_model=DroneModel.CF2X, fault_aware=fault_aware)
        else:
            self.pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)

        self._env = None
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def set_reward_weights(self, w):
        self.w = np.asarray(w, dtype=np.float64)

    def set_max_fault(self, pct):
        self.max_fault_pct = float(np.clip(pct, self.niveles[0], self.niveles[-1]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # El PID se reusa entre episodios (se crea una sola vez en __init__);
        # sin este reset, los términos integral/derivativo del episodio anterior
        # contaminan el arranque del nuevo, igual que generar_datos_pid_falla_leve.py
        # crea un controlador nuevo por episodio.
        self.pid_controller.reset()

        # Igual que el experto (CSV combinado falla+nominal): la mitad de los
        # episodios son sin falla (0%), para que el alumno tambien demuestre
        # como vuela limpio, no solo como recupera fallas.
        if np.random.random() < PROB_NOMINAL:
            fault_pct = 0.0
        else:
            niveles_disponibles = [n for n in self.niveles if n <= self.max_fault_pct + 1e-9]
            fault_pct = float(np.random.choice(niveles_disponibles))
        self.severidad = 1.0 - fault_pct

        # A y B independientes, uniformes en la caja de vuelo -- igual que
        # generar_datos_pid_falla_leve*.py y fault_env_residual*.py, para que
        # la distribucion de rutas coincida con la del CSV del experto.
        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            b_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
                break
        self.punto_A   = np.array([a_xy[0], a_xy[1], Z_SUELO])
        self.punto_B   = np.array([b_xy[0], b_xy[1], Z_SUELO])
        self.waypoints = generar_waypoints(self.punto_A, self.punto_B)
        self.wp_medio_crucero = max(1, (len(self.waypoints) - 2) // 2)

        if self._env is None:
            self._env = nueva_env(self.punto_A, gui=self.gui)
        else:
            self._env.INIT_XYZS = self.punto_A.reshape(1, 3)
        obs_raw, _ = self._env.reset()

        self.wp_idx     = 0
        self.paso       = 0
        self.action_rpm = np.ones((1, 4)) * HOVER_RPM
        self.obs_raw    = obs_raw

        self.dist_prev_z    = distancia_z(obs_raw[0][0:3], self.punto_B, self.escalas)
        self.rpm_anterior   = np.ones(4, dtype=np.float64) * HOVER_RPM
        self.vel_z_anterior = float(obs_raw[0][12])
        self.falla_activa   = False
        self.falla_activada = False  # sticky: una vez True, se queda asi

        return self._build_obs(obs_raw, self.waypoints[0]), {}

    def step(self, ppo_delta_norm):
        self.paso += 1

        ta = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]

        if (not self.falla_activada
                and _fase_de_waypoint(self.wp_idx, len(self.waypoints)) == 1
                and self.wp_idx >= self.wp_medio_crucero):
            self.falla_activada = True
        falla_activa = self.falla_activada
        if self.fault_aware:
            factors = np.ones(4)
            if falla_activa:
                factors[MOTOR_FALLA] = self.severidad
            self.pid_controller.set_motor_effectiveness(factors)

        # PID genera la RPM base a partir del ultimo estado conocido (previo a este paso)
        base_rpm, _, _ = self.pid_controller.computeControlFromState(
            control_timestep=1.0/CTRL_FREQ,
            state=self.obs_raw[0],
            target_pos=ta,
            target_rpy=np.zeros(3),
        )
        base_rpm = np.clip(base_rpm, MIN_RPM, MAX_RPM)

        # PPO aprende el delta residual sobre esa base
        delta        = np.asarray(ppo_delta_norm, dtype=np.float64) * DELTA_MAX
        rpm_comandado = np.clip(base_rpm + delta, MIN_RPM, MAX_RPM)

        # Falla activa desde que falla_activada queda en True (ver arriba)
        rpm_final = rpm_comandado.copy()
        if falla_activa:
            rpm_final[MOTOR_FALLA] *= self.severidad
        self.action_rpm = rpm_final.reshape(1, 4)

        # Aplicar la accion AHORA y obtener el estado resultante (en sincronia con rpm_final)
        obs_raw, _, term, trunc, _ = self._env.step(self.action_rpm)
        pos     = obs_raw[0][0:3].copy()
        rpy     = obs_raw[0][7:10]
        ang_vel = obs_raw[0][13:16]
        vel     = obs_raw[0][10:13]
        self.obs_raw = obs_raw
        es_caida_ahora = es_caida(obs_raw, pos, self.paso, self.wp_idx, len(self.waypoints))

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
            info["falla_activada"] = self.falla_activada
            info["fault_pct"] = 1.0 - self.severidad
        return self._build_obs(obs_raw, ta_obs), reward, done, False, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _build_obs(self, obs_raw, ta):
        estado_norm = normalizar_estado(obs_raw, ta, self.stats)
        fault_pct  = 1.0 - self.severidad if self.falla_activa else 0.0
        fault_info = np.array([float(self.falla_activa), fault_pct], dtype=np.float32)
        model_pred = np.zeros(4, dtype=np.float32)
        return np.concatenate([estado_norm, model_pred, fault_info]).astype(np.float32)
