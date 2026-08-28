"""
Entorno RL para compensación de falla de motor (política residual).

Mamba actúa como controlador base durante todo el episodio.
PPO aprende únicamente a redistribuir empuje después de T_FALLA_SEG.

Observación (24-dim):
    18  estado normalizado (pos, vel, rpy, ang_vel, target, error)
     4  salida cruda de Mamba (pred antes de desnormalizar)
     1  falla_activa (0 antes de falla, 1 después)
     1  fault_pct (porcentaje de potencia perdida)
Accion (4-dim) en [-1, 1] -> delta en [-DELTA_MAX, +DELTA_MAX]
"""

import sys
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import (
    cargar_stats, normalizar_estado, desnormalizar_accion,
    generar_waypoints, nueva_env,
    STATS_PATH, Z_SUELO, Z_CRUCERO,
    CTRL_FREQ, HOVER_RPM, UMBRAL_WAYPOINT, INPUT_COLS,
    MIN_RPM, MAX_RPM,
    MOTOR_FALLA, T_FALLA_SEG, DURACION_FALLA_SEG,
    es_caida, es_aterrizaje,
    MambaDrone, LSTMDrone,
)

XY_LIM   = 2.0   # rango de vuelo igual al del dataset de Mamba
MIN_DIST = 0.8   # distancia mínima horizontal A→B

MAMBA_MODEL_PATH = str(_ROOT / "results" / "modelo_mamba.pth")
LSTM_MODEL_PATH  = str(_ROOT / "results" / "modelo_lstm.pth")
DELTA_MAX    = 1500
DURACION_SEG = DURACION_FALLA_SEG


class FaultResidualEnv(gym.Env):
    """
    PPO aprende el delta de compensacion sobre la accion de Mamba.
    La severidad se aleatoriza en cada reset() para generalizar la politica.

    Antes de T_FALLA_SEG : Mamba controla solo, delta = 0, reward = 0.
    Desde  T_FALLA_SEG   : falla activa, PPO aplica delta, reward activo.
    """

    def __init__(self, severidad=0.95, gui=False, t_falla_override=None, modelo="mamba"):
        super().__init__()
        self.gui       = gui
        self.severidad = severidad
        self.max_pasos = int(DURACION_SEG * CTRL_FREQ)
        t_falla_seg    = t_falla_override if t_falla_override is not None else T_FALLA_SEG
        self.t_falla   = int(t_falla_seg * CTRL_FREQ)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats  = cargar_stats(STATS_PATH)

        if modelo == "lstm":
            self.base_model = LSTMDrone()
            model_path = LSTM_MODEL_PATH
        else:
            self.base_model = MambaDrone()
            model_path = MAMBA_MODEL_PATH
        self.base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.base_model.to(self.device)
        self.base_model.eval()

        self._env          = None
        self.severidad     = 1.0 - 0.02
        self.max_fault_pct = 0.02
        self.model_pred    = np.zeros(4, dtype=np.float32)
        self._delta_acum   = []   # |delta_norm| por step post-falla

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def set_max_fault(self, pct):
        self.max_fault_pct = float(np.clip(pct, 0.02, 0.80))

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Aleatorizar severidad en cada episodio
        fault_pct      = np.random.uniform(0.02, self.max_fault_pct)
        self.severidad = 1.0 - fault_pct

        # Trayectoria aleatoria dentro del volumen de Mamba ([-2,2]m, dist>=0.8m)
        while True:
            a_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            b_xy = np.random.uniform(-XY_LIM, XY_LIM, size=2)
            if np.linalg.norm(a_xy - b_xy) >= MIN_DIST:
                break
        self.punto_A  = np.array([a_xy[0], a_xy[1], Z_SUELO])
        self.punto_B  = np.array([b_xy[0], b_xy[1], Z_SUELO])
        self.waypoints = generar_waypoints(self.punto_A, self.punto_B)

        if self._env is None:
            self._env = nueva_env(self.punto_A, gui=self.gui)
        else:
            self._env.INIT_XYZS = self.punto_A.reshape(1, 3)

        obs_raw, _ = self._env.reset()
        self._delta_acum          = []
        self.bono_recuperacion_dado = False
        self.ventana    = deque(
            [np.zeros(len(INPUT_COLS), dtype=np.float32)] * 50, maxlen=50
        )
        self.action_rpm = np.ones((1, 4)) * HOVER_RPM
        self.paso       = 0
        self.wp_idx     = 0
        self.model_pred = np.zeros(4, dtype=np.float32)
        self.obs_raw    = obs_raw
        self.prev_dist  = 0.0

        return self._build_obs(obs_raw), {}

    def step(self, ppo_delta_norm):
        # 1. Avanzar simulacion con la accion del paso anterior
        obs_raw, _, term, trunc, _ = self._env.step(self.action_rpm)
        pos = obs_raw[0][0:3].copy()
        self.paso += 1

        # 2. Actualizar ventana temporal con la nueva observacion
        ta = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
        self.ventana.append(normalizar_estado(obs_raw, ta, self.stats))

        # 3. Mamba genera accion base
        x = torch.tensor(
            np.array(self.ventana), dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.base_model(x).cpu().numpy()[0]
        self.model_pred = pred
        base_rpm = np.clip(desnormalizar_accion(pred, self.stats), MIN_RPM, MAX_RPM)

        # 4. PPO solo actua despues de la falla
        if self.paso >= self.t_falla:
            delta        = ppo_delta_norm * DELTA_MAX
            cmd_rpm      = np.clip(base_rpm + delta, MIN_RPM, MAX_RPM)
            action_final = cmd_rpm.copy()
            action_final[MOTOR_FALLA] *= self.severidad        # falla sobre cmd ya corregido
            self._delta_acum.append(float(np.mean(np.abs(ppo_delta_norm))))
        else:
            action_final = base_rpm                            # sin falla, sin PPO

        self.action_rpm = action_final.reshape(1, 4)
        self.obs_raw    = obs_raw

        # 5. Reward solo post-falla
        reward  = 0.0
        done    = False
        outcome = None

        if self.paso >= self.t_falla:
            dt   = 1.0 / CTRL_FREQ
            rpy  = obs_raw[0][7:10]
            dist = float(np.linalg.norm(ta - pos))

            # Costo temporal: evita que se quede flotando sin decidir
            reward -= 0.01

            z_objetivo = ta[2]
            vel_z = float(obs_raw[0][12])
            ang_vel = obs_raw[0][13:16]
            en_fase_aterrizaje = z_objetivo < Z_CRUCERO * 0.5

            # Progreso hacia el waypoint (clippeado para evitar saltos grandes)
            if self.paso == self.t_falla:
                self.prev_dist = dist
            progress = float(np.clip(self.prev_dist - dist, -0.10, 0.10))
            self.prev_dist = dist

            # Estabilidad (siempre activa)
            reward -= 2.0 * (abs(rpy[0]) + abs(rpy[1])) * dt
            reward -= 0.5 * float(np.sum(np.abs(ang_vel))) * dt

            if en_fase_aterrizaje:
                # Penalización fuerte de altitud para forzar el descenso
                reward -= 8.0 * abs(pos[2] - z_objetivo) * dt
                # Recompensar explícitamente el descenso controlado
                if vel_z < 0:
                    reward += 4.0 * abs(vel_z) * dt
                # Progreso 3D hacia Bs (cubre también corrección de XY si el drone derivó)
                reward += 8.0 * progress
            else:
                # Crucero: progreso horizontal + penalización suave de altitud
                reward += 8.0 * progress
                reward -= 3.0 * abs(pos[2] - z_objetivo) * dt
                if vel_z < -0.2:
                    reward -= 4.0 * abs(vel_z) * dt
                # Zona de emergencia (caída no deseada)
                if pos[2] < Z_CRUCERO * 0.5:
                    if vel_z < -0.3:
                        reward -= 4.0 * abs(vel_z) * dt
                    reward -= 8.0 * progress

            # Bono único de recuperación (solo en crucero, no durante aterrizaje)
            pasos_post = self.paso - self.t_falla
            tilt  = abs(rpy[0]) + abs(rpy[1])
            omega = float(np.linalg.norm(ang_vel))
            estable = tilt < 0.20 and omega < 1.5 and pos[2] > Z_CRUCERO * 0.8
            if (
                not self.bono_recuperacion_dado
                and int(0.5 * CTRL_FREQ) <= pasos_post <= int(3.0 * CTRL_FREQ)
                and estable
                and not en_fase_aterrizaje
            ):
                reward += 5.0
                self.bono_recuperacion_dado = True

            # Waypoint alcanzado
            if dist < UMBRAL_WAYPOINT:
                reward += 10.0
                self.wp_idx += 1
                if self.wp_idx >= len(self.waypoints):
                    reward += 100.0
                    done    = True
                    outcome = "llego"
                else:
                    ta_nuevo = self.waypoints[self.wp_idx]
                    self.prev_dist = float(np.linalg.norm(ta_nuevo - pos))

            # Eventos terminales
            if not done:
                if es_caida(obs_raw, pos, self.paso):
                    reward -= 120.0
                    done    = True
                    outcome = "cayo"
                elif es_aterrizaje(obs_raw, pos, self.paso):
                    reward += 40.0
                    done    = True
                    outcome = "aterrizo"

        # Timeout: penaliza quedarse sin resolver nada
        if (self.paso >= self.max_pasos or term or trunc) and not done:
            reward -= 30.0
            done    = True
            outcome = "tiempo"

        info = {}
        if done:
            mision_pct = 100.0 * self.wp_idx / max(len(self.waypoints), 1)
            if self.wp_idx >= len(self.waypoints):
                mision_pct = 100.0
            info["outcome"]     = outcome if outcome else "tiempo"
            info["mision_pct"]  = mision_pct
            info["delta_medio"] = float(np.mean(self._delta_acum)) if self._delta_acum else 0.0
            info["severidad"]   = self.severidad
            info["punto_A"]     = self.punto_A.tolist()
            info["punto_B"]     = self.punto_B.tolist()

        return self._build_obs(obs_raw), reward, done, False, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── Observacion ──────────────────────────────────────────────────────────

    def _build_obs(self, obs_raw):
        ta          = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
        estado_norm = normalizar_estado(obs_raw, ta, self.stats)
        fault_pct    = 1.0 - self.severidad
        falla_activa = float(self.paso >= self.t_falla)
        fault_info   = np.array([falla_activa, fault_pct], dtype=np.float32)
        return np.concatenate([estado_norm, self.model_pred, fault_info]).astype(np.float32)
