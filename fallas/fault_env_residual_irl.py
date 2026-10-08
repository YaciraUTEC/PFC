

import json
import sys
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))
sys.path.insert(0, str(Path(__file__).parent))

from comparar_base import (  # noqa: E402
    cargar_stats, normalizar_estado, desnormalizar_accion,
    generar_waypoints, nueva_env,
    STATS_PATH, Z_SUELO, Z_CRUCERO,
    CTRL_FREQ, HOVER_RPM, UMBRAL_WAYPOINT, INPUT_COLS,
    MIN_RPM, MAX_RPM,
    MOTOR_FALLA, T_FALLA_MIN, T_FALLA_MAX, DURACION_FALLA_SEG,
    es_caida, es_aterrizaje, TRAYECTORIAS,
    MambaDrone, LSTMDrone,
)
from irl_features import phi, distancia_z, cargar_escalas  # noqa: E402

XY_LIM   = 2.0   # rango de vuelo igual al del dataset de Mamba
MIN_DIST = 0.8   # distancia mínima horizontal A→B

MAMBA_MODEL_PATH   = str(_ROOT / "results" / "modelo_mamba.pth")
LSTM_MODEL_PATH    = str(_ROOT / "results" / "modelo_lstm.pth")
# Permite override via variable de entorno IRL_WEIGHTS_PATH (set por entrenar_rl.py)
import os as _os
IRL_WEIGHTS_PATH   = _os.environ.get("IRL_WEIGHTS_PATH", str(_ROOT / "results" / "irl_weights.json"))
DETECCION_MODEL_PATH = str(_ROOT / "results" / "modelo_deteccion_mamba.pth")
STATS_DETECCION_PATH = str(_ROOT / "results" / "stats_deteccion.json")
DELTA_MAX    = 1500
DURACION_SEG = DURACION_FALLA_SEG

# ── Recompensa "manual2" (y parte de tarea de "hibrido") ────────────────────
# Prioridades: 1) sobrevivir y estabilizar, 2) mantener altura, 3) avanzar.
R_VIVO            = 0.05    # por paso post-falla: seguir en el aire vale más que caer
K_TILT            = 2.0     # × (|roll|+|pitch|) × dt
K_OMEGA           = 0.5     # × |ω| × dt
K_ALT             = 3.0     # × |z − z_waypoint| × dt
Z_MIN_CRUCERO     = 0.3     # m — por debajo, en crucero, "arrastrarse" anula R_VIVO
R_SUELO           = -0.05
K_PROG            = 8.0     # × metros avanzados hacia el waypoint
PROG_CLIP         = 0.10    # m por paso
K_DELTA           = 0.01    # × ||delta_norm||² — correcciones suaves
R_WAYPOINT        = 10.0
R_LLEGO           = 100.0
R_ATERRIZO_EN_B   = 40.0    # solo si aterriza a < DIST_ATERRIZO_B (horizontal) del destino
R_ATERRIZO_LEJOS  = -60.0   # aterrizar lejos del destino ya no es una salida "segura"
DIST_ATERRIZO_B   = 0.3
R_CAIDA           = -120.0
# Timeout en el aire: 0 (ya cobró R_VIVO en cada paso)

# ── Parte IRL de "hibrido" ──────────────────────────────────────────────────
# Solo features de estilo de vuelo. Las de tarea (destino, velocidad, progreso,
# caída) las cubre la parte manual, y en irl_weights_2.json tenían signo invertido.
FEATURES_TAREA       = {"proximidad_objetivo", "velocidad", "progreso", "penalizacion_caida"}
MU_EXPERTO_HIBRIDO   = str(_ROOT / "results" / "mu_experto_mamba.npy")
ALPHA_HIBRIDO        = 0.05


def _cargar_pesos_irl(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["weights"], dtype=np.float64), data["feature_names"]


def _estado_crudo(obs_raw, wp_actual):
    """Vector de estado (18,) SIN normalizar, en el mismo orden que INPUT_COLS
    — usado por el detector de falla, que se normaliza con stats_deteccion.json
    (distribución de vuelo con falla, distinta de stats_normalizacion.json)."""
    estado = obs_raw[0]
    pos = estado[0:3]; rpy = estado[7:10]; vel = estado[10:13]; ang = estado[13:16]
    error = wp_actual - pos
    valores = {
        'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
        'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
        'roll':  rpy[0], 'pitch': rpy[1], 'yaw':   rpy[2],
        'ang_x': ang[0], 'ang_y': ang[1], 'ang_z': ang[2],
        'target_x': wp_actual[0], 'target_y': wp_actual[1], 'target_z': wp_actual[2],
        'err_x': error[0], 'err_y': error[1], 'err_z': error[2],
    }
    return np.array([valores[c] for c in INPUT_COLS], dtype=np.float32)


class FaultResidualEnvIRL(gym.Env):
    """
    PPO aprende el delta de compensación sobre la acción de Mamba.
    La severidad Y el instante de la falla se aleatorizan en cada reset()
    para generalizar la política. La recompensa post-falla es puramente
    w . phi(s,a) (sin ningún término manual).

    Antes de t_falla : Mamba controla solo, delta = 0, reward = 0.
    Desde  t_falla   : falla activa, PPO aplica delta, reward = w . phi(s,a).
    """

    def __init__(self, gui=False, modelo="mamba", t_falla_min=None, t_falla_max=None,
                fault_info_mode="oracle", rutas="aleatorias", falla_fija=None,
                modo_reward="irl", alpha=ALPHA_HIBRIDO):
        super().__init__()
        self.gui       = gui
        # modo_reward:
        #   "irl"     : R = w · φ                                  (original)
        #   "manual2" : R = recompensa manual (ver constantes R_*/K_*)
        #   "hibrido" : R = manual2 + alpha · w⁺ · (φ − μ_experto)
        if modo_reward not in ("irl", "manual2", "hibrido"):
            raise ValueError(f"modo_reward debe ser irl/manual2/hibrido, no {modo_reward!r}")
        self.modo_reward = modo_reward
        self.alpha       = alpha
        # falla_fija: pérdida fija del motor (ej. 0.10 = 10%) en todos los episodios.
        # None = aleatoria entre 2% y max_fault_pct (comportamiento original).
        self.falla_fija = falla_fija
        # "aleatorias": A y B al azar en cada episodio (comportamiento original).
        # "tesis": cada episodio usa una de las 5 TRAYECTORIAS de comparar_base.py,
        #          las mismas que evalúa falla_tesis.py.
        if rutas not in ("aleatorias", "tesis"):
            raise ValueError(f"rutas debe ser 'aleatorias' o 'tesis', no {rutas!r}")
        self.rutas     = rutas
        self.max_pasos = int(DURACION_SEG * CTRL_FREQ)
        self.t_falla_min = T_FALLA_MIN if t_falla_min is None else t_falla_min
        self.t_falla_max = T_FALLA_MAX if t_falla_max is None else t_falla_max
        self.fault_info_mode = fault_info_mode  # "oracle" | "detected"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats  = cargar_stats(STATS_PATH)
        self.escalas = cargar_escalas(self.stats)
        self.w, self.feature_names = _cargar_pesos_irl(IRL_WEIGHTS_PATH)
        if self.modo_reward == "hibrido":
            # w⁺: se anulan las features de tarea y cualquier peso negativo
            self.w_estilo = np.array([
                0.0 if (n in FEATURES_TAREA or wk < 0) else wk
                for n, wk in zip(self.feature_names, self.w)
            ], dtype=np.float64)
            self.mu_experto = np.load(MU_EXPERTO_HIBRIDO)

        if modelo == "lstm":
            self.base_model = LSTMDrone()
            model_path = LSTM_MODEL_PATH
        else:
            self.base_model = MambaDrone()
            model_path = MAMBA_MODEL_PATH
        self.base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.base_model.to(self.device)
        self.base_model.eval()

        self.detector = None
        if self.fault_info_mode == "detected":
            sys.path.insert(0, str(Path(__file__).parent))
            from entrenar_deteccion import MambaDetector  # import diferido (evita cargar torch/mamba dos veces)
            with open(STATS_DETECCION_PATH) as f:
                self.stats_deteccion = json.load(f)
            self.detector = MambaDetector()
            self.detector.load_state_dict(torch.load(DETECCION_MODEL_PATH, map_location=self.device))
            self.detector.to(self.device)
            self.detector.eval()

        self._env          = None
        self.severidad     = 1.0 - 0.02
        self.max_fault_pct = 0.02
        self.model_pred    = np.zeros(4, dtype=np.float32)

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

        # Severidad fija (falla_fija) o aleatoria en cada episodio
        if self.falla_fija is not None:
            fault_pct  = self.falla_fija
        else:
            fault_pct  = np.random.uniform(0.02, self.max_fault_pct)
        self.severidad = 1.0 - fault_pct

        # Falla en instante fijo (3 segundos) o aleatorio según t_falla_min/max
        # Si son iguales, usa ese valor fijo
        if self.t_falla_min == self.t_falla_max:
            t_falla_seg = self.t_falla_min
        else:
            t_falla_seg = np.random.uniform(self.t_falla_min, self.t_falla_max)
        self.t_falla = int(t_falla_seg * CTRL_FREQ)

        if self.rutas == "tesis":
            # Una de las 5 trayectorias fijas de la evaluación, al azar
            tray = TRAYECTORIAS[np.random.randint(len(TRAYECTORIAS))]
            self.punto_A = np.array(tray["A"], dtype=np.float64)
            self.punto_B = np.array(tray["B"], dtype=np.float64)
        else:
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
        self.ventana    = deque(
            [np.zeros(len(INPUT_COLS), dtype=np.float32)] * 50, maxlen=50
        )
        if self.detector is not None:
            self.ventana_deteccion = deque(
                [np.zeros(len(INPUT_COLS), dtype=np.float32)] * 50, maxlen=50
            )
        self.action_rpm = np.ones((1, 4)) * HOVER_RPM
        self.paso       = 0
        self.wp_idx     = 0
        self.prev_dist_wp = 0.0   # se fija en el primer paso post-falla (manual2/hibrido)
        self.model_pred = np.zeros(4, dtype=np.float32)
        self.obs_raw    = obs_raw
        self._delta_acum = []   # |delta_norm| por step post-falla, para info["delta_medio"]

        self.dist_prev_z    = distancia_z(obs_raw[0][0:3], self.punto_B, self.escalas)
        self.rpm_anterior   = np.ones(4, dtype=np.float64) * HOVER_RPM
        self.vel_z_anterior = float(obs_raw[0][12])

        return self._build_obs(obs_raw), {}

    def step(self, ppo_delta_norm):
        # 1. Avanzar simulacion con la accion del paso anterior
        obs_raw, _, term, trunc, _ = self._env.step(self.action_rpm)
        pos = obs_raw[0][0:3].copy()
        vel = obs_raw[0][10:13]
        self.paso += 1

        # 2. Actualizar ventana temporal con la nueva observacion
        ta = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
        self.ventana.append(normalizar_estado(obs_raw, ta, self.stats))
        if self.detector is not None:
            crudo = _estado_crudo(obs_raw, ta)
            normalizado = np.array([
                (crudo[i] - self.stats_deteccion[c][0]) / self.stats_deteccion[c][1]
                for i, c in enumerate(INPUT_COLS)
            ], dtype=np.float32)
            self.ventana_deteccion.append(normalizado)

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

        # 5. Recompensa (solo activa post-falla; ver modo_reward en __init__)
        reward  = 0.0
        done    = False
        outcome = None
        post_falla     = self.paso >= self.t_falla
        manual         = self.modo_reward in ("manual2", "hibrido")
        es_caida_ahora = es_caida(obs_raw, pos, self.paso)
        dist           = float(np.linalg.norm(ta - pos))

        if post_falla:
            rpy = obs_raw[0][7:10]
            ang_vel = obs_raw[0][13:16]

            vec, self.dist_prev_z = phi(
                pos, rpy, ang_vel, vel, ta, self.punto_B, action_final,
                self.rpm_anterior, self.vel_z_anterior, self.dist_prev_z, self.escalas,
                es_caida_ahora=es_caida_ahora,
            )
            if self.modo_reward == "irl":
                reward = float(np.dot(self.w, vec))
            else:
                if self.paso == self.t_falla:
                    self.prev_dist_wp = dist
                reward = self._recompensa_manual2(pos, rpy, ang_vel, ta, dist, ppo_delta_norm)
                if self.modo_reward == "hibrido":
                    # ≈ 0 volando como el experto, negativo si vuela peor
                    reward += self.alpha * float(np.dot(self.w_estilo, vec - self.mu_experto))

        # Los waypoints avanzan durante TODO el vuelo (también antes de la falla),
        # igual que en falla_tesis.py: a los 3 s el dron ya va a mitad de ruta.
        if dist < UMBRAL_WAYPOINT:
            self.wp_idx += 1
            if post_falla and manual:
                reward += R_WAYPOINT
            if self.wp_idx >= len(self.waypoints):
                done    = True
                outcome = "llego"
                if post_falla and manual:
                    reward += R_LLEGO
            elif post_falla and manual:
                self.prev_dist_wp = float(np.linalg.norm(self.waypoints[self.wp_idx] - pos))

        if not done:
            if es_caida_ahora:
                done    = True
                outcome = "cayo"
                if post_falla and manual:
                    reward += R_CAIDA
            elif es_aterrizaje(obs_raw, pos, self.paso):
                done    = True
                outcome = "aterrizo"
                if post_falla and manual:
                    cerca_B = np.linalg.norm(self.punto_B[:2] - pos[:2]) < DIST_ATERRIZO_B
                    reward += R_ATERRIZO_EN_B if cerca_B else R_ATERRIZO_LEJOS

        self.rpm_anterior   = action_final.astype(np.float64)
        self.vel_z_anterior = float(vel[2])

        # Timeout: termina el episodio, sin penalización manual añadida.
        if (self.paso >= self.max_pasos or term or trunc) and not done:
            done    = True
            outcome = "tiempo"

        info = {}
        if done:
            mision_pct = 100.0 * self.wp_idx / max(len(self.waypoints), 1)
            if self.wp_idx >= len(self.waypoints):
                mision_pct = 100.0
            info["outcome"]    = outcome if outcome else "tiempo"
            info["mision_pct"] = mision_pct
            info["delta_medio"] = float(np.mean(self._delta_acum)) if self._delta_acum else 0.0
            info["severidad"]  = self.severidad
            info["t_falla_seg"] = self.t_falla / CTRL_FREQ
            info["punto_A"]    = self.punto_A.tolist()
            info["punto_B"]    = self.punto_B.tolist()

        return self._build_obs(obs_raw), reward, done, False, info

    def _recompensa_manual2(self, pos, rpy, ang_vel, ta, dist, ppo_delta_norm):
        """Parte continua (por paso) de la recompensa manual2. Los eventos
        (waypoint, llegada, aterrizaje, caída) se suman en step()."""
        dt = 1.0 / CTRL_FREQ
        r  = R_VIVO
        r -= K_TILT  * (abs(rpy[0]) + abs(rpy[1])) * dt
        r -= K_OMEGA * float(np.linalg.norm(ang_vel)) * dt
        r -= K_ALT   * abs(pos[2] - ta[2]) * dt
        en_fase_aterrizaje = ta[2] < Z_CRUCERO * 0.5
        if not en_fase_aterrizaje and pos[2] < Z_MIN_CRUCERO:
            r += R_SUELO
        progreso = float(np.clip(self.prev_dist_wp - dist, -PROG_CLIP, PROG_CLIP))
        self.prev_dist_wp = dist
        r += K_PROG * progreso
        r -= K_DELTA * float(np.sum(np.square(ppo_delta_norm)))
        return r

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ── Observacion ──────────────────────────────────────────────────────────

    def _build_obs(self, obs_raw):
        ta          = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
        estado_norm = normalizar_estado(obs_raw, ta, self.stats)

        if self.detector is not None:
            x = torch.tensor(
                np.array(self.ventana_deteccion), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit_activa, pred_pct = self.detector(x)
            falla_activa = float(torch.sigmoid(logit_activa).item() > 0.5)
            fault_pct    = float(np.clip(pred_pct.item(), 0.0, 1.0)) if falla_activa else 0.0
        else:
            fault_pct    = 1.0 - self.severidad
            falla_activa = float(self.paso >= self.t_falla)

        fault_info = np.array([falla_activa, fault_pct], dtype=np.float32)
        return np.concatenate([estado_norm, self.model_pred, fault_info]).astype(np.float32)
