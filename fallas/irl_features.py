"""
Vector de características φ(s,a) para Apprenticeship Learning (Abbeel & Ng, 2004).

Se usa tanto para calcular las expectativas de características del experto PID
(calcular_mu_experto.py, offline sobre el CSV de 800 episodios) como para evaluar
la recompensa aprendida en línea (fault_env_residual_irl.py). Cada componente se
escala por la desviación estándar del dataset (results/stats_normalizacion.json)
para que las ocho features queden en una escala comparable (~O(1)) y el algoritmo
de proyección tenga una geometría bien condicionada — sin esa normalización,
esfuerzo_motores (escala ~RPM, miles) dominaría por completo a proximidad_objetivo
(escala ~metros).
"""

import os
import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(_HERE, "..", "results", "stats_normalizacion.json")

HOVER_RPM = 14300  # igual que comparar_base.HOVER_RPM

FEATURE_NAMES = [
    "proximidad_objetivo",
    "estabilidad_altura",
    "estabilidad_actitud",
    "estabilidad_angular",
    "estabilidad_vertical",
    "esfuerzo_motores",
    "balance_motores",
    "progreso",
]
N_FEATURES = len(FEATURE_NAMES)


def cargar_escalas(stats):
    """stats: dict como el que devuelve comparar_base.cargar_stats (col -> (mean, std))."""
    return {
        "err":   np.array([stats["err_x"][1], stats["err_y"][1], stats["err_z"][1]]),
        "roll":  stats["roll"][1],
        "pitch": stats["pitch"][1],
        "ang":   np.array([stats["ang_x"][1], stats["ang_y"][1], stats["ang_z"][1]]),
        "vel_z": stats["vel_z"][1],
        "motor": np.array([stats["motor_0"][1], stats["motor_1"][1],
                            stats["motor_2"][1], stats["motor_3"][1]]),
    }


def distancia_z(pos, wp_actual, escalas):
    """Distancia al waypoint actual, con cada eje escalado por su desviación estándar."""
    err_z = (np.asarray(wp_actual) - np.asarray(pos)) / escalas["err"]
    return float(np.linalg.norm(err_z))


def phi(pos, rpy, ang_vel, vel_z, wp_actual, rpm, dist_prev_z, escalas):
    """
    Calcula φ(s,a) (8,) y la distancia actual (para pasarla como dist_prev_z
    en el siguiente paso). Todas las entradas son cantidades físicas crudas
    (no normalizadas por z-score de estado, salvo la escala interna de φ).

    pos, rpy, ang_vel : arrays (3,)
    vel_z              : float
    wp_actual          : array (3,)
    rpm                : array (4,) — RPM comandada
    dist_prev_z         : float — distancia_z() del paso anterior (usar 0.0 en el primer paso)
    escalas             : dict de cargar_escalas()

    Devuelve (phi_vec, dist_actual_z).
    """
    dist_actual_z = distancia_z(pos, wp_actual, escalas)

    roll_z  = float(rpy[0]) / escalas["roll"]
    pitch_z = float(rpy[1]) / escalas["pitch"]
    ang_z   = np.asarray(ang_vel) / escalas["ang"]
    velz_z  = float(vel_z) / escalas["vel_z"]
    rpm_dev_z = (np.asarray(rpm) - HOVER_RPM) / escalas["motor"]

    proximidad_objetivo  = -dist_actual_z
    estabilidad_altura   = -abs((float(wp_actual[2]) - float(pos[2])) / np.mean(escalas["err"]))
    estabilidad_actitud  = -(abs(roll_z) + abs(pitch_z))
    estabilidad_angular  = -float(np.linalg.norm(ang_z))
    estabilidad_vertical = -abs(velz_z)
    esfuerzo_motores     = -float(np.mean(np.abs(rpm_dev_z)))
    balance_motores      = -float(np.std(rpm_dev_z))
    progreso             = float(np.clip(dist_prev_z - dist_actual_z, -1.0, 1.0))

    vec = np.array([
        proximidad_objetivo, estabilidad_altura, estabilidad_actitud,
        estabilidad_angular, estabilidad_vertical, esfuerzo_motores,
        balance_motores, progreso,
    ], dtype=np.float64)

    return vec, dist_actual_z
