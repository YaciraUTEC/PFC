
import numpy as np

HOVER_RPM = 14300  # igual que comparar_base.HOVER_RPM
CTRL_FREQ = 48      # igual que comparar_base.CTRL_FREQ
DT = 1.0 / CTRL_FREQ
G = 9.81  

FEATURE_NAMES = [
    "proximidad_objetivo",
    "estabilidad_altura",
    "estabilidad_angular_rp",
    "velocidad",
    "oscilacion",
    "aceleracion_vertical",
    "progreso",
    "penalizacion_caida",
    "recuperacion_altura",
    "accion_residual_magnitud",
    "eficiencia_accion",
]
N_FEATURES = len(FEATURE_NAMES)

CAIDA_PENALTY = -10.0  # Comparable a proximidad_objetivo, suficiente penalización  
LIMITE_FEATURE = -8.0  
LIMITE_FEATURE_RUTA = -30.0  


def cargar_escalas(stats):
    return {
        "err":    np.array([stats["err_x"][1], stats["err_y"][1], stats["err_z"][1]]),
        "ang_rp": np.array([stats["ang_x"][1], stats["ang_y"][1]]),
        "vel":    np.array([stats["vel_x"][1], stats["vel_y"][1], stats["vel_z"][1]]),
        "motor":  np.array([stats["motor_0"][1], stats["motor_1"][1],
                             stats["motor_2"][1], stats["motor_3"][1]]),
    }


def distancia_z(pos, punto_final, escalas):
    err_z = (np.asarray(punto_final) - np.asarray(pos)) / escalas["err"]
    return float(np.linalg.norm(err_z))


def phi(pos, rpy, ang_vel, vel, wp_actual, punto_final, rpm, rpm_anterior,
        vel_z_anterior, dist_prev_z, escalas, es_caida_ahora=False, delta_residual=None):

    dist_actual_z = distancia_z(pos, punto_final, escalas)

    ang_rp_n   = np.asarray(ang_vel[:2]) / escalas["ang_rp"]
    vel_n      = np.asarray(vel) / escalas["vel"]
    rpm_delta_n = (np.asarray(rpm) - np.asarray(rpm_anterior)) / escalas["motor"]
    accel_vertical_n = ((float(vel[2]) - float(vel_z_anterior)) / DT) / escalas["vel"][2]

    # Todas las características normalizadas por sus escalas
    proximidad_objetivo    = max(-dist_actual_z, LIMITE_FEATURE_RUTA)
    estabilidad_altura     = max(-abs((float(wp_actual[2]) - float(pos[2])) / np.mean(escalas["err"])), LIMITE_FEATURE_RUTA)
    estabilidad_angular_rp = max(-float(np.linalg.norm(ang_rp_n)), LIMITE_FEATURE)
    velocidad              = max(-float(np.linalg.norm(vel_n)), LIMITE_FEATURE)
    oscilacion              = max(-float(np.mean(np.abs(rpm_delta_n))), LIMITE_FEATURE)
    aceleracion_vertical   = max(-abs(accel_vertical_n), LIMITE_FEATURE)
    # Progreso: cambio en distancia normalizada (sin clip artificial)
    progreso                = float(dist_prev_z - dist_actual_z)
    # Penalización por caída: normalizada por escala típica de características
    penalizacion_caida      = (CAIDA_PENALTY / np.mean(list(escalas.values())[0])) if es_caida_ahora else 0.0

    # NUEVAS CARACTERÍSTICAS DE RECUPERACIÓN
    # Recuperación de altura: positivo si está subiendo cuando está bajo
    altura_relativa_norm = float(pos[2]) / np.mean(escalas["err"])
    if altura_relativa_norm < 0.5:  # bajo
        recuperacion_altura = float(vel[2]) / escalas["vel"][2]
    else:
        recuperacion_altura = 0.0  # no aplica si está en altura normal
    recuperacion_altura = max(recuperacion_altura, LIMITE_FEATURE)

    # Magnitud de acción residual: cuánto esfuerzo está haciendo PPO
    if delta_residual is not None:
        delta_array = np.asarray(delta_residual, dtype=np.float64)
        delta_mag = float(np.linalg.norm(delta_array) / np.mean(escalas["motor"]))
    else:
        delta_mag = 0.0
    accion_residual_magnitud = max(-delta_mag, LIMITE_FEATURE)  # negativo = penalizar grandes acciones

    # Eficiencia de acción: delta * progreso / max_delta
    # Positivo si genera progreso; negativo si genera regresión
    if delta_residual is not None and progreso != 0.0:
        delta_array = np.asarray(delta_residual, dtype=np.float64)
        delta_mag = float(np.linalg.norm(delta_array) / np.mean(escalas["motor"]))
        eficiencia_accion = float(delta_mag * np.sign(progreso))
    else:
        eficiencia_accion = 0.0
    eficiencia_accion = max(eficiencia_accion, LIMITE_FEATURE)

    vec = np.array([
        proximidad_objetivo, estabilidad_altura, estabilidad_angular_rp,
        velocidad, oscilacion, aceleracion_vertical, progreso,
        penalizacion_caida, recuperacion_altura, accion_residual_magnitud,
        eficiencia_accion,
    ], dtype=np.float64)

    return vec, dist_actual_z


def horizonte_efectivo(T, gamma):
    if gamma >= 1.0:
        return float(T)
    return (1.0 - gamma ** T) / (1.0 - gamma)
