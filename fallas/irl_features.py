"""
Vector de características φ(s,a) para Apprenticeship Learning (Abbeel & Ng, 2004).

Se usa tanto para calcular las expectativas de características del experto PID
(calcular_mu_experto.py, offline sobre el CSV de 800 episodios) como para evaluar
la recompensa aprendida en línea (nominal_flight_env.py, fault_env_residual_irl.py).
Cada componente se escala por la desviación estándar del dataset
(results/stats_normalizacion.json) para que las features queden en una escala
comparable (~O(1)) y el algoritmo de proyección tenga una geometría bien
condicionada.

Diseño alineado con "Learning-Based Passive Fault-Tolerant Control of a Quadrotor
with Rotor Failure" (arXiv:2503.02649): oscilación (Δrpm entre pasos, no desviación
del hover), velocidad 3D completa, aceleración vertical, y velocidad angular
restringida a roll/pitch (yaw se excluye — bajo falla severa, hasta 80% de pérdida
en este proyecto, el yaw puede volverse incontrolable, y el PID experto —que solo
vuela sin falla— nunca visita ese régimen, así que no hay señal útil del experto
para calibrar un peso ahí). `estabilidad_altura` y `progreso` se mantienen aunque
el paper no las tenga: la tarea del paper es solo estabilizarse en un punto fijo;
la de este proyecto es completar una ruta con aterrizaje al final, que el paper no
necesita resolver. `esfuerzo_motores`/`balance_motores`/`estabilidad_actitud`
(versión anterior de 8 features) se eliminaron: la primera se redefine como
oscilación, actitud se descarta por la misma razón que yaw, y balance de motores
no tenía respaldo ni en el paper ni en la tarea.
"""

import numpy as np

HOVER_RPM = 14300  # igual que comparar_base.HOVER_RPM
CTRL_FREQ = 48      # igual que comparar_base.CTRL_FREQ
DT = 1.0 / CTRL_FREQ
G = 9.81  # m/s^2 — escala física para aceleración vertical (no hay un stat "accel"
          # en stats_normalizacion.json, así que se usa una constante física en vez
          # de una desviación estándar del dataset)

FEATURE_NAMES = [
    "proximidad_objetivo",
    "estabilidad_altura",
    "estabilidad_angular_rp",
    "velocidad",
    "oscilacion",
    "aceleracion_vertical",
    "progreso",
    "penalizacion_caida",
]
N_FEATURES = len(FEATURE_NAMES)

CAIDA_PENALTY = -1.0  # magnitud fija del "golpe" al caer (ver phi())


def cargar_escalas(stats):
    """stats: dict como el que devuelve comparar_base.cargar_stats (col -> (mean, std))."""
    return {
        "err":    np.array([stats["err_x"][1], stats["err_y"][1], stats["err_z"][1]]),
        "ang_rp": np.array([stats["ang_x"][1], stats["ang_y"][1]]),
        "vel":    np.array([stats["vel_x"][1], stats["vel_y"][1], stats["vel_z"][1]]),
        "motor":  np.array([stats["motor_0"][1], stats["motor_1"][1],
                             stats["motor_2"][1], stats["motor_3"][1]]),
    }


def distancia_z(pos, punto_final, escalas):
    """Distancia al punto (final o local), con cada eje escalado por su desviación estándar."""
    err_z = (np.asarray(punto_final) - np.asarray(pos)) / escalas["err"]
    return float(np.linalg.norm(err_z))


def phi(pos, rpy, ang_vel, vel, wp_actual, punto_final, rpm, rpm_anterior,
        vel_z_anterior, dist_prev_z, escalas, es_caida_ahora=False):
    """
    Calcula φ(s,a) (8,) y la distancia actual al destino final (para pasarla
    como dist_prev_z en el siguiente paso). Entradas físicas crudas (no
    normalizadas por z-score de estado, salvo la escala interna de φ).

    pos, rpy, ang_vel, vel : arrays (3,)
    wp_actual    : array (3,) — waypoint local activo (perfil crucero/aterrizaje,
                   usado solo por estabilidad_altura)
    punto_final  : array (3,) — destino real del episodio (punto_B, fijo durante
                   todo el episodio; usado por proximidad_objetivo y progreso)
    rpm, rpm_anterior : arrays (4,) — RPM comandada este paso y el paso anterior
    vel_z_anterior     : float — vel[2] del paso anterior
    dist_prev_z  : float — distancia_z() a punto_final del paso anterior (usar
                   distancia_z(pos_inicial, punto_final, escalas) en el primer paso)
    escalas      : dict de cargar_escalas()
    es_caida_ahora : bool — True si este paso es el que dispara es_caida() en el
                   entorno que llama a phi(). El PID (experto) nunca se cae, así
                   que mu_experto en esta componente es ~0 siempre; una política
                   candidata que se cae mucho queda con mu_i bien negativo ahí,
                   lo que hace que la búsqueda le asigne peso positivo de forma
                   automática -- sin esto, nada en la recompensa notaba que
                   caerse es catastrófico (ver resultados de "prueba_05" en
                   results/pruebas_irl/, donde 40-100% de los episodios de
                   evaluación terminaban en caída en cada iteración).

    Devuelve (phi_vec, dist_actual_z).
    """
    dist_actual_z = distancia_z(pos, punto_final, escalas)

    ang_rp_n   = np.asarray(ang_vel[:2]) / escalas["ang_rp"]
    vel_n      = np.asarray(vel) / escalas["vel"]
    rpm_delta_n = (np.asarray(rpm) - np.asarray(rpm_anterior)) / escalas["motor"]
    accel_vertical = (float(vel[2]) - float(vel_z_anterior)) / DT

    proximidad_objetivo    = -dist_actual_z
    estabilidad_altura     = -abs((float(wp_actual[2]) - float(pos[2])) / np.mean(escalas["err"]))
    estabilidad_angular_rp = -float(np.linalg.norm(ang_rp_n))
    velocidad              = -float(np.linalg.norm(vel_n))
    oscilacion              = -float(np.mean(np.abs(rpm_delta_n)))
    aceleracion_vertical   = -abs(accel_vertical / G)
    progreso                = float(np.clip(dist_prev_z - dist_actual_z, -1.0, 1.0))
    penalizacion_caida      = CAIDA_PENALTY if es_caida_ahora else 0.0

    vec = np.array([
        proximidad_objetivo, estabilidad_altura, estabilidad_angular_rp,
        velocidad, oscilacion, aceleracion_vertical, progreso,
        penalizacion_caida,
    ], dtype=np.float64)

    return vec, dist_actual_z


def horizonte_efectivo(T, gamma):
    """
    Suma de los descuentos gamma^t efectivamente usados al acumular phi sobre
    un episodio de T pasos: sum_{t=0}^{T-1} gamma^t = (1 - gamma^T) / (1 - gamma).

    Dividir la suma acumulada de phi entre esto (en vez de dejarla cruda)
    convierte "costo total acumulado en el episodio" en "costo promedio por
    paso, ponderado por el mismo descuento". Sin esto, un episodio corto (por
    ejemplo porque la política se cayó pronto) acumula menos costo total que
    uno largo aunque vuele peor paso a paso, haciendo que mu_bar parezca
    "mejor que el experto" en las features siempre-negativas solo por haber
    durado menos -- fue la causa del estancamiento visto en irl_convergencia.csv
    de varias corridas (proximidad_objetivo/estabilidad_altura/oscilacion, y
    después casi todas las features, quedando en w=0 de forma consistente).

    Cuando gamma=1 (sin descuento) esto se reduce al promedio simple: T.
    """
    if gamma >= 1.0:
        return float(T)
    return (1.0 - gamma ** T) / (1.0 - gamma)
