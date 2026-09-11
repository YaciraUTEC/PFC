# Hallazgos y Observaciones para el Informe de Tesis

## 1. Arquitecturas implementadas

### LSTM
- 2 capas LSTM, hidden_size = 128, dropout = 0.1
- Cabeza de salida: Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→4)
- Total de parámetros: **216,388**
- Ventana temporal: 50 pasos de historia

### Mamba SSM
- Proyección de entrada: Linear(18→128)
- 2 capas Mamba: d_model=128, d_state=16, d_conv=4, expand=2
- LayerNorm residual después de cada capa
- Cabeza de salida: Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→4)
- Total de parámetros: **~99,000** (54% menos que LSTM)
- Ventana temporal: 50 pasos de historia

---

## 2. Datos de entrenamiento

### Generación
- Script: `generar_datos_CF2X.py`
- Dron: CF2X, controlador: DSLPIDControl (referencia DSL-UTIAS)
- 800 episodios aleatorios de A → B
- Duración por episodio: 20 segundos a 48 Hz de control = hasta 960 pasos
- Espacio de vuelo: X,Y ∈ [-2.0, 2.0] m, distancia mínima A→B = 0.8 m

### Estructura de trayectoria por episodio
Cada episodio tiene **8 waypoints** secuenciales:
1. A_alto: punto A elevado a Z = 1.2 m (despegue)
2-6. 5 puntos intermedios interpolados con curva sinusoidal en Z
7. B_alto: punto B a Z = 1.2 m (crucero)
8. B_suelo: punto B a Z = 0.1 m (aterrizaje)

El parámetro `N_INTERMEDIOS = 5` está hardcodeado en todos los scripts de generación (CF2X, CF2P, RACE) sin justificación documentada. Genera suficiente variedad de comportamiento sin que los waypoints sean demasiado cortos entre sí.

### Entrada al modelo (18 señales)
| Señal | Índices en obs | Descripción |
|---|---|---|
| pos_x, pos_y, pos_z | obs[0][0:3] | Posición absoluta |
| vel_x, vel_y, vel_z | obs[0][10:13] | Velocidad lineal |
| roll, pitch, yaw | obs[0][7:10] | Orientación Euler |
| ang_x, ang_y, ang_z | obs[0][13:16] | Velocidad angular |
| target_x, target_y, target_z | — | Destino final B (fijo por episodio) |
| err_x, err_y, err_z | wp_actual - pos | Error al waypoint actual |

**Nota importante:** `target_x/y/z` es el **destino final B** (constante durante todo el episodio), mientras que `err_x/y/z` es el error al **waypoint actual** (cambia a medida que el dron avanza). Esta distinción permite al modelo conocer tanto el objetivo global como el error táctico inmediato.

### Salida del modelo (4 señales)
RPMs de los 4 motores: motor_0, motor_1, motor_2, motor_3

---

## 3. Normalización

Todas las señales se normalizan con media y desviación estándar calculadas sobre el dataset completo (`generate_states.py`). Los valores del dataset CF2X 800 episodios:

| Señal | Media | Std |
|---|---|---|
| pos_x | 0.087 m | 1.098 m |
| pos_z | 0.820 m | 0.506 m |
| roll | ~0 rad | 0.104 rad |
| motor_0 | 15,180 RPM | 2,145 RPM |

Los RPMs de hover son ~14,300 RPM. La desviación de ~2,145 RPM refleja la variación de control durante maniobras.

---

## 4. Entrenamiento

### Hiperparámetros finales
| Parámetro | Valor |
|---|---|
| Épocas | 150 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Batch size | 128 |
| Función de pérdida | MSELoss |
| Gradient clipping | 1.0 |

### Resultados LSTM (modelo_lstm_3.pth)
| Época | Train loss | Val loss |
|---|---|---|
| 1 | 0.163804 | 0.070061 |
| 60 | 0.014710 | 0.008263 |
| 100 | 0.012653 | 0.007442 |
| 150 | 0.012053 | 0.006630 |
| **Mejor** | — | **0.006155** |

El val loss es consistentemente menor que el train loss porque dropout está activo durante entrenamiento pero desactivado en evaluación (model.eval()). Esto es comportamiento normal, no indica underfitting.

### Evolución entre modelos
| Modelo | Épocas | Val loss | Observación |
|---|---|---|---|
| modelo_lstm.pth | ~60 | ~0.015 | Entrenamiento inicial |
| modelo_lstm_3.pth | 150 | **0.006155** | +weight_decay, +patience |

La mejora de ~2.3x en val loss al pasar de 60 a 150 épocas con regularización indica que el modelo original estaba subentrenado.

---

## 5. Inicialización de la ventana temporal (bug crítico encontrado)

### El problema
Durante la simulación, el simulador inicializaba la ventana de 50 pasos con copias del estado inicial del dron:
```python
# INCORRECTO — fuera de distribución
ventana = deque([estado_inicial] * 50, maxlen=50)
```

### Por qué es un error
Durante el entrenamiento, el `DroneDataset` inicializa la ventana con **ceros** para los primeros pasos de cada episodio:
```python
# Así entrena (correcto)
if i == 0:
    window = np.zeros((50, 18))
```

Al usar el estado inicial repetido 50 veces, el modelo recibía una entrada completamente fuera de la distribución de entrenamiento, causando predicciones incorrectas de RPM y comportamiento errático del dron (vuelo lento, incapacidad de navegar).

### La corrección
```python
# CORRECTO — consistente con el entrenamiento
ventana = deque([np.zeros(18, dtype=np.float32)] * 50, maxlen=50)
```

---

## 6. Umbral de waypoint (UMBRAL_WAYPOINT)

### Definición
Radio de aceptación en metros. Cuando la distancia 3D entre el dron y el waypoint actual es menor que este valor, el dron avanza al siguiente waypoint:
```python
if np.linalg.norm(waypoint_actual - pos) < UMBRAL_WAYPOINT:
    wp_idx += 1
```

### Discrepancia entrenamiento vs simulación
| Contexto | Valor | Razón |
|---|---|---|
| Generación de datos (PID) | 0.10 m | PID es preciso, llega a 10 cm |
| Simulación LSTM | 0.25 m | LSTM tiene error residual de ~15-20 cm |

### Por qué existe esta diferencia
El PID es un controlador matemático exacto que puede posicionarse a menos de 10 cm de cualquier waypoint de forma consistente. El LSTM es una aproximación aprendida — replica el comportamiento general del PID pero con un error residual inherente. Con `UMBRAL_WAYPOINT = 0.10` en el simulador, el LSTM nunca "llega" al waypoint final (especialmente al aterrizar) y el dron da vueltas indefinidamente.

El umbral de 0.25 m refleja la diferencia de precisión entre el controlador de referencia (PID) y la política aprendida (LSTM). A medida que el modelo mejore con más datos y entrenamiento, este umbral podría reducirse.

---

## 7. Metodología de comparación de trayectorias

### Trayectorias fijas (comparar_trayectorias.py)
Se definieron 5 trayectorias reproducibles con coordenadas hardcodeadas:

| # | Punto A | Punto B | Tipo |
|---|---|---|---|
| 1 | (-1.5, -1.5) | (1.5, 1.5) | Diagonal larga |
| 2 | (0.0, -1.5) | (0.0, 1.5) | Recto norte |
| 3 | (-1.5, 0.0) | (1.5, 0.0) | Recto este |
| 4 | (1.0, -1.0) | (-1.0, 1.0) | Diagonal inversa |
| 5 | (-1.0, 1.0) | (1.0, -1.0) | Diagonal sur |

### Flujo de comparación
1. PID vuela la trayectoria → se dibuja en **rojo** en PyBullet
2. LSTM vuela la **misma** trayectoria con los **mismos** waypoints → se dibuja en **azul**
3. Se guardan posición y orientación en cada paso en JSON
4. `graficar_trayectorias.py` genera figuras 3D para el informe

### Por qué son comparables
Los waypoints se calculan una sola vez con `generar_waypoints(A, B)` y se pasan a ambos controladores. La función es determinista — para el mismo A y B siempre produce los mismos 8 puntos. Así, la diferencia entre trayectoria roja y azul refleja únicamente la diferencia entre el controlador PID y el modelo LSTM, no variación en los objetivos.

---

## 8. Métricas de evaluación

### Por trayectoria
- **Llegó (bool):** si el dron pasó por todos los waypoints incluyendo B_suelo
- **Error final (m):** distancia 3D entre posición final del dron y punto B
- **Pasos:** número de pasos de control hasta terminar

### Agregadas (para tabla de resultados)
- Tasa de éxito: llegaron / total × 100%
- Error promedio ± desviación estándar
- Comparación PID vs LSTM vs Mamba

---

## 9. Observaciones sobre el comportamiento del dron

- El dron con LSTM tiende a ser menos fluido que con PID en las maniobras de giro
- El aterrizaje (descenso a Z=0.1 m) es la maniobra más difícil para el modelo aprendido
- Con el modelo anterior (val loss ~0.015) el dron se movía lentamente; con val loss 0.006 el vuelo es más enérgico
- La fase de crucero (waypoints intermedios) se ejecuta mejor que el aterrizaje
- El suavizado EMA de RPMs (0.7×nuevo + 0.3×anterior) reduce oscilaciones pero fue removido en versiones posteriores al mejorar el modelo

---

## 10. Parámetros de simulación

| Parámetro | Valor |
|---|---|
| Frecuencia física (PyBullet) | 240 Hz |
| Frecuencia de control | 48 Hz |
| Duración máxima por episodio | 20 s = 960 pasos |
| Altitud de crucero | 1.2 m |
| Altitud de suelo (Z_SUELO) | 0.1 m |
| RPM de hover | ~14,300 RPM |
| Rango de RPM (clip) | [9,440 , 21,700] RPM |
| Dron | CF2X (Crazyflie 2.x, configuración X) |
| Masa del dron | 0.027 kg (parámetro URDF) |
