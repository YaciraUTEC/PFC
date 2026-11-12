# PFC

Control de un cuadricóptero Crazyflie ante pérdida de efectividad en motores.
Esta versión incluye las correcciones de IRL, un DSLPIDControl actualizado y
experimentos de compensación residual con evaluación por rutas reservadas.

## Controlador

- `gym_pybullet_drones/control/DSLPIDControl.py`: fuerza y pares en unidades SI,
  asignación acotada según efectividad de motores, realimentación angular y
  corrección de las integrales frente a saturación.
- `LegacyDSLPIDControl.py`: implementación original para reproducir los ensayos
  históricos. `RobustDSLPIDControl.py` conserva su comportamiento anterior.
- La efectividad suministrada es un factor de **RPM**. El empuje depende de su
  cuadrado. El controlador recibe esa estimación; no implementa un detector.

## Experimentos

- [`experiments/round3/README.txt`](experiments/round3/README.txt): comparación de
  recompensas basadas en CSIL, ME-IRL y RIME, más clonación y recompensa manual,
  usando un compensador LSTM común y cinco semillas de entrenamiento.
- [`experiments/README.txt`](experiments/README.txt): comparación histórica de
  LSTM, Mamba-1/2/3 SISO, GTrXL y LTC con el DSLPID original.
- [`experiments/SOURCES.txt`](experiments/SOURCES.txt): fuentes y alcance de las
  implementaciones de referencia. Se conservan sus licencias.

Son adaptaciones al proyecto; los detalles de cada algoritmo y sus diferencias
respecto de los trabajos originales figuran en los documentos de experimentos.

## Entorno y reproducción local

En el PC de trabajo, código, entorno Conda, temporales y resultados están bajo
`E:\UTEC\Tesis\TesisNicol`. El entorno usado es `conda_env`, dentro de esa carpeta.
`experiments/requirements.txt` registra las dependencias verificadas de los
experimentos portables; no requiere instalar `mamba-ssm` para esas referencias.
Las dependencias del código histórico permanecen en `requirements.txt`.

Desde la raíz del repositorio en PowerShell, con el entorno ya preparado:

```powershell
. .\experiments\local_env.ps1
python -m experiments.round3.common
python -m pytest experiments/tests -q --junitxml=E:\UTEC\Tesis\TesisNicol\revision\pruebas_round3.xml
python -m experiments.round3.run --phase train --seeds 11 22 33 44 55
python -m experiments.round3.freeze_checkpoints
python -m experiments.round3.run --phase test
python -m experiments.round3.analyze
```

La primera orden de Python genera las demostraciones locales necesarias también
para las pruebas de recompensas. En el entorno existente se aprobaron 53 pruebas.
Los checkpoints, datos generados y cachés no se versionan; permanecen en `runs/`.
No se incluyen en esta actualización los nuevos pesos de entrenamiento.

## Interpretación de resultados

El éxito actual significa completar la lista de waypoints con tolerancia de
0,25 m. **No certifica aterrizaje estable ni control perfecto.** La condición
«efectividad exacta» proporciona la información verdadera de la falla sin demora;
la condición estimada introduce sesgo y demora. La detección está asistida por
el simulador. El modo severo del controlador no constituye una implementación
validada del equilibrio giratorio de Mueller–D'Andrea.
## Nueva ronda: aterrizaje controlado y seis arquitecturas

La ronda reproducible se documenta en [experiments/round4/README.txt](experiments/round4/README.txt). Exige contacto suave y un segundo de estabilidad en destino; incluye estimación de efectividad desde movimiento/comandos, memoria persistente y afinado protegido de la imitación. Los resultados anteriores se conservan.

## Actualización del controlador y estimador de la ronda 5

La configuración ensayada, sus componentes y la integración están en
[experiments/DSLPID_ACTUALIZADO.txt](experiments/DSLPID_ACTUALIZADO.txt).
Incluye estimación adaptativa de efectividad, frenado angular de emergencia y
supervisión del descenso. Los resultados finales de esta ronda siguen pendientes.
