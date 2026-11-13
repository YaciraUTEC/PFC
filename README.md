# PFC

Control de un cuadricóptero Crazyflie ante pérdida de efectividad en un motor.

La **ronda 5 terminó** el 12 de noviembre de 2026, con 75 modelos entrenados,
129 bloques de evaluación y auditoría final aprobada.
Los [resultados completos](results/round5/README.md) incluyen informe, tablas,
siete gráficas, tiempos, configuración recomendada y comprobantes de integridad.

## Conclusión de la campaña

La recomendación registrada conserva el DSLPID de referencia de la ronda 4,
sin compensación neuronal. El estimador adaptativo y el perfil completo mejoraron
el éxito con el ruido habitual, pero no superaron la regla de robustez al
duplicarlo. Ninguna familia neuronal demostró una mejora estadística consistente
frente al controlador completo.

La [decisión](results/round5/decision.json) y la
[configuración recomendada](results/round5/recommended_controller.json) fijan el
resultado concreto. La configuración elegida no se deduce del nombre de la clase.

El éxito exige completar la ruta, contacto suave cerca del destino y un segundo
de estabilidad. Los porcentajes expresan pérdida de RPM de un motor; el empuje
depende de su cuadrado. Las conclusiones corresponden al simulador, sensores y
distribución de fallas evaluados.

## Controladores

- [DSLPIDControl](gym_pybullet_drones/control/DSLPIDControl.py): fuerza y momentos
  en unidades SI, asignación acotada según efectividad, realimentación angular y
  corrección de integrales frente a saturación.
- [Estimador y supervisor de descenso](experiments/DSLPID_ACTUALIZADO.txt):
  componentes y configuración ensayada. Su disponibilidad no implica adopción.
- [NMPCINDIControl](gym_pybullet_drones/control/NMPCINDIControl.py): controlador
  independiente con dinámica no lineal, retardo de motores, optimización acotada
  e INDI. Comparte las interfaces computeControl y computeControlFromState.
  Su [dependencia opcional](gym_pybullet_drones/control/requirements-nmpc.txt)
  es CasADi/IPOPT. Tiene 68 pruebas ligeras aprobadas; no participó en la ronda 5
  ni cuenta todavía con validación de desempeño en vuelo en este proyecto.
- LegacyDSLPIDControl conserva la implementación histórica.

La efectividad se suministra como factor de RPM. La detección de falla es externa
a las clases de control y puede provenir de un estimador.

## Arquitecturas, entrenamiento y reproducción

El [protocolo de la ronda 5](experiments/round5/README.txt) documenta LSTM, Mamba,
Mamba-2, Mamba-3 SISO, GTrXL y LTC con memoria persistente, decisión de intervención,
imitación seguida de PPO protegido, ajuste por arquitectura y cinco semillas.
Incluye compensación directa, corrección de efectividad y las adaptaciones
de recompensas CSIL, ME-IRL y RIME.

Las [configuraciones registradas](experiments/round5/configs) y las pruebas de
experiments/tests acompañan al código. Los pesos, datasets y trazas generadas
permanecen en runs/, excluido de Git. El paquete results/round5 conserva los
resultados agregados y la auditoría de la ejecución final.

En el PC de trabajo, código, entorno Conda, cachés, temporales y resultados
permanecen bajo E:\UTEC\Tesis\TesisNicol. Las dependencias de experimentos están
en [experiments/requirements.txt](experiments/requirements.txt); las del código
histórico se mantienen en requirements.txt.

## Histórico y fuentes

- [Ronda 4](experiments/round4/README.txt): aterrizaje estable y comparación de
  seis arquitecturas.
- [Ronda 3](experiments/round3/README.txt): primeras comparaciones de recompensas.
- [Experimentos iniciales](experiments/README.txt).
- [Fuentes y alcance de las implementaciones](experiments/SOURCES.txt).

Los comprobantes archivados contienen las rutas del PC original y sus huellas.
Para otra ejecución se generan sus propios datos y comprobantes siguiendo el
protocolo; el registro final original se conserva como evidencia del estudio.

## Control verificado de la ronda 6

Se publica el [conjunto analitico de control](experiments/round6/CONTROL_VERIFICADO.txt)
de la fase 1: estimador estructurado de una falla, descenso coordinado y
apagado persistente tras contacto. El perfil seleccionado conserva la clase
DSLPIDControl base y agrega estos componentes mediante LandingEnv.

Los [comprobantes de validacion](results/round6/phase1/phase1_conclusion.json)
registran 71 ganancias pareadas y cero perdidas en 24 trayectorias por
condicion. Esta fase esta cerrada; la comparacion neuronal de la ronda 6
sigue en curso y no forma parte de esta publicacion.
