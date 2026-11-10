PRIMERA PRUEBA: DSLPID ACTUALIZADO + APRENDIZAJE DE RECOMPENSA

Todo se ejecuta en E:\UTEC\Tesis\TesisNicol, con el entorno conda_env
alojado en esa carpeta. No hacen falta instalaciones adicionales.

Controlador público
------------------
gym_pybullet_drones/control/DSLPIDControl.py contiene el controlador nuevo.
Mantiene computeControl / computeControlFromState y sus tres resultados.
Las ganancias de actitud se expresan ahora en unidades físicas de fuerza/par,
no en las unidades PWM de la implementación antigua. El constructor permite
ajustar attitude_frequency (frecuencia natural en rad/s) y attitude_damping; setPIDCoefficients sigue
funcionando y valida los valores en unidades SI.

set_motor_effectiveness recibe CUATRO FACTORES DE RPM estimados en [0,1].
Por ejemplo,10% de pérdida de RPM en motor0: [0.9,1,1,1]. La asignación usa
el cuadrado de esos factores. La planta física o el simulador debe multiplicar
las RPM por el factor UNA vez. El controlador no estima la falla por sí solo.

El controlador trabaja con fuerza y pares sin recortar previamente una mezcla
PWM. Respeta [0,21700]RPM en el Crazyflie de esta simulación; pondera fuerza,
roll/pitch y yaw; realimenta las tasas angulares en el cuerpo; limita la fuerza
y la inclinación pedidas y usa back-calculation de las integrales. El diagnóstico
last_diagnostics muestra petición,asignación,desbalance,saturación y emergencia.
El modo severo es un esfuerzo de actitud reducida,no un controlador validado
de equilibrio giratorio y aterrizaje con un motor apagado.

LegacyDSLPIDControl.py conserva el código original. RobustDSLPIDControl sigue
siendo el enfoque robusto anterior y hereda explícitamente del original.
Los experimentos anteriores en experiments/env.py y NominalFlightEnv con
controlador='pid' conservan ese comportamiento histórico. La nueva comparación
está en experiments/round3/env.py y usa EXPLÍCITAMENTE el DSLPID actualizado.
Los modelos antiguos de24 entradas/4 acciones no son compatibles con las
48 entradas/5 acciones nuevas. No se sobreescribieron sus resultados.

Protocolo y ejecución
--------------------
Desde PowerShell:
. 'E:\UTEC\Tesis\TesisNicol\PFC\experiments\local_env.ps1'
python -m experiments.round3.common
python -m experiments.round3.run --phase train --seeds 11 22 33 44 55
python -m experiments.round3.freeze_checkpoints
python -m experiments.round3.run --phase test
python -m experiments.round3.analyze

También se pueden repartir SEMILLAS DISTINTAS en varios procesos. Nunca ejecutar
la misma semilla simultáneamente en dos procesos: ambos escribirían en el mismo
directorio. Los marcadores complete.json permiten saltar tratamientos terminados.
Un tratamiento interrumpido se reinicia desde la clonación inicial; el programa
no promete continuación exacta del flujo de simulación a mitad de PPO.
Las evaluaciones terminadas se guardan atómicamente y pueden continuarse.

La prueba principal se abre solo cuando las20 combinaciones (4 recompensas
incluida manual ×5 semillas) han terminado entrenamiento y selección por
validación. Incluye100 rutas nuevas por nivel de falla. Se evalúan aparte
clonación,controladores sin residual,efectividad exacta y los modelos al terminar
todo el presupuesto. protocol.json conserva condiciones,criterios y cambios
registrados antes de inspeccionar los resultados de prueba.

Qué se está comparando
----------------------
CSIL: núcleo de recompensa coherente y cabeza estacionaria portados a Torch.
ME-IRL: discriminador con potencial esperado bajo ensemble de dinámica;
PPO y rollouts reales,seguido de recompensa congelada.
RIME: reward por preferencias,ensemble,filtro KL y cambios de etiquetas;
preferencias calculadas con métricas de simulación y10% de ruido. No humanas.

El optimizador posterior es el mismo PPO temporal para los tres y la referencia
manual. Las anclas terminales de tarea son explícitas y comunes. Son adaptaciones
controladas al proyecto,no reproducciones completas de los sistemas publicados.
RIME pertenece a aprendizaje de recompensa por preferencias,complementario al
IRL de demostraciones. Se documentan sus diferencias en el informe final.

Resultados
----------
runs/irl_round3 guarda protocolo,datos,pesos,modelos,métricas por vuelo,tiempos,
análisis y figuras. El informe principal se guarda también en:
E:\UTEC\Tesis\TesisNicol\INFORME_DSLPID_IRL_PRIMERA_PRUEBA.txt

El éxito significa completar los waypoints originales; no certifica aterrizaje
estable ni seguridad en hardware real. Los porcentajes de falla son pérdida de
RPM: 50%RPM equivale a75% de pérdida de empuje. Superar la factibilidad del vuelo
normal requiere replantear el objetivo de emergencia y permitir giro,y no solo
cambiar el reward.

Fuentes primarias
-----------------
https://proceedings.mlr.press/v331/zhan26a.html
https://joemwatson.github.io/csil/
https://github.com/google-deepmind/csil
https://proceedings.mlr.press/v235/cheng24k.html
https://github.com/CJReinforce/RIME_ICML2024
Los repositorios oficiales consultados están bajo revision/sources_round3;
sus revisiones exactas figuran en provenance.json.
La evaluación final usa FP32 sin TF32. Los actores con parámetros idénticos
a la clonación comparten sus resultados guardados;los modelos completamente
entrenados se evalúan aparte. evaluation_reuse.json registra cada reutilización.
En esta PC conviene ejecutar como máximo dos grupos de entrenamiento a la vez.
