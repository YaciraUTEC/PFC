RONDA 4: ATERRIZAJE CONTROLADO, ESTIMACION Y MEMORIA
==================================================

Ejecutar exclusivamente en E:\UTEC\Tesis\TesisNicol, con conda_env local.
No requiere nuevas instalaciones.

. .\experiments\local_env.ps1
python -m experiments.round4.data --routes 40 --workers 4
python -m experiments.round4.train --phase hpo
. .\experiments\round4\run_campaign.ps1

El ultimo comando entrena treinta modelos (seis arquitecturas x cinco semillas),
congela los pesos y ejecuta las pruebas nuevas e historicas. Dos trabajadores de
entrenamiento, cada uno con ocho simuladores y cuatro procesos de validacion.
Dos trabajadores de evaluacion, cada uno con ocho simuladores. No lanzar otra
copia del mismo comando sobre los mismos directorios.

Los resultados estan en runs/controlled_landing_round4, ignorado por Git.
data_governor_v2 es el conjunto vigente; data corresponde a una generacion
preliminar interrumpida y NO se usa para entrenar. Los experimentos antiguos
se preservan en runs/irl_round3 y runs/dslpid_comparison_v2.

EXITO
-----
Contacto real en destino; primer contacto vertical/horizontal <=0.25m/s,
|roll| y |pitch| <=12 grados, velocidad angular <=1rad/s y radio <=0.20m.
Despues, un segundo continuo con contacto, velocidad total <=0.10m/s,
|roll| y |pitch| <=10 grados y velocidad angular <=0.5rad/s, altura <=0.055m.
Se registra la velocidad ANTES del impacto a240Hz, pues la velocidad despues
del impulso de contacto puede ocultar una caida rapida.

El objetivo anterior terminaba a10cm del suelo. Se aplica el MISMO supervisor
de descenso a todos los controladores:0.18m/s y corte de motores tras contacto.
Se conserva ademas una evaluacion con perfil original y metrica estricta para
separar el cambio de criterio del cambio necesario en el plan de aterrizaje.
waypoint_reached y stable_landing son metricas separadas.

CONTROL Y ESTIMACION
-------------------
El controlador publico permite thrust_reserve_fraction y emergency_thrust_ratio.
Los valores por defecto .05 y1.02 mantienen las rondas previas. Esta ronda usa
explicitamente .005 y1.0001: la reserva fija anterior eliminaba la autoridad
horizontal cerca de28-29% RPM. No se aumento el limite de21700RPM.

El estimador RLS usa solo movimiento observado, comandos y modelo nominal
calibrado. Estima beta=etaRPM^2, con incertidumbre y ruido de diferencias de
velocidad/gyro .002. No recibe motor averiado, severidad o tiempo de falla.
La simulacion de estado/ruido no equivale a validar IMU/odometria en hardware.
exact es un oraculo; estimated es RLS; biased conserva la subestimacion artificial
de round3 como ablacion. La verdad de la falla aparece SOLO en maestro y metricas.

INTERVENCION Y AFINADO
---------------------
Se generan etiquetas con ramas reales del simulador: intensidades0,.25,.5,1 de
la correccion privilegiada,24ticks,horizonte y coste terminal fisico,penalizacion
de esfuerzo y umbral de mejora. Cada alternativa restaura Bullet,controlador,
integrales,referencia,actuadores,historial,estimador y RNG. Las repeticiones desde
el mismo estado restaurado son identicas; Bullet puede tener diferencias
numericas pequenas frente a continuar SIN restaurar. No se afirma restauracion
bit a bit del estado interno no expuesto del motor fisico.

La red emite direccion y compuerta continua. Una proteccion adicional escrita
con errores/velocidades/asignacion/incertidumbre apaga residual cuando la base
controla bien. Su ablacion usa una semilla por arquitectura y se etiqueta aparte.

BC seguido de recompensa aprendida sin proteccion YA se probo en round3.
Esta ronda no repite esa idea como aporte nuevo. Las seis arquitecturas nuevas
usan reward fisico explicito y afinado PPO con retencion BC y KL a su clonacion.
El critico tiene una red SEPARADA y se calienta8192pasos sin mover el actor.
Las recompensas aprendidas round3 se reevaluan como controladores historicos.

MEMORIA Y COMPARACION
--------------------
LSTM: h/c persistentes, inicializacion ortogonal, forget bias1,1/2capas.
Mamba: estados16/32 y cache convolucional+SSM, sin weight decay en dinamica.
Mamba2: estados32/64,cabezas16/32,normalizacion y cache persistente.
Mamba3: SISO explicito,estados32/64,fase+trapezoide+SSM persistentes. MIMO no
incluido en este ensayo; no se atribuye el resultado a todas las variantes.
GTrXL: dos capas,gates1/2,memoria64/128,warmup y mascara deslizante causal
que hace equivalente el procesamiento por chunks y por tokens.
LTC: ncps puro,dt1/48s,3/6subpasos ODE,positividad implicita,sin hibrido LSTM.

PPO usa secuencias contiguas,mascaras de episodio,estado inicial propio y burn-in.
El BC usa64pasos supervisados y32de contexto; la memoria de despliegue persiste
hasta el final del episodio. Las recurrencias Mamba y gradientes se contrastan
con las referencias previas. Se prueba reconstruccion de probabilidades PPO,
reinicio de memoria y equivalencia de lotes de evaluacion/reutilizacion de pool.

Dos configuraciones por arquitectura,igual presupuesto HPO600updatesBC+65536PPO,
validacion reservada. Cinco semillas71..75,1200updatesBC+524288PPO cada una.
La validacion puede elegir el clon inicial si el afinado no mejora. Antes de
congelar se comprueba tambien0/2/5% en8rutas de validacion y ambas condiciones;
cualquier perdida de una ruta frente aBC hace regresar al clon. Se guardan
la eleccion anterior y los casos en selection_safety/. No se usa test para decidir.
La prueba usa30geometrias NUEVAS por nivel,condicion y semilla. No confundir
150casos de modelos con150geometrias independientes. El bootstrap cruza
semillas de entrenamiento y geometrias, manteniendo severidades emparejadas.

Niveles principales RPM:0,2,5,10,15,20,25,28,29%.
Estres:29.5,30,35,50%. El limite estatico ideal de hover normal es29.7186%RPM;
no garantiza control hasta ese limite ni demuestra recuperacion giratoria.
El enfoque Mueller-DAndrea de emergencia no esta implementado por esta ronda.

Los modelos historicos v2 conservan entradas24 y residualRPM. Su identificacion
de falla se alimenta de eta exacta/estimada; el umbral es1%RPM. Se entrenaron
originalmente con motor0 y ahora se exponen a los cuatro: cambio de distribucion
que debe mencionarse. Round3 conserva entradas48,cinco acciones y gobernador
historico,con el nuevo supervisor y criterio de aterrizaje comunes.

PRECISION DE EVALUACION
----------------------
El entrenamiento usaGPUfloat32. La referencia final calcula las redes en
CPUfloat64 y convierte la accion una sola vez afloat32 en el entorno. Los pesos
originales siguenfloat32 y se verifican susSHA. No se cambia el aprendizaje.
Se detecto sensibilidad a redondeo y lote enMamba2 cerca de29%:mismos desenlaces
pero diferencias de trayectorias. Confloat64,96casos devalidacion de las seis
arquitecturas reprodujeron desenlace,pasos yRMSE entreCPU/GPUy lotes8/16.
Se preservan los diagnosticos FP32 y la paridad float64;no se usaron rutas de test.
Los adaptadores historicos reproducen exactamente su prediccionFP32 antes de
elevar precision;se limpian caches posicionales al cambiar dtype.

ARTEFACTOS
----------
protocol.json;hpo/*_selection.json;models/*/seed_*/{bc,best,final}.pt;
frozen_before_test.json;test/*.json;historical/*.json;success_rates.csv;
historical_rates.csv;paired_changes.json;residual_vs_base.json;
full_budget_changes.json;intervention_metrics.csv;guard_ablation.csv;
estimator_metrics.csv;effectiveness_ablation.csv;outcome_counts.csv;
training_timings.csv;stage_timings.json;results_integrity.json;
comparacion_arquitecturas.png;afinamiento_vs_imitacion.png;INFORME_RONDA4.txt.
Cada cache de evaluacion registra origen e identidad del actor;no se reutilizan
resultados entre actores distintos. Las interrupciones parciales de entrenamiento
se reinician desde su clon: no se promete reanudar exactamente un rollout.

FUENTES PRIMARIAS
-----------------
https://research.tudelft.nl/en/publications/control-of-unknown-quadrotors-from-a-single-throw/
https://proceedings.mlr.press/v119/parisotto20a.html
https://github.com/state-spaces/mamba
https://github.com/mlech26l/ncps
https://arxiv.org/abs/1709.10087
Los bloques importados conservan las licencias de experiments/LICENSE_*.txt.

AUDITORIA FINAL
--------------
python -m experiments.round4.verify_results
python -m pytest experiments/tests -q

La auditoria comprueba 320 archivos con 122760 casos logicos: las rutas comunes,
los niveles, la unicidad de cada caso, las metricas finitas, el contrato registrado
de aterrizaje, las copias de cache y los 147 archivos de pesos congelados.
El numero de registros realmente simulados se informa por separado de los
reutilizados por identidad exacta del actor. analyze ejecuta esta auditoria antes
de producir las tablas y el informe. Las 108 pruebas actuales pasan.

Los intervalos principales son simultaneos para 24 contrastes (seis arquitecturas,
dos condiciones y dos referencias: BC y DSLPID). Los efectos del presupuesto
final y de la proteccion se etiquetan como diagnosticos exploratorios.
