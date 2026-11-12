RONDA 5: REPRODUCIR LA EVALUACIÓN COMPLETA

Todos los datos, cachés, modelos y resultados están bajo E:\UTEC\Tesis\TesisNicol.
No requiere instalaciones nuevas. Desde PFC, cargar experiments/local_env.ps1 y usar
E:\UTEC\Tesis\TesisNicol\conda_env\python.exe.

El protocolo y sus enmiendas previas a prueba están en runs/intervention_round5.
PLAN.txt explica hipótesis, criterios de éxito y cierre. final_evaluation_protocol.json
fija rutas, severidades, intervalos y regla de decisión. Nunca ajustar parámetros
con las rutas de prueba final; no borrar la congelación para incorporar un modelo.

Desde una copia nueva, prepare inicializa protocolos registrados, configura y
verifica los controladores antes de generar datos. Los ajustes iniciales están
enconfigs/reference_inputs.json y no requieren resultados de ronda4.Las marcas
de preregistro copiadas conservan la historia del estudio;los recibos de cada
ejecución guardan sus tiempos reales.Los pilotos de agregación conservan la
pérdida histórica no equilibrada;los75modelos principales usan la equilibrada.

Secuencia (cada comando con el Python indicado):
  -m experiments.round5.prepare
  -m experiments.round5.worker --worker 0
  -m experiments.round5.worker --worker 1
  -m experiments.round5.run_evaluation
Los dos trabajadores pueden ejecutarse en paralelo; esperan a que prepare termine.
El coordinador espera a ambos y luego ejecuta selección reservada, replay de todas
las redes, curvas completas, confirmación independiente, sensibilidad, latencia,
análisis, reproducción del paquete elegido, informe y auditoría. No ejecutarlo
duplicado. Fallos técnicos detienen la etapa; corregir la causa antes de reanudar.
No se afirma reanudación exacta de un simulador a mitad de PPO: un trabajo incompleto
reinicia desde su BC y recompensa inicial congelada; los ya completos se conservan.

Pruebas:
  -m pytest experiments/tests -q --junitxml=runs/intervention_round5/tests_final.xml
La auditoría requiere tests_final_receipt.json con exit_code=0, además del XML.
El coordinador solo crea ALL_COMPLETE.json después de todas las verificaciones.
Los artefactos de salida están excluidos de Git por tamaño; el código está versionado.

Cambios de control:
DSLPIDControl añade emergency_policy: legacy(predeterminado compatible), damped
(frena yaw incluso en modo de emergencia) y balanced(gobernador continuo contrastado).
apply_prepared_q acepta velocidades cuadradas normalizadas y conserva asignación,
límites físicos y antiwindup. El entorno de ronda5 añade observación compartida,
estimador adaptativo de efectividad, pausa del descenso por preparación observable,
compensación directa de motores o corrección de efectividad y compuerta explícita.
La decisión de despliegue NO se deduce del nombre de la clase; usar la configuración
concreta de recommended_controller.json después de ALL_COMPLETE.

Política y entrenamiento:
Memoria recurrente persistente por episodio en LSTM, Mamba, Mamba-2, Mamba-3 SISO,
GTrXL y LTC. Entrenamiento de secuencias96 y calentamiento32, BC1200actualizaciones,
PPO524288pasos. Valor con historia separada del gradiente del actor; recompensa
terminal dominante, escalada por100 para estabilidad del crítico, plazos terminales,
generador estocástico de acciones independiente, regularización hacia BC y KL,
LR con calentamiento y coseno. Dos tasas por arquitectura, dos horizontes GAE(0.97/0.995) y5semillas principales.
Un solo modelo de BC por semilla inicializa los4afinados LSTM(manual,CSIL,MEIRL,RIME).

El maestro contrasta propuestas durante12ticks y termina la misión con el PID.
La etiqueta de activación exige beneficio verificable. DAgger conserva estados de
alumnos de todas las arquitecturas, propuestas aunque la red esté apagada y ráfagas
exploratorias limitadas. Se separan acciones ejecutadas y etiquetas para no entrenar
la dinámica con acciones que no produjeron las transiciones.

El juez usa5predictores de beneficio, margen empírico de validación y abstención
fuera de soporte. No es un certificado formal de seguridad. La confirmación final
prueba daño condicional<1% en mezclas de niveles predefinidas, no en cada estado.
Si ninguna red cumple, se recomienda el PID estructurado aceptado; abstenerse no
se presenta como éxito neuronal. Los resultados de familias(5semillas×60rutas)
se distinguen de la confirmación del checkpoint concreto(1000rutas×5casos).

Salidas para revisión:
INFORME_FINAL_RONDA5.txt, decision.json, recommended_controller.json,
success_rates.csv, training_timings.csv, stage_timings.json y los siete PNG.
final_audit.json conserva cobertura, hashes y comprobantes. frozen_before_test.json
fija código y pesos. Solo report.py/plots.py pueden cambiar después de la congelación;
se conservan sus hashes de presentación. No se sustituyen los modelos de ronda4.

Reproducción del controlador recomendado, después del cierre:
  -m experiments.round5.deployment --level 0.295 --cases 8 --condition estimated
La función make_environment devuelve entorno y política(None si solo PID); reiniciar
la memoria de la política al acabar cada episodio. La evaluación de referencia usa
CPUfloat64 y acciónfloat32. La medición de latencia es local, con núcleos portables.
No es un controlador listo para volar hardware real sin integración y validación.

Fuentes primarias de las adaptaciones:
DAgger: https://proceedings.mlr.press/v15/ross11a.html
CSIL: https://arxiv.org/html/2305.16498v3
MEIRL: https://proceedings.mlr.press/v331/zhan26a.html
RIME: https://proceedings.mlr.press/v235/cheng24k.html
Time limits: https://proceedings.mlr.press/v80/pardo18a.html
Residual RL: https://arxiv.org/abs/1812.03201
SPIBB: https://proceedings.mlr.press/v97/laroche19a.html
Estos métodos se adaptaron a intervención residual y recompensa terminal dominante;
no se pretende reproducir íntegramente sus benchmarks o garantías.

ACELERACIÓN EXACTA DE EVALUACIÓN
replay_cache.py almacena observaciones y resultados base con huellas de código,
activos físicos y observaciones. La red se reproduce en CPUfloat64, tick a tick,
con la misma memoria por episodio. Solo se reutilizan casos donde toda la secuencia
de acciones coincide exactamente con ZERO. Si una acción difiere, se simula desde
el inicio el caso entero con la red. Nunca se reutilizan propuestas shadow ON.
Las pruebas diferenciales cubren las seis familias, estados de compuerta mixtos,
contexto de efectividad, captura, corrupción y escritura concurrente. Los informes
distinguen casos sin intervención y certificados globales de política cero.

CONFIRMACIÓN INCREMENTAL
El estimador solo también se confirma en1000rutas(5000casos),por separado delperfil
completo.Conservarlo si mejora al anterior sin daño protegido; añadir el perfil
completo solo si supera al estimador ya aceptado.Las redes deben superar la
referencia estructurada así aceptada.Se preregistraron estas comparaciones antes
de abrir pruebas finales; máximo12contrastes×3extremos=36.La congelación lo
verifica; no se agrega un contraste retrospectivo fuera de ese presupuesto.

DIAGNÓSTICO CAUSAL DE LA COMPUERTA
Después de fijar modelos/estrategias y antes de abrir prueba final,se evalúan
las propuestas reales de75políticas en4rutas reservadas×5semillas porfamilia
a29.5%estimado.Se consultan arranque,1segundo,recuperación,inicio de descenso
yaproximación:su propia propuestaON12ticks frente a cero,despuésPIDhasta
terminar.Se reportanTP/FP/FN/TNyrescates/pérdidas.No es un oráculo desplegado,
ni un techo global de rendimiento;son consultas correlacionadas de diagnóstico.
Sus datos no ajustan pesos,umbrales ni la regla de adopción.
