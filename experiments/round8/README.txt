RONDA 8 CERRADA - PRESERVACION Y RESCATE CON GTrXL

Resultados: results/round8/final/README.md.
Analisis critico posterior: results/round8/analysis/README.md.

Se reutiliza el actor FlashSAC+GTrXL201 congelado de ronda 6. Se aprendieron cinco
compuertas de tres cabezas: probabilidad de exito con DSLPID, preservacion
condicional a ese exito y rescate condicional a su fallo. No son cinco actores
nuevos ni un nuevo entrenamiento IRL. Solo GTrXL participa como arquitectura.
La compuerta 505 se eligio en desarrollo ANTES de la prueba final.

La politica entregada permite una opcion raw24half: propuesta recalculada a 48 Hz,
escala 0.5 durante hasta 24 pasos, una sola opcion por vuelo y despues DSLPID solo.
Se mantiene el historial mientras espera; exige degradacion observada> 2%, veto
nativo de entrada, beneficio conservador> 1 pp y riesgo pronosticado< 2%.
Estos pronosticos son heuristicas, no un certificado fisico. Se separan de la
cota estadistica de perjuicio y de los criterios estrictos de aterrizaje.

Mapa: factor_gate/fit aprenden probabilidades; active_data/active_scan/acquire
agregan continuaciones completas pareadas por replay natural; policy_v2/v3/
policy_guarded implementan las opciones; loading_compact/execution_compact/
cache_compact ejecutan la representacion equivalente reducida; study/sequential
aplican el protocolo original congelado; report/handoff/finish documentan y
verifican el cierre. Los pilotos y ablaciones negativos se conservan.

PLAN.txt y los AMENDMENT archivados son historicos: no ejecutan una campana al
importar. study conserva el contrato ORIGINAL de cero perdidas observadas.
El analisis posterior propone un contrato distinto pero NO lo implementa ni
cambia sus resultados. No reutilizar un test ya inspeccionado como ciego.

La familia finalmente entregada se ajusto con --full-support; su metadata y
el paquete local verifican este detalle. Para volver a ajustar la compuerta:
python -B -m experiments.round8.fit --seed 505 --folder runs/gtrxl_round8/data/active_merged_v1 --tag reproduccion --full-support
Se requieren los datasets y checkpoints locales originales. Cambiar actor,
opcion, escalas o reglas exige volver a etiquetar y validar la politica completa.

Git publica codigo, pruebas, protocolos, informes, tablas, figuras y proyecciones
por episodio suficientes para recalcular la auditoria. Los checkpoints .pt,
datasets de entrenamiento .npz, historias largas y el entorno no estan en Git.
El paquete local ENTREGA_NICOLE_RONDA8/intento_1.zip contiene esos artefactos
necesarios para la demostracion verificada y la reproduccion con los pesos.
Los comprobantes originales mantienen sus rutas y huellas historicas.

Recalcular estadisticas publicadas sin pesos ni simulacion:
python -B results/round8/analysis/analizar_resultados.py
Solo requiere NumPy y SciPy; escribe en runs/round8_reanalysis.
Dependencias generales: experiments/requirements.txt. Las pruebas completas
usan ademas fixtures locales y modelos; publicacion_verificada.json detalla
su ejecucion. No copiar cierres historicos para fingir ejecuciones nuevas.
