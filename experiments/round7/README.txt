RONDA 7 CERRADA - GTRXL, DEMOSTRACIONES Y COMPUERTA

Estado: estudio terminado el 16 de noviembre de 2026. Los resultados originales
estan en results/round7/final; ALL_COMPLETE.json registra cero trabajo pendiente.
Recomendacion: DSLPID + estimador estructurado + RecoveryDescent, residual OFF.
El candidato experimental no supero los criterios prospectivos de adopcion.

Alcance del codigo
Solo Transformer/GTrXL. Se reutiliza la propuesta FlashSAC de ronda 6 y se estudia
una autorizacion contrafactual, mas demostraciones, refinamiento supervisado y
un filtro predictivo comparativo. No es una nueva comparacion Mamba/Transformer
ni un entrenamiento online FlashSAC completo. El afinado conserva el encoder.
PLAN.txt y FORMULACION_GLOBAL.txt son los protocolos originales previos al test;
su redaccion prospectiva se conserva como evidencia historica.

Mapa de modulos
- common/protocol: rutas de datos, semillas y comprobantes atomicos.
- h0, episode, replay_fork, composed_replay: reproduccion de trayectorias y
  bifurcaciones mediante el historial natural de acciones.
- h1, data, bc, proposal_validation, repair_proposals: adquisicion y ajuste
  de propuestas, comparacion de cantidades y reparacion de supervision.
- gate_data, gate_fit, terminal_options, branches: etiquetas de ambas
  continuaciones hasta terminal y aprendizaje de autorizacion.
- policy, execution, fast_gtrxl: politica compuesta y ejecucion con memoria;
  optimizaciones numericas comprobadas frente al backend original.
- study, continuation_study, screen, h2: ablaciones de compuerta/filtro.
- h4, refine: afinado supervisado anclado y agregacion limitada de datos.
- evaluation, cache, h5, statistics: prueba reservada, confirmacion y estadistica
  agrupada por ruta; cinco replicas no multiplican las rutas independientes.
- h6: latencia de la cadena completa, informe, figuras y cierre.
- runner/power: utilidades de ejecucion y comprobantes. Importar no inicia
  automaticamente una campana.

Metodo valido y diagnosticos
El primer metodo save/restore de PyBullet se rechazo por no conservar el paso
natural siguiente. snapshot/snapshot_diagnostic permanecen como diagnostico
del metodo descartado; no usar sus etiquetas preliminares para entrenar.
El estudio valido usa replay_fork/composed_replay y OUT=gtrxl_round7_replay.
Los controles de prefijo y equivalencia se conservan en los resultados.
Los resultados de desarrollo y los de rutas reservadas se mantienen separados.
Los analisis posteriores del caso de traspaso de control son diagnosticos;
no se ajustaron parametros a partir de esa ruta ni se declara generalizacion.

Dependencias, pesos y rutas
Usar el entorno del proyecto y experiments/requirements.txt, con los requisitos
NMPC opcionales publicados si se regeneran sus demostraciones.
La implementacion conserva los nucleos GTrXL ya publicados y sus licencias.
El entrenamiento y parte de la evaluacion utilizan CUDA; las pruebas pequenas
de CPU indicadas abajo no construyen redes.

Git incluye codigo, protocolos, estadisticas, informes y graficas.
Los checkpoints .pt, datasets .npz e historias completas de vuelos no se suben.
Para repetir el estudio hacen falta los modelos originales GTrXL+FlashSAC
de ronda 6 y los datos/modelos de cada hito que se generan en una nueva ejecucion.
Los hashes originales identifican exactamente esos inputs; un informe agregado
no sustituye a un checkpoint o un dataset ausente.

common.py define ROOT como la raiz de esta copia y OLD como
ROOT.parent/PFC/runs/intervention_round6. Si la copia se llama PFC, OLD coincide
con su carpeta local de ronda 6; en una copia con otro nombre se espera una
carpeta hermana PFC que contenga esos resultados y pesos originales.
LEGACY_OUT=runs/gtrxl_round7 conserva el H1 valido y diagnosticos descartados;
OUT=runs/gtrxl_round7_replay contiene el estudio corregido.
En el PC del estudio todo esta bajo E:\UTEC\Tesis\TesisNicol, en conda_env.
experiments/local_env.ps1 cambia el directorio a PFC: para otra copia hay que
seleccionar de nuevo su raiz despues de cargar ese entorno.

Uso de los resultados publicados
Los archivos dentro de results/round7/final son un archivo historico consultable.
No copiarlos a runs para simular hitos, datos o modelos nuevos. Las rutas absolutas
dentro de los comprobantes corresponden al PC original y se conservan sin cambios.
Para una nueva ejecucion se generan sus propios outputs y comprobantes por hitos;
los modulos no proporcionan una restauracion universal desde informes agregados.

Recomendacion frente a candidato
deployment_recommendation.json fija residual_policy_specs=[]: significa cero
correccion neuronal aunque el entorno tenga gate_mode=learned.
selected_candidate.json identifica la politica ensayada, no una autorizacion
para desplegarla. El perfil analitico de ronda 6 ya publicado puede seguir
usandose para ejecutar DSLPID + estimador + supervisor sin cargar redes.

Verificacion
El estudio original paso 375 pruebas y verifico 250 fuentes y 12 inputs
congelados. La publicacion vuelve a comprobar las huellas e imports locales,
compila los fuentes y ejecuta solo pruebas pequenas de CPU.
Algunas pruebas de test_round7 y las suites historicas requieren modelos,
datasets o CUDA; no ejecutarlas suponiendo que los informes archivados
incluyen esos recursos. El recibo de publicacion distingue la verificacion
nueva de la suite completa original.

Verificacion de esta publicacion: 45 fuentes Python compilan, imports locales
resueltos y seis pruebas pequenas de CPU aprobadas en20.66s. No se construyeron
redes ni se inicio entrenamiento. Recibo: results/round7/final/publication_verification.json.
