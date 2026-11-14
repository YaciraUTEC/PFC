RONDA 6 - CONTROL Y APRENDIZAJE PUBLICADOS

Control analitico: CONTROL_VERIFICADO.txt y configs/verified_control.json.
Implementacion neuronal: models.py, train.py, offpolicy.py, train_offpolicy.py.
Configuracion registrada: configs/training_reference.json.
Fuentes, licencias y huellas: results/round6/training_code/source_snapshot.json.

Esta publicacion agrega el codigo de aprendizaje congelado antes del test.
La evaluacion final de la campana original sigue en curso. No constituye una
recomendacion final de politica ni incorpora las correcciones de compuerta
propuestas despues de observar resultados. No incluye pesos ni datasets.

Mamba-3/GTrXL
Los nucleos existentes de rondas anteriores se conservan; la entrada pasa a
88 variables y se enmascaran las señales de accion/compuerta previa. El modelo
mantiene memoria durante cada episodio. Mamba-3 es la referencia matematica
SISO portable ya publicada, no los kernels MIMO/fusionados oficiales.
BC y PPO entrenan la representacion. FlashSAC/FQL congelan el encoder PPO y
entrenan cabezas de accion y valor sobre memoria y estado fisico.

PPO
train.py usa BC comun, regularizacion BC/KL, warmup/coseno, gamma .999,
GAE .995 y clip .1. Compara nominal/lineal/adaptativo/rango completo,
524288 interacciones por ejecucion y cinco semillas de entrenamiento.
Los episodios conservan su memoria y los deadlines se tratan como terminales.

FlashSAC/FQL
offpolicy.py contiene las cabezas y la compuerta; train_offpolicy.py realiza
su entrenamiento. FlashSAC reutiliza bloques oficiales MIT y critico C51.
FQL adapta las perdidas oficiales a PyTorch con flow matching, integracion
Euler de diez pasos y distillation. Ambos son adaptaciones recurrentes, no
reproducciones de los benchmarks de sus autores. La busqueda prospectiva
hyperparameters.py selecciona sigma/alpha antes de los cinco ajustes finales.
FQL ejecuta 100000 updates offline y luego 524288 interacciones online.
FlashSAC ejecuta 524288 interacciones online con replay e inicializacion PPO.
La compuerta actual compara dos criticos extrinsecos contra residual cero,
con umbral .005. No es un filtro fisico certificado.

Uso y dependencias
Ejecutar los modulos desde la raiz del repositorio, en el entorno del proyecto.
experiments/requirements.txt contiene las dependencias compartidas.
Los entrypoints de entrenamiento usan CUDA explicitamente y requieren una GPU
compatible y PyTorch CUDA. El controlador analitico puede ejecutarse en CPU.
La generacion de demostraciones del profesor NMPC necesita tambien
gym_pybullet_drones/control/requirements-nmpc.txt.
No se necesita instalar los repositorios originales FlashSAC/FQL: los bloques
Flash necesarios estan en vendor_flash y las perdidas FQL estan adaptadas aqui.

Entradas historicas y ejecuciones nuevas
OUT se resuelve respecto a la raiz local: runs/intervention_round6.
No ejecutar estos entrypoints en el PC mientras la campana original este activa:
comparten esa carpeta de resultados. En otra copia limpia se puede entrenar
sobre el controlador historico seleccionado reutilizando como entradas:
results/round6/phase1/phase1_conclusion.json, phase1_closed.json,
phase1_physical_closed.json y phase1_corrections_closed.json.
Se copian esos cuatro comprobantes a runs/intervention_round6, dejando claro
que son la seleccion historica del controlador, no una evaluacion nueva.
No copiar comprobantes de dataset ni de modelos terminados para simular datos
que no estan presentes. Los datos y cierres nuevos deben generarse realmente.

Orden para una nueva ejecucion sobre esa seleccion:
1. python -m experiments.round6.prepare_data
2. python -m experiments.round6.mpc_teacher_data
3. python -m experiments.round6.rescue_teacher_data
4. python -m experiments.round6.train --worker 0
   python -m experiments.round6.train --worker 1
   Los dos workers pueden correr en paralelo; la validacion se serializa.
5. python -m experiments.round6.select --phase 2
6. python -m experiments.round6.hyperparameters
7. python -m experiments.round6.train_offpolicy --worker 0
   python -m experiments.round6.train_offpolicy --worker 1
8. python -m experiments.round6.select --phase 3

prepare_data/teacher generan transiciones y auditorias; train y train_offpolicy
guardan BC, best/latest/final y sus comprobantes. Una politica incompleta
reinicia desde su inicializacion; no se afirma restauracion exacta de un
estado de simulacion interrumpido.
finalize_expert_data es un auxiliar historico previo al conjunto rescue:
no ejecutarlo despues de rescue_teacher_data, que ya publica el fingerprint final.

La orquestacion, prueba final y presentacion de la campana original no forman
parte de esta entrega de entrenamiento. Las rutas, protocolos y fuentes
archivados permiten revisar exactamente que se entreno.

Verificacion
experiments/tests/test_round6.py conserva las pruebas originales de ingenieria.
Algunas construyen redes y requieren CUDA; no se ejecutan automaticamente.
El recibo historico registra 303 pruebas aprobadas antes de abrir test.
La publicacion verifica identidad de fuentes, sintaxis, dependencias locales
y pruebas pequenas de CPU sin instanciar redes ni iniciar entrenamientos.

Comprobacion de publicacion: 28 modulos compilan, imports relativos resueltos,
siete archivos heredados coinciden y cinco pruebas pequenas CPU aprobaron en
6.65s. No se construyeron redes, no se uso GPU ni se modifico la campana activa.
Recibo: results/round6/training_code/publication_verification.json.
