EXPERIMENTO LOCAL — DSLPIDControl + IRL + SEIS COMPENSADORES TEMPORALES

Carpeta raíz: E:\UTEC\Tesis\TesisNicol
Entorno: E:\UTEC\Tesis\TesisNicol\conda_env
Código corregido: PFC; esta documentación describe la comparación histórica.
La actualización del DSLPID y las nuevas recompensas se documentan en experiments/round3/README.txt.

1. Tarea y alcance
La base física y demostradora es exactamente DSLPIDControl.
Cada arquitectura es el extractor temporal del actor-critic PPO que produce cuatro
correcciones de RPM sobre esa base. Es control residual, con información oracle
sobre activación/severidad de falla. La detección autónoma no se evalúa aquí.
Se comparan LSTM, Mamba-1, Mamba-2, Mamba-3 SISO, GTrXL y LTC real (no CfC).
Todos reciben el mismo historial finito de 32 observaciones, a 48 Hz.
Los tamaños se seleccionan cerca de 150000 parámetros por extractor.
Los backends Mamba son referencias PyTorch comprobadas, sin afirmar equivalencia
de velocidad con los kernels CUDA optimizados de Linux.

2. Recompensa
Se aprende una combinación lineal reescalada mediante apprenticeship/proyección,
con pesos no negativos sobre características de signos declarados.
Se excluyen del ajuste accion_residual_magnitud y eficiencia_accion:
el experto base tiene residual cero por definición, sesgando esas comparaciones.
El esfuerzo residual se regulariza explícitamente con coeficiente .02.
Se incluyen prioridades de tarea explícitas comunes a todos: misión +100, caída
-120, aterrizaje cercano +40, lejano -60 y timeout -30.
Estas prioridades y los signos son conocimiento incorporado; no son preferencias
descubiertas exclusivamente desde las demostraciones.
Los pesos, escalas, máscara de aprendizaje, coeficientes y clase base se congelan
juntos en reward.json. La recompensa de estilo se multiplica por .01.

3. Datos y separación experimental
1600 episodios DSLPIDControl; mezcla nominal y fallas leves 2,4,6,8,10%.
Se conservan todos los desenlaces, incluidos los fallos previos a exposición.
Las escalas se calculan únicamente con demostraciones.
La validación de políticas usa semillas desde 50000; la prueba final, desde 70000.
Los mismos escenarios y severidades se reutilizan entre arquitecturas.
La evaluación informa intentos, exposición, misión completa, aterrizaje y caída.
La demostración no cubre todas las fallas del 20%; la transferencia se evalúa
mediante interacción con el simulador y las prioridades de tarea explícitas.

4. Configuración razonada
IRL: 8 iteraciones de 100000 pasos PPO; 200 episodios para comparar expectativas.
PPO de control: 2048 transiciones por actualización; 32 entornos locales y 64 pasos
por entorno, batch 128, 5 épocas, gamma .99, GAE .95, clip .15, target KL .015,
recorte de gradiente .5 y entropía .005.
Se comparan tasas iniciales 1e-4 y 3e-4 en pilotos LSTM de 32768 pasos.
La tasa decae linealmente hasta el 10% de su valor inicial.
Se elige el piloto por éxito de misión en validación, sin usar la prueba final.
Es una selección práctica de configuración inicial, no una prueba de óptimo global.
Se guardan configuraciones completas y curvas para decidir si ampliar el presupuesto.

5. Aceleración sin cambiar el vuelo
El prefijo anterior a la falla es propiedad exclusiva de DSLPIDControl.
Se simula ese prefijo y se actualiza el historial, omitiendo únicamente llamadas
neuronales que no pueden afectar las acciones. Una prueba verifica igualdad de
estados, historial y transiciones respecto del vuelo completo.
Los fallos en el prefijo se devuelven como episodios fallidos; nunca se reintenta
silenciosamente hasta obtener uno exitoso.
En entrenamiento se muestrean fallas; el vuelo nominal no activa el residual.
Los pasos de aprendizaje no incluyen el prefijo físico. La evaluación final sigue
contando intentos completos, con sus pasos de exposición y desenlace.

6. Archivos
experiments/run.py: datos, IRL, pilotos, entrenamiento, evaluación y checkpoints.
experiments/tests/: pruebas de contratos, simulación y referencias numéricas.
experiments/models.py: seis extractores temporales.
experiments/vendor_gtrxl.py: fuente DI-engine adaptada con helpers PyTorch locales.
experiments/SOURCES.txt y LICENSE_gtrxl.txt: fuentes y licencia.
runs/dslpid_comparison_v1/: corrida diagnóstica que reveló el sesgo de residual cero.
runs/dslpid_comparison_v2/: recompensa corregida y comparación principal.
revision/pruebas.xml y dependencias_instaladas.txt: validación y entorno.

7. Reproducir desde PowerShell
. E:\UTEC\Tesis\TesisNicol\PFC\experiments\local_env.ps1
E:\UTEC\Tesis\TesisNicol\conda_env\python.exe -m pytest experiments/tests -q --basetemp E:\UTEC\Tesis\TesisNicol\_tmp\pytest -o cache_dir=E:\UTEC\Tesis\TesisNicol\_cache\pytest
E:\UTEC\Tesis\TesisNicol\conda_env\python.exe -u -m experiments.run --stage all --envs 8 --policy-envs 32 --steps 524288 --test-episodes 50

No ejecutar una segunda copia simultánea de una corrida ya activa.


8. Ejecución local y evaluación final
Mamba-1 usa un scan afín paralelo matemáticamente equivalente a la recurrencia
selectiva; se verifican salida y gradientes contra la recurrencia y la referencia
oficial. LTC conserva ncps.LTC y sus tres subpasos ODE; solo se compila la celda
con PyTorch/Triton para Windows. Se usa el compilador Visual Studio ya instalado.
Los temporales y cachés de compilación quedan dentro de la raíz de esta tarea.
La comparación de tiempos describe estos backends, no los kernels de Linux.
Los checkpoints nuevos se guardan después de actualizar el optimizador PPO.
Al reanudar un checkpoint antiguo se repite únicamente su última colección sin
actualizar; se guarda ese hecho en resume_history. Se mantienen optimizador,
pesos, objetivo total y calendario. Los flujos de entornos reinician su semilla.

La evaluación pareada definitiva está en paired_test.json de cada arquitectura
y baseline_paired.json. experiments/paired.py asigna exactamente las semillas
70000 a 70049 por severidad, independientemente de cuántos procesos se usen o
cuánto dure cada episodio. Los test.json iniciales son diagnósticos preliminares:
sus reinicios automáticos podían mezclar distinto número de rutas por proceso.
La selección del checkpoint permanece basada únicamente en validación; se
conservan los checkpoints elegidos antes de esa evaluación pareada final.

Severidad significa pérdida relativa de RPM del motor 0: RPM_final *= (1-p).
Dado que empuje = kf * RPM^2, la pérdida de empuje es 1-(1-p)^2, no p.
Por ejemplo, 10% RPM representa 19% empuje y 20% RPM representa 36% empuje.
La falla comienza a mitad del crucero y persiste. No se ensayan otros motores,
instantes de falla, ruido externo ni detectores de falla autónomos.

La ronda principal usa una semilla de entrenamiento (42) y 524288 pasos de
aprendizaje por arquitectura. Esto permite una primera comparación; no estima
la variabilidad entre entrenamientos ni demuestra convergencia u óptimo global.
La distancia de proyección IRL final se informa, sin declarar convergencia.

Evaluar los checkpoints ya entrenados con las rutas comunes:
E:\UTEC\Tesis\TesisNicol\conda_env\python.exe -u -m experiments.paired --episodes 50 --envs 8
Definición de éxito: outcome='llego' agota la lista de waypoints, con tolerancia
espacial de 0.25 m, incluido el último waypoint B a z=0.1 m. Puede terminar antes
del contacto con suelo. Las tablas indican éxito de la ruta según ese criterio;
no certifican un aterrizaje completo ni estabilidad sostenida tras contacto.
LTC usa una celda ODE densa de 160 unidades; el argumento común layers=2 del
constructor solo aplica a las otras arquitecturas. Se empareja por número de
parámetros aproximado, no por número de capas ni costo de cómputo.
Los pilotos y la selección de checkpoint usaron semillas iniciales comunes por
proceso y reinicios automáticos. Esa validación no garantiza exactamente la
misma mezcla de rutas, pues las duraciones de episodios varían. La evaluación
final sí fija una semilla por episodio y permite comparación pareada exacta.
La configuración ya elegida se mantiene; no se reelige usando la prueba final.
Para una siguiente ronda conviene usar el evaluador pareado también al validar.
El resultado histórico de Nicol al 4% no se reproduce sin el checkpoint exacto.
La comparación actual mezcla las correcciones y una nueva configuración de
aprendizaje; no cuantifica causalmente cada corrección ni el aporte aislado del
IRL o de la memoria temporal. Esas preguntas requieren ablaciones adicionales.