# Evaluación de control y aprendizaje — ronda 6

## Cómo leer los resultados

Las columnas de falla indican pérdida de RPM de un motor. Cada celda es la proporción de vuelos que terminó con aterrizaje controlado y estable. Redes: cinco entrenamientos sobre las mismas 60 rutas por punto; no son 300 geometrías independientes.

**Por éxito medio, el mejor enfoque aprendido evaluado es FlashSAC + GTrXL.** En la prueba principal recuperó 250 casos que DSLPID no resolvía y perdió 54 que DSLPID sí resolvía: balance neto de 196 éxitos adicionales. Estos recuentos agregan 14.400 casos modelo/ruta de esa familia en todos los niveles y escenarios de la prueba principal. La mejora crítica se confirmó en 200 rutas nuevas: +14,35 puntos porcentuales, intervalo [9,40; 19,30].

La recomendación conservadora de mantener DSLPID responde al requisito prospectivo de no perder ningún éxito de la base, no a que DSLPID tenga una tasa media mayor en las columnas estimadas. El candidato no cumplió esa protección ni al mover el instante de falla. Ambos criterios deben leerse por separado.


**Decisión:** conservar DSLPID actualizado, sin corrección neural en el controlador recomendado.

La familia candidata elegida en validación fue **FLASHSAC + GTrXL**. Su diferencia de éxito frente a la base en los casos severos estimados fue 13.58 puntos porcentuales; intervalo simultáneo [1.75, 25.42]. La confirmación en 200 rutas dio 14.35 puntos, intervalo [9.40, 19.30].

Criterios no cumplidos: conservar todos los éxitos de la base en el test, conservar todos los éxitos en confirmación, conservar los éxitos al cambiar el instante de falla.

La aceptación exige ganancia confirmada, ninguna pérdida observada de éxitos de la base en todos los regímenes evaluados y cómputo compatible con 48 Hz. Los intervalos no prueban ausencia de riesgo.

## Qué cambió y qué se conserva

La coordinación de recuperación, descenso y frenado se evaluó antes del aprendizaje. La configuración elegida obtuvo 71 aterrizajes adicionales y 0 pérdidas frente a la referencia en 1.056 casos de validación. La mejora incluye el observador de una sola falla y la reserva de empuje de 0,1%; no se atribuye únicamente al supervisor ni a una red.

Se corrigió una compuerta que copiaba la intervención anterior: ahora la decisión usa el estado físico y la memoria, con ejemplos explícitos del inicio de la intervención y del retorno al PID. Las demostraciones del profesor NMPC se ejecutaron bajo las mismas limitaciones del alumno. Se conservaron los intentos descartados como diagnóstico y se repitieron las inicializaciones afectadas.

El éxito exige contacto suave, límites de velocidad e inclinación y un segundo estable. Los porcentajes de falla indican pérdida de RPM de un motor, no pérdida directa de empuje. El límite estacionario normal de 29,7186% no prueba imposibilidad de otras maniobras.

## Resultados independientes

Redes: cinco semillas de entrenamiento y 60 rutas por punto (300 casos por modelo de arquitectura y método). Controladores principales: las mismas 60 rutas. NMPC secundario: 12 rutas por punto. Un 100% observado corresponde a esa muestra, no a una garantía universal.

### Efectividad estimada, ruido 1×

| Controlador | 0% | 25% | 28% | 29% | 29,5% | 29,6% | 30% | 35% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLPID referencia | 100.0% | 100.0% | 95.0% | 83.3% | 61.7% | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| DSLPID actualizado | 100.0% | 100.0% | 100.0% | 91.7% | 68.3% | 70.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + Mamba-3 | 100.0% | 100.0% | 100.0% | 91.0% | 67.3% | 64.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + Mamba-3 | 100.0% | 100.0% | 100.0% | 95.7% | 79.7% | 70.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + Mamba-3 | 100.0% | 100.0% | 100.0% | 94.0% | 70.0% | 64.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + GTrXL | 100.0% | 100.0% | 99.3% | 92.0% | 65.7% | 61.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + GTrXL | 100.0% | 100.0% | 100.0% | 98.3% | 85.3% | 78.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + GTrXL | 100.0% | 99.7% | 99.3% | 91.3% | 69.0% | 62.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI rápido | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI mayor horizonte | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Efectividad estimada, ruido 2×

| Controlador | 0% | 25% | 28% | 29% | 29,5% | 29,6% | 30% | 35% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLPID referencia | 100.0% | 100.0% | 95.0% | 85.0% | 68.3% | 46.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| DSLPID actualizado | 100.0% | 100.0% | 100.0% | 91.7% | 68.3% | 68.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + Mamba-3 | 100.0% | 100.0% | 100.0% | 91.0% | 67.0% | 64.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + Mamba-3 | 100.0% | 100.0% | 100.0% | 96.0% | 79.7% | 71.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + Mamba-3 | 100.0% | 100.0% | 100.0% | 93.0% | 67.0% | 60.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + GTrXL | 100.0% | 100.0% | 99.3% | 92.0% | 66.7% | 62.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + GTrXL | 100.0% | 100.0% | 100.0% | 98.3% | 85.3% | 80.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + GTrXL | 100.0% | 99.7% | 99.3% | 92.0% | 68.3% | 63.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI rápido | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 91.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI mayor horizonte | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Efectividad exacta, ruido 1×

| Controlador | 0% | 25% | 28% | 29% | 29,5% | 29,6% | 30% | 35% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLPID referencia | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| DSLPID actualizado | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + Mamba-3 | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 98.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + Mamba-3 | 100.0% | 100.0% | 100.0% | 100.0% | 98.3% | 94.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + Mamba-3 | 100.0% | 100.0% | 100.0% | 100.0% | 97.3% | 95.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + GTrXL | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% | 99.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + GTrXL | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 98.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + GTrXL | 100.0% | 99.3% | 99.0% | 99.3% | 97.3% | 87.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI rápido | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI mayor horizonte | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Efectividad exacta, ruido 2×

| Controlador | 0% | 25% | 28% | 29% | 29,5% | 29,6% | 30% | 35% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSLPID referencia | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| DSLPID actualizado | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + Mamba-3 | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + Mamba-3 | 100.0% | 100.0% | 100.0% | 100.0% | 98.3% | 95.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + Mamba-3 | 100.0% | 100.0% | 100.0% | 100.0% | 98.0% | 95.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| PPO + GTrXL | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| FLASHSAC + GTrXL | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| FQL + GTrXL | 100.0% | 99.3% | 99.0% | 99.0% | 96.7% | 87.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI rápido | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| NMPC sin INDI mayor horizonte | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Cuándo intervenir: compuerta e instante de falla

| Familia | Éxito con compuerta | Éxito sin compuerta | Fallos evitados | Beneficios bloqueados |
|---|---:|---:|---:|---:|
| FLASHSAC + GTrXL | 96.2% | 40.6% | 92 | 3 |
| FLASHSAC + Mamba-3 | 91.9% | 48.1% | 71 | 1 |
| FQL + GTrXL | 81.9% | 1.9% | 128 | 0 |
| FQL + Mamba-3 | 79.4% | 5.6% | 119 | 1 |
| PPO + GTrXL | 86.2% | 84.4% | 7 | 4 |
| PPO + Mamba-3 | 86.2% | 86.9% | 4 | 5 |

Ablación de la tabla: 16 rutas, cinco semillas, 29,5% y 29,6%, efectividad estimada y ruido doble. Los cuatro regímenes están en gate_ablation.csv y gate_summary.csv. Sin compuerta se aplica la propuesta de la red dentro de los mismos límites físicos. Una compuerta útil debe evitar daños sin bloquear demasiadas correcciones beneficiosas.

| Familia | Falla a (s) | Éxito | Base | Diferencia (puntos) | Ganancias / pérdidas |
|---|---:|---:|---:|---:|---:|
| FLASHSAC + GTrXL | 4 | 61.2% | 62.5% | -1.25 | 26 / 28 |
| FLASHSAC + GTrXL | 8 | 21.9% | 0.0% | +21.88 | 35 / 0 |
| FLASHSAC + Mamba-3 | 4 | 51.2% | 62.5% | -11.25 | 21 / 39 |
| FLASHSAC + Mamba-3 | 8 | 5.6% | 0.0% | +5.62 | 9 / 0 |
| FQL + GTrXL | 4 | 28.8% | 62.5% | -33.75 | 11 / 65 |
| FQL + GTrXL | 8 | 2.5% | 0.0% | +2.50 | 4 / 0 |
| FQL + Mamba-3 | 4 | 32.5% | 62.5% | -30.00 | 5 / 53 |
| FQL + Mamba-3 | 8 | 2.5% | 0.0% | +2.50 | 4 / 0 |
| PPO + GTrXL | 4 | 14.4% | 62.5% | -48.12 | 3 / 80 |
| PPO + GTrXL | 8 | 0.6% | 0.0% | +0.62 | 1 / 0 |
| PPO + Mamba-3 | 4 | 20.0% | 62.5% | -42.50 | 4 / 72 |
| PPO + Mamba-3 | 8 | 5.0% | 0.0% | +5.00 | 8 / 0 |

Prueba temporal: efectividad estimada y ruido doble, 16 rutas y cinco semillas por familia. La aceptación también exige cero pérdidas observadas respecto a la base al adelantar o retrasar la falla a 4 s y 8 s. Este requisito se fijó antes de abrir el test; la comprobación no garantiza seguridad fuera de la muestra.

## Schedules del rango de ruido

| Arquitectura | Nominal | Lineal | Adaptativo | Rango completo | Elegido en validación |
|---|---:|---:|---:|---:|---|
| Mamba-3 | 0.9727 | 1.0039 | 1.0023 | 0.9945 | linear |
| GTrXL | 0.9531 | 0.9859 | 0.9609 | 0.9859 | linear |

La tabla anterior contiene el objetivo agregado de validación, no porcentajes de éxito. Las curvas y tablas finales separan cada severidad. La clonación inicial y su regularización usan ruido nominal en las cuatro ramas. El schedule solo modifica el rango de ruido en los vuelos de entrenamiento.

Lineal: ampliar el máximo de 1× a 2× durante el 80% del presupuesto y mantenerlo después. Adaptativo: avanzar o retroceder según el éxito de entrenamiento en cada estrato severo estimado, protegiendo los casos leves. Nominal y rango completo sirven como controles de exposición.

Ruido nominal: desviación de 0,002 m/s en velocidad lineal y 0,002 rad/s en velocidad angular; ruido doble: 0,004 en ambas. No se añadieron ruido de posición, viento ni errores de orientación.

**Límites de ruido solicitados durante el entrenamiento**

| Arquitectura / schedule | Máximo solicitado entre semillas | Rango 2× antes del fin | Rango 2× antes del checkpoint elegido |
|---|---:|---:|---:|
| mamba3 / nominal | 1.00–1.00 | 0/5 | 0/5 |
| mamba3 / linear | 2.00–2.00 | 5/5 | 1/5 |
| mamba3 / adaptive | 1.20–2.00 | 3/5 | 1/5 |
| mamba3 / full | 2.00–2.00 | 5/5 | 3/5 |
| gtrxl / nominal | 1.00–1.00 | 0/5 | 0/5 |
| gtrxl / linear | 2.00–2.00 | 5/5 | 1/5 |
| gtrxl / adaptive | 1.00–2.00 | 3/5 | 1/5 |
| gtrxl / full | 2.00–2.00 | 5/5 | 3/5 |

Se excluye una ampliación decidida justo después del último paso: no cuenta como exposición. El límite se aplica al reiniciar cada episodio; no es una medición del ruido de cada muestra. En PPO, un checkpoint anterior al afinado online conserva únicamente la clonación nominal. El detalle está en noise_exposure.csv.

## Tiempo medido

| Trabajo | Cantidad | Suma de minutos de los trabajos | Mediana por trabajo (min) |
|---|---:|---:|---:|
| Clonación inicial compartida | 10 | 7.0 | 0.7 |
| PPO definitivo | 40 | 1142.3 | 23.9 |
| Pilotos de hiperparámetros | 12 | 170.4 | 14.5 |
| FlashSAC definitivo | 10 | 645.2 | 63.1 |
| FQL definitivo | 10 | 896.0 | 87.4 |

Las sumas incluyen trabajos concurrentes y no equivalen al tiempo de reloj. Los CSV separan validación y espera de turno, codificación, inicialización y aprendizaje offline/online. En PPO la carga de datos se comparte por trabajador y no se cronometró por modelo: las celdas vacías significan no medido o no aplicable, no cero segundos. Los intentos de preparación descartados no cuentan como modelos definitivos.

| Etapa registrada | Minutos de reloj |
|---|---:|
| data | 8.6 |
| final_evaluation | 752.1 |
| mpc_expert_data | 5.4 |
| phase2_selection | 27.2 |
| phase2_training | 522.0 |
| phase3_selection | 40.1 |
| phase3_training | 859.0 |
| rescue_expert_data | 3.1 |
| tests_before_test | 3.3 |
| training_smoke | 4.9 |

Tiempo desde el inicio registrado de la campaña hasta la generación de este informe: 57.52 horas; incluye estudios de desarrollo, esperas y repeticiones de preparación. La primera recolección de datos incluyó una espera por el cierre del control.

## Alcance de FlashSAC, FQL y las arquitecturas

FlashSAC y FQL se evaluaron después de cerrar la elección de control y schedule. Se escogieron hiperparámetros por arquitectura con pilotos y rutas de validación separadas; después se entrenaron cinco semillas. FQL recibió 100.000 actualizaciones offline y cada modelo definitivo 524.288 interacciones online.

Los 60 modelos completan su presupuesto, pero se evalúa el checkpoint elegido por validación. PPO: 20 de 40 mejores checkpoints antes del afinado online; FlashSAC: 0 de 10 mejores checkpoints antes del afinado online; FQL: 4 de 10 mejores checkpoints antes del afinado online. En PPO, cero pasos online significa conservar la clonación inicial; en FQL incluye sus 100.000 actualizaciones offline. No se atribuye al afinado online una mejora que ya existía antes.

Las cabezas usan el encoder recurrente congelado del PPO elegido. Se reutilizan bloques oficiales de FlashSAC y un port de las pérdidas oficiales de FQL. Mamba-3 es una referencia matemática SISO portable; esta evaluación no compara kernels MIMO fusionados ni demuestra una superioridad general de las arquitecturas o algoritmos completos.

La compuerta off-policy compara dos estimaciones de recompensa extrínseca para intervenir frente a no intervenir. La ablación de la compuerta y los cambios del instante de falla se entregan junto con las curvas; no se reajustó ningún modelo después de abrir el test.

## Archivos de respaldo

- `success_rates.csv`: todas las severidades y regímenes, dispersión entre semillas y denominadores.
- `paired_statistics.json`, `confirmation_statistics.json`: comparación pareada e intervalos.
- `gate_ablation.csv` y carpeta `sensitivity`: compuerta e instante de falla.
- `training_times.csv`, `cloning_times.csv`, `hyperparameter_times.csv`, `stage_times.json`: tiempos.
- `final_integrity.json` y `frozen_before_test.json`: verificación de datos, fuentes y modelos.
- `INFORME_FINAL.txt`: detalle completo y limitaciones.

Fuentes: [FlashSAC](https://arxiv.org/abs/2604.04539), [FQL](https://proceedings.mlr.press/v267/park25f.html), [NMPC tolerante a fallas](https://arxiv.org/abs/2109.12886).

## Aclaraciones finales de auditoría

La comprobación final validó 60 modelos definitivos, 12 pilotos y 204.736 registros modelo/ruta. Los registros reutilizan geometrías entre semillas y algunos certificados OFF: no equivalen a ese número de vuelos físicamente independientes. Las 182 fuentes congeladas, incluido el generador del informe, permanecen idénticas; también se verificaron los 132 checkpoints y los 198 checkpoints anteriores.

Los 54 éxitos perdidos por GTrXL + FlashSAC en la prueba principal se clasifican en 23 contactos tempranos, 12 aterrizajes bruscos y 19 episodios sin terminar a tiempo; 48 corresponden al 29,6% de pérdida de RPM. Es una clasificación descriptiva, no una demostración de una única causa. La compuerta reduce muchos fallos, pero valorar una ventaja esperada no certifica conservar el éxito de cada vuelo.

La ablación de la compuerta utiliza 16 rutas; la comparación principal, 60. Por eso el 96,25% del subconjunto de GTrXL + FlashSAC no debe compararse directamente con sus tasas del conjunto principal. Los datos detallados están en [candidate_loss_analysis.json](candidate_loss_analysis.json) y [gate_summary_estimated_noise2.csv](gate_summary_estimated_noise2.csv).


Comparación justa de NMPC en rutas comunes: con efectividad estimada y ruido doble, al reunir 29,5% y 29,6%, DSLPID logra 21/24 aterrizajes, NMPC rápido 23/24 (3 ganancias y 1 pérdida frente a la base) y NMPC de mayor horizonte 24/24 (3 ganancias y ninguna pérdida). Son 12 rutas por severidad. El cálculo dedicado p95 fue de 21,91 ms y 65,72 ms respectivamente, frente a un período de 20,83 ms. La mejor calidad de NMPC de mayor horizonte no lo hace compatible con 48 Hz en esta implementación. En ninguna configuración evaluada hubo éxitos con 30%, 35%, 40% o 50% bajo este criterio de aterrizaje.

Las dos variantes finales del archivo NMPCINDIControl usan INDI desactivado (ganancia 0), elegido en desarrollo. La comparación pareada de NMPC con DSLPID se hizo solo en las mismas 12 rutas; sus ganancias y pérdidas descriptivas están en [nmpc_paired_subset.json](nmpc_paired_subset.json). No se extrapola un 12/12 a las 60 rutas principales.

Hubo una interrupción de la ejecución. El último resultado previo quedó guardado el 14/11/2026 a las 11:54 y se reanudó el 14/11/2026 a las 23:38, hora de Lima. No se conoce la hora exacta de parada; el intervalo entre ese guardado y la reanudación fue de 11.74 horas. Los 117 archivos conservados se validaron antes de continuar. La fila `final_evaluation_receipt` mide solamente el tramo reanudado; antes hubo al menos 5.02 horas hasta el último resultado previo. Ese intervalo sin progreso no se debe sumar como entrenamiento activo. El tiempo total desde el inicio de la campaña sí incluye esta interrupción. Las duraciones de etapas corresponden a los recibos preservados; en etapas reanudadas no equivalen necesariamente al cómputo acumulado de todos los intentos.

La configuración operativa elegida está exportada en [recommended_configuration.json](recommended_configuration.json) y en `recommended_configuration.json`. Conserva el controlador y el supervisor probados, con compensación neural desactivada.

## Contenido publicado

Este directorio conserva informes, tablas, figuras, configuración, protocolos y comprobantes de la evaluación terminada el 15 de noviembre de 2026. [La configuración recomendada](recommended_configuration.json), [la comparación pareada](paired_statistics.json), [la confirmación](confirmation_statistics.json) y [el detalle de los contraejemplos](candidate_loss_analysis.json) permiten revisar la decisión.

Figuras: [efectividad estimada](comparacion_estimated.png), [efectividad exacta](comparacion_exact.png) y [schedules](schedules.png). Tablas completas: [estimada/ruido doble](tabla_estimated_ruido2.csv) y [todos los resultados](success_rates.csv).

Los checkpoints, datasets, caches, logs extensos y las trazas completas por episodio permanecen en el almacenamiento local original, no en este paquete resumido. Los hashes y rutas del comprobante de congelación son históricos; copiar esos recibos a una carpeta vacía no reproduce una corrida ni crea los pesos ausentes. Los resúmenes de fase2/fase3 omiten las trazas voluminosas y declaran el hash del archivo original.

La auditoría histórica aprobó 303 pruebas. `publication_validation.json` y `publication_tests.xml` registran por separado la comprobación de esta copia publicada. `manifest.json` enumera los archivos y sus hashes.

**Verificación de publicación:** 351 pruebas aprobadas, cero fallas, errores o saltos; 158,22 segundos. Las 303 pruebas del cierre histórico se conservan por separado.
