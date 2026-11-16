# Resultados finales de la ronda 7

La ronda 7 terminó el 16 de noviembre de 2026 con cero tareas pendientes.
La recomendación conserva **DSLPID + estimador estructurado + RecoveryDescent,
sin corrección neuronal**. El candidato experimental no cumple los criterios
prospectivos de ganancia confirmada y preservación de éxitos de la base.

| Evaluación | Rutas | Rescates del candidato | Éxitos de DSLPID perdidos |
|---|---:|---:|---:|
| Principal | 64 | 5 | 1 |
| Confirmación | 320 | 15 | 12 |
| Otros instantes de falla | 32 | 32 | 10 |

Los conteos combinan cinco réplicas y condiciones repetidas; no son rutas
independientes adicionales. La mejora crítica estimada es +0,312 puntos
porcentuales en principal y +0,047 en confirmación, con intervalos que incluyen
cero. El filtro y el afinado no justificaron su adopción. La ampliación de
demostraciones sí redujo el error de imitación un 26,61%, sin garantizar por
ello un mejor aterrizaje.

- [Conclusión global](CONCLUSION_GLOBAL_RONDA7.txt) e [informe detallado](INFORME_FINAL.txt).
- [Decisión operativa](deployment_recommendation.json) y [resultados estructurados](final_results.json).
- [Tabla por severidad](tabla_por_falla.csv) e [inventario](inventario_evaluacion.csv).
- [Comparación final](comparacion_final.png), [demostraciones y supervisión](demostraciones_y_supervision.png) y [diagnóstico de intervención](diagnostico_intervencion.png).
- [Integridad original](FINAL_ARTIFACTS_MANIFEST.json), [fuentes congeladas](frozen_before_final.json) y [cierre](ALL_COMPLETE.json).
- [Procedencia de esta publicación](publication_manifest.json).

El manifiesto original verifica los archivos finales indicados. El manifiesto
de publicación añade las huellas de protocolos y diagnósticos complementarios.
Se preservan los bytes originales. Las rutas absolutas en los comprobantes
identifican el PC del estudio. El generador original de la conclusión global
se archiva como report_generator_original.py; conserva sus rutas originales
y no es un punto de entrada de entrenamiento.

Los pesos, datasets e historias completas permanecen locales. El código
está en [experiments/round7](../../../experiments/round7/README.txt).
