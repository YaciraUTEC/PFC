# Ronda 8: preservacion y rescate con GTrXL

**Estudio cerrado.** La compuerta 505 sobre GTrXL201 fue seleccionada antes del
test: 252 rescates y 0 aterrizajes de DSLPID perdidos en 16.384 condiciones
pareadas sobre 2.496 rutas. La cota registrada de perjuicio fue 0,2892% en la
mezcla simulada con una condicion independiente asignada por ruta.

En 29,5–29,6% de perdida de RPM de un motor y efectividad estimada:

| Prueba | DSLPID | Propuesta 505 | Rescates/perdidas |
|---|---:|---:|---:|
| Principal |1082/1536(70,44%)|1165/1536(75,85%)|83/0|
| Confirmacion |3020/4096(73,73%)|3155/4096(77,03%)|135/0|

Otros instantes 4/5/7/8 s: 512 condiciones sin intervencion ni rescates; se preserva
la base. El 30% no se resolvio. Cero perdidas observadas no es seguridad universal.

- [Informe completo](report/attempt_1/CONCLUSION_GLOBAL_RONDA8.txt)
- [Resultados pareados](report/attempt_1/resultados_pareados.csv)
- [Tabla por falla](report/attempt_1/tabla_por_falla.csv)
- [Curva completa](report/attempt_1/comparacion_por_falla.png)
- [Zona critica](report/attempt_1/comparacion_zona_critica.png)
- [Cinco compuertas](report/attempt_1/replicas_de_compuerta.csv)
- [Decision original](final/attempt_1/decision.json)
- [Cierre](ALL_COMPLETE.json)
- [Codigo y reproduccion](../../../experiments/round8/README.txt)
- [Revision critica posterior](../analysis/README.md)

El criterio original exigia preservar todos los aterrizajes observados. Bajo
maximo exito medio, la504 obtiene186 rescates y 12 pérdidas en confirmacion,
frente a 135/0 de505. El analisis posterior cuantifica ese intercambio, la
gravedad y la incertidumbre, sin cambiar retrospectivamente la seleccion.

Las proyecciones JSON por episodio conservan desenlaces, casos, contactos,
exposicion y resumen de intervencion; omiten historiales y recompensas no usados
en la auditoria. Cada una declara la huella del JSON original. No son nuevas
simulaciones. Los comprobantes byte a byte conservan las rutas del PC original;
sus hashes describen el estudio historico, no un cierre nuevo de esta copia.
publication_manifest.json documenta que se copio y que se proyecto.

425 pruebas pasaron en el paquete trasladado original. La publicacion registra
sus propias comprobaciones por separado. Pesos, datasets y ZIP completo quedan
locales bajo E:/UTEC/Tesis/TesisNicol/ENTREGA_NICOLE_RONDA8.
