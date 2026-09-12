# Pruebas de la búsqueda de pesos IRL (Apprenticeship Learning)

Registro de las corridas de `entrenar_irl_apprenticeship.py` durante el proceso
iterativo de diagnóstico y ajuste del vector de features φ y del algoritmo de
búsqueda. Cada carpeta contiene los archivos crudos (`irl_weights.json` y/o
`irl_convergencia.csv`) de esa corrida específica, tal como se generaron.

Orden de features en todos los vectores `w`:
`[proximidad_objetivo, estabilidad_altura, estabilidad_angular_rp, velocidad, oscilacion, aceleracion_vertical, progreso]`

## Resumen comparativo

| Prueba | Config. | Pesos finales (redondeado) | Margen final | Duración media episodio candidato | Conclusión |
|---|---|---|---|---|---|
| **01** — currículo de severidad | 15 iter, 500 pasos/iter, con inyección de falla en vivo + currículo de severidad (2%→80%). Sin currículo de distancia, sin normalización por horizonte. | `[0, 0, 0.57, 0.75, 0, 0, 0.35]` | 18.07 (estancado desde iter 1: 18.14→18.07) | No registrada (aún no existía la columna) | `proximidad_objetivo`, `estabilidad_altura` y `oscilacion` quedaron en 0.0 en **las 15 iteraciones sin excepción**. `velocidad` domina. Margen prácticamente plano — la búsqueda no converge, se estanca desde la primera iteración. |
| **02** — 30 iteraciones, 1000 pasos | Igual que 01 pero con el doble de presupuesto (30 iter, 1000 pasos/iter), para descartar que fuera un problema de entrenamiento insuficiente. | `[0, 0, 0.96, 0.08, 0.25, 0, 0]` | 17.99 | No registrada | Más presupuesto **no** resolvió el estancamiento (18.07→17.99, cambio marginal). Además `progreso` —que sí se mantenía positivo en la prueba 01— también cayó a 0. Confirma que el problema no es de presupuesto de entrenamiento, es estructural. |
| **03** — currículo de distancia A→B | 15 iter, 500 pasos/iter, se agrega currículo de distancia A→B (empieza en 0.8m, sube a ~5.66m) + registro de duración de episodio. Sin normalización por horizonte todavía. | `[0, 0, 0, 0, 0, 0, 1.0]` | 15.96 (primera mejora real: 16.44→15.96) | 113–222 pasos (vs. ~338 del PID) — **nunca se acerca** a la duración del experto, ni siquiera en la iteración con distancia máxima habilitada. | El margen sí se mueve por primera vez, pero el resultado colapsa aún más: **todas** las features menos `progreso` quedan en 0. La columna de duración confirma la hipótesis: los episodios candidatos son sistemáticamente más cortos que los del PID, lo que hace que las features "siempre negativas" (que se acumulan sin cancelarse) parezcan mejores que el experto solo por acumular menos pasos — no por volar mejor. |

## Diagnóstico y arreglo aplicado después de la prueba 03

Las tres pruebas comparten el mismo problema de fondo: `μ_experto` y `μᵢ` se
calculaban como la suma cruda descontada de φ sobre el episodio completo
(`Σ γᵗ·φ(sₜ)`), sin normalizar por cuánto duró ese episodio. Como el PID
(experto) siempre completa episodios largos (~338 pasos) y las políticas
candidatas de la búsqueda terminan mucho antes (ya sea por caerse, o por
currículos que acortan la tarea), un episodio corto acumula menos costo total
en las features "siempre negativas" — pareciendo mejor que el experto sin
serlo realmente. Esto explica por qué el algoritmo de proyección (Abbeel-Ng)
recortaba sistemáticamente esas componentes a 0 (`w ≥ 0`, ver
`entrenar_irl_apprenticeship.restringir_w`).

**Arreglo**: se agregó `irl_features.horizonte_efectivo(T, gamma) =
Σ_{t=0}^{T-1} γᵗ`, y cada episodio se divide entre su propio horizonte
efectivo antes de promediar entre episodios — convirtiendo "costo total
acumulado" en "costo promedio por paso, ponderado por el descuento". Este
arreglo invalida las tres pruebas de esta carpeta (la escala de `μ_experto`
cambió por completo), así que la prueba 04 (pendiente) es la primera corrida
con la normalización aplicada.

## Pendiente

- [ ] Prueba 04: primera corrida con `horizonte_efectivo` aplicado — confirmar
      si `proximidad_objetivo`/`estabilidad_altura`/`oscilacion` dejan de
      quedarse en 0, y si el margen converge por debajo de `eps`.
