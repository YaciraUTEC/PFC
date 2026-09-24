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

## Prueba 04 — normalización por horizonte efectivo

Config: igual que la prueba 03 (15 iter, 500 pasos/iter, currículo de severidad
+ currículo de distancia), más la normalización de φ por `horizonte_efectivo`
(ver commit `25c8adc7`) tanto en `μ_experto` como en `μᵢ`.

- Pesos finales: `[0, 0, 0.26, 0.34, 0, 0, 0.90]`
- Margen: 21.97 (mejora real y consistente desde 24.56 en la iteración 1 —
  a diferencia de las pruebas 01/02, aquí sí converge, aunque lejos de `eps`)
- Duración media de episodio: 119–224 pasos (sigue sin acercarse a los ~338 del PID)

**Mejoró respecto a la prueba 03**: ya no colapsa a una sola feature — se
recuperan `estabilidad_angular_rp` y `velocidad` junto con `progreso`.

**Sigue igual que las pruebas 01-03**: `proximidad_objetivo`, `estabilidad_altura`
y `oscilacion` quedan en exactamente 0.0 en las 15 iteraciones, sin excepción.

**Hipótesis revisada**: la normalización por horizonte quita el sesgo de
"episodio corto = se ve mejor", pero no quita un sesgo distinto introducido
por el currículo de distancia (prueba 03): si las candidatas vuelan, en
promedio, distancias A→B más cortas que las del dataset fijo del PID
(sobre todo en iteraciones tempranas), su `proximidad_objetivo` promedio por
paso se ve mejor simplemente porque la tarea es más fácil, no porque naveguen
mejor. Tampoco se está registrando el desenlace (`outcome`: caída/aterrizaje/
llegó/timeout) de los episodios de evaluación — sin ese dato no se puede
distinguir "se cayó pronto" de "llegó rápido a una meta cercana".

## Prueba 05 — registro de outcome, causa raíz confirmada

Config: igual que la prueba 04 (currículo de severidad + distancia,
normalización por horizonte), más el registro de `outcome` (`cayo`/`aterrizo`/
`llego`/`tiempo`) por episodio de evaluación en `rollout_mu()`.

- Pesos finales: `[0, 0, 0.69, 0, 0, 0, 0.72]`
- Margen: 23.87 (empieza en 26.41, converge de forma consistente — mejor
  tendencia que pruebas anteriores)
- **Outcomes por iteración** (de 10 episodios de evaluación c/u): entre 2 y 10
  terminan en `cayo` en TODAS las 15 iteraciones — dos iteraciones (2 y 11)
  tuvieron 10/10 caídas. `aterrizo` casi nunca ocurre (solo 3 veces en total).

**Esto confirma directamente la causa raíz**: no es (solo) el currículo de
distancia dándoles rutas más fáciles — las candidatas se están cayendo, mucho,
en toda la corrida. Nada en la recompensa penalizaba la caída en sí misma, así
que el algoritmo de proyección nunca tuvo forma de aprender que caerse es malo.

**Arreglo aplicado después de esta prueba** (commit `c20a1efb`): nueva feature
`penalizacion_caida` (8va feature) — vale 0 en cada paso normal y -1.0 en el
paso donde ocurre `es_caida()`. Como el PID nunca se cae en los 800 episodios,
`mu_experto` en esta componente es exactamente 0 — cualquier candidata que se
caiga mucho mostrará `mu_i` negativo ahí, y el mecanismo de búsqueda ya
existente (`w ≥ 0`) le asignará peso positivo automáticamente, sin necesidad
de fijar el peso a mano. Esto invalida esta prueba y las anteriores (el
vector φ pasó de 7 a 8 componentes) — la prueba 06 (pendiente) es la primera
con este arreglo.

## Pendiente

- [x] Registrar `outcome` por episodio en `rollout_mu()` — hecho, es lo que
      reveló las caídas en la prueba 05.

## Prueba 06 — primera corrida con `penalizacion_caida` (CAIDA_PENALTY=-1.0)

Config: igual que prueba 05, más la nueva 8va feature `penalizacion_caida`.

- Pesos finales: `[0, 0, 0.026, 0.408, 0, 0.011, 0.911, 0.053]`
- Margen: 22.26 (mejora leve desde 23.87 de la prueba 05)
- Caídas promedio por iteración: ~6.7 de 10 (67%) — **prácticamente igual**
  que la prueba 05 (~64%), no mejoró.

**El mecanismo funcionó** (peso de `penalizacion_caida` salió positivo, 0.053,
no quedó en 0 como las otras 3 problemáticas) — pero **demasiado chico** para
cambiar el comportamiento real de PPO. Causa: `penalizacion_caida` solo se
activa una vez por episodio (con descuento), mientras que features como
`progreso` se acumulan en cada paso — incluso con ~65-70% de caídas, el
déficit acumulado de una feature "de un solo evento" queda muy por debajo del
de una feature "de cada paso" en términos puramente numéricos.

**Arreglo aplicado** (commit `10d51318`): subir `CAIDA_PENALTY` de -1.0 a
-20.0 — mismo mecanismo, pero con suficiente magnitud para competir con las
demás features en la comparación. No requiere recalcular `mu_experto`/
`mu_escala` (el PID nunca activa esta feature, así que su valor sigue siendo
exactamente 0 sin importar la magnitud).

## Pendiente

- [ ] Prueba 07: primera corrida con `CAIDA_PENALTY=-20.0` — confirmar si
      ahora sí baja la tasa de caídas.
- [ ] Si -20 tampoco alcanza, considerar subir más, o reconsiderar el diseño
      (por ejemplo, aplicar el golpe en más de un paso alrededor de la caída,
      no solo en el instante exacto, para que el descuento no lo atenúe tanto).
