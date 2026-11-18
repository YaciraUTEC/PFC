# Analisis posterior: rescate, perjuicio y gravedad

[Informe critico completo](ANALISIS_CRITICO.txt) ·
[Tasas auditadas](tasas_condicionales.csv) ·
[Rescate/perjuicio](rescate_perjuicio_confirmacion.png) ·
[Sensibilidad al costo](sensibilidad_al_costo.png).

Conclusiones:505 cumple el contrato original de preservacion empirica;504
obtiene mayor exito medio aceptando mas perjuicios. En confirmacion critica,
504 rescata 186/1076(17,29%) y pierde 12/3020(0,397%); 505 rescata 135/1076(12,55%)
y pierde 0. Son condiciones repetidas de 1024 rutas, no 4096 rutas independientes.
Los resultados 75/0 y 137/3 de 505 con/sin guarda pertenecen a desarrollo.

La frontera no sustituye la incertidumbre ni la gravedad: incluso cuando ambos
controladores fallan pueden cambiar consecuencias que una metrica binaria no
cuenta. El informe revisa literatura primaria, F1, costos explicitos, limites de
riesgo, supuestos contrafactuales y posibles protocolos futuros. No implementa
ese protocolo ni cambia el checkpoint recomendado o el test historico.

Reproduccion desde la raiz del repositorio, con NumPy/SciPy:

```text
python -B results/round8/analysis/analizar_resultados.py
```

Escribe JSON/CSV en runs/round8_reanalysis; no necesita modelos ni ejecuta vuelos.
La adaptacion cambia solo las rutas y agrega opciones --source/--output; los
calculos permanecen iguales al script original, archivado con sufijo_original.
La publicacion comprueba igualdad de las salidas numericas. Los generadores
originales de presentacion usan la disposicion local historica; las figuras y
el informe ya generados se publican byte a byte. No ejecutar esos generadores
originales para una reproduccion portable.

fuentes_locales_sha256.json y ORIGINAL_MANIFEST_SHA256.json documentan la carpeta
original de analisis, con sus nombres y rutas originales. No son manifiestos de
esta publicacion; publication_manifest.json en ../final identifica las copias.
Los pesos, fuentes congeladas y ZIP originales se verificaron sin modificarlos.
