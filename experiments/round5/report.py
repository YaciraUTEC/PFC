"""Final Spanish report generated only from the frozen completed campaign."""
import csv,json,datetime
import numpy as np
from .evaluate import OUT
from .inventory import groups,TEST_LEVELS,STRESS
from .plots import main as figures,NAMES
from .inventory import sha
from ..run import save

def read(name):return json.loads((OUT/name).read_text())
def percent(x):return f'{100*x:.2f}%'
def duration(seconds):
    if seconds is None:return 'no registrado'
    s=int(round(seconds));return f'{s//3600} h {(s%3600)//60:02d} min {s%60:02d} s'
def main():
    from .analysis import timings as refresh_timings
    refresh_timings()
    figures();decision=read('decision.json');timings=read('stage_timings.json');manifest=read('confirmation_manifest.json');runtime=read('runtime_selection_complete.json')
    with (OUT/'success_rates.csv').open(encoding='utf-8') as f:rates=list(csv.DictReader(f))
    with (OUT/'training_timings.csv').open(encoding='utf-8') as f:times=list(csv.DictReader(f))
    def rate(kind,name,mode,condition,level):return next(r for r in rates if r['kind']==kind and r['name']==name and r['mode']==mode and r['condition']==condition and abs(float(r['level'])-level)<1e-8)
    def successes(kind,name,mode,condition,levels):return np.mean([float(rate(kind,name,mode,condition,l)['success_percent']) for l in levels])
    lines=[]
    def add(text=''):lines.append(str(text))
    add('EVALUACIÓN FINAL RONDA 5 — CONTROL DE ATERRIZAJE CON FALLA DE UN MOTOR')
    add('Generado: '+datetime.datetime.now(datetime.timezone.utc).isoformat());add()
    add('RECOMENDACIÓN CONCRETA: '+decision['recommended'])
    add('Referencia estructurada aceptada: '+decision['accepted_structured_reference'])
    add('Configuración: '+json.dumps(decision['recommended_specification']['config'],ensure_ascii=False,sort_keys=True))
    if decision['accepted_learning_candidates']:add('Candidatos aprendidos que superaron todas las condiciones: '+', '.join(decision['accepted_learning_candidates']))
    else:add('Ningún candidato aprendido superó todas las condiciones de incorporación. La recomendación es el controlador estructurado indicado, sin compensación neuronal.')
    add('Esta decisión queda cerrada para el presupuesto y la distribución evaluados; no depende de completar otro entrenamiento. No equivale a una garantía de vuelo real ni demuestra inferioridad universal de una arquitectura.')
    add()
    add('1. CONFIRMACIÓN INDEPENDIENTE Y REGLA DE ADOPCIÓN')
    add('Jerarquía fijada antes de prueba: confirmar primero el estimador solo frente al controlador anterior;el perfil completo debe mejorar a esa referencia ya aceptada para incorporar descenso/frenado/reserva.Las redes deben superar finalmente a la referencia estructurada aceptada.Se cubren como máximo36extremos de las comparaciones necesarias,conservando la corrección simultánea predeclarada.')
    add('1000 geometrías nuevas por controlador; en cada una se probaron 28%,29% y29.5% estimados, un nivel leve preseleccionado uniformemente entre0..25% y un nivel con efectividad exacta. 5000 vuelos por controlador. Modelos y compuertas se eligieron solo con validación.')
    add('El controlador debe mejorar el resultado severo con límite inferior pareado>0, mantener el límite superior del daño condicional<1% en ambas mezclas protegidas, no perder éxito neto al duplicar ruido ni anticipar falla, y cumplir latencia p95≤20.83ms. Corrección para36 extremos; bootstrap primario100000remuestras por rutas; los intervalos de familias se calculan por separado.')
    for spec in manifest['specifications']:
        rows=read('confirmation/'+spec['id']+'.json');primary=[r for r in rows if r['role']=='primary']
        add(f"{spec['id']}: {sum(r['stable_landing'] for r in primary)}/{len(primary)} = {percent(np.mean([r['stable_landing'] for r in primary]))}")
        for level in [.28,.29,.295]:
            rr=[r for r in primary if r['level']==level];add(f"  {100*level:g}%: {sum(r['stable_landing'] for r in rr)}/{len(rr)} = {percent(np.mean([r['stable_landing'] for r in rr]))}")
    for label,c in decision['comparisons'].items():
        p=c['primary'];add(f"{label}: ganancia {100*p['mean']:+.2f}pp; intervalo [{100*p['lower']:+.2f},{100*p['upper']:+.2f}]pp; adopción={c['adoption_pass']}.")
        for name,risk in c['risk'].items():add(f"  {name}: {risk['losses']} daños/{risk['baseline_successes']} éxitos de referencia; límite superior {percent(risk['upper'])}.")
        add('  Sensibilidad: '+json.dumps(c['sensitivity_gain'])+f"; cumple={c['sensitivity_pass']}; latencia cumple={c['latency_pass']}.")
    add()
    add('RESULTADO DE ROBUSTEZ QUE DETERMINA LA RECOMENDACIÓN')
    add('Perfil | prueba | éxito en 0% | éxito en 25% | éxito en 29.5% | éxito en 29.6%')
    sensitivity_rows={}
    for spec in manifest['specifications']:
        for mode in ['noise2','clock4']:
            rr=read(f"sensitivity/production_{spec['id']}_{mode}.json");sensitivity_rows[(spec['id'],mode)]=rr
            counts=[]
            for level in [0.,.25,.295,.296]:
                subset=[r for r in rr if r['level']==level];counts.append(f"{sum(r['stable_landing'] for r in subset)}/{len(subset)}")
            add(spec['id']+' | '+mode+' | '+' | '.join(counts))
    add('noise2 duplica el ruido de velocidad y giro;clock4 introduce la falla a los4s. Son24rutas nuevas por nivel, compartidas entre controladores. Estas rutas son distintas de las1000de confirmación: comparar perfiles dentro de cada prueba;no interpretar los porcentajes de ambos conjuntos como un antes/después sobre los mismos vuelos.')
    key=('baseline_selected','noise2')
    if key in sensitivity_rows:
        severe=[r for r in sensitivity_rows[key] if r['level'] in [.295,.296]]
        expired=sum(r['outcome']=='timeout' for r in severe);pause=[r.get('descent_paused_seconds',0.) for r in severe]
        spin=[float(np.linalg.norm(r['terminal_angular_velocity'])) for r in severe]
        add(f"Perfil completo con ruido doble cerca del límite: {expired}/{len(severe)} agotaron el plazo de30s;descenso pausado en promedio{np.mean(pause):.2f}s (rango{min(pause):.2f}–{max(pause):.2f}s);velocidad angular terminal media{np.mean(spin):.2f}rad/s. La supervisión evita continuar el descenso mientras el vehículo sigue sin estar listo,pero en estos casos no consigue completar la tarea. No es un fallo de latencia ni una intervención neuronal: estas pruebas usan solo PID.")
    for label,c in decision['comparisons'].items():
        if c['statistical_pass'] and not c['sensitivity_pass']:
            add(label+': superó el criterio estadístico de ganancia y las mezclas protegidas;no supera la regla adicional de no perder éxito neto en sensibilidad,fijada antes de ver la prueba.')
    original=sensitivity_rows.get(('baseline_original','noise2'));observer=sensitivity_rows.get(('baseline_observer_only','noise2'))
    if original is not None and observer is not None:
        differences=[]
        for level in [.295,.296]:
            a=[r for r in original if r['level']==level];b=[r for r in observer if r['level']==level]
            differences.append(f"{100*level:g}%: {sum(r['stable_landing'] for r in b)-sum(r['stable_landing'] for r in a):+d} aterrizajes de {len(a)}")
        add('Estimador solo frente a referencia con ruido doble: '+', '.join(differences)+'. Esta muestra pequeña no prueba inferioridad estadística universal;la decisión aplica la regla de incorporación acordada sin cambiarla después de ver el resultado.')
    add('La recomendación general de esta evaluación es '+decision['recommended']+'. La ganancia nominal y el cumplimiento de robustez se informan por separado.')
    add()
    add('Reutilización exacta de simulaciones: se reproduce la red paso a paso sobre observaciones de la trayectoria base, con memoria persistente y CPUfloat64. Solo si ejecuta la acción canónica nula en todos los pasos se reutiliza ese vuelo del PID. Cualquier acción distinta obliga a simular de nuevo el caso completo con la red. No se acorta el horizonte ni se omiten casos. Equivalencia, casos mixtos, contexto de efectividad, integridad y concurrencia se verifican con las seis arquitecturas. No se atribuye una mejora neuronal a esos vuelos sin intervención.')
    add('La cota de daño se refiere a la mezcla predeclarada de severidades, condicionada a que la referencia aterrice bien. No certifica cada severidad o región de estados por separado. 100% observado significa todos los vuelos de esta muestra; no perfección fuera de ella.')
    add()
    add('2. TODAS LAS ARQUITECTURAS — CORRECCIÓN DIRECTA Y DE EFECTIVIDAD')
    add('Cada porcentaje por arquitectura/nivel combina5 semillas y60 rutas (300 vuelos); las mismas rutas se emparejan con el PID de60 vuelos. Las semillas no son300 geometrías independientes.')
    for kind,caption in [('residual','CORRECCIÓN DIRECTA DE MOTORES'),('observer','CORRECCIÓN DE EFECTIVIDAD')]:
        add(caption);add('Arquitectura | E25 raw | E28 raw | E29 raw | E29.5 raw | E29.5 elegida | Exacta raw(promedio0..29.5) | ON raw en29.5')
        for name in ['lstm','mamba','mamba2','mamba3','gtrxl','ltc']:
            v=[float(rate(kind,name,'raw','estimated',l)['success_percent']) for l in [.25,.28,.29,.295]]
            sel=float(rate(kind,name,'selected','estimated',.295)['success_percent']);exact=successes(kind,name,'raw','exact',TEST_LEVELS);on=float(rate(kind,name,'raw','estimated',.295)['applied_on_percent'])
            add(f"{NAMES[name]:13s} | "+' | '.join(f'{x:6.2f}%' for x in v+[sel,exact,on]))
        add()
    add('Familia | Ganancia severa raw sobre PID actualizado (pp) | Intervalo simultáneo entre15 familias')
    for r in read('family_simultaneous_intervals.json')['contrasts']:add(f"{r['group']:24s} | {100*r['mean']:+.2f} | [{100*r['lower']:+.2f},{100*r['upper']:+.2f}]")
    add('Configuraciones fijadas en validación; los campos de configuración comunes no utilizados por una arquitectura no se interpretan como mecanismos activos:')
    for profile in read('architecture_profiles.json'):
        add(f"{NAMES[profile['architecture']]}: {profile['total_parameters']} parámetros totales ({profile['policy_encoder_parameters']} del codificador); configuración={profile['configuration']}.")
    add('LSTM: memoria por episodio y sesgo de olvido1. Mamba: estado selectivo y caché convolucional persistentes. Mamba-2: estado por cabezas y escalas de tiempo sin decaimiento de pesos. Mamba-3 SISO: fase rotatoria y recurrencia trapezoidal persistentes. GTrXL: memoria causal deslizante y compuertas GRU. LTC: integración con dt real1/48, restricciones implícitas positivas y número explícito de subpasos. Se usa calentamiento recurrente y critic separado del gradiente del actor en todas.')
    add('Se usan implementaciones portables: Mamba-3 está limitado a SISO. La comparación no mide todos los mecanismos ni los núcleos acelerados oficiales de cada publicación. El presupuesto de ajuste por familia es pequeño y uniforme; no demuestra su óptimo global.')
    add()
    add('3. IMITACIÓN INICIAL, AFINADO PPO Y RECOMPENSAS APRENDIDAS')
    add('BC significa imitación inicial con datos agregados de estados visitados por alumnos. Todos los métodos de recompensa parten exactamente del mismo BC-LSTM de su semilla; el PPO manual utiliza la recompensa diseñada del aterrizaje. CSIL es imitación regularizada; MEIRL usa un conjunto de modelos de dinámica; RIME aprende preferencias automáticas de segmentos. Son adaptaciones locales, no reproducciones completas de los resultados publicados ni tres algoritmos IRL equivalentes.')
    add('Modelo | E28 | E29 | E29.5 | Exacta media')
    comparisons=[('residual','lstm','bc','LSTM solo BC'),('residual','lstm','raw','LSTM BC+PPO manual')]+[('reward',n,'raw',n.upper()) for n in ['csil','meirl','rime']]
    for k,n,m,label in comparisons:
        v=[successes(k,n,m,'estimated',[l]) for l in [.28,.29,.295]]+[successes(k,n,m,'exact',TEST_LEVELS)]
        add(label+' | '+' | '.join(f'{x:.2f}%' for x in v))
    add('Todas las arquitecturas residuales en29.5% estimado | solo BC | mejor PPO reservado | final PPO | semillas con mejor punto=BC inicial')
    for name in ['lstm','mamba','mamba2','mamba3','gtrxl','ltc']:
        v=[successes('residual',name,mode,'estimated',[.295]) for mode in ['bc','raw','final']]
        rr=[r for r in times if r['kind']=='residual' and r['name']==name and r['phase']=='ppo']
        unchanged=sum(int(float(r['selected_steps']))==0 for r in rr)
        add(NAMES[name]+' | '+' | '.join(f'{x:.2f}%' for x in v)+f' | {unchanged}/5')
    add('Para las seis redes se publican además los puntos finales del presupuesto y los mejores puntos intermedios elegidos en validación; no se selecciona retrospectivamente un checkpoint por su curva de prueba.')
    add()
    add('4. CUÁNDO INTERVENIR: COMPUERTA, PROPUESTA Y VETO')
    add('La acción tiene dirección, magnitud y decisión binaria explícita. Al apagarse se ejecuta exactamente el PID; los parámetros continuos inactivos no reciben gradientes de probabilidad espurios. El maestro evalúa una ráfaga de12 pasos y después el resto de la misión con PID, exigiendo beneficio mínimo absoluto y relativo. La compuerta se entrena con clases equilibradas porque los beneficios son raros. La propuesta se conserva aunque la compuerta interna la rechace, para permitir que un juez independiente decida.')
    add('Opciones comparadas antes de prueba: compuerta aprendida, umbral mayor, veto conservador, propuesta autorizada por juez y abstención completa. El juez usa información medida; su calibración empírica no es una prueba formal de seguridad.')
    for r in runtime['groups']:
        strategy=r['chosen_strategy'];add(f"{r['group']['id']}: {strategy['id']}; pérdidas en pantalla protegida={r['full_screen_losses']}; "+('solo PID, no se atribuye éxito a la red' if strategy['off'] else 'se permite compensación'))
    proposals=read('paired_proposal_diagnostics.json')
    add(f"Datos de entrenamiento/calibración: {proposals['unique_query_modes']} consultas-modo; {proposals['baseline_failure_queries']} continuaciones base fallidas y {proposals['queries_with_rescue_proposal']} con alguna propuesta capaz de rescatar el aterrizaje. No son vuelos o geometrías independientes ni resultados de prueba final.")
    for r in proposals['rows']:
        if r['split']=='validation' and r['source']=='student':add(f"Propuestas reservadas {r['student_architecture']}/{r['mode']}: {r['proposals']}, beneficios={r['beneficial']}, ganancias negativas={r['negative_gain']}, rescates binarios={r['binary_rescues']}, pérdidas binarias={r['binary_losses']}.")
    current=read('oracle_proposal_audit_complete.json')
    add('Diagnóstico de propuestas de los75modelos ya entrenados:4rutas reservadas por semilla,29.5%estimado;consulta enarranque,1segundo,trasfalla,iniciodedescensoyaproximación.Se compara su propia propuestaON12ticks con cero,y luego PID hasta terminar.No es el controlador desplegado ni un techo de todo control posible.')
    for r in current['groups']:
        add(f"{r['group']}: consultas={r['queries']}, propuestas beneficiosas={r['beneficial']}, propuestas que rescatan={r['binary_rescue_proposals']}, propuestas que perjudican={r['binary_loss_proposals']}; TP={r.get('true_positive',0)},FP={r.get('false_positive',0)},FN={r.get('false_negative',0)},TN={r.get('true_negative',0)}; precisiónON={r['precision_when_on']};recall={r['recall_benefits']}.")
    add('Juez: '+json.dumps(read('judge_calibration.json'),ensure_ascii=False))
    add()
    add('5. ABLACIONES Y CAUSAS OBSERVADAS')
    add('Variante | E28 | E29 | E29.5 | E29.6 | E30 | E35 | E50')
    labels=['original','selected','balanced_governor','ready_only','observer_only','yaw_only','reserve_only','without_observer','without_ready','without_yaw','without_reserve']
    for name in labels:add(name+' | '+' | '.join(f"{successes('baseline',name,'baseline','estimated',[l]):.2f}%" for l in [.28,.29,.295]+STRESS))
    add('Estimación en29.5% estimado | MAE eta del filtro | MAE eta aplicada | cobertura nominal95% del filtro | fracción con primer error<2% | tiempo condicional(segundos)')
    for kind,name,mode,label in [('baseline','original','baseline','DSLPID anterior'),('baseline','selected','baseline','DSLPID actualizado')]+[('observer',a,'raw',NAMES[a]) for a in ['lstm','mamba','mamba2','mamba3','gtrxl','ltc']]:
        r=rate(kind,name,mode,'estimated',.295)
        add(label+' | '+' | '.join(r[k] or 'no disponible' for k in ['failed_motor_eta_mae','applied_failed_motor_eta_mae','estimated_95pct_coverage','estimate_within_2pct_fraction','conditional_estimation_recovery_seconds']))
    add('success_rates.csv incluye también MAE de los cuatro motores, amplitud antes/después de la falla, alarmas del filtro antes de la falla, fracción de acciones que modificaron realmente los motores y esfuerzo en velocidades cuadradas normalizadas. Activar la compuerta no equivale necesariamente a cambiar motores cuando la asignación satura.')
    add('La cobertura se mide frente a la efectividad verdadera solo como métrica del filtro base; no es una banda calibrada para la corrección neuronal. El tiempo de recuperación se condiciona a haber alcanzado el error<2%; se informa también su fracción para no ocultar casos censurados.')
    add('ready=supervisión de descenso; observer=estimación adaptativa; yaw=frenado angular en emergencia; reserve=conservación de margen de empuje. Las ablaciones individuales y de retirada permiten observar interacciones, sin asumir efectos aditivos.')
    failures=read('failure_analysis.json')
    for r in failures:
        if r['mode']=='raw' and r['condition']=='estimated' and r['level']==.295:
            add(f"{r['kind']}_{r['name']}: pérdidas pareadas={r['paired_losses']}; violación angular={r['angular_violations']}, vertical={r['vertical_violations']}, horizontal={r['horizontal_violations']}, radio={r['radius_violations']}, inclinación={r['tilt_violations']}; pérdidas dominadas por yaw={r['yaw_dominant_losses']}; compuerta OFF último segundo={r['off_during_last_precontact_second']}; desenlaces={r['outcomes']}")
    add('Las infracciones se superponen. Compensación OFF antes del contacto no prueba por sí sola un defecto de activación: también puede faltar autoridad o la propuesta puede empeorar el vuelo. Las propuestas pareadas y las ablaciones aportan la evidencia adicional; no se infiere causalidad solo de una traza.')
    add()
    add('6. LÍMITES Y DEFINICIÓN DE ÉXITO')
    add('Falla expresada como pérdida de RPM de un motor:29.5%RPM implica50.30% de pérdida de empuje de ese motor porque empuje∝RPM². El umbral estático aproximado29.7186% del modelo corresponde a vuelo estacionario normal con equilibrio de momentos; no es un límite universal de descenso transitorio. Se reportan29.6,30,35 y50% aunque fallen. No se cambia la tarea por vuelo giratorio; recuperar con un motor apagado exigiría otro régimen y otro controlador, no más entrenamiento de este mismo objetivo.')
    physical=read('physical_limits.json');near=next(r for r in physical['levels'] if r['rpm_loss']==.295)
    add(f"Comprobación con parámetros reales del modelo: masa={physical['mass_kg']:.4f}kg; hover={physical['hover_rpm']:.2f}RPM; máximo={physical['max_rpm']:.0f}RPM; límite estático={100*physical['static_normal_hover_rpm_loss_limit']:.5f}%. En29.5%, la relación empuje equilibrado máximo/peso es{near['balanced_max_thrust_to_weight']:.6f}: margen vertical ideal{100*(near['balanced_max_thrust_to_weight']-1):.3f}%, e inclinación máxima estática ideal{near['ideal_static_hover_tilt_degrees']:.3f}°. No incluye margen dinámico ni error de estimación.")
    add('Antes del primer impacto: velocidad vertical y horizontal≤0.25m/s, inclinación≤12°, velocidad angular≤1rad/s y radio≤0.20m. Después:1segundo de contacto estable, velocidad≤0.10m/s, inclinación≤10°, velocidad angular≤0.5rad/s y altura≤0.055m. Llegar al punto o tocar el suelo no basta. Límite de30s es fallo terminal.')
    add('Todos usan la misma medición ruidosa de velocidad y giro; posición y orientación se consideran medidas exactas. Ruido gaussiano nominal0.002, prueba secundaria al doble. Un solo motor afectado. Toda conclusión es para este simulador, estos sensores, rutas y distribución de falla.')
    add('La precisión de contacto se fijó en960Hz (vuelo240Hz, control48Hz),200iteraciones y residuo1e-12. Se contrastó contra1920Hz/400iteraciones, restauración de estados y replay CPU/GPU. Los cambios numéricos se aplican también a la referencia: no se atribuye su efecto a una red ni se compara directamente el porcentaje nuevo con el histórico de otra precisión/sensor.')
    add()
    add('7. TIEMPOS Y PRESUPUESTO REAL')
    for name,seconds in timings['stages'].items():add(name+': '+duration(seconds))
    add('El primer ensayo corto de búsqueda se interrumpió y archivó al ampliar el horizonte GAE; no se usó para seleccionar modelos. Los24ensayos posteriores parten de inicialización idéntica dentro de cada arquitectura. Los recibos originales de datos/juez se conservaron sin volver a entrenarlos. Los tiempos por etapa pueden solaparse; el calendario desde el inicio de datos comunes se informa por separado y no incluye la implementación previa.')
    add('Duración individual por método/fase: media y rango entre semillas; incluye simulación y validación según la fase. La suma de procesos no es tiempo de calendario porque hubo dos trabajadores simultáneos.')
    for kind in ['residual','observer','reward']:
        names=sorted({r['name'] for r in times if r['kind']==kind})
        for name in names:
            for phase in ['bc','ppo']:
                rr=[r for r in times if r['kind']==kind and r['name']==name and r['phase']==phase]
                if not rr:continue
                v=[float(r['seconds']) for r in rr];validation=[float(r['validation_seconds'] or 0) for r in rr]
                add(f"{kind}/{name}/{phase}: n={len(rr)}, media {duration(np.mean(v))}, rango {duration(min(v))}–{duration(max(v))}; validación media {duration(np.mean(validation))}; aprendizaje/carga de recompensa media {duration(np.mean([float(r['reward_seconds'] or 0) for r in rr]))}; preparación media {duration(np.mean([float(r['setup_seconds'] or 0) for r in rr]))}.")
    add('Los15 modelos de recompensas reutilizan el BC del LSTM correspondiente; no se suma como un nuevo entrenamiento supervisado. Se entrenaron30 residuales+30 correctores de efectividad+15 métodos de recompensa. Cada PPO principal completó524288pasos; BC principal1200actualizaciones;24búsquedas cortas y12pilotos se reportan aparte.')
    for row in read('latency_complete.json')['results']:add(f"{row['id']}: control completo p95={1000*row['p95_seconds']:.3f}ms, red p95={1000*row['network_p95_seconds']:.3f}ms; cumple48Hz={row['realtime_pass']}.")
    add()
    add('8. REPRODUCCIÓN Y ARCHIVOS')
    add('Recomendación ejecutable: recommended_controller.json; evaluación: decision.json; curvas completas: success_rates.csv; costos: training_timings.csv y stage_timings.json; integridad: final_audit.json.')
    add('Desde la raíz PFC, activar experiments/local_env.ps1 y ejecutar el Python de conda_env con -m experiments.round5.deployment. Reconstruye el PID, estimador, supervisión y política exactos del paquete recomendado. La política conserva memoria por episodio y se reinicia al terminar.')
    add('El controlador conserva0..21700RPM, considera efectividad estimada, asignación saturada y antiwindup. La rama heredada conserva su valor predeterminado para compatibilidad. El artefacto recomendado identifica explícitamente las opciones nuevas.')
    add()
    add('9. EVIDENCIA TEÓRICA UTILIZADA')
    for title,url in [('DAgger','https://proceedings.mlr.press/v15/ross11a.html'),('CSIL','https://arxiv.org/html/2305.16498v3'),('MEIRL','https://proceedings.mlr.press/v331/zhan26a.html'),('RIME','https://proceedings.mlr.press/v235/cheng24k.html'),('Time limits in RL','https://proceedings.mlr.press/v80/pardo18a.html'),('Residual RL','https://arxiv.org/abs/1812.03201'),('SPIBB','https://proceedings.mlr.press/v97/laroche19a.html')]:add(title+': '+url)
    target=OUT/'INFORME_FINAL_RONDA5.txt';target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figure_names=['curvas_residuales.png','curvas_estimadores.png','curvas_recompensas.png','efectividad_exacta_y_estimada.png','intervalos_familias.png','confirmacion_independiente.png','ablaciones_y_limites.png']
    save(OUT/'report_complete.json',dict(report=str(target),report_sha256=sha(target),figures=[dict(path=str(OUT/n),sha256=sha(OUT/n)) for n in figure_names],presentation_sources={n:sha(__import__('pathlib').Path(__file__).with_name(n)) for n in ['report.py','plots.py']}))
    print('FINAL_REPORT',str(target),decision['recommended'],flush=True)
if __name__=='__main__':main()
