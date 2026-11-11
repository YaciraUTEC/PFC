"""Readable Spanish report and descriptive diagnostics for the frozen campaign."""
from collections import Counter
import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .env import LEVELS,STRESS
from .models import ARCHS
from .train import SEEDS

NAMES=dict(lstm='LSTM',mamba='Mamba',mamba2='Mamba-2',mamba3='Mamba-3 (SISO)',gtrxl='GTrXL',ltc='LTC',
    legacy='PID original',robust='PID robusto previo',updated_round3='DSLPID ronda 3',updated='DSLPID actualizado')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def finite_mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(values)) if values else float('nan')
def diagnostics(out):
    intervention=[];guard=[];estimation=[];biased=[];failures=[]
    for condition in ['exact','estimated']:
        for arch in ARCHS:
            rows=[r for seed in SEEDS for r in read(out/'test'/f'{arch}_{seed}_best_{condition}.json') if r['level'] in LEVELS]
            intervention.append(dict(architecture=arch,condition=condition,model_cases=len(rows),
                **{key:finite_mean(rows,key) for key in ['requested_gate','mean_gate','gate_prefault','gate_postfault',
                    'guard_off_fraction','effort','gate_variation','tracking_rmse','saturation']}))
            for outcome,count in Counter(r['outcome'] for r in rows).items():
                failures.append(dict(architecture=arch,condition=condition,outcome=outcome,count=count))
            protected=read(out/'test'/f'{arch}_71_best_{condition}.json')
            protected={(r['level'],r['seed']):r for r in protected if r['level'] in LEVELS}
            unguarded=read(out/'test'/f'{arch}_71_no_guard_{condition}.json')
            other={(r['level'],r['seed']):r for r in unguarded}
            assert set(protected)==set(other)
            guard.append(dict(architecture=arch,condition=condition,training_seed=71,model_cases=len(other),
                guarded_success_percent=100*np.mean([r['stable_landing'] for r in protected.values()]),
                unguarded_success_percent=100*np.mean([r['stable_landing'] for r in other.values()]),
                paired_gain= sum(protected[k]['stable_landing'] and not other[k]['stable_landing'] for k in other),
                paired_loss= sum(other[k]['stable_landing'] and not protected[k]['stable_landing'] for k in other)))
    for condition in ['exact','estimated']:
        for controller in ['legacy','robust','updated_round3','updated']:
            rows=[r for r in read(out/'test'/f'baseline_{controller}_{condition}.json') if r['level'] in LEVELS]
            for outcome,count in Counter(r['outcome'] for r in rows).items():
                failures.append(dict(architecture=controller,condition=condition,outcome=outcome,count=count))
    for condition in ['exact','estimated','biased']:
        rr=read(out/'test'/f'baseline_updated_{condition}.json')
        assert len(rr)==30*len(LEVELS+STRESS)
        for level in LEVELS+STRESS:
            rows=[r for r in rr if r['level']==level]
            biased.append(dict(condition=condition,level=level,success_percent=100*finite_mean(rows,'stable_landing'),
                tracking_rmse=finite_mean(rows,'tracking_rmse'),saturation=finite_mean(rows,'saturation'),
                eta_mae=finite_mean(rows,'eta_mae'),failed_motor_eta_mae=finite_mean(rows,'failed_motor_eta_mae')))
            if condition=='estimated':
                recovered=[r['first_estimate_within_2pct_seconds'] for r in rows if r['first_estimate_within_2pct_seconds'] is not None]
                estimation.append(dict(level=level,cases=len(rows),eta_mae=finite_mean(rows,'eta_mae'),
                    affected_motor_postfault_eta_mae=finite_mean(rows,'failed_motor_eta_mae'),
                    first_entry_below_2pct_cases=len(recovered),
                    first_entry_median_seconds=float(np.median(recovered)) if recovered else float('nan')))
    for name,records in [('intervention_metrics',intervention),('guard_ablation',guard),('estimator_metrics',estimation),
                         ('effectiveness_ablation',biased),('outcome_counts',failures)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    return intervention,guard,estimation

def stage_timings(out):
    result=dict(data_seconds=read(out/'data_complete.json')['seconds'],
        mild_selection_seconds=read(out/'selection_safety_complete.json')['seconds'],
        training_campaign_wall_seconds=read(out/'training_campaign_complete.json')['seconds'],
        joint_evaluation_active_seconds=read(out/'test_complete.json')['seconds'])
    evaluation=read(out/'test_complete.json')
    if 'calendar_seconds' in evaluation:
        result['joint_evaluation_calendar_seconds']=evaluation['calendar_seconds']
        result['evaluation_interruption_gap_seconds']=evaluation['inactive_gap_seconds']
    first=out/'hpo'/'lstm_0';r=read(first/'result.json')
    result['hpo_seconds']=max(p.stat().st_mtime for p in (out/'hpo').glob('*_selection.json'))-(first/'bc.pt').stat().st_mtime+r['timing']['seconds']
    return result

def render(out,df,historical,differences,base_differences,final_differences,timings,latency):
    intervention,guard,estimation=diagnostics(out)
    stage=stage_timings(out)
    (out/'stage_timings.json').write_text(json.dumps(stage,indent=2),encoding='utf-8')
    lines=[]
    def add(text=''):lines.append(text)
    add('RONDA 4: ATERRIZAJE CONTROLADO, EFECTIVIDAD Y SEIS ARQUITECTURAS')
    add('Generado en Lima: '+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5))).isoformat())
    add()
    add('CONCLUSIÓN PRINCIPAL DEL ENSAYO')
    observed={arch:float(np.mean([r['delta_pp'] for r in base_differences if r['architecture']==arch])) for arch in ARCHS}
    descriptive_best=max(observed,key=observed.get)
    if all(r['delta_pp']<=0 for r in base_differences):
        add('Ninguna arquitectura neuronal mejoró el promedio observado del DSLPID actualizado en los niveles primarios de 20%, 25%, 28% y 29%, ni con efectividad exacta ni estimada. Para esta configuración y distribución conservaría DSLPID con estimación RLS como referencia práctica.')
    add('El mejor promedio neuronal observado, combinando ambas condiciones, corresponde a '+NAMES[descriptive_best]+'. Esta descripción de tasas no demuestra superioridad estadística de una arquitectura; los contrastes e intervalos simultáneos aparecen más adelante.')
    add()
    add('LECTURA DE LOS RESULTADOS')
    add('Éxito significa aterrizar, no solamente alcanzar un punto de la trayectoria. Las tablas principales muestran el modelo elegido exclusivamente con validación, que puede ser su propia clonación (BC) si el afinado no aportó una mejora.')
    add('En las tablas principales de arquitecturas nuevas, cada celda neuronal combina cinco semillas de entrenamiento y 30 geometrías nuevas comunes: 150 casos de modelos, pero 30 geometrías independientes. El controlador base tiene 30 vuelos por celda. Se incluyen todos los intentos, incluso choques, contacto temprano y tiempo agotado.')
    add('La evaluación aplica a todos el mismo plan de aterrizaje corregido. El perfil anterior acababa a 10 cm del suelo; se añade descenso de 0.18 m/s y corte de motores tras contacto en destino. También se publica la métrica estricta con el perfil original para separar ambos cambios.')
    add()
    add('CRITERIO DE ATERRIZAJE')
    add('Contacto real en un radio de 20 cm del destino, después de completar la ruta. En el primer contacto: |velocidad vertical| <= 0.25 m/s, velocidad horizontal <= 0.25 m/s, |roll| y |pitch| cada uno <= 12 grados y velocidad angular <= 1 rad/s.')
    add('Después se exige un segundo continuo en contacto, velocidad total <= 0.10 m/s, |roll| y |pitch| cada uno <= 10 grados, velocidad angular <= 0.5 rad/s y altura del centro <= 5.5 cm. Una caída rápida no se convierte en éxito por quedar inmóvil tras chocar: se conserva la velocidad previa al impacto a 240 Hz.')
    for condition,label in [('exact','EFECTIVIDAD EXACTA (ORÁCULO)'),('estimated','EFECTIVIDAD ESTIMADA EN LÍNEA')]:
        add();add(label)
        add('Porcentaje de aterrizajes estables. Columnas: '+', '.join(f'{100*l:g}%' for l in LEVELS+STRESS)+' de pérdida de RPM en un motor.')
        for arch in ['legacy','robust','updated_round3','updated']+ARCHS:
            tab=df[(df.architecture==arch)&(df.condition==condition)&df['mode'].isin(['base','protected_selected'])].sort_values('level')
            add(f'{NAMES[arch]:24s} '+ ' | '.join(f'{v:5.1f}' for v in tab.success_percent))
    add();add('QUÉ APORTÓ EL AFINADO PROTEGIDO')
    add('Comparaciones pareadas promediadas en 20%, 25%, 28% y 29% de pérdida de RPM. Se cruzan remuestreos de cinco semillas y 30 geometrías, manteniendo las severidades emparejadas. Los intervalos simultáneos controlan la familia de 24 contrastes primarios: seis arquitecturas x dos condiciones x dos referencias (BC y DSLPID). Los intervalos son evidencia experimental en estas rutas; no certifican seguridad.')
    for condition in ['exact','estimated']:
        add(condition+':')
        for arch in ARCHS:
            bc=next(r for r in differences if r['architecture']==arch and r['condition']==condition)
            base=next(r for r in base_differences if r['architecture']==arch and r['condition']==condition)
            fmt=lambda r:f"{r['delta_pp']:+.2f} pp [{r['ci_family24'][0]:+.2f}, {r['ci_family24'][1]:+.2f}]"
            add(f"  {NAMES[arch]}: frente a su BC {fmt(bc)}; frente a DSLPID {fmt(base)}")
    add('Conclusión de los contrastes primarios:')
    for condition in ['exact','estimated']:
        above_base=[NAMES[r['architecture']] for r in base_differences if r['condition']==condition and r['ci_family24'][0]>0]
        above_bc=[NAMES[r['architecture']] for r in differences if r['condition']==condition and r['ci_family24'][0]>0]
        add('  '+condition+': superioridad frente a DSLPID con intervalo simultáneo enteramente positivo: '+(', '.join(above_base) if above_base else 'ninguna arquitectura demostrada en esta muestra')+'.')
        add('  '+condition+': mejora del afinado frente a su BC con ese mismo criterio: '+(', '.join(above_bc) if above_bc else 'ninguna arquitectura demostrada en esta muestra')+'.')
    add('La ausencia de superioridad demostrada no prueba equivalencia. Las tasas y diferencias puntuales describen estos vuelos; las pruebas cercanas al límite se presentan además por separado como diagnóstico.')
    add('Pasos de PPO del modelo seleccionado, en orden de semillas 71, 72, 73, 74, 75; cero significa que se conserva BC:')
    for arch in ARCHS:add(f"  {NAMES[arch]}: "+', '.join(str(t['selected_steps']) for t in timings if t['architecture']==arch))
    safety=read(out/'selection_safety_complete.json')
    add(f"La comprobación prospectiva de validación en 0%, 2% y 5%, ambas condiciones, devolvió {safety['rollbacks']} modelos a BC por perder al menos una ruta que su clon sí resolvía. Esta decisión precedió a la prueba final. Ver selection_safety/.")
    add('Diagnóstico exploratorio del PPO al agotar los 524288 pasos, frente a BC (IC 95% sin ajuste múltiple):')
    for r in final_differences:add(f"  {NAMES[r['architecture']]} / {r['condition']}: {r['delta_pp']:+.2f} pp [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    add();add('CUÁNDO Y CUÁNTO INTERVIENE')
    add('La red aprende dirección e intensidad continua a partir de ramas reales de simulación con intensidades 0, 0.25, 0.5 y 1 durante 24 ticks. El maestro tiene efectividad exacta; el estudiante no. Se exige mejora de coste físico >= max(0.02, 3% del coste base) y se penalizan esfuerzo y cambios de intervención.')
    add('Se añade una protección explícita por errores, velocidades, postura, asignación e incertidumbre. Esta regla apaga el residual cuando la base controla bien y es distinta de la compuerta aprendida. La tabla promedia episodios de los nueve niveles principales; una compuerta alta no implica una corrección grande si su dirección tiene poca amplitud.')
    for r in intervention:add(f"  {NAMES[r['architecture']]} / {r['condition']}: compuerta solicitada={r['requested_gate']:.3f}, aplicada={r['mean_gate']:.3f}, antes/después de falla={r['gate_prefault']:.3f}/{r['gate_postfault']:.3f}, protección apagada={100*r['guard_off_fraction']:.1f}%, esfuerzo={r['effort']:.5f}")
    add('Ablación exploratoria de la protección, solamente semilla 71, mismos nueve niveles y rutas (no demuestra un efecto causal de todas las semillas):')
    for r in guard:add(f"  {NAMES[r['architecture']]} / {r['condition']}: con protección={r['guarded_success_percent']:.1f}%, sin ella={r['unguarded_success_percent']:.1f}%; vuelos recuperados/perdidos={r['paired_gain']}/{r['paired_loss']}")
    add();add('ESTIMACIÓN DE EFECTIVIDAD Y DSLPID')
    add('RLS multivariable estima beta = eta_RPM² usando solo movimiento observado, comandos y el modelo nominal de masa, inercia y geometría. No recibe motor afectado, severidad ni tiempo de falla. Las diferencias de velocidad y giro incluyen ruido 0.002. La incertidumbre se entrega a la política.')
    add('Errores del motor afectado después de la falla, en vuelos del DSLPID actualizado. Se promedian episodios, incluyendo transitorios. La primera entrada bajo 2 puntos porcentuales no implica convergencia permanente:')
    for r in estimation:
        if .10<=r['level']<=.29:add(f"  {100*r['level']:g}% RPM: MAE={100*r['affected_motor_postfault_eta_mae']:.2f} pp; primera entrada <2 pp en {r['first_entry_below_2pct_cases']}/30 vuelos, mediana={r['first_entry_median_seconds']:.3f} s")
    add('effectiveness_ablation.csv incluye también la subestimación artificial antigua, separada del RLS real. No se trata de otro entrenamiento.')
    add('DSLPID conserva asignación física limitada, efectividad y saturación. Se parametriza la reserva de empuje: la anterior del 5% anulaba la autoridad horizontal cerca del techo. Esta ronda usa explícitamente 0.5% y umbral de emergencia 1.0001 mg, sin aumentar 21700 RPM. Los valores públicos por defecto 5% y 1.02 mg se preservan para reproducir resultados históricos.')
    add();add('IRL E IMITACIÓN')
    add('La combinación BC -> recompensa aprendida sin protección ya se evaluó en la ronda 3. Esta ronda no la repite como aporte nuevo: las seis arquitecturas nuevas usan recompensa física explícita y PPO protegido con retención de demostraciones, KL a su clon y crítico separado, calentado 8192 pasos. Por tanto, sus resultados no se atribuyen al IRL.')
    add('Se reevaluaron las recompensas históricas manual, CSIL, MEIRL y RIME, tanto selección por validación como presupuesto final, con la misma métrica estricta. La ronda 3 conserva cinco semillas por método (150 casos de modelos por celda); v2 conserva un modelo por arquitectura (30 casos por celda). Las tablas completas siguen a continuación; historical_rates.csv conserva también la proximidad al waypoint para distinguirla de aterrizar.')
    hist=pd.DataFrame(historical)
    for version in ['v2','round3']:
        for condition in ['exact','estimated']:
            add(f'{version} / {condition}; mismas 13 columnas de severidad:')
            sub=hist[(hist.version==version)&(hist.condition==condition)]
            for name in sub.model.unique():add(f"  {name:18s} "+' | '.join(f'{v:5.1f}' for v in sub[sub.model==name].sort_values('level').success_percent))
    add('Algunos timeouts tras contacto suave reflejan el criterio angular estricto y la dinámica de contacto simulada. En una reproducción del caso exacto 440012 al 20%, con motores ya apagados y contacto presente en todos los ticks, la velocidad máxima fue 0.0165 m/s y el ángulo máximo 0.142 grados, pero hubo picos de giro de 0.658 rad/s, por encima de 0.5. Se conserva como fallo; no se relaja el criterio después de ver test. Ver contact_stability_diagnostic.json. Este comportamiento limita la interpretación de pequeñas diferencias de tasas como mejoras de control en vuelo.')
    add('Los modelos v2 usaban residual de RPM, 24 entradas, ventana finita y motor 0 en entrenamiento. Ahora se exponen a cuatro motores y al aterrizaje corregido. Esta reevaluación mide compatibilidad y generalización; no aísla únicamente la arquitectura. La ronda 3 conserva 48 entradas y gobernador histórico.')
    add('Métrica estricta aplicada al perfil original, sin el nuevo descenso:')
    for condition in ['exact','estimated']:
        rr=read(out/'test'/f'metric_only_original_profile_{condition}.json')
        add('  '+condition+': '+', '.join(f'{100*l:g}% RPM -> {100*np.mean([r["stable_landing"] for r in rr if r["level"]==l]):.1f}% éxito' for l in [0.,.25,.29]))
    add();add('ARQUITECTURAS Y ENTRENAMIENTO')
    add('Todas conservan memoria durante el episodio y la reinician al terminar. PPO usa secuencias contiguas, estado inicial propio, máscaras y burn-in. LSTM: estado h/c e inicialización ortogonal. Mamba: caché convolucional y SSM. Mamba-2: estados y cabezas ajustados. Mamba-3: SISO con fase, integración trapezoidal y SSM persistentes; no se evaluó MIMO. GTrXL: dos capas, compuertas, memoria y calentamiento. LTC: puro, dt físico 1/48 s, subpasos ODE y positividad implícita, sin híbrido LSTM.')
    add('Se probaron dos configuraciones por arquitectura con igual presupuesto: 600 actualizaciones BC y 65536 pasos PPO por candidata. Las configuraciones seleccionadas solo con validación son:')
    relevant=dict(lstm=['width','layers','lr'],mamba=['width','layers','state','lr'],
        mamba2=['width','layers','state','head','lr'],mamba3=['width','layers','state','head','lr'],
        gtrxl=['width','layers','memory','gate_bias','lr'],ltc=['width','ode','lr'])
    for arch in ARCHS:
        cfg=read(out/'hpo'/f'{arch}_selection.json')['config']
        add(f"  {NAMES[arch]}: "+json.dumps({key:cfg[key] for key in relevant[arch]},ensure_ascii=False))
    add('Entrenamiento final: cinco semillas por arquitectura, 1200 actualizaciones BC + 524288 pasos PPO cada una; 15 728 640 pasos PPO en total. Learning rate por arquitectura, caída coseno hasta el 10%, límite KL, retención BC de 1 a 0.3 y crítico separado. Se igualaron pasos, no parámetros ni tiempo de cómputo.')
    add();add('LÍMITES Y PRECISIÓN')
    add('El límite estático ideal de vuelo nivelado sin giro es aproximadamente 29.7186% de pérdida de RPM en un motor. Se ensayan además 29.5%, 30%, 35% y 50%. Estar debajo del límite no garantiza maniobrar ni aterrizar. 25% de pérdida de RPM equivale a 43.75% de pérdida de empuje de ese motor; 29.5% RPM equivale a 50.30% de empuje perdido y 50% RPM equivale a 75% de empuje perdido.')
    add('Las fallas se activan a mitad del crucero y afectan un motor con reducción permanente. Falta probar instantes arbitrarios, pérdidas intermitentes, recuperación del motor y fallas múltiples; la compuerta aprendida se evalúa bajo esa distribución.')
    add('Esta ronda no implementa la recuperación giratoria Mueller-DAndrea ni demuestra vuelo normal con un motor apagado. Los resultados son simulados, con física nominal conocida y estado del simulador; falta validar sensores, batería, parámetros inciertos y hardware real.')
    add('El entrenamiento usa GPU float32. La referencia de inferencia usa CPU float64 y convierte la acción una sola vez a float32. Los pesos originales no se alteraron. Se detectó sensibilidad al redondeo/lote en Mamba-2 cerca del límite: el diagnóstico float64 reprodujo 96 casos de validación entre CPU/GPU y lotes 8/16 sin diferencias en desenlaces, pasos ni RMSE. Esta decisión precedió a test y no acredita robustez física.')
    add('Las latencias describen estas implementaciones portables en float64. No comparan los kernels oficiales fusionados ni demuestran que una familia de arquitecturas sea universalmente más rápida o mejor. La búsqueda de dos configuraciones por familia es limitada y no aísla el efecto de cada ajuste.')
    add('Los SHA de los 147 archivos de pesos congelados se verificaron antes y después de evaluar. La caché puede reutilizar vuelos entre actores exactamente idénticos, dejando trazabilidad .reuse.json; las copias no cuentan como nuevos vuelos físicos independientes.')
    add();add('TIEMPOS REALES POR ETAPA')
    for name,seconds in stage.items():add(f'  {name}: {seconds:.1f} s ({seconds/60:.1f} min, {seconds/3600:.2f} h)')
    add('El inicio de HPO se reconstruye a partir de la fecha del primer archivo BC y su duración registrada; los demás tiempos proceden de los cronómetros de campaña.')
    add('Los tiempos de pared de etapas paralelas no se obtienen sumando los de cada modelo. test_complete e historical_complete describen la misma evaluación conjunta.')
    evaluation_record=read(out/'test_complete.json')
    if evaluation_record.get('attempts',1)>1:
        add('La evaluación necesitó dos intentos por una interrupción de comunicación con un simulador. Se conservaron 310 archivos válidos y los 147 SHA de pesos antes de retomar solo los resultados faltantes. El tiempo activo aproximado suma ambos intentos; el tiempo calendario incluye además la pausa entre ellos. No se suman estas dos medidas entre sí. Los criterios, pesos y selección no cambiaron.')

    add('Detalle por semilla en segundos, con validación separada:')
    add(pd.DataFrame(timings).to_string(index=False))
    add('Latencia de una aeronave, entrada y comando a CPU; se calcula solo el actor y se excluyen estimación y asignación. La columna parameters cuenta el modelo completo, incluido el crítico separado:')
    add(pd.DataFrame(latency)[['architecture','device','parameters','p50_ms','p95_ms','p99_ms']].to_string(index=False))
    add();add('VERIFICACIÓN DEL CÓDIGO Y DEL IMPACTO')
    add('Pasaron 108 pruebas del código. Además, tres caídas físicas verificaron la integración del detector: el contacto suave pudo completar un segundo estable y dos impactos rápidos fueron rechazados.')
    physical=read(out/'physical_landing_contract.json')
    for r in physical:add(f"  {r['label']}: velocidad previa={r['first_contact']['vertical_speed']:.3f} m/s, posterior={r['post_contact_vertical_speed']:.3f} m/s, éxito={r['stable_landing']}")
    add('Estos ensayos verifican el criterio, no añaden rutas a las tasas de éxito del estudio. Ver physical_landing_contract.json y results_integrity.json.')
    add();add('ARTEFACTOS PARA AUDITAR')
    add('success_rates.csv; historical_rates.csv; paired_changes.json; residual_vs_base.json; full_budget_changes.json; intervention_metrics.csv; guard_ablation.csv; estimator_metrics.csv; effectiveness_ablation.csv; outcome_counts.csv; training_timings.csv; stage_timings.json; comparacion_arquitecturas.png; limite_fisico.png; afinamiento_vs_imitacion.png; test/; historical/.')
    add();add('FUENTES PRIMARIAS')
    lines.extend(['https://research.tudelft.nl/en/publications/control-of-unknown-quadrotors-from-a-single-throw/',
        'https://proceedings.mlr.press/v119/parisotto20a.html','https://github.com/state-spaces/mamba',
        'https://github.com/mlech26l/ncps','https://arxiv.org/abs/1709.10087'])
    return '\n'.join(lines)+'\n'
