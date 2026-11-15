"""Final auditable scientific tables, paired conclusions and publication figures."""
import csv,json,time,math,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import OUT,LEVELS,SEEDS,save
from .models import ARCHS
from .select import SCHEDULES,model_entries
from ..round5.statistics import crossed_simultaneous

def read(path):return json.loads(Path(path).read_text())
def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def split_name(path):
    label,condition,noise=path.stem.rsplit('_',2);return label,condition,float(noise[5:])
def groups():
    out=defaultdict(list)
    for p in sorted((OUT/'test').glob('*.json')):
        data=read(p);label,condition,noise=split_name(p)
        e=data.get('entry')
        family=f"{e['kind']}_{e['arch']}_{e['schedule']}" if e else label
        for r in data['episodes']:out[(family,condition,noise,r['level'])].append(dict(r,training_seed=None if e is None else e['seed']))
    return out

def rates(group):
    rows=[]
    for (family,condition,noise,level),rr in sorted(group.items()):
        seeds=sorted({r['training_seed'] for r in rr if r['training_seed'] is not None})
        sr=[100*np.mean([r['stable_landing'] for r in rr if r['training_seed']==s]) for s in seeds]
        rows.append(dict(family=family,condition=condition,noise_multiplier=noise,rpm_loss_percent=100*level,
            success_percent=100*float(np.mean([r['stable_landing'] for r in rr])),seed_sd_percent=float(np.std(sr,ddof=1)) if len(sr)>1 else 0.,
            model_cases=len(rr),route_geometries=len({r['seed'] for r in rr}),training_seeds=len(seeds),
            applied_on_percent=100*float(np.mean([r['applied_on_fraction'] for r in rr])),mean_amplitude=float(np.mean([r['mean_gate'] for r in rr])),
            certified_zero_cases=sum(r.get('zero_replay_certified',False) for r in rr),
            timeout_cases=sum(r['outcome']=='timeout' for r in rr),hard_landing_cases=sum(r['outcome']=='hard_landing' for r in rr),
            mean_solve_p95_ms=1000*float(np.mean([r['solve_seconds_p95'] for r in rr])),mean_pause_seconds=float(np.mean([r['descent_paused_seconds'] for r in rr]))))
    csvwrite(OUT/'success_rates.csv',rows);return rows

def primary_statistics(group):
    p2=read(OUT/'phase2_conclusion.json');names=[f'{method}_{arch}_{p2["schedule_by_arch"][arch]}' for arch in ARCHS for method in ['ppo','flashsac','fql']]
    reference={(c,n,l,r['seed']):r for (f,c,n,l),rr in group.items() if f=='controller_selected' for r in rr}
    matrices=[];protections=[]
    for family in names:
        keyed={(c,n,l,r['training_seed'],r['seed']):r for (f,c,n,l),rr in group.items() if f==family for r in rr}
        matrices.append([[[int(keyed[('estimated',n,l,s,route)]['stable_landing'])-int(reference[('estimated',n,l,route)]['stable_landing']) for n in [1.,2.] for l in [.295,.296]] for route in range(612000,612060)] for s in SEEDS])
        losses_mild=losses_exact=gains=total_losses=severe_estimated_losses=0
        for (c,n,l,s,route),r in keyed.items():
            ref=reference[(c,n,l,route)];loss=ref['stable_landing'] and not r['stable_landing'];gains+=int(r['stable_landing'] and not ref['stable_landing'])
            total_losses+=int(loss);severe_estimated_losses+=int(loss and c=='estimated' and l>=.28)
            losses_mild+=int(loss and l<=.25);losses_exact+=int(loss and c=='exact' and l<=.296)
        protections.append(dict(family=family,total_losses=total_losses,severe_estimated_losses=severe_estimated_losses,mild_losses=losses_mild,exact_losses=losses_exact,gains_all_levels=gains))
    simultaneous=crossed_simultaneous(np.asarray(matrices),replicates=10000,seed=619)
    for i,name in enumerate(names):simultaneous['contrasts'][i].update(family=name,**{k:v for k,v in protections[i].items() if k!='family'})
    save(OUT/'paired_statistics.json',simultaneous)
    return simultaneous,names

def confirmation_statistics():
    baseline={};candidate=defaultdict(dict)
    for p in (OUT/'confirmation').glob('*.json'):
        data=read(p);lab,c,n=split_name(p)
        if lab=='baseline':
            for r in data['episodes']:baseline[(c,n,r['level'],r['seed'])]=r
        elif data.get('entry'):
            e=data['entry']
            for r in data['episodes']:candidate[e['seed']][(c,n,r['level'],r['seed'])]=r
    diffs=[];mild_losses=exact_losses=total_losses=0
    for s in SEEDS:
        rr=candidate[s];dif=[]
        for route in range(613000,613200):dif.append([int(rr[('estimated',n,l,route)]['stable_landing'])-int(baseline[('estimated',n,l,route)]['stable_landing']) for n in [1.,2.] for l in [.295,.296]])
        diffs.append(dif)
        for key,r in rr.items():
            b=baseline[key];loss=b['stable_landing'] and not r['stable_landing'];total_losses+=int(loss);mild_losses+=int(loss and key[2]<=.25);exact_losses+=int(loss and key[0]=='exact')
    stats=crossed_simultaneous(np.asarray([diffs]),replicates=10000,seed=621)
    result=dict(**stats['contrasts'][0],training_seeds=5,routes=200,total_losses=total_losses,mild_losses=mild_losses,exact_losses=exact_losses)
    save(OUT/'confirmation_statistics.json',result);return result

def sensitivity_statistics():
    baseline={};grouped=defaultdict(list)
    for path in (OUT/'sensitivity').glob('baseline_clock*.json'):
        clock=float(path.stem.rsplit('clock',1)[1])
        for r in read(path)['episodes']:baseline[(clock,r['seed'],r['level'])]=r
    for path in (OUT/'sensitivity').glob('*.json'):
        d=read(path);entry=d.get('entry')
        if not entry:continue
        family=f"{entry['kind']}_{entry['arch']}_{entry['schedule']}"
        for r in d['episodes']:grouped[(family,float(d['clock']))].append(dict(r,training_seed=entry['seed']))
    rows=[]
    for (family,clock),rr in sorted(grouped.items()):
        gains=losses=base_success=success=0
        for r in rr:
            b=baseline[(clock,r['seed'],r['level'])];base_success+=int(b['stable_landing']);success+=int(r['stable_landing'])
            gains+=int(r['stable_landing'] and not b['stable_landing']);losses+=int(b['stable_landing'] and not r['stable_landing'])
        rows.append(dict(family=family,fault_clock_seconds=clock,cases=len(rr),routes=len({r['seed'] for r in rr}),training_seeds=len({r['training_seed'] for r in rr}),
            success_percent=100*success/len(rr),baseline_success_percent=100*base_success/len(rr),paired_difference_pp=100*(success-base_success)/len(rr),gains=gains,losses=losses))
    csvwrite(OUT/'fault_clock_sensitivity.csv',rows);save(OUT/'fault_clock_sensitivity.json',dict(rows=rows,scope='Estimated effectiveness,noise2,29.5/29.6%;same routes/5model seeds;descriptive paired diagnostic,not independent physical repetitions.'))
    return rows

def summarize_gate(rows,destination='gate_summary.csv'):
    grouped=defaultdict(list)
    for r in rows:grouped[r['family']].append(r)
    output=[]
    for family,rr in sorted(grouped.items()):
        n=sum(r['routes'] for r in rr)
        output.append(dict(family=family,cases=n,gated_success_percent=100*sum(r['gated_success'] for r in rr)/n,
            ungated_success_percent=100*sum(r['ungated_success'] for r in rr)/n,
            prevented_failures=sum(r['prevented_failures'] for r in rr),missed_benefits=sum(r['missed_benefits'] for r in rr)))
    csvwrite(OUT/destination,output);return output

def requested_noise_limits(decisions,steps,initial):
    last=0;upper=float(initial);mass=0.;maximum=None
    for d in decisions:
        when=int(d['steps'])
        if when>steps:break
        assert when>=last
        if when>last:
            mass+=(when-last)*upper;maximum=upper if maximum is None else max(maximum,upper)
        last=when;upper=float(d['upper'])
    if steps>last:
        mass+=(steps-last)*upper;maximum=upper if maximum is None else max(maximum,upper)
    return dict(maximum_during_steps=maximum,mean_requested_upper=mass/steps if steps else None,upper_after_last_decision=upper)

def curriculum_exposure(timings):
    timing_by_path={r['path']:r for r in timings};rows=[]
    for e in model_entries(2)+model_entries(3):
        folder=Path(e['path']).parent;d=read(folder/'complete.json');decisions=read(folder/'schedule.json')
        initial=2. if e['schedule']=='full' else 1.
        selected_steps=timing_by_path[str(folder)]['best_checkpoint_online_steps']
        trained=requested_noise_limits(decisions,d['steps'],initial);selected=requested_noise_limits(decisions,selected_steps,initial)
        rows.append(dict(kind=e['kind'],arch=e['arch'],seed=e['seed'],schedule=e['schedule'],training_steps=d['steps'],selected_online_steps=selected_steps,
            trained_max_requested_upper=trained['maximum_during_steps'],trained_mean_requested_upper=trained['mean_requested_upper'],upper_after_training=trained['upper_after_last_decision'],
            selected_max_requested_upper=selected['maximum_during_steps'],selected_mean_requested_upper=selected['mean_requested_upper']))
    csvwrite(OUT/'noise_exposure.csv',rows);return rows

def timing():
    rows=[]
    for entry in model_entries(2)+model_entries(3):
        p=Path(entry['path']).with_name('complete.json')
        d=read(p);best_metadata=torch.load(entry['path'],map_location='cpu',weights_only=False)['metadata']
        rows.append(dict(path=str(p.parent),phase='PPO' if 'phase2' in p.parts else d.get('method','offpolicy'),
            arch=entry['arch'],seed=d.get('seed'),schedule=d.get('schedule'),steps=d['steps'],best_checkpoint_online_steps=best_metadata.get('steps'),
            total_seconds=d['seconds'],validation_and_wait_seconds=d.get('validation_seconds'),data_seconds=d.get('data_seconds'),encode_seconds=d.get('encode_seconds'),
            head_initial_seconds=d.get('initial_seconds'),offline_seconds=d.get('offline_seconds'),online_with_validation_seconds=d.get('online_with_validation_seconds',d['seconds'])))
    csvwrite(OUT/'training_times.csv',rows)
    receipts={p.stem:read(p) for p in OUT.glob('*_receipt.json')};save(OUT/'stage_times.json',receipts)
    return rows,receipts

def readable_summary(table,recommendation,p1,p2,p3,times,receipts,latency,stats,confirm,shifts=None,gate_rows=None,exposure=None):
    """Readable Spanish decision record, alongside detailed machine-readable data."""
    bc=[]
    for arch in ARCHS:
        for seed in SEEDS:
            path=OUT/'bc'/arch/f'seed_{seed}'/'bc_timing.json'
            d=read(path);bc.append(dict(arch=arch,seed=seed,seconds=d['seconds'],updates=d['updates']))
    csvwrite(OUT/'cloning_times.csv',bc)
    pilots=[]
    for path in sorted((OUT/'hpo').rglob('complete.json')):
        d=read(path);pilots.append(dict(arch=d['arch'],method=d['method'],seconds=d['seconds'],steps=d['steps'],offline_updates=d['offline_updates']))
    csvwrite(OUT/'hyperparameter_times.csv',pilots)
    selected=p2['schedule_by_arch']
    names={'dslpid_reference':'DSLPID referencia','controller_selected':'DSLPID actualizado'}
    for arch in ARCHS:
        for method in ['ppo','flashsac','fql']:
            names[f'{method}_{arch}_{selected[arch]}']=f"{method.upper()} + {'Mamba-3' if arch=='mamba3' else 'GTrXL'}"
    for name in read(OUT/'nmpc_test_plan.json')['names']:
        gain=p1['all_configs'][name].get('nmpc_options',{}).get('indi_gain',1.)
        names[name]=('NMPC sin INDI' if gain==0 else 'NMPC + INDI')+(' rápido' if 'fast' in name else ' mayor horizonte')
    exposure_lines=['**Límites de ruido solicitados durante el entrenamiento**','',
        '| Arquitectura / schedule | Máximo solicitado entre semillas | Rango 2× antes del fin | Rango 2× antes del checkpoint elegido |',
        '|---|---:|---:|---:|']
    for arch in ARCHS:
        for sch in SCHEDULES:
            rr=[r for r in exposure or [] if r['kind']=='ppo' and r['arch']==arch and r['schedule']==sch]
            if not rr:continue
            maxima=[r['trained_max_requested_upper'] for r in rr]
            exposure_lines.append(f"| {arch} / {sch} | {min(maxima):.2f}–{max(maxima):.2f} | {sum(v>=2-1e-9 for v in maxima)}/{len(rr)} | {sum(r['selected_max_requested_upper'] is not None and r['selected_max_requested_upper']>=2-1e-9 for r in rr)}/{len(rr)} |")
    exposure_lines += ['', 'Se excluye una ampliación decidida justo después del último paso: no cuenta como exposición. El límite se aplica al reiniciar cada episodio; no es una medición del ruido de cada muestra. En PPO, un checkpoint anterior al afinado online conserva únicamente la clonación nominal. El detalle está en noise_exposure.csv.']
    exposure_text='\n'.join(exposure_lines)
    criterion_names=dict(runtime_48hz='tiempo de cálculo',primary_confirmed_gain='ganancia confirmada en el test',independent_confirmation_gain='ganancia en confirmación',
        protected_primary='conservar todos los éxitos de la base en el test',protected_confirmation='conservar todos los éxitos en confirmación',protected_shifted_fault='conservar los éxitos al cambiar el instante de falla')
    failed=[criterion_names[k] for k,v in recommendation.get('acceptance_checks',{}).items() if not v]
    criterion_text='Criterios no cumplidos: '+', '.join(failed)+'.' if failed else 'Cumplió los criterios previstos.'
    adopted=recommendation['learned_family']
    lines=['# Evaluación de control y aprendizaje — ronda 6','',
        '**Decisión:** conservar '+names['controller_selected']+(' y añadir '+names[adopted]+'.' if adopted else ', sin corrección neural en el controlador recomendado.'),'',
        f"La familia candidata elegida en validación fue **{names[recommendation['candidate_family']]}**. Su diferencia de éxito frente a la base en los casos severos estimados fue {100*recommendation['primary']['mean']:.2f} puntos porcentuales; intervalo simultáneo [{100*recommendation['primary']['lower']:.2f}, {100*recommendation['primary']['upper']:.2f}]. La confirmación en 200 rutas dio {100*confirm['mean']:.2f} puntos, intervalo [{100*confirm['lower']:.2f}, {100*confirm['upper']:.2f}].",'',
        criterion_text,'',
        'La aceptación exige ganancia confirmada, ninguna pérdida observada de éxitos de la base en todos los regímenes evaluados y cómputo compatible con 48 Hz. Los intervalos no prueban ausencia de riesgo.','',
        '## Qué cambió y qué se conserva','',
        f"La coordinación de recuperación, descenso y frenado se evaluó antes del aprendizaje. La configuración elegida obtuvo {p1['gains']} aterrizajes adicionales y {p1['losses']} pérdidas frente a la referencia en 1.056 casos de validación. La mejora incluye el observador de una sola falla y la reserva de empuje de 0,1%; no se atribuye únicamente al supervisor ni a una red.",'',
        'Se corrigió una compuerta que copiaba la intervención anterior: ahora la decisión usa el estado físico y la memoria, con ejemplos explícitos del inicio de la intervención y del retorno al PID. Las demostraciones del profesor NMPC se ejecutaron bajo las mismas limitaciones del alumno. Se conservaron los intentos descartados como diagnóstico y se repitieron las inicializaciones afectadas.','',
        'El éxito exige contacto suave, límites de velocidad e inclinación y un segundo estable. Los porcentajes de falla indican pérdida de RPM de un motor, no pérdida directa de empuje. El límite estacionario normal de 29,7186% no prueba imposibilidad de otras maniobras.','',
        '## Resultados independientes','',
        'Redes: cinco semillas de entrenamiento y 60 rutas por punto (300 casos por modelo de arquitectura y método). Controladores principales: las mismas 60 rutas. NMPC secundario: 12 rutas por punto. Un 100% observado corresponde a esa muestra, no a una garantía universal.','']
    for condition in ['estimated','exact']:
        for noise in [1.,2.]:
            lines += [f"### Efectividad {'estimada' if condition=='estimated' else 'exacta'}, ruido {noise:g}×",'',
                '| Controlador | 0% | 25% | 28% | 29% | 29,5% | 29,6% | 30% | 35% | 40% | 50% |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for family,label in names.items():
                cells=[]
                for level in [0.,25.,28.,29.,29.5,29.6,30.,35.,40.,50.]:
                    rr=[r for r in table if r['family']==family and r['condition']==condition and r['noise_multiplier']==noise and abs(r['rpm_loss_percent']-level)<1e-7]
                    cells.append(f"{rr[0]['success_percent']:.1f}%" if rr else '—')
                lines.append('| '+label+' | '+' | '.join(cells)+' |')
            lines.append('')
    lines += ['## Cuándo intervenir: compuerta e instante de falla','',
        '| Familia | Éxito con compuerta | Éxito sin compuerta | Fallos evitados | Beneficios bloqueados |',
        '|---|---:|---:|---:|---:|']
    for r in gate_rows or []:
        lines.append(f"| {names.get(r['family'],r['family'])} | {r['gated_success_percent']:.1f}% | {r['ungated_success_percent']:.1f}% | {r['prevented_failures']} | {r['missed_benefits']} |")
    lines += ['', 'Ablación de la tabla: 16 rutas, cinco semillas, 29,5% y 29,6%, efectividad estimada y ruido doble. Los cuatro regímenes están en gate_ablation.csv y gate_summary.csv. Sin compuerta se aplica la propuesta de la red dentro de los mismos límites físicos. Una compuerta útil debe evitar daños sin bloquear demasiadas correcciones beneficiosas.','',
        '| Familia | Falla a (s) | Éxito | Base | Diferencia (puntos) | Ganancias / pérdidas |',
        '|---|---:|---:|---:|---:|---:|']
    for r in shifts or []:
        lines.append(f"| {names.get(r['family'],r['family'])} | {r['fault_clock_seconds']:g} | {r['success_percent']:.1f}% | {r['baseline_success_percent']:.1f}% | {r['paired_difference_pp']:+.2f} | {r['gains']} / {r['losses']} |")
    lines += ['', 'Prueba temporal: efectividad estimada y ruido doble, 16 rutas y cinco semillas por familia. La aceptación también exige cero pérdidas observadas respecto a la base al adelantar o retrasar la falla a 4 s y 8 s. Este requisito se fijó antes de abrir el test; la comprobación no garantiza seguridad fuera de la muestra.','',
        '## Schedules del rango de ruido','',
        '| Arquitectura | Nominal | Lineal | Adaptativo | Rango completo | Elegido en validación |',
        '|---|---:|---:|---:|---:|---|']
    for arch in ARCHS:
        values=[f"{p2['summary'][arch][sch]['mean_score']:.4f}" for sch in SCHEDULES]
        lines.append('| '+('Mamba-3' if arch=='mamba3' else 'GTrXL')+' | '+' | '.join(values)+' | '+selected[arch]+' |')
    lines += ['', 'La tabla anterior contiene el objetivo agregado de validación, no porcentajes de éxito. Las curvas y tablas finales separan cada severidad. La clonación inicial y su regularización usan ruido nominal en las cuatro ramas. El schedule solo modifica el rango de ruido en los vuelos de entrenamiento.','',
        'Lineal: ampliar el máximo de 1× a 2× durante el 80% del presupuesto y mantenerlo después. Adaptativo: avanzar o retroceder según el éxito de entrenamiento en cada estrato severo estimado, protegiendo los casos leves. Nominal y rango completo sirven como controles de exposición.','',
        'Ruido nominal: desviación de 0,002 m/s en velocidad lineal y 0,002 rad/s en velocidad angular; ruido doble: 0,004 en ambas. No se añadieron ruido de posición, viento ni errores de orientación.','',
        exposure_text,'','## Tiempo medido','',
        '| Trabajo | Cantidad | Suma de minutos de los trabajos | Mediana por trabajo (min) |',
        '|---|---:|---:|---:|']
    for label,jobs in [('Clonación inicial compartida',bc),('PPO definitivo',[r for r in times if r['phase']=='PPO']),('Pilotos de hiperparámetros',pilots),('FlashSAC definitivo',[r for r in times if r['phase']=='flashsac']),('FQL definitivo',[r for r in times if r['phase']=='fql'])]:
        seconds=[r.get('seconds',r.get('total_seconds')) for r in jobs]
        lines.append(f"| {label} | {len(jobs)} | {sum(seconds)/60:.1f} | {np.median(seconds)/60:.1f} |")
    lines += ['', 'Las sumas incluyen trabajos concurrentes y no equivalen al tiempo de reloj. Los CSV separan validación y espera de turno, codificación, inicialización y aprendizaje offline/online. En PPO la carga de datos se comparte por trabajador y no se cronometró por modelo: las celdas vacías significan no medido o no aplicable, no cero segundos. Los intentos de preparación descartados no cuentan como modelos definitivos.','',
        '| Etapa registrada | Minutos de reloj |','|---|---:|']
    for name,d in receipts.items():lines.append(f"| {name.removesuffix('_receipt')} | {d['seconds']/60:.1f} |")
    elapsed=(time.time()-read(OUT/'campaign_started.json')['started_epoch'])/3600
    lines += ['',f"Tiempo desde el inicio registrado de la campaña hasta la generación de este informe: {elapsed:.2f} horas; incluye estudios de desarrollo, esperas y repeticiones de preparación. La primera recolección de datos incluyó una espera por el cierre del control.",'',
        '## Alcance de FlashSAC, FQL y las arquitecturas','',
        'FlashSAC y FQL se evaluaron después de cerrar la elección de control y schedule. Se escogieron hiperparámetros por arquitectura con pilotos y rutas de validación separadas; después se entrenaron cinco semillas. FQL recibió 100.000 actualizaciones offline y cada modelo definitivo 524.288 interacciones online.','',
        'Los 60 modelos completan su presupuesto, pero se evalúa el checkpoint elegido por validación. '+ '; '.join(f"{label}: {sum(r.get('best_checkpoint_online_steps')==0 for r in times if r['phase']==method)} de {sum(r['phase']==method for r in times)} mejores checkpoints antes del afinado online" for method,label in [('PPO','PPO'),('flashsac','FlashSAC'),('fql','FQL')])+'. En PPO, cero pasos online significa conservar la clonación inicial; en FQL incluye sus 100.000 actualizaciones offline. No se atribuye al afinado online una mejora que ya existía antes.','',
        'Las cabezas usan el encoder recurrente congelado del PPO elegido. Se reutilizan bloques oficiales de FlashSAC y un port de las pérdidas oficiales de FQL. Mamba-3 es una referencia matemática SISO portable; esta evaluación no compara kernels MIMO fusionados ni demuestra una superioridad general de las arquitecturas o algoritmos completos.','',
        'La compuerta off-policy compara dos estimaciones de recompensa extrínseca para intervenir frente a no intervenir. La ablación de la compuerta y los cambios del instante de falla se entregan junto con las curvas; no se reajustó ningún modelo después de abrir el test.','',
        '## Archivos de respaldo','',
        '- `success_rates.csv`: todas las severidades y regímenes, dispersión entre semillas y denominadores.',
        '- `paired_statistics.json`, `confirmation_statistics.json`: comparación pareada e intervalos.',
        '- `gate_ablation.csv` y carpeta `sensitivity`: compuerta e instante de falla.',
        '- `training_times.csv`, `cloning_times.csv`, `hyperparameter_times.csv`, `stage_times.json`: tiempos.',
        '- `final_integrity.json` y `frozen_before_test.json`: verificación de datos, fuentes y modelos.',
        '- `INFORME_FINAL.txt`: detalle completo y limitaciones.','',
        'Fuentes: [FlashSAC](https://arxiv.org/abs/2604.04539), [FQL](https://proceedings.mlr.press/v267/park25f.html), [NMPC tolerante a fallas](https://arxiv.org/abs/2109.12886).']
    text='\n'.join(lines)
    (OUT/'RESUMEN_RESULTADOS.md').write_text(text,encoding='utf-8')
    (OUT.parent.parent.parent/'RESULTADOS_RONDA6.md').write_text(text,encoding='utf-8')

def gate_ablation(group):
    rows=[]
    for p in (OUT/'ablations').glob('*.json'):
        d=read(p);e=d['entry'];_,c,n=split_name(p);family=f"{e['kind']}_{e['arch']}_{e['schedule']}"
        for l in [.295,.296]:
            selected={r['seed']:r for r in group[(family,c,n,l)] if r['training_seed']==e['seed'] and r['seed']<612016}
            rr=[r for r in d['episodes'] if r['level']==l]
            rows.append(dict(family=family,training_seed=e['seed'],condition=c,noise=n,level=l,routes=len(rr),
                gated_success=sum(selected[r['seed']]['stable_landing'] for r in rr),ungated_success=sum(r['stable_landing'] for r in rr),
                prevented_failures=sum(selected[r['seed']]['stable_landing'] and not r['stable_landing'] for r in rr),
                missed_benefits=sum(r['stable_landing'] and not selected[r['seed']]['stable_landing'] for r in rr)))
    csvwrite(OUT/'gate_ablation.csv',rows);return rows

def plots(rows):
    p2=read(OUT/'phase2_conclusion.json');control=read(OUT/'phase1_conclusion.json');nmpc=read(OUT/'nmpc_test_plan.json')['names']
    labels={'dslpid_reference':'DSLPID referencia','controller_selected':'DSLPID actualizado'}
    for arch in ARCHS:
        for method in ['ppo','flashsac','fql']:labels[f'{method}_{arch}_{p2["schedule_by_arch"][arch]}']=method.upper()+' + '+('Mamba-3' if arch=='mamba3' else 'GTrXL')
    for name in nmpc:
        gain=control['all_configs'][name].get('nmpc_options',{}).get('indi_gain',1.)
        labels[name]=('NMPC sin INDI' if gain==0 else 'NMPC + INDI')+(' (rápido)' if 'fast' in name else ' (mayor horizonte)')
    families=list(labels);colors=plt.get_cmap('tab10').colors;plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    created=[]
    for condition in ['estimated','exact']:
        fig,axes=plt.subplots(2,2,figsize=(16,10),sharey=True)
        for i,noise in enumerate([1.,2.]):
            for j,zoom in enumerate([False,True]):
                ax=axes[i,j]
                for k,family in enumerate(families):
                    rr=sorted([r for r in rows if r['family']==family and r['condition']==condition and r['noise_multiplier']==noise and (not zoom or 28<=r['rpm_loss_percent']<=30)],key=lambda r:r['rpm_loss_percent'])
                    if not rr:continue
                    x=[r['rpm_loss_percent'] for r in rr];y=[r['success_percent'] for r in rr]
                    ax.plot(x,y,marker='o',ms=3,lw=1.6,color=colors[k%10],label=labels[family],linestyle='--' if family.startswith('fql') else '-')
                ax.set_ylim(-3,103);ax.set_xlim(27.8,30.2) if zoom else ax.set_xlim(0,50)
                ax.set_title(f"Ruido {'nominal' if noise==1 else 'doble'} · {'detalle del límite' if zoom else 'rango completo'}")
                ax.set_xlabel('Pérdida de RPM del motor (%)');ax.set_ylabel('Aterrizajes controlados y estables (%)')
        handles,ll=axes[0,0].get_legend_handles_labels();fig.legend(handles,ll,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.005),fontsize=9)
        fig.suptitle('Efectividad '+('estimada' if condition=='estimated' else 'exacta')+' · comparación final',fontsize=15)
        fig.text(.5,.092,'Principales: 60 rutas/punto, 5 semillas por red. NMPC secundarios: 12 rutas/punto. Los puntos son mediciones.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.13,1,.95]);base=OUT/f'comparacion_{condition}'
        for ext in ['png','svg']:fig.savefig(str(base)+'.'+ext,dpi=180,bbox_inches='tight')
        plt.close(fig);created.append(str(base)+'.png')
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharey=True)
    for i,arch in enumerate(ARCHS):
        for j,noise in enumerate([1.,2.]):
            ax=axes[i,j]
            for k,sch in enumerate(SCHEDULES):
                rr=sorted([r for r in rows if r['family']==f'ppo_{arch}_{sch}' and r['condition']=='estimated' and r['noise_multiplier']==noise and 28<=r['rpm_loss_percent']<=30],key=lambda r:r['rpm_loss_percent'])
                ax.plot([r['rpm_loss_percent'] for r in rr],[r['success_percent'] for r in rr],marker='o',label={'nominal':'Nominal','linear':'Lineal','adaptive':'Adaptativo','full':'Rango completo'}[sch])
            ax.set_title(f"{arch} · ruido {noise:g}×");ax.set_xlabel('Pérdida de RPM (%)');ax.set_ylabel('Aterrizajes estables (%)');ax.set_ylim(-3,103);ax.legend()
    fig.suptitle('Schedules del rango de ruido · PPO · cinco semillas');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/f'schedules.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);created.append(str(OUT/'schedules.png'));return created

def acceptance_checks(primary,confirmation,shifted,runtime):
    return dict(runtime_48hz=runtime<=1000/48,primary_confirmed_gain=primary['lower']>0,independent_confirmation_gain=confirmation['lower']>0,
        protected_primary=primary['total_losses']==0,protected_confirmation=confirmation['total_losses']==0,
        protected_shifted_fault=all(r['losses']==0 for r in shifted))

def main():
    group=groups();table=rates(group);stats,names=primary_statistics(group);confirm=confirmation_statistics();times,receipts=timing();exposure=curriculum_exposure(times);ablation=gate_ablation(group);shifts=sensitivity_statistics();gate_rows=summarize_gate(ablation);critical_gate_rows=summarize_gate([r for r in ablation if r['condition']=='estimated' and r['noise']==2.],destination='gate_summary_estimated_noise2.csv')
    p1=read(OUT/'phase1_conclusion.json');p2=read(OUT/'phase2_conclusion.json');p3=read(OUT/'phase3_conclusion.json')
    candidate=p3['representative'];family=f"{candidate['kind']}_{candidate['arch']}_{candidate['schedule']}";primary=next(r for r in stats['contrasts'] if r['family']==family)
    latency=read(OUT/'latency.json');runtime=latency['controllers']['selected']['p95_ms']+latency['neural'][candidate['arch']+'_'+candidate['kind']]['p95_ms']
    candidate_shift=[r for r in shifts if r['family']==family];assert {r['fault_clock_seconds'] for r in candidate_shift}=={4.,8.}
    checks=acceptance_checks(primary,confirm,candidate_shift,runtime)
    accepted=all(checks.values())
    recommendation=dict(controller=p1['selected'],learned_family=family if accepted else None,representative=candidate if accepted else None,
        reason='Independent paired gain confirmed without observed protected-case harm.' if accepted else 'The validation-selected candidate did not meet all prospective gain/protection requirements; retain base controller and abstention.',
        candidate_family=family,primary=primary,confirmation=confirm,sum_component_p95_ms=runtime,acceptance_checks=checks,shifted_fault=candidate_shift)
    save(OUT/'recommendation.json',recommendation);figures=plots(table)
    readable_summary(table,recommendation,p1,p2,p3,times,receipts,latency,stats,confirm,shifts,critical_gate_rows,exposure)
    total_logical=sum(len(v) for k,v in group.items() if not k[0].startswith(('dslpid','controller','nmpc')))
    zero=sum(r.get('zero_replay_certified',False) for rr in group.values() for r in rr)
    lines=['RESULTADO FINAL RONDA6','',f"Control elegido: {p1['selected']}",f"Familia aprendida aceptada: {family if accepted else 'NINGUNA; conservar el control base'}",recommendation['reason'],'',
        'FASE1: COORDINACION Y CONTROL',f"Ganancias/perdidas pareadas en validacion: {p1['gains']}/{p1['losses']}.",
        'Se implementaron y compararon supervisor continuo, fases de recuperacion/descenso/frenado, limites de aceleracion angular, asignacion de empuje y filtrado. Solo se conserva la version que gano la comparacion.',
        'El observador estructurado supone una sola falla. No se presenta como estimador general multifalla.',
        'NMPC por defecto y sus ajustes se evaluaron aparte. Una version INDI0 se identifica como NMPC sin compensacion INDI. Horizonte/retardo/gain son adaptaciones al simulador; no reproducen el hardware del paper.',
        'NMPC secundario tiene12rutas porpunto, frente a60 en principales.0/12 no prueba imposibilidad fisica ni exito poblacional nulo. Se rechazo para48Hz una variante que supera20.83ms porcalculo.','',
        'FASE2: SCHEDULES DE RUIDO',json.dumps(p2['summary'],indent=2),f"Seleccion por arquitectura: {p2['schedule_by_arch']}",
        'BC y regularizacion de imitacion se restringen a ruido nominal en todas las ramas de fase2;solo los rollouts online amplian el ruido. Las ramas no ven ruido doble por la clonacion antes del curriculum.',
        'Lineal: ampliar upper1->2durante80%delpresupuesto y mantener. Adaptativo: ampliar/reducir segun exito de entrenamiento en cada estrato severo estimado y proteger casos faciles. Nominal y full son controles de exposicion.',
        'sigma nominal=.002m/s en velocidad lineal y.002rad/s en giros; doble=.004en ambas. Posicion/orientacion siguen exactas. Se muestrea un multiplicador uniforme[.5,upper] por episodio en entrenamiento.','',
        'FASE3: FLASH SAC Y FLOW Q-LEARNING',json.dumps(p3['summary'],indent=2),'PARAMETROS ELEGIDOS ANTES DE LOS MODELOS FINALES',json.dumps(read(OUT/'hpo_selection.json'),indent=2),
        'Encoders Mamba-3/GTrXL congelados desde PPO elegido; cabezas off-policy entrenadas con replay de acciones ejecutadas. Memoria de cada episodio reconstruida desdecero. Mamba-3 es referencia matematica portable, no benchmark de kernels fusionados.',
        'Compuerta extrinseca separada: compara Q(a) frente a Q(0) usando recompensa real/100, umbral.005 y ambos criticos. La entropia SAC no decide si aterrizar mejor. Es una estimacion aprendida, no garantia formal de seguridad.',
        'FlashSAC reutiliza bloques oficiales MIT, critico distribucional, BN/RMS/weight normalization y normalizacion de retorno; FQL porta a PyTorch el algoritmo oficial con flujoBC, distillation y Q. Gamma.999 porhorizonte largo, FQL4x512 conalpha seleccionado en validacion, Flashactor128/critic256con2bloques. Se igualan interacciones, no parametros ni segundos.',
        'Se incorporaron demostraciones adicionales de un profesor NMPC calculado fuera de tiempo real, con estado medido y efectividad estimada;accionesresidualesproyectadasyejecutadas. Se repitieron las inicializacionesanteriores con el conjunto comun ampliado, identificado por hash.',
        'PPO/FlashSAC/FQL usan la misma recompensa de tarea y aterrizaje estricto. Estas pruebas no comparan nuevos metodos IRL.','',
        'EVIDENCIA INDEPENDIENTE',json.dumps(stats,indent=2),json.dumps(confirm,indent=2),'CRITERIOS DE ACEPTACION',json.dumps(checks,indent=2),'SENSIBILIDAD AL INSTANTE DE FALLA',json.dumps(shifts,indent=2),'COMPuERTA',json.dumps(gate_rows,indent=2),
        f'Casos logicos de redes: {total_logical}; certificados OFF reutilizados: {zero}. No son todos simulaciones independientes.',
        'Las tablas distinguen casos por modelo,60rutas y5semillas. Intervalos primarios: bootstrap cruzado rutas/semillas con banda simultanea para6familias. La confirmacion usa200rutas nuevas.',
        'Ablacion de compuerta en16rutas,5semillas y dos severidades; sensibilidad con inicio de falla4s/8s. VerCSV y JSON, sin ajustes posteriores a test.','',
        'LIMITES DEL RESULTADO','El limite29.7186% corresponde al vuelo estacionario normal con momentos equilibrados. No es una demostracion de imposibilidad para descenso transitorio o vuelo giratorio.50%RPM significa75%deperdida de empuje en ese motor.',
        'Se incluyeron experiencias de entrenamiento a30%,35%y50%, con probabilidad total8%porreinicio, ademas de los niveles principales. No es una busqueda exhaustiva de maniobras giratorias. Ver exposicion real por nivel en complete.json.',
        'El exito sigue exigiendo contacto suave y un segundo estable. Mantenerse girando en vuelo no cuenta. No se relajan umbrales para presentar mejores tasas.',
        'Los resultados se limitan a esta planta simulada y este modelo de ruido; no prueban vuelo real.','',
        'TIEMPOS REALES POR ETAPA',json.dumps(receipts,indent=2),'TIEMPO DE CONTROL DEDICADO',json.dumps(latency,indent=2),
        'training_times.csv separa datos, representacion, inicializacion, offline, online y validacion con espera de turno segun el metodo; vacio significa no medido/no aplicable. Las sumas de trabajos concurrentes no equivalen al tiempo de reloj de la campaña.','',
        'FUENTES PRIMARIAS','FlashSAC https://arxiv.org/abs/2604.04539 ; codigo https://github.com/Holiday-Robot/FlashSAC',
        'FQL https://proceedings.mlr.press/v267/park25f.html ; codigo https://github.com/seohongpark/fql',
        'NMPC tolerante a fallas https://arxiv.org/abs/2109.12886',
        'Ver frozen_before_test.json para commits, hashes, activos y modelos anteriores preservados.']
    text='\n'.join(lines);(OUT/'INFORME_FINAL.txt').write_text(text,encoding='utf-8');(OUT.parent.parent.parent/'INFORME_RONDA6.txt').write_text(text,encoding='utf-8')
    for c in ['exact','estimated']:
        for noise in [1.,2.]:
            rr=[r for r in table if r['condition']==c and r['noise_multiplier']==noise]
            families=sorted({r['family'] for r in rr});wide=[]
            for level in LEVELS:
                row={'falla_RPM_percent':100*level}
                for f in families:row[f]=next((r['success_percent'] for r in rr if r['family']==f and abs(r['rpm_loss_percent']-100*level)<1e-8),None)
                wide.append(row)
            csvwrite(OUT/f'tabla_{c}_ruido{noise:g}.csv',wide)
    ready=dict(created_epoch=time.time(),figures=figures,recommendation=recommendation,rate_rows=len(table),training_models=len(times),
        figure_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in figures})
    save(OUT/'report_ready_for_visual_review.json',ready)
    save(OUT/'status.json',dict(stage='awaiting_visual_review',status='evaluation_complete'))
    print('REPORT_READY_FOR_VISUAL_REVIEW',figures,flush=True)
    # The root agent must inspect all figures before the campaign can complete.
    while not (OUT/'presentation_verified.json').exists():time.sleep(5)
    verified=read(OUT/'presentation_verified.json');assert verified['passed']
    save(OUT/'report_complete.json',dict(finished_epoch=time.time(),presentation_verified=True))
if __name__=='__main__':main()
