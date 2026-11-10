"""Paired held-out report. Run only after round3.run --phase test completes."""
from pathlib import Path
import json,csv,datetime,hashlib,xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import OUT,SEEDS,LEVELS,save
METHODS=['bc','manual','csil','meirl','rime']
LABELS={'bc':'Clonación LSTM','manual':'PPO recompensa manual','csil':'CSIL adaptado','meirl':'ME-IRL adaptado','rime':'RIME adaptado'}
COLORS={'bc':'#666666','manual':'#e69f00','csil':'#0072b2','meirl':'#009e73','rime':'#cc79a7'}
TASK=OUT.parents[2]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def matrix(rows,key='mission_success',seeds=range(200000,200100)):
    lookup={(float(r['level']),int(r['seed'])):r for r in rows}
    return np.array([[float(lookup[(level,seed)][key]) for seed in seeds] for level in LEVELS])
def paired_ci(diff,levels,seed=9103,repeats=20000):
    a=diff[:,levels,:].mean(1);rng=np.random.default_rng(seed);vals=[]
    for _ in range(repeats//500):
        si=rng.integers(a.shape[0],size=(500,a.shape[0]));ri=rng.integers(a.shape[1],size=(500,a.shape[1]))
        vals.extend(a[si[:,:,None],ri[:,None,:]].mean((1,2)))
    vals=np.array(vals)
    return dict(mean=float(a.mean()),ci95=np.quantile(vals,[.025,.975]).tolist(),ci_simultaneous=np.quantile(vals,[.05/6,1-.05/6]).tolist())
def aggregate(rows):
    out=[]
    for level in LEVELS:
        rr=[r for r in rows if r['level']==level]
        out.append(dict(level=level,n=len(rr),success=float(np.mean([r['mission_success'] for r in rr])),exposed=sum(r['exposed'] for r in rr),
            rmse=float(np.mean([r['tracking_rmse'] for r in rr])),effort=float(np.mean([r['effort'] for r in rr])),gate=float(np.mean([r['mean_gate'] for r in rr])),
            saturation=float(np.mean([r['saturation'] for r in rr])),peak_tilt_degrees=float(np.rad2deg(max(r['max_tilt'] for r in rr)))))
    return out

def main():
    assert read(OUT/'status.json')['stage']=='complete'
    if not (OUT/'reward_validation_diagnostics.json').exists():
        from .diagnose_rewards import main as diagnose_rewards
        diagnose_rewards()
    if not (OUT/'policy_validation_diagnostics.json').exists():
        from .diagnose_policy import main as diagnose_policy
        diagnose_policy()
    base=read(OUT/'baselines_test.json');controllers={}
    for controller in ['legacy','robust','updated','oracle_teacher']:
        for condition in ['exact','uncertain']:
            rows=[r for r in base if r['controller']==controller and r['condition']==condition]
            if rows:controllers[controller+'_'+condition]=rows
    rows={};mats={};exact={};selection={};timings=[]
    for method in METHODS:
        rows[method]=[];mm=[];ex=[];selection[method]=[]
        for seed in SEEDS:
            sd=OUT/f'seed_{seed}';d=sd if method=='bc' else sd/method
            rr=read(d/('bc_test.json' if method=='bc' else 'test.json'));rows[method].extend([dict(r,training_seed=seed) for r in rr]);mm.append(matrix(rr))
            er=read(d/('bc_exact_test.json' if method=='bc' else 'exact_test.json'));ex.append(matrix(er,seeds=range(200100,200120)))
            if method!='bc':
                choice=read(d/'selection.json');selection[method].append(dict(seed=seed,**choice))
                timing=read(d/'complete.json');timings.append(dict(method=method,seed=seed,**timing))
        mats[method]=np.stack(mm);exact[method]=np.stack(ex)
    baseline=matrix(controllers['updated_uncertain']);statistics={}
    for method in ['csil','meirl','rime']:
        x=mats[method];harm=baseline[None,:,:]-x
        route_harm=np.any(harm[:,:3,:]>0,axis=(0,1));n=int(route_harm.sum());N=len(route_harm)
        statistics[method]=dict(vs_manual=paired_ci(x-mats['manual'],[3,4,5,6]),vs_bc=paired_ci(x-mats['bc'],[3,4,5,6]),
            vs_updated=paired_ci(x-baseline[None,:,:],[3,4,5,6]),mild_harm=paired_ci(harm,[0,1,2]),
            distinct_routes_with_any_mild_harm=n,conservative_route_harm_upper95=float(beta.ppf(.95,n+1,N-n)) if n<N else 1.)
        z=statistics[method]
        z['passes_registered_gate']=z['vs_manual']['ci95'][0]>0 and z['mild_harm']['ci95'][1]<.05
        z['passes_simultaneous_gate']=z['vs_manual']['ci_simultaneous'][0]>0 and z['mild_harm']['ci_simultaneous'][1]<.05
        z['demonstrates_gain_over_cloning']=z['vs_bc']['ci_simultaneous'][0]>0
    final_mats={}
    for method in ['manual','csil','meirl','rime']:
        final_mats[method]=np.stack([matrix(read(OUT/f'seed_{seed}'/method/'final_test.json'),seeds=range(200000,200030)) for seed in SEEDS])
    final_vs_bc={method:paired_ci(value-mats['bc'][:,:,:30],[3,4,5,6]) for method,value in final_mats.items()}
    exact_baseline=matrix(read(OUT/'baseline_exact_secondary.json'),seeds=range(200100,200120))
    summary={k:aggregate(v) for k,v in rows.items()};controller_summary={k:aggregate(v) for k,v in controllers.items()}
    save(OUT/'analysis.json',dict(summary=summary,controller_summary=controller_summary,paired_statistics=statistics,selection=selection,timings=timings,
        exact_success={k:v.mean((0,2)).tolist() for k,v in exact.items()},exact_baseline_success=exact_baseline.mean(1).tolist(),final_success={k:v.mean((0,2)).tolist() for k,v in final_mats.items()},final_vs_bc_paired_statistics=final_vs_bc,bc_same_final_cases_success=mats['bc'][:,:,:30].mean((0,2)).tolist(),confidence='crossed paired bootstrap: resample training seeds and route IDs independently,retain paired levels;20000 draws; simultaneous98.33% intervals for3 methods vs one reference;not a guarantee about real hardware'))
    with (OUT/'success_rates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['method','rpm_loss','success_rate','training_seeds','unique_routes_per_level'])
        for method in METHODS:
            for j,level in enumerate(LEVELS):w.writerow([method,level,mats[method][:,j,:].mean(),5,100])
        for method,rr in controllers.items():
            for r in aggregate(rr):w.writerow([method,r['level'],r['success'],1,100])
    conceptual_cases=30240;simulated_cases=5040
    for seed in SEEDS:
        for method in METHODS:
            d=OUT/f'seed_{seed}' if method=='bc' else OUT/f'seed_{seed}'/method
            for stem,cases in [('bc_test' if method=='bc' else 'test',700),('bc_exact_test' if method=='bc' else 'exact_test',140)]:
                p=d/(stem+'_timing.json');t=read(p) if p.exists() else {}
                simulated_cases+=int(t.get('simulated_cases',cases))
            if method!='bc':simulated_cases+=210
    save(OUT/'evaluation_counts.json',dict(model_case_combinations=conceptual_cases,preserved_simulated_cases=simulated_cases,reused_identical_actor_results=conceptual_cases-simulated_cases,partial_discarded_runs_not_counted=True))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    x=np.array(LEVELS)*100
    for name,label,color,style in [('legacy_uncertain','DSLPID original','#999999','--'),('robust_uncertain','Robust anterior','#d55e00','--'),('updated_uncertain','DSLPID actualizado · estimación imperfecta','#0072b2','-'),('updated_exact','DSLPID actualizado · efectividad exacta','#009e73',':')]:
        axes[0].plot(x,[r['success']*100 for r in controller_summary[name]],style,marker='o',color=color,label=label)
    for method in METHODS:
        mean=mats[method].mean((0,2))*100
        axes[1].plot(x,mean,marker={'bc':'s','manual':'x','csil':'o','meirl':'^','rime':'d'}[method],linestyle={'bc':'--','manual':':','csil':'-','meirl':'--','rime':'-.'}[method],color=COLORS[method],label=LABELS[method],markersize=5)
    for ax in axes:
        ax.set_xlabel('Pérdida de RPM en un motor (%)');ax.set_ylim(-2,103);ax.set_xticks(x);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
    axes[0].set_ylabel('Rutas completadas (%)');axes[0].set_title('Cambio del controlador · 100 rutas por nivel')
    axes[1].set_title('Comparación de recompensas · 5 semillas × 100 rutas')
    fig.suptitle('Primera prueba reservada: DSLPID actualizado y recompensas aprendidas',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'comparacion_principal.png');fig.savefig(OUT/'comparacion_principal.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));order=['csil','meirl','rime']
    for i,m in enumerate(order):
        c=statistics[m]['vs_manual'];lo,hi=np.array(c['ci_simultaneous'])*100;mean=c['mean']*100
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color=COLORS[m],label=LABELS[m])
    ax.axvline(0,color='black',lw=1);ax.set_yticks(range(3),[LABELS[m] for m in order]);ax.set_xlabel('Diferencia en éxito frente a PPO manual (puntos porcentuales)')
    ax.set_title('Fallas de 10–25% · intervalos simultáneos de 98,33%');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'diferencias_pareadas.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for method in METHODS:ax.plot(x,[np.sqrt(r['effort']) for r in summary[method]],marker='o',color=COLORS[method],label=LABELS[method])
    ax.set_xlabel('Pérdida de RPM en un motor (%)');ax.set_ylabel('RMS residual relativo a los límites de fuerza/par');ax.set_title('Intervención aplicada · comparar junto con el éxito');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'intervencion_residual.png');plt.close(fig)
    lines=[]
    def put(s=''):lines.append(s)
    put('DSLPID ACTUALIZADO Y TRES OPCIONES DE APRENDIZAJE DE RECOMPENSA')
    put('Primera prueba local terminada · '+datetime.datetime.now().astimezone().isoformat());put()
    winners=[m for m,z in statistics.items() if z['passes_simultaneous_gate'] and z['demonstrates_gain_over_cloning']]
    if winners:put('CONCLUSIÓN: '+', '.join(LABELS[m] for m in winners)+' supera el control con recompensa manual y la clonación en las condiciones ensayadas. Revisar también la prueba con efectividad exacta antes de adoptar.')
    else:put('CONCLUSIÓN: esta primera prueba no demuestra una ventaja suficiente de las recompensas aprendidas sobre los comparadores para recomendar IRL como mejora confirmada. El controlador actualizado y la clonación siguen siendo los comparadores de referencia. Un resultado negativo aquí se refiere a estas adaptaciones, presupuesto y condiciones, no invalida los algoritmos originales.')
    put();put('CAMBIO DEL CONTROLADOR')
    put('El DSLPID público calcula fuerza y pares sin pasar primero por PWM recortado. Usa tasas angulares medidas en el cuerpo, inercias del URDF y compensación giroscópica. La asignación acotada pondera fuerza, roll/pitch y yaw, conoce la efectividad estimada de cada motor y devuelve el error de asignación a las integrales. Se limita la fuerza e inclinación solicitadas y se cancela la integral yaw ante una falla. El modo severo pide actitud reducida como mejor esfuerzo; no implementa un equilibrio giratorio validado de Mueller–D’Andrea.')
    put('η es un factor de RPM. El empuje usa η²; la planta multiplica las RPM una sola vez. No se incrementó el máximo de 21.700 RPM. El mínimo nuevo permite 0 RPM en simulación, mientras el controlador original conservaba su mínimo PWM. Ese cambio forma parte del diseño nuevo. El controlador original se conservó en LegacyDSLPIDControl.py.')
    put();put('CANDIDATOS Y EVIDENCIA PUBLICADA')
    put('1. ME-IRL, L4DC 2026: integra una distribución de transición en el potencial del reward; evidencia en MuJoCo estocástico y Atari. https://proceedings.mlr.press/v331/zhan26a.html')
    put('   Prueba local: ensemble de7 modelos gaussianos,5 élites,potencial esperado y discriminador causal con logπ;65536 pasos de descubrimiento por semilla. Se usa PPO y solo transiciones reales; después se congela R(s) para el compensador común. No es reproducción completa de SAC con rollouts sintéticos del artículo. El modelo de dinámica se ajusta al conjunto inicial de entrenamiento.')
    put('2. CSIL, NeurIPS 2023 Spotlight: evidencia en control continuo, Adroit y robomimic; obtiene una recompensa coherente de la razón entre política clonada y prior. https://joemwatson.github.io/csil/')
    put('   Prueba local: port Torch de las características periódicas estacionarias y cabeza gaussiana heteroscedástica PSD,128 features; recompensa 0,1 log(q/p). Se conserva el núcleo de la recompensa; el optimizador posterior es el PPO común, con normalización y anclas de tarea, por lo que no se transfieren las garantías de coherencia del actor-critic original.')
    put('3. RIME, ICML 2024: evidencia de robustez a preferencias ruidosas en locomoción y manipulación. https://proceedings.mlr.press/v235/cheng24k.html')
    put('   Es aprendizaje de recompensa por preferencias, complementario al IRL clásico. Prueba local: ensemble3,Bradley–Terry,calentamiento con entropía kNN,filtro KL y cambios de etiquetas como en el código oficial. Preferencias en segmentos32 del mismo escenario,calculadas con métricas físicas y10% de probabilidad de invertir la etiqueta. No son preferencias humanas. Se aprende de un lote fijo y se usa PPO posterior; no reproduce todo el SAC/exploración intrínseca de RIME.')
    put('La selección combina investigación reciente sobre dinámica con métodos de recompensa maduros y reproducibles. No afirma que sean los tres mejores algoritmos universales ni que los artículos demuestren recuperación de drones. También se revisaron PIRO (NeurIPS2025),SFM (ICLR2025) y trabajos de2026; SFM no entrega una recompensa explícita y PIRO es un candidato sólido de2025;esta ronda prioriza contrastar dinámica,recompensa desde clonación y preferencias ruidosas con tres ensayos. No se deduce que PIRO sea inferior. Esta ronda compara tres núcleos con adaptaciones declaradas.')
    put();put('DISEÑO DE LA PRUEBA')
    data=read(OUT/'data_episodes.json');records=data['episodes']
    put(f"Se generaron {len(records)} vuelos y {np.load(OUT/'data.npz')['s'].shape[0]} transiciones. Entrenamiento:24 geometrías ×7 fallas ×3 variantes; validación de reward:6 geometrías separadas. Las210 demostraciones del maestro completaron su ruta. Solo las168 de entrenamiento alimentaron la clonación y los métodos basados en experto; no se ocultaron los fallos de los otros controladores.")
    put('El maestro usa efectividad verdadera después de la misma demora de detección que el alumno. En despliegue de la simulación,alumno y DSLPID solo reciben una estimación que subestima la pérdida por un factor uniforme0,45–0,90,con demora de2–6 ticks (42–125ms). El motor afectado varía entre los cuatro. No se implementó un estimador físico de efectividad;se ensaya este modelo de error conocido. La detección está asistida por el simulador,con la demora indicada y sin falsas alarmas;esta prueba no valida un detector autónomo.')
    put('Política común:LSTM64 de1 capa,historial32 (0,667s),acciones gaussianas acotadas por tanh. Cinco semillas11,22,33,44,55,con la misma clonación inicial para cada tratamiento pareado. Residual de fuerza/pares en unidades físicas y una compuerta continua; no se impone apagarla en casos nominales.')
    put('Presupuesto posterior:262144 pasos por modelo,8 entornos,rollout256,batch256,5 épocas,learning rate2e-4→2e-5,gamma0,995,KL objetivo0,015. Reward y políticas se ajustan solo con datos de entrenamiento. Se selecciona entre clonación inicial y checkpoints usando35 casos de validación,disjuntos de la prueba final. La normalización del estilo usa únicamente entrenamiento. Todos comparten anclas explícitas:+10 misión,−10 caída,−5 aterrizaje prematuro/timeout y penalización de esfuerzo.')
    put('Prueba principal:100 geometrías nuevas por cada uno de7 niveles; las mismas en todos los modelos. Cada método aprendido usa5 semillas ×100 rutas/nivel:500 evaluaciones por nivel,pero100 geometrías únicas. La condición de éxito es completar los waypoints originales; no equivale a aterrizaje estable. La reserva se amplió de50 a100 antes de observar resultados. Hay20 rutas adicionales por nivel con efectividad exacta como prueba secundaria.')
    put();put('RESULTADOS: PORCENTAJE DE RUTAS COMPLETADAS')
    put('Método'.ljust(37)+''.join(f'{v*100:>8g}%' for v in LEVELS))
    for name in ['legacy_uncertain','robust_uncertain','robust_exact','updated_uncertain','updated_exact']:
        label={'legacy_uncertain':'DSLPID original','robust_uncertain':'Robust anterior · estimado','robust_exact':'Robust anterior · exacto','updated_uncertain':'DSLPID nuevo · estimado','updated_exact':'DSLPID nuevo · exacto'}[name]
        put(label.ljust(37)+''.join(f"{r['success']*100:9.1f}" for r in controller_summary[name]))
    for m in METHODS:put(LABELS[m].ljust(37)+''.join(f'{v:9.1f}' for v in mats[m].mean((0,2))*100))
    put();put('EFECTIVIDAD EXACTA: PRUEBA SECUNDARIA (5 semillas ×20 rutas/nivel)')
    for m in METHODS:put(LABELS[m].ljust(37)+''.join(f'{v:9.1f}' for v in exact[m].mean((0,2))*100))
    put('DSLPID nuevo exacto · mismos20 casos'.ljust(37)+''.join(f'{v:9.1f}' for v in exact_baseline.mean(1)*100))
    put();put('MODELO AL TERMINAR TODO EL PRESUPUESTO (5 semillas ×30 rutas/nivel,subconjunto reservado)')
    put('Clonación · mismos30 casos'.ljust(37)+''.join(f'{v:9.1f}' for v in mats['bc'][:,:,:30].mean((0,2))*100))
    for m in ['manual','csil','meirl','rime']:put(LABELS[m].ljust(37)+''.join(f'{v:9.1f}' for v in final_mats[m].mean((0,2))*100))
    for m,z in final_vs_bc.items():
        lo,hi=np.array(z['ci95'])*100
        put(f"Final {LABELS[m]} frente a clonación en los mismos30 casos y niveles10–25%: {z['mean']*100:+.2f} puntos,IC95% [{lo:+.2f},{hi:+.2f}]. Diagnóstico adicional,no criterio primario de selección.")
    put('Estos modelos finales se evalúan aparte del checkpoint seleccionado para distinguir mejora real por aprendizaje de conservar la clonación inicial.')
    put();put('DIFERENCIAS PAREADAS EN FALLAS10–25%')
    for m,z in statistics.items():
        for ref,key in [('PPO manual','vs_manual'),('clonación','vs_bc')]:
            t=z[key];lo,hi=np.array(t['ci_simultaneous'])*100
            put(f"{LABELS[m]} frente a {ref}: {t['mean']*100:+.2f} puntos; intervalo simultáneo98,33% [{lo:+.2f},{hi:+.2f}].")
        t=z['mild_harm'];lo,hi=np.array(t['ci95'])*100
        put(f"  Pérdida en casos0–5% frente a DSLPID nuevo: {t['mean']*100:+.2f} puntos; IC95% [{lo:+.2f},{hi:+.2f}]. Rutas con algún perjuicio leve en alguna semilla:{z['distinct_routes_with_any_mild_harm']}/100; cota conservadora95% de ese evento:{z['conservative_route_harm_upper95']*100:.2f}%.")
        chosen=selection[m];put('  Checkpoints elegidos:'+','.join(f"semilla{r['seed']}→{r['steps']} pasos" for r in chosen)+'. El paso0 significa clonación sin mejora atribuible a RL.')
    put('Los intervalos remuestrean tanto las5 semillas de entrenamiento como los100 identificadores de ruta,manteniendo pareados todos los niveles. Se reporta el umbral registrado95% y una versión simultánea98,33% para las3 comparaciones con un mismo referente. Cinco semillas no garantizan estimar bien eventos raros; la cota conservadora por ruta se presenta aparte. No deben tratarse las500 evaluaciones de un nivel como500 geometrías independientes.')
    put();put('DIAGNÓSTICOS DE RECOMPENSA E IMITACIÓN (VALIDACIÓN,NO PRUEBA DE VUELO)')
    rd=read(OUT/'reward_validation_diagnostics.json')['diagnostics'];pd=read(OUT/'policy_validation_diagnostics.json')['rows']
    for m in ['csil','meirl','rime']:
        q=[r['style_preference_accuracy'] for r in rd if r['method']==m]
        put(f"{LABELS[m]}: acierto medio al ordenar segmentos según métricas físicas de estilo:{np.mean(q)*100:.1f}%. Etiquetas calculadas,no humanas;se excluyen empates. No sustituye el éxito en vuelo.")
    for m in METHODS:
        q=[r['residual_rmse_post_detection'] for r in pd if r['method']==m]
        put(f"{LABELS[m]}: RMSE medio de comando residual frente al maestro en estados reservados:{np.mean(q):.5f} (normalizado). Para los métodos PPO se usa el modelo final;para clonación,el inicial. Cambiar la imitación no demuestra por sí mismo empeorar la tarea.")
    put();put('TIEMPOS MEDIDOS EN ESTA PC')
    put(f"Generación de datos:{data['seconds']:.1f}s. Piloto de controladores:{read(TASK/'runs/controller_round3_pilot.json')['seconds']:.1f}s.")
    for seed in SEEDS:put(f"Clonación LSTM semilla{seed}:{read(OUT/f'seed_{seed}/bc_timing.json')['seconds']:.1f}s.")
    for r in timings:
        rt=r['reward_timing'];reward_seconds=float(rt.get('seconds',0))
        if r['method']=='meirl':reward_seconds=float(rt['dynamics']['seconds'])+float(rt['behavioral_clone']['seconds'])+float(rt['discovery_seconds'])
        validation=r.get('validation_seconds');valtext='incluida,sin desglose' if validation is None else f"{validation:.1f}s"
        put(f"{r['method']} semilla{r['seed']}: reward{reward_seconds:.1f}s; política+validación{r['policy_seconds']:.1f}s; validación{valtext}; total del tratamiento{r['total_seconds']:.1f}s; pasos{r['actual_steps']}.")
    put('Hasta tres grupos de semillas se ejecutaron en paralelo. Estos tiempos son duraciones observadas bajo esa carga,no benchmarks aislados. El total por tratamiento incluye inicialización y guardado; la generación de datos y clonación compartida se contabilizan aparte. ME-IRL consume65536 pasos extra de descubrimiento por semilla,además de los262144 del compensador. La primera semilla manual registra validación incluida sin desglose. La comprobación de concurrencia interrumpió RIME55 tras al menos32768 pasos;se reinició ese tratamiento desde la clonación,reutilizando su recompensa congelada. Los tiempos del tratamiento terminado no incluyen el intento descartado;infrastructure_events.json y los registros conservan ese costo. El tiempo de ajuste de reward indicado es el original aunque se haya reutilizado en un reinicio.')
    put();put('VALIDACIÓN, ARTEFACTOS Y LÍMITES')
    xml=ET.parse(TASK/'revision/pruebas_round3.xml').getroot();put(f"Pruebas automatizadas:{sum(int(x.attrib.get('tests',0)) for x in xml.iter('testsuite'))}; revisar revision/pruebas_round3.xml. La evaluación en FP32 con TF32 desactivado reproduce resultados y pasos entre1,4 y16 procesos en35 casos de validación;la diferencia máxima de RMSE es menor de6e-8m (fp32_batch_parity.json). El transporte compacto también coincide con el historial completo (compact_evaluation_parity.json).")
    put('La compatibilidad de la interfaz pública de ganancias se completó durante la ejecución: sus salidas predeterminadas y las del Robust histórico coincidieron exactamente en1000 solicitudes variadas por controlador (api_default_parity.json). No cambió la configuración de entrenamiento.')
    put(f"Se comparan {conceptual_cases} combinaciones modelo-caso. Se conservan {simulated_cases} simulaciones completas y se reutilizan {conceptual_cases-simulated_cases} resultados de actores exactamente idénticos,certificados por hash en frozen_before_test.json y evaluation_reuse.json. Esto evita simular dos veces la misma política;no añade muestras independientes. Los20 modelos al finalizar todo el presupuesto se simulan aparte. No incluye intentos parciales descartados.")
    put('Archivos:modelos best.zip y final.zip,checkpoints de clonación bc.zip,pesos de recompensa reward.pt,protocol.json con sus modificaciones previas a abrir la reserva,provenance.json,datos de episodios y resultados por vuelo. analysis.json contiene estadísticas completas;success_rates.csv y las figuras facilitan revisar o exportar los resultados.')
    put('Las recompensas nuevas son redes,no los once pesos lineales antiguos. El éxito final depende también del DSLPID,las demostraciones,la clonación y las anclas de tarea. No debe atribuirse todo el porcentaje final a IRL. Tampoco deben compararse estos porcentajes directamente con los de las seis arquitecturas anteriores: cambiaron el controlador,la representación residual,la estimación de efectividad y las rutas.')
    put('El máximo físico,la saturación y la imposibilidad de hover convencional al50% de pérdida de RPM no desaparecen aprendiendo un reward. Para vuelo nivelado sin pares ni giro,el límite de este modelo está cerca de29,7% de pérdida de RPM. El50% implica75% de pérdida de empuje del motor afectado. Una solución de emergencia que permita girar y aterrice de forma controlada debe evaluarse como otro objetivo,con equilibrio reducido,estimación y métricas específicas. El piloto de10 casos no demuestra una recuperación validada al50%.')
    put('No se ensayó hardware real,retardo del motor,batería,ruido de IMU,estimación real de falla ni otras distribuciones de error del estimador. La prueba secundaria exacta mide un cambio de condición,pero no sustituye una campaña amplia fuera de distribución. Un resultado débil de estas adaptaciones de PPO no permite descartar los algoritmos originales con sus optimizadores nativos.')
    text='\n'.join(lines)+'\n';(TASK/'INFORME_DSLPID_IRL_PRIMERA_PRUEBA.txt').write_text(text,encoding='utf-8');(OUT/'report.txt').write_text(text,encoding='utf-8')
    print(json.dumps(dict(statistics=statistics,selected_steps={k:[r['steps'] for r in v] for k,v in selection.items()},summary={k:[round(r['success']*100,1) for r in v] for k,v in summary.items()}),ensure_ascii=False),flush=True)
if __name__=='__main__':main()