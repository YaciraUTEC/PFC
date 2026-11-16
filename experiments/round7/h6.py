"""Frozen-study latency, scientific figures and a complete audit report.
Reporting never changes a policy or selects parameters on held-out flights.
"""
import argparse,time,csv,platform
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .common import OUT,ROOT,OLD,LEGACY_OUT,LEVELS,read,save,sha,config,silent,old_model,close_milestone
from .episode import infer
from .policy import TransformerPolicy
from .statistics import paired
from .h5 import verify
from ..round6.env import LandingEnv


def latency():
    verify();torch.set_num_threads(1);start=time.time();chosen=read(OUT/'selected_candidate.json')['entries']
    specs=[('DSLPID',None),('previous_GTrXL',dict(proposer=str(old_model(201)),mode='native'))]+[('candidate_'+str(301+i),s) for i,s in enumerate(chosen)]
    results=[]
    for name,spec in specs:
        dest=OUT/'latency'/name/'complete.json'
        if dest.exists():results.append(read(dest));continue
        model=None if spec is None else TransformerPolicy.from_spec(spec).cpu().double().eval()
        if model is not None:
            from .execution import settings
            if settings()['fast_gtrxl']:
                from .fast_gtrxl import install
                model=install(model)
        elapsed=[];inference=[];base=[];metadata=[];strata=[]
        for seed,level in [(719970,.295),(719971,.296)]:
            with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=2.,fault_clock=4.))
            physics=[0.];original=e._impact_step
            def measured_physics(rpm):
                before=time.perf_counter_ns()
                try:return original(rpm)
                finally:physics[0]+=(time.perf_counter_ns()-before)/1e6
            e._impact_step=measured_physics
            try:
                with silent():obs,_=e.reset(seed=seed,options=dict(level=level))
                state=None;tick=0
                while True:
                    postfault=e.fault_step is not None and not e.touchdown_seen
                    physics[0]=0.;before=time.perf_counter_ns();a,state=infer(model,obs,state);mid=time.perf_counter_ns()
                    diag=getattr(model,'last_diagnostics',{}) if model is not None else {}
                    attempted=bool(diag.get('authorized_delta',torch.zeros(1,4)).abs().max()>.002)
                    obs,_,d,t,info=e.step(a);after=time.perf_counter_ns();tick+=1
                    if tick>24:
                        neural=(mid-before)/1e6;control=(after-mid)/1e6-physics[0]
                        assert control>=0
                        inference.append(neural);base.append(control);elapsed.append(neural+control);strata.append((postfault,attempted,bool(a[5]>0)))
                    if d or t:metadata.append(dict(seed=seed,level=level,outcome=info['outcome'],steps=tick));break
            finally:e.close()
        values=np.asarray(elapsed);result=dict(name=name,samples=len(values),p50_ms=float(np.quantile(values,.5)),p95_ms=float(np.quantile(values,.95)),p99_ms=float(np.quantile(values,.99)),max_ms=float(values.max()),inference_p95_ms=float(np.quantile(inference,.95)),controller_observer_supervisor_p95_ms=float(np.quantile(base,.95)),deadline_ms=1000/48,deadline_miss_fraction=float(np.mean(values>1000/48)),flights=metadata,
            scope='Measured complete inference + control/observer/supervisor/observation bookkeeping in one process; plant integration excluded, first24ticks warmup excluded per flight. Quantiles of the complete chain, not sums of component quantiles; empirical timing, not hard-real-time proof.')
        masks=np.asarray(strata,bool);groups={}
        for j,label in enumerate(['postfault_flight','screen_attempt','applied_intervention']):
            v=values[masks[:,j]]
            groups[label]=dict(samples=len(v),p95_ms=float(np.quantile(v,.95)) if len(v) else None,p99_ms=float(np.quantile(v,.99)) if len(v) else None)
        result['conditional_latency']=groups;result['guarded_p95_ms']=max([result['p95_ms']]+[g['p95_ms'] for g in groups.values() if g['samples']>=24]);result['conditional_scope']='Also inspect active phases so frequent OFF decisions cannot hide intervention cost; guarded p95 includes strata with at least24 samples.'
        save(dest,result);results.append(result);print('FULL_CHAIN_LATENCY',name,round(result['p95_ms'],4),flush=True)
    verify();close_milestone('h6_latency','Complete-chain timing finished without training workers; all five selected replicas measured.',results=results,seconds=time.time()-start)


def load_phase(label):
    folder=OUT/'final'/label;base=read(folder/'baseline.json')
    candidate=[read(folder/f'candidate_{s}'/'results.json') for s in range(301,306)]
    references=[read(folder/f'reference_{s}'/'results.json') for s in range(201,206)] if label!='confirmation' else []
    return base,candidate,references


def rows_table(base,candidates,references):
    rows=[]
    for name,models in [('DSLPID',[base]),('GTrXL anterior',references),('GTrXL ronda7',candidates)]:
        if not models:continue
        for condition in ['exact','estimated']:
            for noise in [1.,2.]:
                for level in LEVELS:
                    per=[];counts=[]
                    for model in models:
                        rr=[r for r in model['episodes'] if r['condition']==condition and r['noise']==noise and r['level']==level]
                        if rr:per.append(float(np.mean([r['stable_landing'] for r in rr])));counts.append(len(rr))
                    if not per:continue
                    rows.append(dict(controller=name,condition=condition,noise=noise,failure_percent=level*100,success_percent=100*np.mean(per),model_min_percent=100*min(per),model_max_percent=100*max(per),model_replicas=len(per),route_clusters=counts[0],logical_flights=sum(counts)))
    return rows


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figures(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    for ax,(condition,noise) in zip(axes.flat,[('exact',1.),('exact',2.),('estimated',1.),('estimated',2.)]):
        for name,color in [('DSLPID','#20252b'),('GTrXL anterior','#bd5c14'),('GTrXL ronda7','#007d9c')]:
            rr=[r for r in table if r['controller']==name and r['condition']==condition and r['noise']==noise];x=np.arange(len(rr));y=[r['success_percent'] for r in rr]
            ax.plot(x,y,'o-',label=name,color=color,lw=2,markersize=4)
            if name!='DSLPID':ax.fill_between(x,[r['model_min_percent'] for r in rr],[r['model_max_percent'] for r in rr],color=color,alpha=.12)
        ax.set_title(('Efectividad exacta' if condition=='exact' else 'Efectividad estimada')+f' | ruido x{noise:g}')
        ax.set_ylim(-3,103);ax.set_xticks(range(len(LEVELS)),[f'{l*100:g}' for l in LEVELS]);ax.set_xlabel('Perdida RPM en un motor (%) - niveles evaluados');ax.set_ylabel('Aterrizaje estable (%)')
    axes[0,0].legend(loc='lower left',fontsize=9);fig.suptitle('Prueba reservada: 64 rutas por condicion y nivel; cinco replicas GTrXL',fontsize=14)
    fig.text(.5,.012,'Banda = minimo/maximo entre replicas (no intervalo de confianza). Niveles en posiciones equidistantes.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.95]);p=OUT/'comparacion_final.png';fig.savefig(p,dpi=180);plt.close(fig)
    smoke=read(OUT/'h1_repair_screen_conclusion.json')['results'];scale=read(OUT/'h1_repaired_scale_conclusion.json');fig,axes=plt.subplots(1,3,figsize=(17,5))
    names={'original':'Mezcla original','teacher_gate':'Solo maestro + compuerta','teacher_physical':'Propuesta fisica separada','teacher_direct':'Salida fisica directa'}
    axes[0].bar(range(4),[r['final_active_mse'] for r in smoke],color=['#a0a5aa','#788b99','#3995a0','#007d9c']);axes[0].set_xticks(range(4),[names[r['recipe']] for r in smoke],rotation=17,ha='right');axes[0].set_ylabel('Error cuadratico de correccion fisica');axes[0].set_title('Ablacion de supervision (1 semilla, 1200 updates)')
    for group in [0,256]:
        values=[r['final_active_mse'] for r in scale['results'] if r['groups']==group];j=[0,256].index(group)
        axes[1].bar(j,np.mean(values),color=['#bd5c14','#007d9c'][j],alpha=.65);axes[1].scatter(np.full(len(values),j)+np.linspace(-.05,.05,len(values)),values,color='black',s=22)
    axes[1].set_xticks([0,1],['Datos anteriores','+256 escenarios']);axes[1].set_ylabel('Error cuadratico de correccion fisica');axes[1].set_title('Supervision reparada (3 semillas, 3600 updates)')
    initial=read(OUT/'h1_proposal_selection_conclusion.json');rr=initial['route_mean_mse_by_scale'];axes[2].plot(range(4),[rr[str(g)] for g in [0,16,64,256]],'o-',color='#bd5c14');axes[2].set_xticks(range(4),['0','16','64','256']);axes[2].set_xlabel('Escenarios adicionales');axes[2].set_ylabel('Error cuadratico de correccion fisica');axes[2].set_title('Supervision original (3 semillas)')
    fig.suptitle('Fidelidad en 8 rutas de maestros reservadas; no mide exito de aterrizaje');fig.tight_layout();p2=OUT/'demostraciones_y_supervision.png';fig.savefig(p2,dpi=180);plt.close(fig)
    return [p,p2]


def trace_diagnostics(path):
    if path is None:return None
    with np.load(path) as z:
        n=len(z['observation']);authorized=z['policy_authorized_delta'] if 'policy_authorized_delta' in z else np.zeros((n,4));final=z['policy_final_delta'] if 'policy_final_delta' in z else np.zeros((n,4));use=np.abs(authorized).max(-1)>.002;requested=np.abs(final).max(-1)>.002;applied=np.abs(z['projected_motor_delta']).max(-1)>1e-7 if 'projected_motor_delta' in z else requested
        first=np.flatnonzero(applied);result=dict(steps=n,authorized_steps=int(use.sum()),final_requested_steps=int(requested.sum()),applied_steps=int(applied.sum()),applied_scope='exact post-saturation motor delta' if 'projected_motor_delta' in z else 'requested delta; older trace lacks exact projection',first_intervention_seconds=float(first[0]/48) if len(first) else None,interpretation='First activation is a timeline marker, not proof that this isolated action caused the terminal failure.')
        for key in ['predicted_harm','predicted_rescue','predicted_base_success']:
            field='policy_'+key
            if field in z and use.any():result[key+'_when_authorized']=dict(mean=float(z[field][use].mean()),max=float(z[field][use].max()),min=float(z[field][use].min()))
        for key in ['screen_veto','interrupted']:
            if 'policy_'+key in z:result[key+'_steps']=int(z['policy_'+key].sum())
        if 'true_eta_before' in z and 'estimated_eta_before' in z:
            err=np.abs(z['true_eta_before']-z['estimated_eta_before']);result['effectiveness_error_max']=float(err.max());result['effectiveness_error_mean_during_intervention']=float(err[applied].mean()) if applied.any() else None;result['effectiveness_error_max_during_intervention']=float(err[applied].max()) if applied.any() else None
        return result


def report():
    verify();assert read(OUT/'h5_frozen_test_conclusion.json')['closed'];lat=read(OUT/'h6_latency_conclusion.json')
    phases={k:load_phase(k) for k in ['primary','confirmation','shifted']};stats={}
    for label,(base,models,refs) in phases.items():
        stats[label]=dict(candidate=paired(models,base),reference=paired(refs,base) if refs else None,
            baseline_success=float(np.mean([r['stable_landing'] for r in base['episodes']])),candidate_success=float(np.mean([r['stable_landing'] for m in models for r in m['episodes']])))
    p=stats['primary']['candidate'];c=stats['confirmation']['candidate'];losses=sum(stats[k]['candidate']['losses'] for k in stats);times=[r for r in lat['results'] if r['name'].startswith('candidate_')]
    development_losses=read(OUT/'selected_candidate.json')['development']['losses']
    accepted=bool(development_losses==0 and p['lower95']>0 and c['lower95']>0 and losses==0 and max(r.get('guarded_p95_ms',r['p95_ms']) for r in times)<1000/48)
    decision=dict(accepted=accepted,development_protection=development_losses==0,development_losses=development_losses,recommendation='GTrXL ronda7' if accepted else 'DSLPID + estimador estructurado + RecoveryDescent; residual OFF',positive_primary=p['lower95']>0,positive_confirmation=c['lower95']>0,zero_new_losses=losses==0,complete_chain_deadline=max(r.get('guarded_p95_ms',r['p95_ms']) for r in times)<1000/48,observed_losses=losses,scope='Decision for this frozen empirical study; not a proof of universal safety or the impossibility of future learned improvements.')
    save(OUT/'deployment_recommendation.json',dict(recommendation=decision['recommendation'],learned_candidate_accepted=accepted,base_controller='DSLPIDControl',base_configuration=config(),residual_policy_specs=read(OUT/'selected_candidate.json')['entries'] if accepted else [],landing_criteria=read(OUT/'frozen_before_final.json')['criteria'],evidence=decision,scope='An empty residual_policy_specs list explicitly means zero neural correction. The separately saved selected_candidate.json is an experimental candidate, not an authorization to deploy it.'))
    table=rows_table(*phases['primary']);write_csv(OUT/'tabla_por_falla.csv',table);plots=figures(table)
    inventory=[];counts=Counter()
    for label,(base,models,refs) in phases.items():
        counts['baseline_flights']+=len(base['episodes'])
        for family,entries in [('candidate',models),('reference',refs)]:
            for index,m in enumerate(entries):
                row=dict(phase=label,family=family,seed=(301 if family=='candidate' else 201)+index,logical_cases=m['logical_cases'],fresh_simulations=m['fresh_simulations'],verified_off_reuse=m['verified_off_reuse'],identical_policy_reuse=m.get('identical_policy_reuse',0),seconds=m['seconds'])
                assert row['logical_cases']==row['fresh_simulations']+row['verified_off_reuse']+row['identical_policy_reuse']
                inventory.append(row)
                for field in ['logical_cases','fresh_simulations','verified_off_reuse','identical_policy_reuse']:counts[field]+=row[field]
    write_csv(OUT/'inventario_evaluacion.csv',inventory)
    timing=[]
    for path in sorted(OUT.glob('*_receipt.json')):
        r=read(path)
        if 'seconds' in r:timing.append(dict(stage=path.stem,seconds=r['seconds'],exit_code=r.get('exit_code'),started_epoch=r.get('started_epoch'),finished_epoch=r.get('finished_epoch'),duration_scope='Wall time for the subprocess stage; may include dependency waits.'))
    for path in sorted((LEGACY_OUT/'proposal_repair').rglob('complete.json')):
        r=read(path);timing.append(dict(stage='retained_H1/'+str(path.parent.relative_to(LEGACY_OUT)),seconds=r['seconds'],exit_code=0,started_epoch=r.get('started_epoch'),finished_epoch=r.get('finished_epoch')))
    for path in sorted(OUT.glob('*_conclusion.json')):
        r=read(path)
        if 'seconds' in r:timing.append(dict(stage=path.stem,seconds=r['seconds'],duration_scope='Duration reported by the milestone itself; consult the receipt for whether acquisition, validation or calibration is included.'))
    for path in sorted(LEGACY_OUT.glob('*_receipt.json')):
        r=read(path)
        if 'seconds' in r:timing.append(dict(stage='preliminary_snapshot_diagnostic/'+path.stem,seconds=r['seconds'],exit_code=r.get('exit_code'),duration_scope='Preliminary labeling/evaluation before the natural-execution restoration check; not final-policy evidence.'))
    for folder in ['bc_curve','refinement']:
        for path in sorted(((LEGACY_OUT if folder=='bc_curve' else OUT)/folder).rglob('complete.json')):
            r=read(path);end=r.get('finished_epoch',path.stat().st_mtime);timing.append(dict(stage=str(path.parent),seconds=r['seconds'],exit_code=0,started_epoch=r.get('started_epoch',end-r['seconds']),finished_epoch=end,epoch_scope='File modification time used where the original receipt retained duration only.'))
    for path in sorted(OUT.glob('demonstration_audit_*.json')):
        r=read(path);end=path.stat().st_mtime;timing.append(dict(stage=path.stem,seconds=r['seconds'],exit_code=0,started_epoch=end-r['seconds'],finished_epoch=end,epoch_scope='File modification time; reuses existing pilot groups.'))
    save(OUT/'timing_audit.json',dict(stages=timing,scope='Stage durations may overlap; do not sum them as wall-clock time. Process receipts may include dependency waits (not pure training). Milestone and fit receipts give their measured computation/validation scopes. Original BC fits include full-flight validation; repairs include action-fidelity validation. Demonstration generation includes pilot reuse.'))
    failures=[]
    for label,(base,models,refs) in phases.items():
        lookup={tuple(r[k] for k in ['seed','level','condition','noise','clock']):r for r in base['episodes']}
        for seed,m in zip(range(301,306),models):
            for r in m['episodes']:
                b=lookup[tuple(r[k] for k in ['seed','level','condition','noise','clock'])]
                if b['stable_landing']!=r['stable_landing']:
                    failures.append(dict(phase=label,model_seed=seed,route=r['seed'],level=r['level'],condition=r['condition'],noise=r['noise'],clock=r['clock'],kind='rescue' if r['stable_landing'] else 'loss',outcome=r['outcome'],trace_path=r.get('trace_path'),first_contact=r.get('first_contact'),steps=r['episode_steps'],stable_seconds=r['stable_seconds'],policy_diagnostics=trace_diagnostics(r.get('trace_path'))))
    save(OUT/'changed_outcomes.json',failures)
    support={}
    for split in ['train','calibration']:
        records=[r for path in sorted((OUT/'gate_data'/split).glob('*.json')) for r in read(path).get('records',[])]
        summary=read(OUT/f'gate_data_{split}_complete.json');assert len(records)==summary['samples']
        query=lambda r:(r['seed'],r['level'],r['tick'],r['proposer'])
        classes={}
        for label,name in enumerate(['both_success','rescue','harm','both_fail']):
            rr=[r for r in records if r['label']==label]
            classes[name]=dict(paired_option_future_records=len(rr),query_states=len({query(r) for r in rr}),route_clusters=len({r['seed'] for r in rr}))
        support[split]=dict(route_clusters=len({r['seed'] for r in records}),query_states=len({query(r) for r in records}),option_queries=len({(*query(r),r['option']) for r in records}),paired_option_future_records=len(records),classes=classes,scope='Classes can coexist at the same queried state under different actions/futures. Route clusters, query states, option queries and future records are distinct counts; only routes are independent scenario clusters.')
    save(OUT/'label_support_audit.json',support)
    result=dict(decision=decision,statistics=stats,latency=lat,execution_counts=dict(counts),selection=read(OUT/'selected_candidate.json'),label_support=support,plots=[str(p) for p in plots],test_receipt=read(OUT/'h5_tests_receipt.json'),frozen_integrity_verified=True,execution_settings=read(OUT/'execution_settings.json'))
    save(OUT/'final_results.json',result)
    quantity=read(OUT/'h1_repaired_scale_conclusion.json');means=quantity['means'];data_audit=read(OUT/'demonstration_audit_256.json');gate_audit=read(OUT/'h2_gate_model_prediction_diagnostics.json');h3=read(OUT/'h3_filter_ablation_conclusion.json');h4=read(OUT/'h4_protected_finetuning_conclusion.json')
    lines=['RONDA 7: GTRXL, DEMOSTRACIONES Y AUTORIZACION','', 'DECISION: '+decision['recommendation'], 'Candidato supera todos los criterios prospectivos: '+str(accepted),'',
        'Falla (%) significa perdida de RPM de un motor; la fuerza depende aproximadamente del cuadrado de esa velocidad. Exito: contacto dentro de todos los limites y 1s estable antes de30s; criterio sin cambios.',
        'Solo se trabajo con Transformer/GTrXL. La arquitectura base de control permanece congelada. No se entreno Mamba ni se representa el afinado supervisado como un nuevo entrenamiento online FlashSAC.','',
        'H1: ampliar datos con etiquetas incompatibles produjo colapso. Se verificaron84 pares de entrada identica con orden base0 y orden maestra distinta; eliminar esos pares solamente no basto. La ablacion separo la perdida fisica de la autorizacion y evaluo una salida de cuatro correcciones fisicas.',
        'Curva original0/16/64/256 y reparacion conservadas completas. La seleccion de cantidad uso fidelidad en maestros de8 rutas independientes. Eso no autoriza a concluir que el error de accion equivale a exito de vuelo.',
        'H2: etiquetas de continuaciones completas y estados naturales reconstruidos por el historial exacto de acciones, dos futuros por consulta, opciones de politica realimentada de1/24/96ticks y hasta terminal, con veto nativo y vuelta a base al cerrar opciones breves. La red no recibe identidad de ruta, motor fallado ni efectividad verdadera en modo estimado.',
        'Verificacion adicional: save/restore de PyBullet no conservaba el siguiente paso natural. Se rechazo esa fuente de etiquetas, se conservaron los diagnosticos previos y se regenero toda autorizacion por prefijos reproducidos exactamente, sin modificar el vuelo consultado.',
        'H3: ablacion propuesta/compuerta/filtro. El filtro es una aproximacion predictiva comparativa; no se atribuyen garantias formales de alcanzabilidad.',
        'H4: una ronda de afinado supervisado de cabezas, encoder congelado y anclaje al modelo anterior, seguida de agregacion de estados del alumno compuesto seleccionado y recalibracion; ablacion2x2 cabeza/compuerta cuando procede. No es una reproduccion completa de DAgger ni un ensayo de un nuevo algoritmo SAC.',
        'H5: fuentes, modelos y criterio congelados antes de64 rutas principales,320 de confirmacion y32 con otros instantes; ninguna seleccion sobre esos resultados. Cinco replicas neuronales; niveles/ruidos repetidos se agrupan por ruta para incertidumbre.','']
    lines+=['RESUMEN DE HITOS:',f"H1: {data_audit['groups']} escenarios nuevos, {data_audit['successful_proposals']} episodios maestros exitosos, {data_audit['active_demonstration_episodes']} demostraciones activas. MSE con supervision reparada: datos anteriores {means['0']:.8f}, ampliados {means['256']:.8f}; mejora {100*(1-means['256']/means['0']):.2f}%.",'H2: diagnostico probabilistico '+str(gate_audit['heldout']), 'H3: configuracion seleccionada en ablacion y control de continuidad no negativa '+h3['selected']['name']+'; metricas '+str(h3['selected']['metrics']), 'H4: familia seleccionada '+h4['selected_family']+'; metricas '+str(h4['selected']['metrics']), 'Fuente del candidato final: '+str(result['selection']['source'])+'; configuracion '+str(result['selection']['entries'][0]),'']
    lines+=['DETALLE DE DESARROLLO (no son las rutas de prueba):']
    for row in read(OUT/'h2_counterfactual_gate_conclusion.json')['results']:
        m=row['metrics'];lines.append(f"  H2 {row['name']}: exito {100*m['success']:.3f}%, rescates {m['gains']}, perdidas {m['losses']}, diferencia critica {100*m['critical_difference']:.3f}pp.")
    for row in h3['results']:
        m=row['metrics'];lines.append(f"  H3 {row['name']}: exito {100*m['success']:.3f}%, rescates {m['gains']}, perdidas {m['losses']}, diferencia critica {100*m['critical_difference']:.3f}pp.")
    for name,row in h4['comparisons'].items():
        lines.append('  H4 '+name+': '+str(row['metrics']))
    for split in ['train','calibration']:
        audit=read(OUT/f'gate_data_{split}_complete.json');lines.append(f"  Etiquetas finales {split}: {audit['groups']} grupos, {audit['samples']} comparaciones, clases {audit['labels']}, {audit['prefix_replay_checks']} comprobaciones de prefijo, error maximo {audit['maximum_prefix_error']}; los futuros y opciones no son rutas independientes.")
    lines+=['Soporte efectivo de las etiquetas: '+str(support), 'Dos futuros no certifican una probabilidad de dano del0.5%. El umbral de probabilidad aprendido y la penalizacion por desacuerdo son reglas de seleccion, no cotas de seguridad demostradas.', '', 'RESULTADOS EN RUTAS RESERVADAS:']
    for label in stats:
        x=stats[label]['candidate'];lines.append(f"{label}: diferencia critica estimada {100*x['mean']:.3f} pp, IC95% [{100*x['lower95']:.3f}, {100*x['upper95']:.3f}]; rescates {x['gains']}, perdidas {x['losses']} (exacta {x['exact_losses']}, leves {x['mild_losses']}); rutas {x['route_clusters']}, replicas5.")
        lines.append(f"  Rutas protegidas perjudicadas {x['harmed_route_clusters']}/{x['protected_route_clusters']}; cota superior unilateral95% {100*x['harmed_route_upper95']:.3f}% (independencia de rutas, no garantia universal).")
        if stats[label]['reference'] is not None:
            ref=stats[label]['reference'];lines.append(f"  Referencia GTrXL+FlashSAC anterior: diferencia critica estimada {100*ref['mean']:.3f}pp, IC95% [{100*ref['lower95']:.3f}, {100*ref['upper95']:.3f}]; rescates {ref['gains']}, perdidas {ref['losses']}.")
    lines+=['', 'LATENCIA:']+[f"{r['name']}: cadena completa p95={r['p95_ms']:.3f} ms, p99={r['p99_ms']:.3f} ms; p95 protegido por fases={r.get('guarded_p95_ms',r['p95_ms']):.3f} ms; limite20.833ms." for r in lat['results']]
    lines+=['', 'EJECUCIONES: '+str(dict(counts)), 'Reuso OFF exige identidad de accion comprobada en toda la historia base. No se cuentan registros logicos reutilizados como nuevas simulaciones independientes. Si la seleccion coincide exactamente con la referencia congelada, su evaluacion se comparte y se contabiliza aparte como identical_policy_reuse.',
        'Fuentes teoricas: DAgger https://proceedings.mlr.press/v15/ross11a.html ; DPRL https://proceedings.mlr.press/v258/sharma25a.html ; gatekeeper https://arxiv.org/abs/2508.09963 ; filtros predictivos https://arxiv.org/abs/1812.05506 . Se emplearon ideas, no se trasladaron garantias sin sus hipotesis.',
        '', 'Archivos: deployment_recommendation.json (configuracion recomendada), final_results.json, label_support_audit.json, tabla_por_falla.csv, comparacion_final.png, demostraciones_y_supervision.png, changed_outcomes.json, inventario_evaluacion.csv, timing_audit.json.']
    (OUT/'INFORME_FINAL.txt').write_text('\n'.join(lines),encoding='utf-8')
    close_milestone('h6_report','All frozen comparisons, uncertainty, changed outcomes, latency and actual-execution accounting are reported.',decision=decision,plots={str(p):sha(p) for p in plots})
    print('FINAL_DECISION',decision,flush=True)


def finish():
    verify();report=read(OUT/'h6_report_conclusion.json');visual=read(OUT/'presentation_verified.json')
    for path,digest in report['plots'].items():assert visual['plots'][path]==digest and sha(path)==digest
    required=['h0_reproduction','h0_gate_pilot','h0_natural_fork','h0_composed_method','h0_composed_acquisition_pilot','h2_natural_labels','h2_execution_equivalence','h1_supervision_repair','h2_counterfactual_gate','h3_filter_ablation','h3_continuation_ablation','h4_protected_finetuning','h5_execution_preflight','h5_frozen_test','h6_latency','h6_report']
    for name in required:assert read(OUT/(name+'_conclusion.json'))['closed'],name
    assert read(OUT/'h5_tests_receipt.json')['exit_code']==0
    save(OUT/'ALL_COMPLETE.json',dict(complete=True,finished_epoch=time.time(),remaining_required_work=0,decision=read(OUT/'final_results.json')['decision'],milestones=required))
    save(OUT/'status.json',dict(stage='all_complete',status='complete',remaining_required_work=0));print('ROUND7_ALL_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['latency','report','finish'],required=True);a=p.parse_args();globals()[a.phase]()
