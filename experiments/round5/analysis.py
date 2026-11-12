"""Frozen decision rules and full paired analysis of the completed campaign."""
import csv,json,time,math,datetime
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from .evaluate import OUT
from .inventory import TEST_LEVELS,core,sha
from .statistics import crossed_simultaneous,compare_confirmation
from ..run import save

def read(path):return json.loads(Path(path).read_text())
def keyed(rows):return {(r['level'],r['seed']):r for r in rows}
def finite_mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None];return None if not values else float(np.mean(values))
def csvfile(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        fields=list(rows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main_rates():
    manifest=read(OUT/'final_test_manifest.json')['logical'];aggregate=defaultdict(list);paired=defaultdict(lambda:dict(gains=0,losses=0,loss_rows=[]))
    for request in manifest:
        kind=request['kind'];mode=request.get('mode','baseline');name=request['name']
        for condition in request['conditions']:
            rows=read(OUT/'test'/f"{request['label']}_{condition}.json")
            baseline=keyed(read(OUT/'test'/f'baseline_selected_{condition}.json'))
            for row in rows:
                key=(kind,name,mode,condition,row['level']);aggregate[key].append(dict(row,training_seed=request['seed'],zero_policy=request['zero_policy']))
                ref=baseline[(row['level'],row['seed'])]['stable_landing'];p=paired[key]
                if row['stable_landing'] and not ref:p['gains']+=1
                if ref and not row['stable_landing']:p['losses']+=1;p['loss_rows'].append(row)
    rates=[];failure=[]
    for key,rows in sorted(aggregate.items()):
        kind,name,mode,condition,level=key;p=paired[key];losses=p['loss_rows']
        rate=dict(kind=kind,name=name,mode=mode,condition=condition,level=level,success_percent=100*float(np.mean([r['stable_landing'] for r in rows])),
            model_cases=len(rows),route_geometries=len({r['seed'] for r in rows}),training_seeds=0 if kind=='baseline' else len({r['training_seed'] for r in rows}),
            gains_vs_selected=p['gains'],losses_vs_selected=p['losses'],zero_policy_cases=sum(r['zero_policy'] or r.get('zero_replay_certified',False) for r in rows),globally_zero_policy_cases=sum(r['zero_policy'] for r in rows),
            applied_on_percent=100*finite_mean(rows,'applied_on_fraction'),mean_amplitude=finite_mean(rows,'mean_gate'),
            eta_mae_all_motors=finite_mean(rows,'eta_mae'),applied_eta_mae_all_motors=finite_mean(rows,'applied_eta_mae'),effectual_percent=100*finite_mean(rows,'effectual_fraction'),mean_q_delta_energy=finite_mean(rows,'q_delta_energy'),mean_prefault_amplitude=finite_mean(rows,'gate_prefault'),mean_postfault_amplitude=finite_mean(rows,'gate_postfault'),mean_prefault_filter_false_alarm_ticks=finite_mean(rows,'pre_fault_false_alarm_ticks'),failed_motor_eta_mae=finite_mean(rows,'failed_motor_eta_mae'),applied_failed_motor_eta_mae=finite_mean(rows,'applied_failed_motor_eta_mae'),
            tracking_rmse=finite_mean(rows,'tracking_rmse'),mean_episode_seconds=finite_mean(rows,'episode_steps')/48,estimated_95pct_coverage=finite_mean(rows,'estimated_95pct_coverage'),estimate_within_2pct_fraction=float(np.mean([r.get('first_estimate_within_2pct_seconds') is not None for r in rows])),conditional_estimation_recovery_seconds=finite_mean(rows,'first_estimate_within_2pct_seconds'))
        rates.append(rate)
        violations=Counter();yaw_dominant=0;off_before=0
        for r in losses:
            first=r.get('first_contact')
            if first:
                for field,limit in [('vertical_speed',.25),('horizontal_speed',.25),('tilt',math.radians(12)),('angular_speed',1.),('radius',.2)]:violations[field]+=int(first[field]>limit)
                omega=np.asarray(first.get('body_angular_velocity',[0.,0.,0.]));norm2=float(omega@omega)
                yaw_dominant+=int(norm2>0 and omega[2]**2/norm2>=.9)
            off_before+=int(r.get('precontact_trace_available',False) and r['precontact_applied_on_fraction']==0)
        failure.append(dict(kind=kind,name=name,mode=mode,condition=condition,level=level,paired_losses=len(losses),
            angular_violations=violations['angular_speed'],horizontal_violations=violations['horizontal_speed'],vertical_violations=violations['vertical_speed'],radius_violations=violations['radius'],tilt_violations=violations['tilt'],
            yaw_dominant_losses=yaw_dominant,off_during_last_precontact_second=off_before,outcomes=dict(Counter(r['outcome'] for r in losses))))
    csvfile('success_rates.csv',rates);save(OUT/'failure_analysis.json',failure);return rates

def family_intervals():
    from .inventory import groups
    baseline=keyed(read(OUT/'test/baseline_selected_estimated.json'));deltas=[];names=[]
    for group in groups():
        seed_values=[]
        for entry in group['entries']:
            rows=keyed(read(OUT/'test'/f"{group['id']}_{entry['seed']}_raw_estimated.json"))
            seed_values.append([[int(rows[(level,seed)]['stable_landing'])-int(baseline[(level,seed)]['stable_landing']) for level in [.28,.29,.295]] for seed in range(529000,529060)])
        deltas.append(seed_values);names.append(group['id'])
    result=crossed_simultaneous(deltas)
    for label,row in zip(names,result['contrasts']):row['group']=label
    save(OUT/'family_simultaneous_intervals.json',result);return result

def sensitivity_gain(candidate,reference):
    result={}
    for mode in ['noise2','clock4']:
        c=keyed(read(OUT/'sensitivity'/f'production_{candidate}_{mode}.json'));r=keyed(read(OUT/'sensitivity'/f'production_{reference}_{mode}.json'))
        assert set(c)==set(r)
        result[mode]=float(np.mean([int(c[k]['stable_landing'])-int(r[k]['stable_landing']) for k in r]))
    return result

STRUCTURED_ORDER=['baseline_observer_only','baseline_selected','baseline_balanced']
def comparison_pairs(ids):
    refs=['baseline_original']+[name for name in STRUCTURED_ORDER if name in ids]
    pairs=[(name,reference) for j,name in enumerate(refs) for reference in refs[:j]]
    pairs.extend((name,reference) for name in ids if not name.startswith('baseline_') for reference in refs)
    return pairs

def choose_structured(ids,comparisons):
    accepted='baseline_original'
    for name in STRUCTURED_ORDER:
        if name in ids and comparisons[f'{name}/{accepted}']['adoption_pass']:accepted=name
    return accepted

def decide():
    manifest=read(OUT/'confirmation_manifest.json');specifications=manifest['specifications'];spec={s['id']:s for s in specifications};latency={r['id']:r for r in read(OUT/'latency_complete.json')['results']}
    confirmed={name:read(OUT/'confirmation'/f'{name}.json') for name in spec};references=[name for name in spec if name.startswith('baseline_')];comparisons={}
    pairs=comparison_pairs(spec);assert len(pairs)*3<=36,'Predeclared endpoint budget exceeded'
    for name,reference in pairs:
        comparison=compare_confirmation(confirmed[name],confirmed[reference]);sensitivity=sensitivity_gain(name,reference)
        comparison.update(candidate=name,reference=reference,sensitivity_gain=sensitivity,sensitivity_pass=all(v>=-1e-12 for v in sensitivity.values()),latency_pass=latency[name]['realtime_pass'])
        comparison['adoption_pass']=comparison['statistical_pass'] and comparison['sensitivity_pass'] and comparison['latency_pass']
        comparisons[f'{name}/{reference}']=comparison
    original='baseline_original';accepted_base=choose_structured(spec,comparisons)
    learned=[name for name in spec if not name.startswith('baseline_') and comparisons[f'{name}/{accepted_base}']['adoption_pass']]
    accepted=accepted_base
    if learned:accepted=max(learned,key=lambda name:(comparisons[f'{name}/{accepted_base}']['success'],-latency[name]['p95_seconds']))
    result=dict(accepted_structured_reference=accepted_base,recommended=accepted,recommended_specification=spec[accepted],
        comparisons=comparisons,accepted_learning_candidates=learned,rejected_learning_candidates=[name for name in spec if not name.startswith('baseline_') and name not in learned],
        structured_rule='Accept estimator-only first if it passes against original;accept full selectedcore only if it passes against that accepted reference.Then evaluate learned controllers against accepted structured reference.',rule='Predeclared confirmed primary gain>0 lower bound;conditional harm upper<1percent in both protected mixtures;no negative sensitivity net gain;full control p95<=1/48s.',
        interpretation='Recommendation for the concrete frozen configuration and this simulation/sensor/fault distribution.Not an intrinsic ranking of architecture families.')
    save(OUT/'decision.json',result)
    chosen=spec[accepted];bundle=dict(id=accepted,configuration=chosen['config'],checkpoint=chosen['path'],checkpoint_sha256=None if chosen['path'] is None else sha(chosen['path']),
        gate_logit_threshold=chosen['threshold'],proposal_only=chosen['shadow'],judge_path=chosen['config'].get('judge_path'),
        judge_sha256=None if not chosen['config'].get('judge_path') else sha(chosen['config']['judge_path']),inference=dict(device='cpu',dtype='float64',action_dtype='float32'),
        decision_file=str(OUT/'decision.json'),primary_levels=[.28,.29,.295],maximum_training_fault_rpm_fraction=.295,deadline_seconds=30)
    save(OUT/'recommended_controller.json',bundle);return result

def timings():
    model_rows=[]
    from .inventory import groups
    for group in groups():
        for entry in group['entries']:
            directory=Path(entry['directory'])
            for phase,file in [('bc','bc_timing.json'),('ppo','complete.json')]:
                path=directory/file
                if not path.exists():continue
                r=read(path);model_rows.append(dict(kind=group['kind'],name=group['name'],seed=entry['seed'],phase=phase,seconds=r['seconds'],validation_seconds=r.get('validation_seconds',0),updates=r.get('updates'),environment_steps=r.get('steps'),selected_steps=r.get('selected_steps'),reward_seconds=r.get('reward_seconds',0),setup_seconds=r.get('setup_seconds',0),ppo_seconds=r.get('ppo_seconds')))
    csvfile('training_timings.csv',model_rows)
    stages={}
    for name in ['collect_initial','bc_pilots','collect_dagger','hpo']:
        path=OUT/f'{name}_stage.json'
        if path.exists():stages[name]=read(path)['seconds']
    for label,file in [('runtime_selection','runtime_selection_complete.json'),('actual_proposal_oracle','oracle_proposal_audit_complete.json'),('inference_audit','inference_audit_complete.json'),('final_evaluation','final_test_complete.json'),('confirmation','confirmation_complete.json'),('sensitivity','sensitivity_complete.json')]:
        if (OUT/file).exists():stages[label]=read(OUT/file)['seconds']
    for label in ['recommended_replay','latency']:
        receipt=OUT/f'{label}_receipt.json'
        if receipt.exists():stages[label]=read(receipt)['seconds']
    if (OUT/'judge_calibration.json').exists():stages['judge_fit']=read(OUT/'judge_calibration.json').get('seconds')
    for label,file in [('core_engineering_valid_version','core_selection.json'),('continuous_governor_validation','structural_challenger.json'),('estimator_only_validation','observer_fallback_selection.json'),('paired_physics_restoration','paired_physics_audit.json'),('final_unit_suite','tests_final_receipt.json')]:
        if (OUT/file).exists():stages[label]=read(OUT/file).get('seconds')
    first=read(OUT/'collect_initial_stage.json');start=datetime.datetime.fromisoformat(first['finished_utc']).timestamp()-first['seconds']
    stages['calendar_elapsed_initial_data_to_reporting']=time.time()-start
    receipts=[r for w in [0,1] for r in read(OUT/f'worker_{w}_receipts.json')]
    intervals=sorted((r['started_epoch'],r['finished_epoch']) for r in receipts);total=0.;left=None;right=None
    for a,b in intervals:
        if left is None:left,right=a,b
        elif a<=right:right=max(right,b)
        else:total+=right-left;left,right=a,b
    if left is not None:total+=right-left
    stages['full_training_wall_union']=total;stages['training_process_seconds_sum']=sum(r['seconds'] for r in receipts)
    save(OUT/'stage_timings.json',dict(stages=stages,model_count=len({(r['kind'],r['name'],r['seed']) for r in model_rows}),interpretation='Model times include simulation;validation separated where available.Wall union accounts for overlap of the two workers.Process-time sum is not calendar duration.'))

def architecture_profiles():
    from .models import load_agent
    profiles=[]
    from .inventory import groups
    for group in groups():
        if group['kind']!='residual':continue
        entry=group['entries'][0];m=load_agent(entry['path'])
        profiles.append(dict(architecture=group['name'],configuration=m.cfg,total_parameters=sum(p.numel() for p in m.parameters()),
            policy_encoder_parameters=sum(p.numel() for p in m.encoder.parameters()),actor_parameters=sum(p.numel() for p in m.actor.parameters()),critic_parameters=sum(p.numel() for p in m.critic.parameters()),
            selected_hpo=read(OUT/'hpo'/f"{group['name']}_selection.json"),persistent_episode_memory=True,sequence_ticks=96,burnin_ticks=32))
    save(OUT/'architecture_profiles.json',profiles)

def physical_limits():
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
    c=DSLPIDControl(DroneModel.CF2X);eta=float(c.HOVER_RPM/c._command_max)
    levels=[]
    for level in [.25,.28,.29,.295,.296,.30,.35,.50]:
        ratio=float(((1-level)*c._command_max/c.HOVER_RPM)**2)
        levels.append(dict(rpm_loss=level,affected_motor_thrust_loss=1-(1-level)**2,balanced_max_thrust_to_weight=ratio,
            ideal_static_hover_tilt_degrees=None if ratio<1 else float(np.degrees(np.arccos(1/ratio)))))
    save(OUT/'physical_limits.json',dict(mass_kg=c.MASS,gravity_newtons=c.GRAVITY,kf=c.KF,max_rpm=c._command_max,hover_rpm=c.HOVER_RPM,
        static_normal_hover_rpm_loss_limit=1-eta,levels=levels,assumptions='Fixed modeled mass,zero roll/pitch/yaw moments,normal hover,single weak motor,bounded RPM;not a reachability bound for transient descent or spinning flight.'))

def proposal_diagnostics():
    counts=defaultdict(Counter);queries={}
    for file in sorted((OUT/'data').glob('*/route_*.json')):
        for r in read(file)['branches']:
            arch='initial' if r['student_model'] is None else Path(r['student_model']).parent.name
            key=(r['split'],r['mode'],r['source'],arch);c=counts[key]
            c['proposals']+=1;c['beneficial']+=int(r['beneficial']);c['negative_gain']+=int(r['gain']<0)
            before=r['base_outcome']=='stable_landing';after=r['outcome']=='stable_landing'
            c['binary_rescues']+=int(after and not before);c['binary_losses']+=int(before and not after)
            c['both_stable']+=int(before and after)
            qkey=(r['split'],r['mode'],r['student_model'],r['seed'],r['condition'],r['level'],r['t'])
            q=queries.setdefault(qkey,dict(base_failure=not before,can_rescue=False,any_benefit=False))
            q['can_rescue']|=after and not before;q['any_benefit']|=r['beneficial']
    rows=[dict(split=k[0],mode=k[1],source=k[2],student_architecture=k[3],**v) for k,v in sorted(counts.items())]
    summary=dict(rows=rows,unique_query_modes=len(queries),baseline_failure_queries=sum(q['base_failure'] for q in queries.values()),queries_with_rescue_proposal=sum(q['can_rescue'] for q in queries.values()),queries_with_beneficial_proposal=sum(q['any_benefit'] for q in queries.values()),
        interpretation='Training/calibration data only;multiple correlated queries per route.An oracle among the available12tick proposals followed by base control,not the optimal control law or final-test evidence.')
    save(OUT/'paired_proposal_diagnostics.json',summary)

def main():
    start=time.time();rates=main_rates();intervals=family_intervals();decision=decide();timings();architecture_profiles();physical_limits();proposal_diagnostics()
    save(OUT/'analysis_complete.json',dict(seconds=time.time()-start,rate_rows=len(rates),family_contrasts=len(intervals['contrasts']),recommended=decision['recommended'],completed=True))
    print('DECISION',decision['recommended'],'accepted_base',decision['accepted_structured_reference'],flush=True)
if __name__=='__main__':main()
