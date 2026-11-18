"""Post-hoc read-only analysis of closed R8 outcomes; no policy training/selection."""
import json,csv,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import beta,binomtest
ROOT=Path('E:/UTEC/Tesis/TesisNicol');SRC=ROOT/'PFC_round8/runs/gtrxl_round8';DEST=ROOT/'ANALISIS_RESCATE_PERJUICIO_20261117';INPUT={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);INPUT[str(p)]=digest(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,obj):(DEST/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def key(r):return tuple(r[k] for k in ['seed','level','condition','noise','clock'])
def scope(row,s):return s=='global' or (row['condition']=='estimated' and row['level'] in [.295,.296])
def summarise(b,c,s):
 rows=[x for x in b['episodes'] if scope(x,s)];cc={key(x):x for x in c['episodes']};assert len(cc)==len(c['episodes']);g=[];l=[];bs=0
 for x in rows:
  y=cc[key(x)];bs+=bool(x['stable_landing'])
  if not x['stable_landing'] and y['stable_landing']:g.append(key(x))
  if x['stable_landing'] and not y['stable_landing']:l.append(key(x))
 n=len(rows);bf=n-bs;gg=len(g);ll=len(l);h=ll/bs if bs else None;r=gg/bf if bf else None
 return dict(cases=n,routes=len({x['seed'] for x in rows}),base_successes=bs,base_failures=bf,gains=gg,losses=ll,rescue_percent=100*r if r is not None else None,harm_percent=100*h if h is not None else None,net_success_gain_pp=100*(gg-ll)/n,baseline_failure_percent=100*bf/n,base_success_percent=100*bs/n,candidate_success_percent=100*(bs+gg-ll)/n,rescued_routes=len({k[0] for k in g}),harmed_routes=len({k[0] for k in l}),loss_outcomes=dict(Counter(cc[k]['outcome'] for k in l))),set(g),set(l)
def csvwrite(name,rows):
 with (DEST/name).open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
frozen=SRC/'final/attempt_1/frozen.json';frozen_sha=digest(frozen);datasets={};tables=[]
dev=SRC/'development/checkpoint_selection/with_severity_sweep'
for label,family in [('dev_guarded505','guarded_raw'),('dev_unguarded505','expanded_raw')]:
 b=read(dev/family/'baseline.json');c=read(dev/family/'seed_505/results.json');assert {key(x) for x in b['episodes']}=={key(x) for x in c['episodes']};datasets[label]=(b,c)
 for s in ['global','critical_estimated']:
  m,_,_=summarise(b,c,s);tables.append(dict(dataset=label,gate=505,scope=s,**{k:v for k,v in m.items() if k!='loss_outcomes'}))
for phase in ['primary','confirmation']:
 b=read(SRC/f'final/attempt_1/{phase}/baseline.json')
 for gate in range(501,506):
  c=read(SRC/f'final/attempt_1/{phase}/gate_{gate}/results.json');datasets[f'{phase}_{gate}']=(b,c)
  for s in ['global','critical_estimated']:
   m,_,_=summarise(b,c,s);tables.append(dict(dataset=phase,gate=gate,scope=s,**{k:v for k,v in m.items() if k!='loss_outcomes'}))
csvwrite('tasas_condicionales.csv',tables)
overlaps={}
for name,a,z in [('development_guard_vs_no_guard','dev_guarded505','dev_unguarded505'),('confirmation505_vs504','confirmation_505','confirmation_504')]:
 b,c=datasets[a];_,d=datasets[z];assert {key(x):x['stable_landing'] for x in b['episodes']}=={key(x):x['stable_landing'] for x in datasets[z][0]['episodes']}
 ma,ga,la=summarise(b,c,'global');mz,gz,lz=summarise(b,d,'global');ca={key(x):x for x in c['episodes']};cz={key(x):x for x in d['episodes']}
 overlaps[name]=dict(first=a,second=z,first_metrics=ma,second_metrics=mz,shared_rescues=len(ga&gz),second_only_rescues=len(gz-ga),first_only_rescues=len(ga-gz),second_minus_first_net_gains=mz['gains']-ma['gains'],second_minus_first_losses=mz['losses']-ma['losses'],utility_cost_break_even=(mz['gains']-ma['gains'])/(mz['losses']-ma['losses']),direct_second_wins=sum(not x['stable_landing'] and cz[k]['stable_landing'] for k,x in ca.items()),direct_second_loses=sum(x['stable_landing'] and not cz[k]['stable_landing'] for k,x in ca.items()))
# Fair independent episode subsample: conditions assigned before outcomes for op505.
alloc=read(SRC/'final/attempt_1/allocations.json');assigned=[x for x in alloc['assigned'] if 850000<=x['seed']<851024];assert len(assigned)==1024 and len({x['seed'] for x in assigned})==1024
independent=[]
for gate in range(501,506):
 b,c=datasets[f'confirmation_{gate}'];bb={key(x):x for x in b['episodes']};cc={key(x):x for x in c['episodes']};bs=[bb[key(x)]['stable_landing'] for x in assigned];cs=[cc[key(x)]['stable_landing'] for x in assigned];n=sum(bs);g=sum(not x and y for x,y in zip(bs,cs));l=sum(x and not y for x,y in zip(bs,cs));p=binomtest(g,g+l,.5,alternative='greater').pvalue
 upper=lambda alpha:float(beta.ppf(1-alpha,l+1,n-l)) if l<n else 1.
 independent.append(dict(gate=gate,scenarios=1024,baseline_successes=n,baseline_failures=1024-n,gains=g,losses=l,rescue_percent=100*g/(1024-n),harm_percent=100*l/n,harm_upper95_percent=100*upper(.05),harm_upper_family95_five_models_percent=100*upper(.05/5),harm_upper_joint95_five_models_two_claims_percent=100*upper(.05/10),mcnemar_one_sided_p_fixed_sample=float(p),benefit_p_bonferroni_five_models=min(1.,5*float(p))))
csvwrite('submuestra_independiente_confirmacion.csv',independent)
lookup504={key(x):x for x in datasets['confirmation_504'][1]['episodes']};lookup505={key(x):x for x in datasets['confirmation_505'][1]['episodes']}
wins=sum(lookup504[key(x)]['stable_landing'] and not lookup505[key(x)]['stable_landing'] for x in assigned);losses=sum(lookup505[key(x)]['stable_landing'] and not lookup504[key(x)]['stable_landing'] for x in assigned);direct_p=float(binomtest(wins,wins+losses,.5,alternative='greater').pvalue)
independent_direct=dict(scenarios=len(assigned),wins504=wins,losses504=losses,net504_minus505=wins-losses,one_sided_p=direct_p,bonferroni_four_comparisons_to_development_selected505_p=min(1.,4*direct_p),scope='Exploratory fixed-sample exact paired comparison, same preassigned condition per route. Adjusted over four alternative frozen gates versus the previously selected505; does not certify a harm tolerance or adjust additional searched objectives/groups.')
# Descriptive paired route bootstrap for same 4 critical settings per scenario.
b,c=datasets['confirmation_505'];rows=[x for x in b['episodes'] if scope(x,'critical_estimated')];seeds=sorted({x['seed'] for x in rows});index={s:i for i,s in enumerate(seeds)};stats=np.zeros((len(seeds),12),dtype=np.int64);lookups={g:{key(x):x for x in datasets[f'confirmation_{g}'][1]['episodes']} for g in range(501,506)}
for x in rows:
 i=index[x['seed']];bs=bool(x['stable_landing']);stats[i,0]+=not bs;stats[i,1]+=bs
 for j,g in enumerate(range(501,506)):
  cs=bool(lookups[g][key(x)]['stable_landing']);stats[i,2+2*j]+=not bs and cs;stats[i,3+2*j]+=bs and not cs
assert np.all(stats[:,:2].sum(1)==4)
rng=np.random.default_rng(918271);samples=[]
for first in range(0,10000,125):samples.append(stats[rng.integers(len(stats),size=(min(125,10000-first),len(stats)))].sum(1))
samples=np.concatenate(samples);ci={}
for j,g in enumerate(range(501,506)):
 gain=samples[:,2+2*j];loss=samples[:,3+2*j];ci[str(g)]={name:[float(v) for v in np.quantile(values,[.025,.975])] for name,values in dict(rescue_percent=100*gain/samples[:,0],harm_percent=100*loss/samples[:,1],net_success_gain_pp=100*(gain-loss)/samples[:,:2].sum(1)).items()}
delta=100*((samples[:,8]-samples[:,9])-(samples[:,10]-samples[:,11]))/samples[:,:2].sum(1)
bootstrap=dict(replicates=10000,route_clusters=len(stats),conditions_per_route=4,metrics=ci,net504_minus505_gain_pp_ci95=[float(v) for v in np.quantile(delta,[.025,.975])],scope='Post-hoc descriptive pointwise paired route-bootstrap, not simultaneous/model-selection adjusted certification. Zero empirical harm produces degenerate bootstrap intervals; never interpret them as a zero population-risk bound.')
# Loss severity, separate actual recorded contacts from unclassified impact.
audit=read(SRC/'report/attempt_1/cambios_de_resultado.json');severity=[]
for label,value in audit.items():
 if not label.startswith(('primary_','confirmation_')):continue
 loss=[x for x in value['changed'] if x['kind']=='loss'];contacts=[x['candidate_contact'] for x in loss if x['candidate_contact'] is not None]
 severity.append(dict(dataset=label,losses=len(loss),harmed_routes=len({x['case']['seed'] for x in loss}),outcomes=dict(Counter(x['candidate_outcome'] for x in loss)),known_first_contacts=len(contacts),unknown_first_contacts=len(loss)-len(contacts),max_vertical_contact_speed=max((x['vertical_speed'] for x in contacts),default=None),max_angular_contact_speed=max((x['angular_speed'] for x in contacts),default=None),max_contact_radius=max((x['radius'] for x in contacts),default=None)))
rawdev=read(SRC/'h8_rejected_expanded_raw505_losses.json')
both_fail=[];contact_switches=[]
for phase in ['primary','confirmation']:
 for gate in [504,505]:
  b,c=datasets[f'{phase}_{gate}'];old={key(x):x for x in b['episodes']};cnt=Counter((old[key(x)]['outcome'],x['outcome']) for x in c['episodes'] if not old[key(x)]['stable_landing'] and not x['stable_landing']);both_fail.append(dict(phase=phase,gate=gate,both_fail=sum(cnt.values()),changed_failure_outcomes={a+' -> '+z:n for (a,z),n in cnt.items() if a!=z}))
  if phase=='primary' and gate==505:
   for x in c['episodes']:
    base=old[key(x)]
    if base['outcome']=='timeout' and x['outcome'] in ['hard_landing','early_contact']:contact_switches.append(dict(case={k:x[k] for k in ['seed','level','condition','noise','clock']},baseline_outcome=base['outcome'],candidate_outcome=x['outcome'],baseline_first_contact=base['first_contact'],candidate_first_contact=x['first_contact'],baseline_finish_seconds=base['episode_steps']/48,candidate_finish_seconds=x['episode_steps']/48))
save('transiciones_entre_fallos.json',both_fail);save('timeout_a_contacto505.json',contact_switches)
contacts=[x['candidate_first_contact'] for x in contact_switches if x['candidate_first_contact']];blind_spot=dict(cases=len(contact_switches),all_at_30_percent=all(x['case']['level']==.3 for x in contact_switches),known_contacts=len(contacts),maximum_recorded_angular_contact_speed=max(x['angular_speed'] for x in contacts),maximum_recorded_vertical_contact_speed=max(x['vertical_speed'] for x in contacts),scope='Outcome-category changes inside both-fail cells, not lost successful landings. These are not automatically all worse: baseline timeout is not a physical-safety certificate; an agreed severity metric is needed.')
protected_sample_sizes=[dict(harm_tolerance_percent=100*eps,alpha=alpha,independent_zero_loss_protected_flights_required=math.ceil(math.log(alpha)/math.log1p(-eps))) for eps in [.001,.005,.01] for alpha in [.05,.0125,.01]]
utilities=[dict(gate=t['gate'],lambda_harm=cost,score=t['gains']-cost*t['losses'],scope='Observed confirmation counts only; scalar utility assumes equal value per rescue and equal cost per lost baseline landing.') for cost in [1.,2.,4.25,5.,10.,20.] for t in tables if t['dataset']=='confirmation' and t['scope']=='global']
critical={t['gate']:t for t in tables if t['dataset']=='confirmation' and t['scope']=='critical_estimated'};domain=[]
for q in [.01,.05,.1,1076/4096,.4]:
 for g in [504,505]:
  t=critical[g];res=t['rescue_percent']/100;harm=t['harm_percent']/100;domain.append(dict(hypothetical_baseline_failure_percent=100*q,gate=g,expected_net_success_gain_pp=100*(q*res-(1-q)*harm),assumption='Illustration only: the conditional rescue/harm distributions remain fixed at the observed critical-estimated confirmation distribution. Not an estimate of real-world rates.'))
r4=critical[504]['rescue_percent']/100;r5=critical[505]['rescue_percent']/100;h4=critical[504]['harm_percent']/100
out=dict(scope='Analysis only; original R8, frozen policies, delivered ZIP and formal conclusions remain unchanged. Existing tests now inspected are exploratory for any changed objective or model selection.',tables=tables,overlaps=overlaps,independent_direct504_vs505=independent_direct,independent_confirmation_subsample=independent,descriptive_route_bootstrap=bootstrap,formal_loss_severity=severity,both_fail_transitions=both_fail,binary_metric_blind_spot=blind_spot,unguarded505_development_losses=rawdev['changed'],protected_sample_sizes=protected_sample_sizes,utility_scores=utilities,prevalence_sensitivity=domain,critical504_vs505_equal_cost_prevalence_break_even_percent=100*h4/(r4-r5+h4),zero_loss_acceptance_example=dict(true_harm_percent=.1,independent_protected_flights=5000,probability_pass_zero_losses=(1-.001)**5000),registered_selected505_independent_evidence=read(SRC/'final/attempt_1/sequential_initial.json'))
assert (overlaps['development_guard_vs_no_guard']['first_metrics']['gains'],overlaps['development_guard_vs_no_guard']['second_metrics']['gains'],overlaps['development_guard_vs_no_guard']['second_metrics']['losses'])==(75,137,3)
assert (overlaps['development_guard_vs_no_guard']['shared_rescues'],overlaps['development_guard_vs_no_guard']['second_only_rescues'],overlaps['development_guard_vs_no_guard']['first_only_rescues'])==(73,64,2)
for row in tables:
 q=row['baseline_failure_percent']/100;r=row['rescue_percent']/100;h=row['harm_percent']/100
 assert abs(100*(q*r-(1-q)*h)-row['net_success_gain_pp'])<1e-12
assert digest(frozen)==frozen_sha
for p,d in INPUT.items():assert digest(p)==d,'Input changed while reading: '+p
save('analisis_recalculado.json',out);save('fuentes_locales_sha256.json',dict(inputs=INPUT,frozen_sha256=frozen_sha,read_only_original_results=True));csvwrite('sensibilidad_coste.csv',utilities);csvwrite('sensibilidad_frecuencia_fallo.csv',domain)
print('ALL_SCREENSHOT_COUNTS_CHECKED',overlaps)
print('INDEPENDENT_CONFIRMATION',independent)
print('PAIRED_CRITICAL_NET504_MINUS505_CI95_PP',bootstrap['net504_minus505_gain_pp_ci95'])
print('SEVERITY',severity)
print('PROTECTED_SAMPLE_SIZES',protected_sample_sizes)
print('PREVALENCE_BREAK_EVEN_PERCENT',out['critical504_vs505_equal_cost_prevalence_break_even_percent'])

print('INDEPENDENT_DIRECT504_VS505',independent_direct)
