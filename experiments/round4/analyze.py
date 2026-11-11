"""Artifact analysis. Run only after all frozen final evaluations finish."""
import json,time,datetime,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import OUT
from .models import ARCHS
from .train import SEEDS
from .env import LEVELS,STRESS
from ..run import save
NAMES=dict(lstm='LSTM',mamba='Mamba',mamba2='Mamba-2',mamba3='Mamba-3 (SISO)',gtrxl='GTrXL',ltc='LTC')
COLORS=['#1764ab','#269450','#d68111','#a14996','#c74747','#1b9996']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def matrix(arch,mode,condition,metric='stable_landing'):
    mats=[]
    for seed in SEEDS:
        rows=read(OUT/'test'/f'{arch}_{seed}_{mode}_{condition}.json')
        mats.append(np.asarray([[r[metric] for r in rows if r['level']==level] for level in LEVELS+STRESS],float))
    return np.asarray(mats)
def bootstrap_delta(x,reference,levels,iterations=10000):
    # x/reference: training seed,level,route;shared route/severity remain paired.
    delta=(x-reference)[:,levels].mean(1)
    rng=np.random.default_rng(604731)
    si=rng.integers(0,delta.shape[0],(iterations,delta.shape[0]))
    ri=rng.integers(0,delta.shape[1],(iterations,delta.shape[1]))
    boot=delta[si[:,:,None],ri[:,None,:]].mean((1,2))*100
    return dict(delta_pp=float(delta.mean()*100),ci95=np.quantile(boot,[.025,.975]).tolist(),
        ci_family12=np.quantile(boot,[.05/(2*12),1-.05/(2*12)]).tolist(),
        ci_family24=np.quantile(boot,[.05/(2*24),1-.05/(2*24)]).tolist(),
        resampling='independent crossing of5 training seed IDs and30 route IDs;paired levels kept together',iterations=iterations)
def main():
    assert (OUT/'test_complete.json').exists() and (OUT/'historical_complete.json').exists()
    from .verify_results import verify
    verify()
    frozen=read(OUT/'frozen_before_test.json')
    for r in frozen['checkpoints']:assert hashlib.sha256((OUT/r['path']).read_bytes()).hexdigest()==r['sha256']
    from .latency import run as benchmark_latency
    latency=read(OUT/'inference_latency.json') if (OUT/'inference_latency.json').exists() else benchmark_latency()
    rows=[];differences=[];base_differences=[];final_differences=[];timings=[];summary={}
    for condition in ['exact','estimated']:
        for a in ARCHS:
            bc=matrix(a,'bc',condition);best=matrix(a,'best',condition);final=matrix(a,'final',condition)
            for mode,arr in [('bc',bc),('protected_selected',best),('protected_final',final)]:
                for j,level in enumerate(LEVELS+STRESS):
                    rows.append(dict(architecture=a,condition=condition,mode=mode,level=level,
                        success_percent=float(arr[:,j].mean()*100),success_count=int(arr[:,j].sum()),
                        model_cases=int(arr[:,j].size),route_geometries=30,training_seeds=5))
            sel=[LEVELS.index(l) for l in [.20,.25,.28,.29]]
            differences.append(dict(architecture=a,condition=condition,comparison='selected protected minus own BC',**bootstrap_delta(best,bc,sel)))
            base_rows=read(OUT/'test'/f'baseline_updated_{condition}.json')
            base_matrix=np.asarray([[r['stable_landing'] for r in base_rows if r['level']==l] for l in LEVELS+STRESS],float)
            base_ref=np.broadcast_to(base_matrix,best.shape)
            base_differences.append(dict(architecture=a,condition=condition,comparison='selected minus updated DSLPID',**bootstrap_delta(best,base_ref,sel)))
            final_differences.append(dict(architecture=a,condition=condition,comparison='full budget minus own BC,exploratory',**bootstrap_delta(final,bc,sel)))
            exact=matrix(a,'best',condition,metric='effort')
            summary[(a,condition)]=dict(heavy_success=float(best[:,sel].mean()*100),effort=float(exact.mean()))
        for controller in ['legacy','robust','updated_round3','updated']:
            rr=read(OUT/'test'/f'baseline_{controller}_{condition}.json')
            for l in LEVELS+STRESS:
                sub=[r for r in rr if r['level']==l]
                rows.append(dict(architecture=controller,condition=condition,mode='base',level=l,
                    success_percent=float(np.mean([r['stable_landing'] for r in sub])*100),
                    success_count=int(sum(r['stable_landing'] for r in sub)),model_cases=len(sub),route_geometries=30,training_seeds=0))
    historical=[]
    for version,names in [('v2',ARCHS+[a+'_final' for a in ARCHS]),('round3',['bc','manual','csil','meirl','rime','manual_final','csil_final','meirl_final','rime_final'])]:
        for name in names:
            for condition in ['exact','estimated']:
                files=[OUT/'historical'/f'v2_{name}_{condition}.json'] if version=='v2' else [OUT/'historical'/f'round3_{seed}_{name}_{condition}.json' for seed in [11,22,33,44,55]]
                rr=[r for f in files for r in read(f)]
                for level in LEVELS+STRESS:
                    sub=[r for r in rr if r['level']==level]
                    historical.append(dict(version=version,model=name,condition=condition,level=level,
                        success_percent=float(np.mean([r['stable_landing'] for r in sub])*100),
                        waypoint_percent=float(np.mean([r['waypoint_reached'] for r in sub])*100),model_cases=len(sub)))
    pd.DataFrame(historical).to_csv(OUT/'historical_rates.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(OUT/'success_rates.csv',index=False)
    save(OUT/'paired_changes.json',differences)
    save(OUT/'residual_vs_base.json',base_differences)
    save(OUT/'full_budget_changes.json',final_differences)
    for a in ARCHS:
        for seed in SEEDS:
            d=OUT/'models'/a/f'seed_{seed}';bc=read(d/'bc_timing.json');p=read(d/'complete.json')
            timings.append(dict(architecture=a,seed=seed,bc_seconds=bc['seconds'],ppo_seconds=p['seconds'],
                ppo_validation_seconds=p['validation_seconds'],ppo_excluding_validation=p['seconds']-p['validation_seconds'],
                steps=p['steps'],selected_steps=p['selected_steps']))
    pd.DataFrame(timings).to_csv(OUT/'training_timings.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.4),sharey=True)
    for ax,condition,title in zip(axes,['exact','estimated'],['Efectividad exacta (oráculo)','Efectividad estimada en línea']):
        baseline=df[(df.architecture=='updated')&(df.condition==condition)&(df.level<=.29)]
        ax.plot(baseline.level*100,baseline.success_percent,'k--',lw=2.4,label='DSLPID sin residual (30 rutas)')
        for a,color in zip(ARCHS,COLORS):
            tab=df[(df.architecture==a)&(df.condition==condition)&(df['mode']=='protected_selected')&(df.level<=.29)]
            ax.plot(tab.level*100,tab.success_percent,marker='o',markersize=4,color=color,label=NAMES[a])
        ax.set(title=title,xlabel='Pérdida de RPM en un motor (%)',ylim=(-3,103),xticks=[0,5,10,15,20,25,29])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Aterrizajes controlados y estables (%)')
    axes[1].legend(loc='lower left',fontsize=9)
    fig.suptitle('Redes: 5 semillas × 30 rutas por nivel y condición',fontsize=13)
    fig.text(.5,.01,'Contacto suave + radio ≤20 cm + velocidades/postura limitadas +1 s estable. Modelos elegidos solo por validación.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'comparacion_arquitecturas.png',dpi=180);fig.savefig(OUT/'comparacion_arquitecturas.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.2),sharey=True)
    for ax,condition,title in zip(axes,['exact','estimated'],['Efectividad exacta','Efectividad estimada']):
        sub=df[(df.condition==condition)&df.level.between(.25,.30)]
        base=sub[sub.architecture=='updated']
        ax.plot(base.level*100,base.success_percent,'k--',lw=2.4,label='DSLPID sin residual (30 rutas)')
        for arch,color in zip(ARCHS,COLORS):
            tab=sub[(sub.architecture==arch)&(sub['mode']=='protected_selected')]
            ax.plot(tab.level*100,tab.success_percent,marker='o',color=color,label=NAMES[arch])
        ax.axvline(29.7186,color='gray',ls=':',lw=1.6,label='Límite estático ideal')
        ax.set(title=title,xlabel='Pérdida de RPM en un motor (%)',xticks=[25,28,29,29.5,30],ylim=(-3,103))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Aterrizajes controlados y estables (%)')
    axes[1].legend(loc='lower left',fontsize=8)
    fig.suptitle('Cerca del límite: margen físico y aterrizaje estable')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'limite_fisico.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for offset,condition,color in [(-.13,'exact','#1764ab'),(.13,'estimated','#d68111')]:
        xx=np.arange(6)+offset;rr=[next(x for x in differences if x['architecture']==a and x['condition']==condition) for a in ARCHS]
        yy=np.array([r['delta_pp'] for r in rr]);ci=np.array([r['ci_family24'] for r in rr])
        ax.errorbar(xx,yy,yerr=np.maximum(np.stack([yy-ci[:,0],ci[:,1]-yy]),0),fmt='o',capsize=4,color=color,
            label='Exacta' if condition=='exact' else 'Estimada')
    ax.axhline(0,color='gray',lw=1);ax.set(xticks=np.arange(6),xticklabels=[NAMES[a] for a in ARCHS],ylabel='Cambio frente a su propia imitación (puntos porcentuales)',
        title='Afinado protegido: promedio en 20%, 25%, 28%, 29% (IC simultáneo, 24 contrastes)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'afinamiento_vs_imitacion.png',dpi=180);plt.close(fig)
    from .report import render
    text=render(OUT,df,historical,differences,base_differences,final_differences,timings,latency)
    (OUT/'INFORME_RONDA4.txt').write_text(text,encoding='utf-8')
    (ROOT_PARENT()/'INFORME_ATERRIZAJE_ARQUITECTURAS_RONDA4.txt').write_text(text,encoding='utf-8')
    save(OUT/'analysis.json',dict(differences=differences,residual_vs_base=base_differences,full_budget_differences=final_differences,timings=timings,models_unchanged=True,
        bootstrap_note='Primary24contrasts:selected vsBC and vsDSLPID,6architectures x2conditions;use ci_family24. Full-budget diagnostics exploratory ci95.',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(df[(df['mode'].isin(['base','protected_selected']))&df.level.isin([.25,.28,.29])].to_string(index=False),flush=True)
def ROOT_PARENT():return Path(__file__).resolve().parents[3]
if __name__=='__main__':main()
