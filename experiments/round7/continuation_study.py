"""Finite development ablation of continuing a predicted-negative intervention."""
import time,shutil
from .common import OUT,read,save,close_milestone
from .study import run,rank
from .evaluation import cases


def main():
    if (OUT/'h3_continuation_ablation_conclusion.json').exists():return
    initial=read(OUT/'h3_filter_ablation_conclusion.json');assert initial['closed'];start=time.time()
    backup=OUT/'h3_initial_filter_ablation_conclusion.json'
    if not backup.exists():shutil.copy2(OUT/'h3_filter_ablation_conclusion.json',backup)
    base=read(OUT/'filter_flights/baseline.json');cc=cases(range(710012,710020),levels=[0.,.25,.295,.296]);gates=read(OUT/'h2_counterfactual_gate_conclusion.json')['selected_by_source'];rows=[]
    save(OUT/'continuation_flights/baseline.json',base)
    for source in ['flashsac','bc']:
        spec=dict(gates[source]['spec'],continuation_gain=0.)
        rows.append(run(source+'_nonnegative_continuation',spec,cc,base,'continuation_flights'))
        rows.append(run(source+'_nonnegative_continuation_screen',dict(spec,screen=dict(allowance=0.,uncertainty_multiplier=1.)),cc,base,'continuation_flights'))
    all_rows=initial['results']+rows
    # A nonnegative continuation is preferred only when the prespecified flight
    # protection/gain rank ties; it never overrides a worse measured rank.
    best=max(all_rows,key=lambda r:(rank(r),r['spec'].get('mode')=='counterfactual' and r['spec'].get('continuation_gain',-.005)>=0))
    close_milestone('h3_continuation_ablation','Ablacion finita de detener la opcion cuando el beneficio predicho deja de ser no negativo; las rutas diagnosticadas son desarrollo, ninguna prueba final se utilizo.',diagnosis='Route710017, both noise settings:12/23 active steps had negative predicted benefit under the original-.005 continuation tolerance; this association alone does not establish causality.',initial_selected=initial['selected'],selected=best,new_comparisons=rows,seconds=time.time()-start)
    close_milestone('h3_filter_ablation','Ablacion3x2 inicial y cuatro controles de continuacion no negativa completados; mantener la mejor configuracion segun proteccion y ganancia de vuelos, antes de cinco semillas.',cases=len(cc),routes=8,selected=best,results=all_rows,initial_ablation=str(backup),initial_seconds=initial['seconds'],continuation_ablation_seconds=time.time()-start,seconds=initial['seconds']+time.time()-start,screen_is_not_formal_reachability=True)
if __name__=='__main__':main()
