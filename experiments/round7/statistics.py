"""Crossed route/model resampling; repeated settings never count as new routes."""
import numpy as np
from scipy.stats import beta
from .evaluation import key


def crossed_interval(matrix,replicates=20000,seed=771):
    d=np.asarray(matrix,float);assert d.ndim==2 and d.shape[0]==5
    rng=np.random.default_rng(seed);out=[];m,r=d.shape
    for first in range(0,replicates,250):
        n=min(250,replicates-first);mi=rng.integers(m,size=(n,m));ri=rng.integers(r,size=(n,r))
        out.extend(d[mi[:,:,None],ri[:,None,:]].mean((1,2)))
    return dict(mean=float(d.mean()),lower95=float(np.quantile(out,.025)),upper95=float(np.quantile(out,.975)),replicates=replicates,model_seeds=m,route_clusters=r)


def paired(candidate_results,baseline,condition='estimated'):
    b={key(r):r for r in baseline['episodes']};routes=sorted({r['seed'] for r in baseline['episodes']});matrix=[];gains=losses=mild=exact=0;harmed=set();protected=set()
    for data in candidate_results:
        rows=data['episodes'];assert len(rows)==len(b)
        values=[]
        for route in routes:
            rr=[r for r in rows if r['seed']==route and r['condition']==condition and r['level'] in [.295,.296]]
            assert rr
            values.append(np.mean([int(r['stable_landing'])-int(b[key(r)]['stable_landing']) for r in rr]))
        matrix.append(values)
        for r in rows:
            base=b[key(r)];gain=not base['stable_landing'] and r['stable_landing'];loss=base['stable_landing'] and not r['stable_landing']
            gains+=gain;losses+=loss;mild+=loss and r['level']<=.25;exact+=loss and r['condition']=='exact'
            if base['stable_landing']:protected.add(r['seed'])
            if loss:harmed.add(r['seed'])
    n=len(protected);k=len(harmed);upper=float(beta.ppf(.95,k+1,n-k)) if 0<=k<n else 1.
    return dict(**crossed_interval(matrix),gains=int(gains),losses=int(losses),mild_losses=int(mild),exact_losses=int(exact),protected_route_clusters=n,harmed_route_clusters=k,
        harmed_route_upper95=upper,upper_bound_scope='Any lost baseline success among the tested model/regime combinations on a protected route; assumes independent route clusters; not per-tick or universal safety.')
