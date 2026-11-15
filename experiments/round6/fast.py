from .phase1 import BASE,run_job
from .evaluate import OUT,save
VARIANTS={}
for horizon,iterations,limit in [(8,12,.012),(10,16,.015)]:
 name=f'nmpc_fast_{horizon}'
 VARIANTS[name]=dict(BASE,control_kind='nmpc',manager='coordinated',fault_braking=True,filter_tau=0.,mpc_stride=1,
  nmpc_options=dict(horizon=horizon,integration_substeps=5,motor_time_constant=.01,solver_maxiter=iterations,solver_time_limit=limit,indi_gain=0.,velocity_weights=[3.,3.,6.],angular_rate_weights=[.5,.5,.3],fault_yaw_rate_fraction=1.))
def main():
 for n,c in VARIANTS.items():run_job(n,c,'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.5])
 save(OUT/'phase1'/'fast_configs.json',VARIANTS)
if __name__=='__main__':main()
