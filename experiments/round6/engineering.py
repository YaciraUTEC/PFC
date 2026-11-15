"""Engineering-only NMPC integration screen; no validation or test routes."""
import argparse,json
from .phase1 import BASE,run_job
from .evaluate import OUT,save
VARIANTS={
 'coordinated_margin':dict(BASE,control_kind='coordinated',manager='coordinated',filter_tau=.04,yaw_priority=1.,emergency='damped',reserve=.001),
 'reserve_only':dict(BASE,reserve=.001),
 'nmpc_matched':dict(BASE,control_kind='nmpc',manager='coordinated',fault_braking=True,filter_tau=0.,mpc_stride=1,
    nmpc_options=dict(horizon=12,integration_substeps=5,motor_time_constant=.01,solver_maxiter=60,solver_time_limit=.10,indi_gain=0.,angular_rate_weights=[.5,.5,.3],fault_yaw_rate_fraction=1.)),
 'nmpc_matched_indi':dict(BASE,control_kind='nmpc',manager='coordinated',fault_braking=True,filter_tau=0.,mpc_stride=1,
    nmpc_options=dict(horizon=12,integration_substeps=5,motor_time_constant=.01,solver_maxiter=60,solver_time_limit=.10,indi_gain=.3,angular_rate_weights=[.5,.5,.3],fault_yaw_rate_fraction=1.)),
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',default='all');a=p.parse_args()
 for n,c in VARIANTS.items():
  if a.group=='pid' and n.startswith('nmpc'):continue
  if a.group=='nmpc' and not n.startswith('nmpc'):continue
  run_job(n,c,'engineering',range(610008,610012),[0.,.25,.295,.35,.5],conditions=[('exact',1.),('estimated',2.)])
if __name__=='__main__':main()
