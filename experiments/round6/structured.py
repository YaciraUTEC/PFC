from .phase1 import BASE,run_job
from .evaluate import OUT,save
VARIANTS={
 'structured_only':dict(BASE,estimator='structured',reserve=.001),
 'structured_governor':dict(BASE,estimator='structured',manager='coordinated',reserve=.001),
 'structured_coordinated':dict(BASE,control_kind='coordinated',estimator='structured',manager='coordinated',filter_tau=0.,yaw_priority=1.,emergency='damped',reserve=.001,yaw_acceleration_limit=3.,coordinate_thrust=True),
}
def main():
 for n,c in VARIANTS.items():run_job(n,c,'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.5])
 save(OUT/'phase1'/'structured_configs.json',VARIANTS)
if __name__=='__main__':main()
