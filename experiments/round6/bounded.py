from .engineering import VARIANTS as EXTRA
from .phase1 import BASE,run_job,VARIANTS
from .evaluate import OUT,save
BOUNDED={}
for acceleration in [1.5,3.,6.]:
 name=f'coordinated_bounded_{acceleration:g}'
 BOUNDED[name]=dict(BASE,control_kind='coordinated',manager='coordinated',filter_tau=0.,yaw_priority=1.,emergency='damped',reserve=.001,yaw_acceleration_limit=acceleration,coordinate_thrust=True)
BOUNDED['governor_margin']=dict(BASE,manager='coordinated',reserve=.001)
def main():
 for name,cfg in BOUNDED.items():
  run_job(name,cfg,'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.5])
 save(OUT/'phase1'/'bounded_configs.json',BOUNDED)
if __name__=='__main__':main()
