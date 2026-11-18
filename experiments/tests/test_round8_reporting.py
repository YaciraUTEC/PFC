from experiments.round8.report import counts

def r(seed,success,noise=1.,active=0,steps=480):return dict(episode_steps=steps,seed=seed,level=.296,condition='estimated',noise=noise,clock=None,stable_landing=success,policy_summary=dict(actual_motor_intervention_steps=active))
def test_report_keeps_rescues_and_lost_landings_separate_despite_positive_net_gain():
 b=dict(episodes=[r(1,False),r(2,False),r(3,True),r(4,True)]);c=dict(episodes=[r(4,True),r(3,False,active=24),r(1,True,active=24),r(2,True,active=24)]);x=counts(c,b);assert x['gains']==2 and x['losses']==1;assert x['difference_pp']==25.;assert x['active_cases']==3

def test_report_uses_the_complete_paired_regime_key():
 b=dict(episodes=[r(1,False),r(1,True,2.)]);c=dict(episodes=[r(1,True,2.),r(1,True)]);x=counts(c,b);assert x['gains']==1 and x['losses']==0;assert x['base_successes']==1 and x['candidate_successes']==2


def test_report_distinguishes_all_protected_finish_delay_from_actual_interventions():
 b=dict(episodes=[r(1,True),r(2,True),r(3,False)]);c=dict(episodes=[r(1,True),r(2,True,active=24,steps=528),r(3,True,active=24,steps=900)]);x=counts(c,b);assert x["protected_active_cases"]==1;assert x["mean_protected_finish_delay_seconds"]==.5;assert x["mean_protected_active_finish_delay_seconds"]==1.
