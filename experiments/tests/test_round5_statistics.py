import numpy as np
from experiments.round5.statistics import harm_upper,route_interval,crossed_simultaneous,compare_confirmation

def test_sixty_routes_cannot_certify_one_percent_unseen_harm():
    assert harm_upper(0,60)>.04
    assert harm_upper(0,1000)<.01
    assert harm_upper(10,1000)>.01

def test_route_bootstrap_keeps_severities_paired():
    data=np.tile(np.array([-1.,0.,1.]),(100,1))
    r=route_interval(data,replicates=1000)
    assert r['mean']==r['lower']==r['upper']==0

def test_crossed_identical_policies_have_no_improvement():
    r=crossed_simultaneous(np.zeros((3,5,60,3)),replicates=1000)
    assert all(c['mean']==c['lower']==c['upper']==0 for c in r['contrasts'])

def test_crossed_bootstrap_has_uncertainty_over_training_seeds():
    d=np.zeros((1,5,60,3));d[0,0]=1.
    r=crossed_simultaneous(d,replicates=3000)
    assert r['contrasts'][0]['lower']<.2<r['contrasts'][0]['upper']

def test_class_balancing_preserves_rare_positive_gradient():
    import torch
    from experiments.round5.losses import clone_loss
    def gradient(negatives):
        mu=torch.zeros(1,negatives+1,6,requires_grad=True);target=torch.zeros_like(mu);target[...,4:]=-1.;target[0,0,5]=1
        mask=torch.ones(1,negatives+1);loss=clone_loss(mu,target,mask,mask);loss.backward();return float(mu.grad[0,0,5])
    np.testing.assert_allclose(gradient(10),gradient(1000),atol=1e-8)
    assert gradient(1000)<-.04

def test_impact_instrumentation_preserves_original_criterion():
    from types import SimpleNamespace
    from experiments.round4.landing import LandingMonitor
    from experiments.round5.diagnostics import attach_impact_diagnostics
    s=np.zeros(20);s[6]=1.;s[2]=.014;incoming=s.copy();incoming[13:16]=[.2,.3,2.]
    env=SimpleNamespace(monitor=LandingMonitor());reference=LandingMonitor();attach_impact_diagnostics(env)
    a=env.monitor.update(s,np.zeros(3),True,True,impact_state=incoming);b=reference.update(s,np.zeros(3),True,True,impact_state=incoming)
    assert a==b and a[1]
    np.testing.assert_allclose(env.monitor.first['body_angular_velocity'],incoming[13:16])
