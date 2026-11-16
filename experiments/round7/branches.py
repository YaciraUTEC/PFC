"""Counterfactual labels use strict terminal outcomes, never bootstrapped return."""
import numpy as np
from ..round6.env import ZERO
from .episode import action_from_delta,terminal_info

def outcome_class(base_success,candidate_success):
    # both_success, rescue, harm, both_fail
    return 0 if base_success and candidate_success else 1 if candidate_success else 2 if base_success else 3

def continuation(e,delta=None,ticks=0):
    action=ZERO.copy() if delta is None else action_from_delta(delta)
    first=e.t
    for step in range(1440-e.t):
        _,reward,d,tr,info=e.step(action if step<ticks else ZERO)
        if d or tr:return dict(terminal_info(info),continuation_steps=e.t-first)
    raise AssertionError('A strict terminal outcome must occur by the mission deadline')


def feedback_continuation(e,model,memory,initial_delta,alpha,duration):
    """Feedback at every tick under the native veto; then DSLPID to deadline."""
    import torch
    from ..round4.models import tree_map
    from .policy import propose
    state=tree_map(lambda t:t.detach().clone(),memory);delta=np.asarray(initial_delta,float)*alpha;first=e.t
    for step in range(1440-e.t):
        action=action_from_delta(delta) if step<duration else ZERO
        obs,_,done,truncated,info=e.step(action)
        if done or truncated:return dict(terminal_info(info),continuation_steps=e.t-first)
        if step+1<duration:
            with torch.no_grad():raw,on,state=propose(model,torch.as_tensor(obs[-1:],dtype=torch.float64),state)
            delta=raw[0].numpy()*alpha if bool(on[0]) else np.zeros(4)
    raise AssertionError('Feedback option must finish at a strict terminal')
