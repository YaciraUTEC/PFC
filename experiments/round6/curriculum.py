"""Episode-level noise-range curricula; update decisions use training flights only."""
from collections import deque
import numpy as np
from .env import LandingEnv

class NoiseSchedule:
    def __init__(self,kind,budget):
        if kind not in ['nominal','linear','adaptive','full']:raise ValueError(kind)
        self.kind=kind;self.budget=budget;self.upper=2. if kind=='full' else 1.;self.history=deque(maxlen=240);self.decisions=[]
    def update(self,steps,episodes):
        self.history.extend(episodes)
        previous=self.upper;reason='fixed'
        if self.kind=='linear':self.upper=1+min(1.,steps/(.8*self.budget));reason='linear_ramp_then_plateau'
        elif self.kind=='adaptive':
            cells={l:[r['stable_landing'] for r in self.history if np.isclose(r['level'],l) and r.get('effectiveness_condition','estimated')=='estimated'] for l in [.28,.29,.295]}
            mild=[r['stable_landing'] for r in self.history if r['level']<=.25]
            enough=all(len(v)>=4 for v in cells.values()) and len(mild)>=8
            if enough:
                worst=min(np.mean(v) for v in cells.values());healthy=float(np.mean(mild))
                if worst>=.75 and healthy>=.95:self.upper=min(2.,self.upper+.10);reason='advance_all_supported_fault_strata'
                elif worst<.50 or healthy<.85:self.upper=max(1.,self.upper-.15);reason='retreat_after_training_regression'
                else:reason='hold'
            else:reason='insufficient_training_episodes'
        row=dict(steps=int(steps),upper=float(self.upper),previous=float(previous),reason=reason)
        self.decisions.append(row);return self.upper

class CurriculumEnv(LandingEnv):
    def __init__(self,*,training_seed,**kw):
        super().__init__(**kw);self.pending_upper=1.;self.curriculum_rng=np.random.default_rng(training_seed+48531)
    def set_noise_upper(self,upper):self.pending_upper=float(upper)
    def reset(self,seed=None,options=None):
        # Sample once per episode. All controller/observer/policy inputs still
        # share each measured velocity/gyro sample; no privileged filtering.
        multiplier=float(self.curriculum_rng.uniform(.5,self.pending_upper))
        self._sensor_sigma=.002*multiplier;self.config['noise_multiplier']=multiplier
        if options is None:options=dict(level=float(self.curriculum_rng.choice([0.,.02,.10,.25,.28,.29,.295,.296,.30,.35,.50],p=[.15,.08,.07,.12,.14,.14,.12,.10,.04,.025,.015])))
        o,i=super().reset(seed=seed,options=options);return o,i
