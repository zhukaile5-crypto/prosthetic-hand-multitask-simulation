"""Force-target intents, separated from the sole actor and observation state.

Ratio mechanism is a migration of the frozen implementation, not an exact
paper reproduction. These classes never output alpha/actuator actions or
claim that a desired force is a predicted/guaranteed future force.
"""
from collections import deque
from collections.abc import Mapping
import numpy as np
from .observation import number,array,gain

ACQUISITION=('CLOSING','ACQUIRE','CONTACT_ESTABLISHMENT','LIFT','LIFTING')
RELEASE_PHASES=('RELEASE','RELEASING','MONITOR')


def project_force(target,upper,budget):
    y=array(target,(5,),'target',finite=True);cap=array(upper,(5,),'upper',finite=True)
    if np.any(cap<0):raise ValueError('force upper limits must be nonnegative')
    budget=number(budget,'budget',0.)
    result=np.clip(y,0.,cap)
    if result.sum()<=budget:return result
    lo,hi=0.,max(0.,float(np.max(y)))+float(np.max(cap))
    for _ in range(28):
        mid=.5*(lo+hi)
        if np.clip(y-mid,0.,cap).sum()>budget:lo=mid
        else:hi=mid
    return np.clip(y-.5*(lo+hi),0.,cap)


class FixedIntent:
    def __init__(self,gains):
        self.base=number(gain(gains,'base',None),'base',0.)
        self.margin=number(gain(gains,'force_margin',.12),'force_margin',0.)
        self.acquisition=array(gain(gains,'acquisition_force',(2.,)*5),(5,),'acquisition_force',finite=True)
        if np.any(self.acquisition<0):raise ValueError('acquisition_force must be nonnegative')

    def _finish(self,state,raw,limits5,budget,debug):
        cap=array(limits5,(5,),'limits5',finite=True)
        if np.any(cap<0):raise ValueError('limits must be nonnegative')
        budget=number(budget,'budget',0.)
        desired=project_force(raw,np.maximum(0.,cap-self.margin),max(0.,budget-2*self.margin))
        debug.update(unprojected_target5=np.asarray(raw).tolist(),desired_force_is_prediction=False,
            output_role='force target only; common actor owns execution/antiwindup',projection_changed=bool(not np.array_equal(desired,raw)))
        return dict(desired_normal5=desired,debug=debug)

    def step(self,state,phase,limits5,budget,dt=.02):
        number(dt,'intent dt',1e-12)
        weights=array(state.nominal_weights,(5,),'nominal_weights',finite=True)
        if np.any(weights<=0):raise ValueError('nominal weights must be positive')
        phase=str(phase).upper()
        raw=np.zeros(5) if phase in RELEASE_PHASES else (self.acquisition if phase in ACQUISITION else self.base)*weights
        return self._finish(state,raw,limits5,budget,dict(label='release_preempts_intent' if phase in RELEASE_PHASES else 'fixed_force_intent'))


class RatioIntent(FixedIntent):
    def __init__(self,gains):
        super().__init__(gains)
        self.floor=number(gain(gains,'ratio_floor',None),'ratio_floor',0.)
        self.kp=number(gain(gains,'ratio_kp',None),'ratio_kp',0.)
        self.ki=number(gain(gains,'ratio_ki',None),'ratio_ki',0.)
        self.boost_rate=number(gain(gains,'slip_boost_rate',None),'slip_boost_rate',0.)
        if self.floor>5 or self.base<self.floor:raise ValueError('Require base>=ratio_floor and floor<=5')
        self.scalar=self.base;self.boost=0.;self.target_ratio=None
        self.samples=deque(maxlen=5);self.stable_since=None;self.integral=0.
        self.required_group=None;self._last_time=None
        self.last_raw_target=None

    def step(self,state,phase,limits5,budget,dt=.02):
        dt=number(dt,'intent dt',1e-12);now=number(state.time,'state.time',0.)
        limits=array(limits5,(5,),'limits5',finite=True)
        if np.any(limits<0):raise ValueError('limits must be nonnegative')
        number(budget,'budget',0.)
        if self._last_time is not None and now<self._last_time-1e-9:raise ValueError('Intent clock rewind')
        phase=str(phase).upper();weights=array(state.nominal_weights,(5,),'nominal_weights',finite=True)
        if np.any(weights<=0):raise ValueError('nominal weights must be positive')
        if phase in RELEASE_PHASES:
            return self._finish(state,np.zeros(5),limits5,budget,dict(label='release_preempts_intent',memory_updated=False))
        valid=np.asarray(state.valid,bool);contact=np.asarray(state.contact,bool)
        if valid.shape!=(5,) or contact.shape!=(5,):raise ValueError('state masks must be length5')
        n=array(state.delayed_normal,(5,),'delayed_normal');shear=array(state.shear_magnitude,(5,),'shear_magnitude')
        supported=contact&valid&np.isfinite(n)&np.isfinite(shear)
        # Once supported fingers establish the task, dropout never shrinks the
        # denominator. New/unaffiliated contact suspends learning as well.
        if self.required_group is None and supported.sum()>=3:self.required_group=supported.copy()
        group=self.required_group
        observable=bool(group is not None and valid[group].all() and contact[group].all() and np.isfinite(n[group]).all()
            and np.isfinite(shear[group]).all() and not np.any(contact&~group))
        duplicate=self._last_time is not None and now<=self._last_time+1e-9
        self._last_time=now
        update=not duplicate and (observable or self.required_group is None and valid.all())
        ratio=float(np.mean(self.samples)) if self.samples else 0.
        slipping=bool(np.isfinite(state.slip_speed) and state.slip_speed>.004 and supported.sum()>=2)
        label='ratio_observation_unavailable'
        if update:
            if supported.sum()>=2:
                value=float(shear[supported].sum()/max(.25,n[supported].sum()))
                self.samples.append(value)
            ratio=float(np.mean(self.samples)) if self.samples else 0.
            if phase in ACQUISITION:
                self.last_raw_target=self.acquisition*weights
                return self._finish(state,self.acquisition*weights,limits5,budget,
                    dict(label='shared_acquisition_reference',ratio=ratio,target_ratio=self.target_ratio,
                         memory_updated=True,learning_scope='ratio samples only during acquisition; no PI/boost/initialization'))
            if self.target_ratio is None:
                label='ratio_initialization'
                if slipping or supported.sum()<3:
                    if slipping:self.boost+=self.boost_rate*dt
                    self.stable_since=None
                else:
                    self.boost*=np.exp(-dt/1.5)
                    if self.stable_since is None:self.stable_since=now
                    if now-self.stable_since>=1.:
                        self.target_ratio=max(.025,ratio);self.scalar=self.base+self.boost
                target=self.base+self.boost
            else:
                label='ratio_feedback';error=ratio-self.target_ratio
                integral=np.clip(self.integral+error*dt,-.35,.75)
                raw=self.scalar+self.kp*error+self.ki*integral
                if self.floor<=raw<=5. or raw>5. and error<0. or raw<self.floor and error>0.:
                    self.integral=float(integral)
                target=float(np.clip(raw,self.floor,5.))
        else:
            # No PI integration, boost or target learning on missing data.
            # A broken initial stable interval cannot accumulate through dropout.
            if not duplicate and self.target_ratio is None:self.stable_since=None
            target=self.base+self.boost if self.target_ratio is None else self.scalar
        raw_target=(self.acquisition if phase in ACQUISITION else target)*weights
        if not update and phase not in ACQUISITION and self.last_raw_target is not None:
            raw_target=self.last_raw_target.copy()
        elif update:self.last_raw_target=np.asarray(raw_target).copy()
        if phase in ACQUISITION:label='shared_acquisition_reference'
        return self._finish(state,raw_target,limits5,budget,dict(label=label,ratio=ratio,
            target_ratio=self.target_ratio,ratio_integral=self.integral,ratio_scalar=self.scalar,
            initialization_boost=self.boost,ratio_observable=observable,memory_updated=update,
            required_group=None if group is None else group.tolist(),
            initialization_scope='1s observed stable contact; frozen ratio PI conditional integration; actor saturation is separately reconciled'))

    def checkpoint(self):
        """Target-memory only. Shared actor may explicitly roll back its proposal.

        This is not automatic actuator antiwindup and never consumes alpha.
        Restore is a new V12 execution contract, separate from frozen scalar PI.
        """
        return dict(scalar=self.scalar,boost=self.boost,target_ratio=self.target_ratio,
            samples=list(self.samples),stable_since=self.stable_since,integral=self.integral,
            required_group=None if self.required_group is None else self.required_group.copy(),
            last_time=self._last_time,last_raw_target=None if self.last_raw_target is None else self.last_raw_target.copy())

    def restore(self,memory):
        for name in ('scalar','boost','target_ratio','stable_since','integral'):setattr(self,name,memory[name])
        self.samples=deque(memory['samples'],maxlen=5)
        self.required_group=None if memory['required_group'] is None else memory['required_group'].copy()
        self._last_time=memory['last_time']
        self.last_raw_target=None if memory['last_raw_target'] is None else memory['last_raw_target'].copy()
