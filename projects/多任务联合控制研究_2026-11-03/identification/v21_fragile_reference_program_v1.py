"""Paired holding-reference program and single-selection gate for offline preparation.

This module never owns an actuator or a physics clock. A future experimental
session can submit the selected reference through the original V21 single
commit and pass the real typed receipt back to the gate.
"""
from dataclasses import dataclass

import numpy as np

from v13_a1.codec import REFERENCE_AXES21, Reference21
from v13_joint.plant import ExecutionReceipt21
from v14_coord.design import CONTROL_PERIOD_S, reference_rates


SCHEMA = 'v21_fragile_holding_reference_program_v1'
ARMS = ('zero', 'combined')
RAMP_TICKS, HOLD_TICKS = 4, 4
RATE_UTILIZATION = .75
CLOCK_TOL = 1e-9
AUTHORITY_TOL = 1e-12
# These are the nonzero preparation synergy coefficients for TH and MF,
# from foundation_config.OPEN/CLOSED. Negative alpha points toward open.
GROUPS = (
    (('rh_A_THJ5', .847, .2),),
    (('rh_A_MFJ3', .6, 0.), ('rh_A_MFJ0', 2., 0.)),
)
WAVE = tuple(np.r_[np.arange(1,RAMP_TICKS+1)/RAMP_TICKS,
                   np.ones(HOLD_TICKS),
                   np.arange(RAMP_TICKS-1,-1,-1)/RAMP_TICKS])


def vector(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (21,) or not np.isfinite(a).all():
        raise ValueError('finite reference authority with 21 named axes required')
    return a


def tick(time_s):
    if not np.isfinite(time_s):
        raise ValueError('finite time required')
    n = round(time_s / CONTROL_PERIOD_S)
    if abs(n * CONTROL_PERIOD_S - time_s) > CLOCK_TOL:
        raise ValueError('reference issue must be on the 5 ms control grid')
    return n


@dataclass(frozen=True)
class HoldingProgramPair:
    origin_tick: int
    generation: int
    carrier21: tuple
    peak_delta21: tuple
    alpha_reduction: tuple

    def target(self, arm, offset):
        if arm not in ARMS or not 0 <= offset < len(WAVE):
            raise ValueError('unknown program arm or offset')
        carrier = np.array(self.carrier21)
        if arm == 'combined':
            carrier += WAVE[offset] * np.array(self.peak_delta21)
        return Reference21(carrier,(self.origin_tick+offset)*CONTROL_PERIOD_S)


def build_pair(origin, lower, upper, *, generation):
    """Compute a frozen ramp from the origin authority and the V21 holding rates."""
    if not isinstance(origin,Reference21) or type(generation) is not int or generation<0:
        raise ValueError('typed origin and explicit generation required')
    l,u=vector(lower),vector(upper)
    a=origin.values
    if np.any(l>u) or np.any(a<l-AUTHORITY_TOL) or np.any(a>u+AUTHORITY_TOL):
        raise ValueError('origin outside current holding authority')
    rates=reference_rates('holding')
    displacement=np.zeros(21)
    alphas=[]
    for group in GROUPS:
        axes=np.array([REFERENCE_AXES21.index(name) for name,_,_ in group])
        synergy=np.array([coefficient for _,coefficient,_ in group])
        open_at=np.array([opened for _,_,opened in group])
        distance=a[axes]-np.maximum(l[axes],open_at)
        if np.any(distance<=AUTHORITY_TOL):
            raise ValueError('unexposed: declared open direction unavailable')
        alpha=RATE_UTILIZATION*float(min(np.min(distance/synergy),
                             np.min(rates[axes]*CONTROL_PERIOD_S*RAMP_TICKS/synergy)))
        displacement[axes]=-alpha*synergy
        alphas.append(alpha)
    return HoldingProgramPair(tick(origin.time_s),generation,tuple(a),tuple(displacement),tuple(alphas))


class HoldingProgramGate:
    """Select at most one command per tick and cancel on authority or receipt changes."""

    def __init__(self, program, arm):
        if not isinstance(program,HoldingProgramPair) or arm not in ARMS:
            raise ValueError('explicit frozen program and arm required')
        self.program=program
        self.arm=arm
        self.status='ready'
        self.offset=0
        self.expected=np.array(program.carrier21)
        self.pending=None
        self.receipts=[]

    def select(self,current,lower,upper,*,mode,generation):
        if self.pending is not None:
            raise RuntimeError('await the sole real receipt before selecting again')
        if self.status not in ('ready','running'):
            return None
        if not isinstance(current,Reference21):
            raise TypeError('typed current reference required')
        now=tick(current.time_s)
        if now<self.program.origin_tick and self.status=='ready':
            return None
        if mode!='holding' or generation!=self.program.generation:
            self.status='cancelled_mode_or_generation'
            return None
        if now!=self.program.origin_tick+self.offset or not np.array_equal(current.values,self.expected):
            self.status='cancelled_missing_tick_or_changed_current'
            return None
        goal=self.program.target(self.arm,self.offset)
        step=reference_rates('holding')*CONTROL_PERIOD_S
        lo=np.maximum(vector(lower),current.values-step)
        hi=np.minimum(vector(upper),current.values+step)
        if (np.any(lo>hi) or np.any(goal.values<lo-AUTHORITY_TOL)
                or np.any(goal.values>hi+AUTHORITY_TOL)):
            self.status='cancelled_fresh_authority'
            return None
        self.pending=goal
        self.status='running'
        return goal

    def accept(self,receipt):
        if self.pending is None:
            raise RuntimeError('no selected program command awaiting confirmation')
        if not isinstance(receipt,ExecutionReceipt21):
            raise TypeError('real typed execution receipt required')
        self.receipts.append(receipt)
        goal=self.pending
        self.pending=None
        same_clock=all(abs(port.time_s-goal.time_s)<=CLOCK_TOL
                       for port in (receipt.requested,receipt.executed,receipt.engine))
        identical=(np.array_equal(receipt.requested.values,goal.values)
                   and np.array_equal(receipt.executed.values,goal.values))
        encoded=receipt.executed.values.copy()
        encoded[18]+=receipt.engine.lift_bias_m
        engine_identical=np.array_equal(encoded,receipt.engine.values)
        if (not same_clock or not identical or receipt.mode!='holding'
                or receipt.mode_generation!=self.program.generation
                or not engine_identical or receipt.execution_changed or receipt.stale_plan_replaced):
            self.status='cancelled_receipt_mismatch'
            return
        self.expected=receipt.executed.values.copy()
        self.offset+=1
        if self.offset==len(WAVE):
            self.status='complete'
