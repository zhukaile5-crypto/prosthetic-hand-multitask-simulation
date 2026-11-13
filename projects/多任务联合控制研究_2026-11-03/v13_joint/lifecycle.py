"""Release reference proposal from actual21 entry; no plant or engine writer."""
from dataclasses import dataclass
import numpy as np
from v13_a1.codec import Reference21,finite_scalar,finite_array
from .design import (NONTHUMB_CLEAR_TIME_S,THUMB_TUCK_TIME_S,THUMB_TUCK_UNTIL_S,THUMB_UNFOLD_TIME_S,LIFT_DESCENT_TIME_S)

@dataclass(frozen=True)
class ReleaseEntry:
    started_s:float
    reference21:Reference21
    mode_generation:int
    def __post_init__(self):
        now=finite_scalar(self.started_s,'release started')
        if now<0 or not isinstance(self.reference21,Reference21):raise ValueError('Actual nonnegative reference21 release entry required')
        if abs(self.reference21.time_s-now)>1e-9:raise ValueError('Release entry/reference snapshot clock mismatch')
        if isinstance(self.mode_generation,bool) or not isinstance(self.mode_generation,(int,np.integer)) or self.mode_generation<1:raise ValueError('Positive mode generation required')

def release_nominal(entry,query_time_s,open18,lower21,upper21):
    if not isinstance(entry,ReleaseEntry):raise TypeError('Explicit complete actual reference entry required')
    t=finite_scalar(query_time_s,'release reference query');elapsed=t-entry.started_s
    if elapsed<0:raise ValueError('Cannot evaluate release before entry')
    start=entry.reference21.values;open_pose=finite_array(open18,(18,),'declared open18')
    result=start.copy()
    # New independent axes also clear: never reconstruct current posture from alpha.
    blend=float(np.clip(elapsed/NONTHUMB_CLEAR_TIME_S,0.,1.))
    result[5:18]=start[5:18]+blend*(open_pose[5:18]-start[5:18])
    tucked=np.array([-1.03,1.21,.13,-.02,.985]);opened=np.array([-1.,.1,0.,0.,0.])
    if elapsed<THUMB_TUCK_TIME_S:result[:5]=start[:5]+np.clip(elapsed/THUMB_TUCK_TIME_S,0.,1.)*(tucked-start[:5])
    elif elapsed<THUMB_TUCK_UNTIL_S:result[:5]=tucked
    else:result[:5]=tucked+np.clip((elapsed-THUMB_TUCK_UNTIL_S)/THUMB_UNFOLD_TIME_S,0.,1.)*(opened-tucked)
    result[18]=start[18]*(1.-np.clip(elapsed/LIFT_DESCENT_TIME_S,0.,1.))
    result[19:]=start[19:]
    lo=finite_array(lower21,(21,),'reference lower21');hi=finite_array(upper21,(21,),'reference upper21')
    if np.any(lo>hi):raise ValueError('Actual reference authority required')
    return Reference21(np.clip(result,lo,hi),t)
