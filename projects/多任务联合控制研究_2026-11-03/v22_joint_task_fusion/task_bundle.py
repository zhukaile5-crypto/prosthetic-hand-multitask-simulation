"""One finite contact-task bundle, with observable scope kept explicit.

True reference-coordinate limits stay in the domain. Normal inward-correction
slack and acquired-motion projection are simultaneous soft tasks. Neither is
future contact force or full object twist. No optimizer or engine lives here.
The first prototype keeps the V21 witnessed motion domain for causal comparison.
"""
from dataclasses import dataclass
import numpy as np

def _array(value,shape,name):
    a=np.asarray(value,float)
    if a.shape==(0,) and shape[0]==0 and len(shape)==2:a=np.empty(shape)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(name+' requires known finite '+str(shape))
    result=a.copy();result.setflags(write=False);return result

@dataclass(frozen=True)
class JointTaskBundle:
    A:np.ndarray
    b:np.ndarray
    G:np.ndarray
    h:np.ndarray
    E:np.ndarray
    f:np.ndarray
    lower:np.ndarray
    upper:np.ndarray
    correction_rows:int
    motion_rows:int
    motion_cost_scale:float

    def problem(self,initial_x):
        x=_array(initial_x,(len(self.lower),),'actual feasible reference anchor')
        return dict(kind='least_squares',A=self.A,b=self.b,G=self.G,h=self.h,E=self.E,f=self.f,
            lower=self.lower,upper=self.upper,initial_x=x,previous_frozen=[],atol=1e-9,rtol=1e-8,
            role='JOINT_CORRECTION_AND_ACQUIRED_PROJECTION_REFERENCE_INTENT',
            actual_future_force_guaranteed=False,object_twist_observed=False,
            runtime_connected=False,physical_superiority_verified=False)

def assemble(*,lower,upper,normal_cost_A,normal_target_b,projection_A,projection_target_b,
             original_base_G,original_base_h,original_fixed_E,original_fixed_f,
             witnessed_motion_G,witnessed_motion_h):
    """Join compatible finite tasks without freezing optimal correction slack.

    Base G/E must contain the ORIGINAL L1 variable bounds (including eta),
    normal rows, and genuinely fixed coordinates; never the optimized eta
    freeze. Witnessed motion rows are kept unchanged. Callers retain all
    original rows and independently verify primal/KKT at the original units.

    Normal rows already carry original pressure-share square-root factors.
    Projection uses mean squared error across explicitly observed rows. Both
    task families have declared dimensionless reference-authority units and
    fixed coefficient1. This is an explicit candidate choice, not an optimized
    weight or a certificate of equivalent physical safety/utility.
    """
    lo=np.asarray(lower,float);n=len(lo)
    if lo.ndim!=1 or not 1<=n<=256:raise ValueError('declared coordinate dimension1..256 required')
    lo=_array(lo,(n,),'lower');hi=_array(upper,(n,),'upper')
    if np.any(lo>hi):raise ValueError('empty original coordinate domain')
    nb=np.asarray(normal_target_b,float);mb=np.asarray(projection_target_b,float)
    nb=_array(nb,nb.shape,'normal target');mb=_array(mb,mb.shape,'projection target')
    if nb.ndim!=1 or mb.ndim!=1 or not len(nb) or not len(mb):
        raise ValueError('positive-contact and explicitly available projection tasks required')
    N=_array(normal_cost_A,(len(nb),n),'normal cost');M=_array(projection_A,(len(mb),n),'projection cost')
    baseh=np.asarray(original_base_h,float);mh=np.asarray(witnessed_motion_h,float);ef=np.asarray(original_fixed_f,float)
    baseh=_array(baseh,baseh.shape,'base h');mh=_array(mh,mh.shape,'motion h');ef=_array(ef,ef.shape,'fixed f')
    if any(a.ndim!=1 for a in (baseh,mh,ef)):raise ValueError('known vector bounds required')
    G0=_array(original_base_G,(len(baseh),n),'base G');Gm=_array(witnessed_motion_G,(len(mh),n),'motion G')
    E=_array(original_fixed_E,(len(ef),n),'fixed E')
    scale=1/float(np.sqrt(len(mb)))
    A=_array(np.vstack((N,scale*M)),(len(nb)+len(mb),n),'joint A')
    b=_array(np.r_[nb,scale*mb],(len(nb)+len(mb),),'joint b')
    G=_array(np.vstack((G0,Gm)),(len(baseh)+len(mh),n),'complete G')
    h=_array(np.r_[baseh,mh],(len(baseh)+len(mh),),'complete h')
    return JointTaskBundle(A,b,G,h,E,ef,lo,hi,len(nb),len(mb),scale)
