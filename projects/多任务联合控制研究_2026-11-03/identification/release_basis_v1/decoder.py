"""Two-coordinate native release decoder; no pressure/safety model.

The path coordinate is elapsed native path time (s), not a holding synergy.
The independent lift coordinate is a public robot reference (m).  All 18
finger references are still decoded and recorded in their actual units.
"""
from dataclasses import dataclass
import numpy as np

FINGER_AXES=tuple('rh_A_'+n for n in ('THJ5','THJ4','THJ3','THJ2','THJ1',
    'FFJ4','FFJ3','FFJ0','MFJ4','MFJ3','MFJ0','RFJ4','RFJ3','RFJ0','LFJ5','LFJ4','LFJ3','LFJ0'))
COMMAND_AXES=FINGER_AXES+('lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')
PATHS=('native2d','fixed_height')


def vector(value,n,label):
    x=np.asarray(value,float)
    if x.shape!=(n,) or not np.isfinite(x).all():raise ValueError(f'{label} requires {n} finite values')
    return x.copy()


@dataclass(frozen=True)
class DecodedRelease:
    finger_reference:np.ndarray
    lift_reference:float
    wrist_reference:np.ndarray
    alpha:np.ndarray
    path_time_s:float

    @property
    def command21(self):return np.r_[self.finger_reference,self.lift_reference,self.wrist_reference]


class ReleasePath2D:
    def __init__(self,*,entry_alpha,entry_thumb,entry_lift,entry_wrist,open_pose,closed_pose,
                 finger_lower,finger_upper,release_open_rate):
        self.entry_alpha=vector(entry_alpha,5,'entry alpha')
        self.entry_thumb=vector(entry_thumb,5,'actual entry thumb references')
        self.entry_lift=float(entry_lift);self.entry_wrist=vector(entry_wrist,2,'entry wrists')
        self.open_pose=tuple(vector(v,n,'open pose') for v,n in zip(open_pose,(5,3,3,3,4)))
        self.closed_pose=tuple(vector(v,n,'closed pose') for v,n in zip(closed_pose,(5,3,3,3,4)))
        if len(self.open_pose)!=5 or len(self.closed_pose)!=5:raise ValueError('Five actual finger groups required')
        self.lower=vector(finger_lower,18,'finger actuator lower');self.upper=vector(finger_upper,18,'finger actuator upper')
        self.rate=float(release_open_rate)
        if not np.isfinite([self.entry_lift,self.rate]).all() or self.rate<=0 or np.any(self.lower>self.upper):
            raise ValueError('Finite native lift/rate and real actuator intervals required')
        if abs(self.closed_pose[0][0]-self.open_pose[0][0])<1e-12:raise ValueError('Native thumb diagnostic coordinate is singular')

    @classmethod
    def from_plant(cls,p):
        if p.release_started is None:raise ValueError('Actual native release entry must be latched first')
        aids=np.concatenate(p.actuator_groups)
        return cls(entry_alpha=p.release_alpha,entry_thumb=p.release_thumb,entry_lift=p.release_lift,
            entry_wrist=p.wrist_ref,open_pose=p.open_pose,closed_pose=p.closed_pose,
            finger_lower=p.model.actuator_ctrlrange[aids,0],finger_upper=p.model.actuator_ctrlrange[aids,1],
            release_open_rate=p.config.release_open_rate)

    def native_lift(self,path_time_s):
        # Match the native expression/order exactly; it is part of the decoder
        # equivalence check, not an independently invented lowering profile.
        return self.entry_lift*(1.-np.clip(float(path_time_s)/.6,0.,1.))

    def decode(self,path_time_s,lift_reference):
        elapsed=float(path_time_s);lift=float(lift_reference)
        if not np.isfinite([elapsed,lift]).all() or elapsed<0.:raise ValueError('Finite nonnegative path time and finite lift reference required')
        alpha=self.entry_alpha.copy()
        alpha[1:]=np.maximum(0.,self.entry_alpha[1:]-self.rate*elapsed)
        tucked=np.array([-1.03,1.21,.13,-.02,.985]);full_open=np.array([-1.,.1,0.,0.,0.])
        if elapsed<.8:thumb=self.entry_thumb+np.clip(elapsed/.8,0,1)*(tucked-self.entry_thumb)
        elif elapsed<1.4:thumb=tucked
        else:thumb=tucked+np.clip((elapsed-1.4)/.8,0,1)*(full_open-tucked)
        # Native clips the thumb explicitly. Its nonthumb release path does
        # not do a second hidden clip; validation rejects any real violation.
        fingers=[np.clip(thumb,self.lower[:5],self.upper[:5])]
        for i in range(1,5):fingers.append(self.open_pose[i]+alpha[i]*(self.closed_pose[i]-self.open_pose[i]))
        alpha[0]=(thumb[0]-self.open_pose[0][0])/(self.closed_pose[0][0]-self.open_pose[0][0])
        command=np.concatenate(fingers)
        if np.any(command<self.lower-1e-10) or np.any(command>self.upper+1e-10):
            raise ValueError('Native path outside actual actuator authority; no silent clipping')
        return DecodedRelease(command,lift,self.entry_wrist.copy(),alpha,elapsed)

    def at(self,elapsed,path):
        if path not in PATHS:raise ValueError('Unknown registered release path')
        return self.decode(elapsed,self.native_lift(elapsed) if path=='native2d' else self.entry_lift)


def make_plant_class():
    """Import the frozen plant only when constructing a zero-step/real probe.

    Native is evaluated first as the exact oracle at EVERY application. The
    independent decoder must match the entire native ctrl in A, and all finger
    and wrist ctrl in B, before any subsequent integration can take place.
    """
    from hand_plant import HandPlant

    class BasisHandPlant(HandPlant):
        def __init__(self,config,path):
            if path not in PATHS:raise ValueError('Registered path required')
            self.basis_path=path;self.basis_decoder=None;self.basis_rows=[]
            self.basis_apply_count=0;self.basis_mismatch_count=0;self.basis_first_mismatch=None
            super().__init__(config)

        def begin_release(self):
            fresh=self.release_started is None
            super().begin_release()
            if fresh:self.basis_decoder=ReleasePath2D.from_plant(self)

        def _apply_ctrl(self):
            HandPlant._apply_ctrl(self)
            if self.release_started is None:return
            if self.basis_decoder is None:raise RuntimeError('Missing exact actual release entry')
            native=self.data.ctrl.copy();native_lift=float(self.lift_ref);native_alpha=self.alpha.copy()
            elapsed=self.time-self.release_started;decoded=self.basis_decoder.at(elapsed,self.basis_path)
            aids=np.concatenate(self.actuator_groups)
            wrist=np.array([self.aid('rh_A_WRJ2'),self.aid('rh_A_WRJ1')],int)
            gravity=self.model.body_subtreemass[self.hand]*9.81/1500.
            candidate=native.copy();candidate[aids]=decoded.finger_reference
            candidate[wrist]=decoded.wrist_reference;candidate[self.lift_actuator]=decoded.lift_reference+gravity
            identical=np.array_equal(candidate,native) if self.basis_path=='native2d' else np.array_equal(candidate[np.r_[aids,wrist]],native[np.r_[aids,wrist]])
            identical=bool(identical and np.array_equal(decoded.alpha,native_alpha))
            self.basis_apply_count+=1
            if not identical:
                self.basis_mismatch_count+=1
                if self.basis_first_mismatch is None:self.basis_first_mismatch=dict(time=self.time,physical_time=float(self.data.time),
                    native_ctrl=native.tolist(),decoded_ctrl=candidate.tolist(),native_alpha=native_alpha.tolist(),decoded_alpha=decoded.alpha.tolist())
                raise AssertionError('2D decoder failed native exactness BEFORE integration')
            self.data.ctrl[:]=candidate;self.lift_ref=decoded.lift_reference;self.wrist_ref=decoded.wrist_reference.copy();self.alpha=decoded.alpha.copy()
            native21=np.r_[native[aids],native_lift,native[wrist]]
            self.basis_rows.append(np.r_[self.time,float(self.data.time),elapsed,decoded.command21,native21,candidate,native])

    return BasisHandPlant
