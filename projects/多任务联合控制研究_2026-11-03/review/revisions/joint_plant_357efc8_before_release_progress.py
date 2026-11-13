"""Independent full21 commit/mode execution using the frozen A1 sole writer.

No new force model or physics loop. Preparation retains original A1 proposals.
After activation, committed21 is authoritative at every physical substep.
"""
from dataclasses import dataclass
import numpy as np
import time
from v13_a1.plant import A1Plant
from v13_a1.codec import Reference21,EnginePort21,finite_array
from .design import reference_rates
from .lifecycle import ReleaseEntry,release_nominal

@dataclass(frozen=True)
class ExecutionReceipt21:
    requested:Reference21
    executed:Reference21
    engine:EnginePort21
    mode:str
    mode_generation:int
    execution_changed:bool
    stale_plan_replaced:bool

class JointPlant(A1Plant):
    def reset(self):
        next_generation=getattr(self,'mode_generation',-1)+1
        self._joint_committed=None;self._joint_mode=None;self.mode_generation=next_generation
        self._joint_anchor_time=None;self._joint_anchor_reference=None
        self.joint_release_entry=None;self.last_execution_receipt21=None
        self.actor_callback_total_ns=0;self.actor_callback_calls=0
        return super().reset()

    def reference_snapshot(self):
        source=self._joint_committed if self._joint_committed is not None else self.a1_last_proposal.reference21
        # This is current held-value observation time, not its first issue time.
        return Reference21(source.values,self.time)

    def engine_snapshot(self):
        return EnginePort21(self.data.ctrl[self.actor21.indices21].copy(),self.time,self.actor21.lift_bias_m)

    def reference_authority21(self):
        lo=self.actor21.ctrl_range21[:,0].copy();hi=self.actor21.ctrl_range21[:,1].copy()
        lo[18]-=self.actor21.lift_bias_m;hi[18]-=self.actor21.lift_bias_m
        lo[18]=max(0.,lo[18]);hi[18]=min(self.config.lift_target,hi[18])
        for i,name in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
            j=self.model.joint(name).id
            if self.model.jnt_limited[j]:lo[i]=max(lo[i],self.model.jnt_range[j,0]);hi[i]=min(hi[i],self.model.jnt_range[j,1])
        if not np.isfinite(np.r_[lo,hi]).all() or np.any(lo>hi):raise ValueError('Unknown/empty full21 reference authority')
        return lo,hi

    def begin_release(self):
        if self.release_started is None:
            actual=self.reference_snapshot()
            encoded=self.actor21.encode(actual).values
            observed=self.engine_snapshot().values
            if encoded.tobytes()!=observed.tobytes():raise RuntimeError('Release entry command authority disagrees with observed robot ports')
            super().begin_release() # Metadata only; original method writes no engine.
            self.mode_generation+=1;self._joint_mode='release'
            self._joint_committed=actual
            self.joint_release_entry=ReleaseEntry(self.time,actual,self.mode_generation)
            self._apply_ctrl()

    def release_proposal(self,query_time_s):
        if self.joint_release_entry is None:raise ValueError('Release must preempt before its proposal')
        lo,hi=self.reference_authority21()
        return release_nominal(self.joint_release_entry,query_time_s,np.concatenate(self.open_pose),lo,hi)

    def commit_reference21(self,reference,*,mode,expected_generation):
        if not isinstance(reference,Reference21):raise TypeError('Typed full21 proposal required')
        if abs(reference.time_s-self.time)>1e-9:raise ValueError('Proposal issue clock must match current public robot clock')
        if mode not in ('holding','release','monitor'):raise ValueError('Unknown execution mode')
        if isinstance(expected_generation,bool) or not isinstance(expected_generation,(int,np.integer)) or expected_generation<0:raise ValueError('Explicit nonnegative mode generation required')
        if self.release_started is None and mode in ('release','monitor'):raise ValueError('Release must be requested before release-mode commit')
        stale=expected_generation!=self.mode_generation
        requested=reference
        if self.release_started is not None and (stale or mode not in ('release','monitor')):
            ref=self.release_proposal(self.time+self.config.control_dt)
            reference=Reference21(ref.values,self.time);mode='release';stale=True
        elif stale:
            # A stale holding plan cannot overwrite current independent posture.
            reference=self.reference_snapshot()
        current=self.reference_snapshot().values
        if self._joint_anchor_time is not None and self.time<self._joint_anchor_time:raise ValueError('Public command clock rewind requires an explicit reset')
        if self._joint_anchor_time!=self.time:
            self._joint_anchor_time=self.time;self._joint_anchor_reference=current.copy()
        anchor=self._joint_anchor_reference
        lo,hi=self.reference_authority21();step=reference_rates(mode)*self.config.control_dt
        lower=np.maximum(lo,anchor-step);upper=np.minimum(hi,anchor+step)
        if np.any(lower>upper):raise ValueError('Current command cannot reach the declared authority in one control tick')
        values=np.clip(reference.values,lower,upper)
        executed=Reference21(values,self.time)
        self._joint_mode=mode
        self._joint_committed=executed
        self.lift_ref=float(values[18]);self.wrist_ref=values[19:].copy()
        self._apply_ctrl()
        changed=requested.values.tobytes()!=values.tobytes()
        receipt=ExecutionReceipt21(requested,executed,self.a1_last_engine_port21,mode,self.mode_generation,changed,stale)
        self.last_execution_receipt21=receipt
        return receipt

    def _apply_ctrl(self):
        began=time.perf_counter_ns()
        try:
            if getattr(self,'_joint_committed',None) is None:return super()._apply_ctrl()
            reference=Reference21(self._joint_committed.values,self.time)
            self.a1_last_engine_port21=self.actor21.apply(self.data,reference)
            self.data.ctrl[self.obstacle_motor]=float(np.clip(self.env.get('obstacle_x',0.),-.4,.1))
        finally:
            self.actor_callback_total_ns+=time.perf_counter_ns()-began
            self.actor_callback_calls+=1

    def set_closure(self,alpha):
        if self._joint_committed is not None:raise RuntimeError('Legacy alpha cannot overwrite committed full21 posture')
        return super().set_closure(alpha)

    def set_motion(self,lift=None,wrist=None):
        if self._joint_committed is not None:raise RuntimeError('Use one typed21 commit for motion and fingers')
        return super().set_motion(lift,wrist)
