"""Isolate V9 force-projection and additional risk-motion-hold packages.

P1 delegates the frozen V9 step. P0 returns the baseline's actual action
without novel-contact masks, reserve projection, or memory rollback. Baseline
pause/release/finite mechanics remain. No environment/variant-specific gains.
"""
from copy import deepcopy
import numpy as np
from v9.controller import JointTaskController
from vendor.frozen_controllers import ControlAction

VARIANTS={'P0H0':(False,False),'P0H1':(False,True),
          'P1H0':(True,False),'P1H1':(True,True)}

class ComponentController(JointTaskController):
    def __init__(self,calibration,limits,budget,gains,component_variant):
        if component_variant not in VARIANTS:raise ValueError('Unknown predeclared component variant')
        self.component_variant=component_variant
        self.projection_enabled,self.risk_hold_enabled=VARIANTS[component_variant]
        self.counterfactual_risk_hold=False
        super().__init__(calibration,limits,budget,gains)

    def _risk(self,known):
        # Preserve the ORIGINAL hysteresis as evidence even when its additional
        # motion authority is disabled. This never changes force observations.
        if not self.risk_hold_enabled:
            self.recovery_motion_hold=self.counterfactual_risk_hold
        caps,budget,evidence=super()._risk(known)
        self.counterfactual_risk_hold=bool(self.recovery_motion_hold)
        if not self.risk_hold_enabled:self.recovery_motion_hold=False
        return caps,budget,evidence

    def _annotate(self,action):
        action.debug=dict(action.debug or {})
        action.debug['component_ablation']=dict(variant=self.component_variant,
            projection_package=self.projection_enabled,
            additional_risk_motion_hold=self.risk_hold_enabled,
            counterfactual_risk_hold=self.counterfactual_risk_hold,
            actual_additional_risk_hold=bool(self.recovery_motion_hold),
            package_scope='projection includes novel mask and reference antiwindup; not separate novel-mask causality')
        return action

    def step(self,frame,phase=None,command='hold'):
        if self.projection_enabled:
            return self._annotate(super().step(frame,phase,command))
        # This branch has no projection call, novel mask, minimum reserve,
        # reference-memory copy/rollback, or late permission modification.
        if command=='release' or self.task_request.get('mode') in ('release','monitor') or str(phase or frame.phase).upper() in ('RELEASE','RELEASING'):
            self.release_latched=True
        action=self.nominal.step(frame,phase,'release' if self.release_latched else command)
        self.desired=self.nominal.desired.copy();self.predicted[:]=np.nan
        if self.release_latched:
            self.follower.reset();self.motion_reference=self.executed_motion.copy();self.interval_action[:]=0.
            return self._annotate(ControlAction(action.delta_closure.copy(),self.predicted.copy(),
                'component_native_release',False,dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='native release outside held allocation')))
        known,novel=self._known();caps,budget,risk=self._risk(known)
        paused=bool(self.recovery_motion_hold or action.pause_transport)
        self.motion_reference=self.follower.step(self.task_request,pause_transport=paused)
        self.interval_action[:5]=self.nominal.interval_action.copy()
        self.last_decision=self.nominal.last_decision
        debug=deepcopy(action.debug or {})
        debug.update(force_path='unchanged original ratio actual action',force_projection_applied=False,
            reference_memory_rollback=False,recovery_motion_hold=self.recovery_motion_hold,
            recovery_evidence=risk,new_unaffiliated_contact=novel.tolist(),known_grasp_support=known.tolist(),
            pause_transport=paused,prediction_available=False,
            prediction_unavailable_reason='ablation exposes no physical force forecast',
            actual_subinterval_delta_closure=action.delta_closure.tolist())
        return self._annotate(ControlAction(action.delta_closure.copy(),self.predicted.copy(),
            'component_unmodified_ratio',paused,debug))
