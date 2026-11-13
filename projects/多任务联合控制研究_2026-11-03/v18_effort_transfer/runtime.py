"""Independent V18 release; frozen holding, devices, plant and evaluators."""
from pathlib import Path
import hashlib
import numpy as np
from v17b_full_evidence.runtime import ReleaseSession as FrozenSession
from v17b_full_evidence.runtime import METHODS, MOTION_MODES, source_files as parent_files
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .design import SOURCE_NAMES, SCHEMA
from .adapter import EffortContactAdapter
from .supervisor import ReleaseSupervisor

ROOT=Path(__file__).resolve().parents[1]


def source_files(methods=None):
    files=dict(parent_files(methods))
    files.update({'v18_effort_transfer/'+n:ROOT/'v18_effort_transfer'/n for n in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source_files(methods).items()}


class ReleaseSession(FrozenSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.tick!=0 or self.time!=0. or self.sensors.samples!=0:
            raise ValueError('new public adapter must bind before acquisition or physics')
        self.release_adapter=EffortContactAdapter(self.plant.model,
            open18=np.concatenate(self.plant.open_pose),receiver_mount=self.receiver_mount)
        self.research_sources=source_identity([self.method])

    def _lifecycle_step(self,obs):
        if self.v16_release_policy=='coordinated' and self.v16_supervisor is None:
            entry=self.plant.joint_release_entry
            if entry is None:raise ValueError('release must preempt holding before native task binding')
            lower,upper=self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal=release_clearance_goal(entry,
                preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower,mechanical_upper21=upper)
            prior=self.release_adapter.begin_release(entry.started_s)
            observer=TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3,prior=prior)
            self.v16_supervisor=ReleaseSupervisor(entry,open18=self.release_clearance_goal.reference18,
                observer=observer,robot_model=self.plant.model,
                limits5=self.controller.limits,budget=self.controller.budget,
                force_margin=self.controller.g.force_margin)
            self.release_driver=self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value=super().metadata()
        value.update(schema=SCHEMA,benchmark_id='v18_native_effort_transfer_development',
            decision_algorithm='current_native_effort_tasks_with_reference_debt_governor',
            v17b_change_role='frozen_parent_only_V18_release_is_a_new_algorithm',
            research_source_sha256=self.research_sources,
            frozen_v17b_source_sha256={k:self.research_sources[k] for k in parent_files([self.method])},
            controller_scope='Frozen assigned holding; current 5ms native-effort release task and debt governor',
            v18_release_design=dict(phases=['SEARCH','CAPTURE','TRANSFER','SEPARATE','UNSUPPORTED_EXIT'],
                native_task='single shifted mechanical residual, normal intent and soft posture effort',
                nominal_and_braking='explicit public robot q/v/u, software rate and recovery-budget governor',
                tangent='delayed per-digit device-plane signed effort intent; no passivity certificate',
                holding_unchanged=True,physical_model_and_limits_unchanged=True,
                original_metrics_deadline_and_failure_retention_unchanged=True,
                future_pressure_prediction_available=False,placement_certified=False))
        return value
