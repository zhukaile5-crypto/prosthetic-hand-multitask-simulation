"""V19 release integration, with all frozen holding and acquisition preserved."""
from pathlib import Path
import hashlib
import numpy as np
from v18_effort_transfer.runtime import ReleaseSession as FrozenSession
from v18_effort_transfer.runtime import METHODS,MOTION_MODES,source_files as parent_files
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .adapter import ContactMotionAdapter
from .supervisor import ReleaseSupervisor
from .design import SOURCE_NAMES,SCHEMA

ROOT=Path(__file__).resolve().parents[1]


def source_files(methods=None):
    files=dict(parent_files(methods))
    files.update({'v19_contact_reference/'+n:ROOT/'v19_contact_reference'/n for n in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source_files(methods).items()}


class ReleaseSession(FrozenSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.tick!=0 or self.time!=0. or self.sensors.samples!=0:
            raise ValueError('bind explicit reference tasks before any acquisition or physics')
        self.release_adapter=ContactMotionAdapter(self.plant.model,
            open18=np.concatenate(self.plant.open_pose),receiver_mount=self.receiver_mount)
        self.research_sources=source_identity([self.method])

    def _lifecycle_step(self,obs):
        if self.v16_release_policy=='coordinated' and self.v16_supervisor is None:
            entry=self.plant.joint_release_entry
            if entry is None:raise ValueError('actual release request must preempt holding')
            lower,upper=self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal=release_clearance_goal(entry,
                preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower,mechanical_upper21=upper)
            prior=self.release_adapter.begin_release(entry.started_s)
            observer=TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3,prior=prior)
            self.v16_supervisor=ReleaseSupervisor(entry,open18=self.release_clearance_goal.reference18,
                observer=observer,robot_model=self.plant.model,limits5=self.controller.limits,
                budget=self.controller.budget,force_margin=self.controller.g.force_margin)
            self.release_driver=self.v16_supervisor;self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value=super().metadata()
        value.update(schema=SCHEMA,benchmark_id='v19_explicit_reference_contact_transfer_development',
            decision_algorithm='joint_reference_progress_normal_intent_and_current_native_effort',
            research_source_sha256=self.research_sources,
            frozen_v18_source_sha256={k:self.research_sources[k] for k in parent_files([self.method])},
            controller_scope='Frozen assigned holding; explicit21 reference and native branch task in5ms release',
            v19_release_design=dict(reference_geometry_explicit=True,current_native_branch_not_global_linearization=True,
                absolute_projection_includes_coacquired_robot_velocity=True,
                robot_tangent_task_independent_of_external_projection_validity=True,
                normal_unload_and_clearance_progress_joint=True,physical_model_and_metrics_unchanged=True,
                future_motion_or_force_prediction_available=False,placement_certified=False))
        return value
