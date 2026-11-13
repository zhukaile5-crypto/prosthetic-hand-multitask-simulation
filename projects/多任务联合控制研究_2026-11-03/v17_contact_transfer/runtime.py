"""Independent contact-task release with a shared motion-domain instrument.

Frozen holding, physical model, actuator writer and evaluator stay unchanged.
All policies get the same public domain metadata; only coordinated release uses
the new approach/unloading/motion tasks. No object truth enters these tasks.
"""
from pathlib import Path
import hashlib
import numpy as np
from v16b_release_goal.runtime import ReleaseSession as FrozenGoalSession
from v16b_release_goal.runtime import METHODS,MOTION_MODES,source_files as parent_files
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .motion_domain_sensors import MotionDomainSensors
from .motion_adapter import PublicMotionAdapter
from .supervisor import ReleaseSupervisor

ROOT=Path(__file__).resolve().parents[1]
NEW_FILES=('runtime.py','run_development.py','supervisor.py','task_solver.py','hierarchy.py','motion_domain_sensors.py','motion_adapter.py')


def source_files(methods=None):
    files=dict(parent_files(methods))
    files.update({'v17_contact_transfer/'+n:ROOT/'v17_contact_transfer'/n for n in NEW_FILES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in source_files(methods).items()}


class ReleaseSession(FrozenGoalSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.sensors.samples!=0 or self.tick!=0 or self.time!=0.:
            raise ValueError('Shared motion domain must be bound before first acquisition/integration')
        self.sensors=MotionDomainSensors(self.sensors)
        self.release_adapter=PublicMotionAdapter(self.plant.model,
            open18=np.concatenate(self.plant.open_pose),receiver_mount=self.receiver_mount)
        self.research_sources=source_identity([self.method])

    def _lifecycle_step(self,obs):
        if self.v16_release_policy=='coordinated' and self.v16_supervisor is None:
            entry=self.plant.joint_release_entry
            if entry is None:raise ValueError('Release request must preempt contact-task control')
            lower,upper=self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal=release_clearance_goal(entry,
                preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower,mechanical_upper21=upper)
            prior=self.release_adapter.begin_release(entry.started_s)
            observer=TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3,prior=prior)
            self.v16_supervisor=ReleaseSupervisor(entry,open18=self.release_clearance_goal.reference18,observer=observer)
            self.release_driver=self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value=super().metadata()
        value.update(schema='v17_public_contact_task_transfer_trial_v1',
            benchmark_id='v17_public_contact_task_transfer_development',
            frozen_v16b_source_sha256={key:self.research_sources[key] for key in parent_files([self.method])},
            v17_motion_domain_instrument='Existing delayed slip valid domain exposed from same acquisition; original measurements/RNG unchanged; all policies receive it',
            v17_release_design=dict(
                phases=['support_approach','contact_unloading','free_clear_and_retreat','monitor_unconfirmed'],
                progress='observed-pressure-weighted normal withdrawal in declared length-proposal geometry',
                motion_task='coacquired external contact-average projected velocity correction when measured domain is valid',
                posture_goal='frozen release terminal18 only as secondary preference',
                current_robot_dynamics_or_future_pressure_not_predicted=True,
                object_pose_or_identity_not_controller_input=True,
                original_deadlines_metrics_and_physical_limits_unchanged=True))
        value['controller_scope']='Frozen candidateA/legacy holding as assigned; shared motion domain and contact-task coordinated release as assigned'
        value['coordination_contract']['release_policy']=self.v16_release_policy
        return value
