"""Task-goal correction around the frozen V16 whole release architecture.

This is an implementation correction, not a new control gain or contribution.
V16 source123 and all failed records stay frozen; only the goal binding changes.
"""
from pathlib import Path
import hashlib
import numpy as np
from v16_release_geometry.runtime import ReleaseSession as FrozenReleaseSession
from v16_release_geometry.runtime import METHODS,MOTION_MODES,source_files as parent_files
from v15_support.runtime import source_files as original_files
from v16_release_geometry.transfer_observer import TransferObserver
from v16_release_geometry.supervisor import ReleaseSupervisor
from .goal import release_clearance_goal

ROOT=Path(__file__).resolve().parents[1]


def source_files(methods=None):
    files=dict(parent_files(methods))
    files.update({'v16b_release_goal/'+n:ROOT/'v16b_release_goal'/n for n in ('goal.py','runtime.py','run_development.py')})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in source_files(methods).items()}


class ReleaseSession(FrozenReleaseSession):
    def __init__(self,*args,**kwargs):
        self.release_clearance_goal=None
        super().__init__(*args,**kwargs)
        self.research_sources=source_identity([self.method])

    def _lifecycle_step(self,obs):
        if self.v16_release_policy=='coordinated' and self.v16_supervisor is None:
            entry=self.plant.joint_release_entry
            if entry is None:raise ValueError('Release must preempt before binding its goal')
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
        value.update(schema='v16b_explicit_release_clearance_goal_trial_v1',
            benchmark_id='v16b_goal_contract_correction_development',
            release_clearance_goal=None if self.release_clearance_goal is None else self.release_clearance_goal.to_public(),
            task_goal_correction_is_not_algorithm_contribution=True)
        value['frozen_v15_source_sha256']={key:self.research_sources[key] for key in original_files([self.method])}
        value['frozen_v16_source_sha256']={key:self.research_sources[key] for key in parent_files([self.method])}
        return value
