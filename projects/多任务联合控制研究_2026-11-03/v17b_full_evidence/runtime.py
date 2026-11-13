"""V17 decisions with complete optional motion-task evidence.

This is a logging-only namespace. Parent sources, instruments, execution,
holding/release policy, physical parameters and evaluators stay frozen.
"""
from pathlib import Path
import hashlib
import numpy as np
from v17_contact_transfer.runtime import ReleaseSession as FrozenSession
from v17_contact_transfer.runtime import METHODS, MOTION_MODES, source_files as parent_files
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .supervisor import ReleaseSupervisor

ROOT = Path(__file__).resolve().parents[1]
NEW_FILES = ('runtime.py', 'supervisor.py', 'run_development.py')


def source_files(methods=None):
    files = dict(parent_files(methods))
    files.update({'v17b_full_evidence/' + name: ROOT / 'v17b_full_evidence' / name for name in NEW_FILES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in source_files(methods).items()}


class ReleaseSession(FrozenSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.research_sources = source_identity([self.method])

    def _lifecycle_step(self, obs):
        if self.v16_release_policy == 'coordinated' and self.v16_supervisor is None:
            # Literal parent construction with only the evidence-owner class changed.
            entry = self.plant.joint_release_entry
            if entry is None:
                raise ValueError('Release request must preempt contact-task control')
            lower, upper = self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal = release_clearance_goal(entry,
                preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower, mechanical_upper21=upper)
            prior = self.release_adapter.begin_release(entry.started_s)
            observer = TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3, prior=prior)
            self.v16_supervisor = ReleaseSupervisor(entry,
                open18=self.release_clearance_goal.reference18, observer=observer)
            self.release_driver = self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value = super().metadata()
        value.update(schema='v17b_complete_motion_evidence_trial_v1',
            benchmark_id='v17b_logging_only_v17_prospective_development',
            frozen_v17_source_sha256={key: self.research_sources[key] for key in parent_files([self.method])},
            v17b_change_role='logging_only_copy_of_existing_public_motion_matrix_and_target_after_decision',
            decision_algorithm='unchanged_V17_contact_transfer')
        return value
