"""Same-information three-arm release integration, with frozen V20 retained."""
from pathlib import Path
import hashlib
import numpy as np
from v20_contact_exit.runtime import ReleaseSession as FrozenSession
from v20_contact_exit.runtime import METHODS, MOTION_MODES, source_files as parent_files
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .supervisor import ReleaseSupervisor
from .prefix_supervisor import ReleaseSupervisor as PrefixOnlySupervisor
from .design import SOURCE_NAMES, SCHEMA, RELEASE_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def source_files(methods=None):
    files = dict(parent_files(methods))
    files.update({'v21_stable_handoff/'+n: ROOT/'v21_stable_handoff'/n for n in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in source_files(methods).items()}


class ReleaseSession(FrozenSession):
    def __init__(self, *args, release_variant='v21', **kwargs):
        if release_variant not in RELEASE_VARIANTS:
            raise ValueError('explicit registered release variant required')
        self.release_variant = release_variant
        super().__init__(*args, **kwargs)
        if self.tick != 0 or self.time != 0. or self.sensors.samples != 0:
            raise ValueError('bind variant before acquisition or physics')
        self.research_sources = source_identity([self.method])

    def _lifecycle_step(self, obs):
        if self.release_variant == 'v20':
            return super()._lifecycle_step(obs)
        if self.v16_release_policy == 'coordinated' and self.v16_supervisor is None:
            entry = self.plant.joint_release_entry
            if entry is None:
                raise ValueError('actual release must preempt holding')
            lower, upper = self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal = release_clearance_goal(entry,
                preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower, mechanical_upper21=upper)
            prior = self.release_adapter.begin_release(entry.started_s)
            observer = TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=self.release_adapter.receiver_half_size_xy_m,
                receiver_up_axis3=self.release_adapter.receiver_up_axis3, prior=prior)
            cls = PrefixOnlySupervisor if self.release_variant == 'prefix' else ReleaseSupervisor
            self.v16_supervisor = cls(entry,
                open18=self.release_clearance_goal.reference18, observer=observer,
                robot_model=self.plant.model, limits5=self.controller.limits,
                budget=self.controller.budget, force_margin=self.controller.g.force_margin)
            self.release_driver = self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value = super().metadata()
        # These inherited dictionaries describe earlier active coordinators;
        # keep their provenance under one explicitly historical field.
        historical = {k: value.pop(k) for k in ('v19_release_design', 'v20_release_design') if k in value}
        value.update(schema=SCHEMA, benchmark_id='v21_same_information_handoff_development',
            release_variant=self.release_variant, research_source_sha256=self.research_sources,
            frozen_v20_source_sha256={k: self.research_sources[k] for k in parent_files([self.method])},
            historical_release_designs=historical,
            decision_algorithm={'v20': 'unchanged_original_V20',
                'prefix': 'original_V20_tasks_with_certified_prefix_commit',
                'v21': 'one_sided_normal_motion_budget_exit_path_certified_prefix'}[self.release_variant],
            controller_scope='Frozen holding/devices/plant; registered release variant only',
            v21_release_design=dict(same_information_all_variants=True,
                active_variant=self.release_variant, physical_model_and_evaluation_unchanged=True,
                full_hand_auxiliary_local_geometry_available=False,
                future_motion_or_force_prediction_available=False, placement_certified=False))
        return value
