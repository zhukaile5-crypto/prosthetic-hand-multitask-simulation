"""Three registered release formulations on the unchanged V21 execution layer.

The v21 arm constructs the actual frozen V21 supervisor. Backend/joint use the
new supervisor and differ only by its explicit task-formulation argument.
The parent physical session, holding controller, adapter, path memory, actor,
permissions, public information and evaluator are retained without re-sampling.
"""
from pathlib import Path
import hashlib
import numpy as np
from .numerics import ensure_numeric_environment, numerical_identity
ensure_numeric_environment()
from v20_contact_exit.runtime import ReleaseSession as FrozenSession
from v20_contact_exit.runtime import METHODS, MOTION_MODES
from v21_stable_handoff.runtime import source_files as frozen_v21_files
from v21_stable_handoff.supervisor import ReleaseSupervisor as FrozenV21Supervisor
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .supervisor import ReleaseSupervisor
from .design import SOURCE_NAMES, SCHEMA, RELEASE_VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def source_files(methods=None):
    # Pin the full actually executed V21 closure plus EVERY V22 source. Missing
    # new numerical sources fail identity construction; there is no soft skip.
    files = dict(frozen_v21_files(methods))
    files.update({'v22_joint_task_fusion/'+n: ROOT/'v22_joint_task_fusion'/n for n in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in source_files(methods).items()}


class ReleaseSession(FrozenSession):
    def __init__(self, *args, release_variant='joint', **kwargs):
        if release_variant not in RELEASE_VARIANTS:
            raise ValueError('explicit registered v21/backend/joint variant required')
        self.release_variant = release_variant
        self.numerical_sources = numerical_identity()
        super().__init__(*args, **kwargs)
        if self.tick != 0 or self.time != 0. or self.sensors.samples != 0:
            raise ValueError('bind variant before acquisition or physics')
        self.research_sources = source_identity([self.method])
        if numerical_identity() != self.numerical_sources:
            raise ValueError('numerical identity changed during zero-step construction')

    def _lifecycle_step(self, obs):
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
            cls = FrozenV21Supervisor if self.release_variant == 'v21' else ReleaseSupervisor
            options = {} if self.release_variant == 'v21' else dict(formulation=self.release_variant)
            self.v16_supervisor = cls(entry,
                open18=self.release_clearance_goal.reference18, observer=observer,
                robot_model=self.plant.model, limits5=self.controller.limits,
                budget=self.controller.budget, force_margin=self.controller.g.force_margin, **options)
            self.release_driver = self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value = super().metadata()
        historical = {k: value.pop(k) for k in ('v19_release_design', 'v20_release_design') if k in value}
        active = getattr(self, 'v16_supervisor', None)
        value.update(schema=SCHEMA, benchmark_id='v22_same_information_joint_task_development',
            release_variant=self.release_variant, research_source_sha256=self.research_sources,
            numerical_identity=self.numerical_sources,
            frozen_v21_source_sha256={k: self.research_sources[k] for k in frozen_v21_files([self.method])},
            historical_release_designs=historical,
            decision_algorithm={'v21': 'frozen_original_V21_supervisor_and_tasks',
                'backend': 'original_V21_task_rules_with_new_finite_backend',
                'joint': 'joint_normal_motion_exit_path_tasks_with_same_finite_backend'}[self.release_variant],
            active_release_supervisor_class=None if active is None else type(active).__module__+'.'+type(active).__qualname__,
            controller_scope='Frozen V21 holding/devices/plant/authority/evaluation; registered release tasks only',
            v22_release_design=dict(same_information_all_variants=True,
                active_variant=self.release_variant,
                declared_task_formulation=None if self.release_variant == 'v21' else self.release_variant,
                declared_coordinated_supervisor_is_frozen_v21=self.release_variant == 'v21',
                receiver_policy=self.v16_release_policy,
                coordinated_release_supervisor_bound=active is not None,
                actual_task_formulation=None if active is None else getattr(active, 'formulation', None),
                physical_model_and_evaluation_unchanged=True,
                full_hand_auxiliary_local_geometry_available=False,
                future_motion_or_force_prediction_available=False,
                mechanical_reference_bounds_are_not_contact_force_certification=True,
                placement_certified=False))
        return value
