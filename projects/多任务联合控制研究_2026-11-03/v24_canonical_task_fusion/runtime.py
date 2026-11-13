"""Three explicit task roles with unchanged V23 physical/execution rules.

v23_joint directly constructs the frozen V23 joint supervisor. The two
canonical roles use this package's task solver, sharing all public observations,
physical permissions, holding policy, path memory, sole actor and evaluator.
"""
from pathlib import Path
import hashlib
import numpy as np
from v22_joint_task_fusion.numerics import ensure_numeric_environment, numerical_identity
ensure_numeric_environment()
from v20_contact_exit.runtime import ReleaseSession as FrozenSession
from v20_contact_exit.runtime import METHODS, MOTION_MODES
from v21_stable_handoff.runtime import source_files as frozen_v21_files
from v22_joint_task_fusion.runtime import source_files as frozen_v22_files
from v23_task_execution.runtime import source_files as frozen_v23_files
from v23_task_execution.supervisor import ReleaseSupervisor as FrozenV23Supervisor
from v16b_release_goal.goal import release_clearance_goal
from v16_release_geometry.transfer_observer import TransferObserver
from .supervisor import ReleaseSupervisor
from .design import SOURCE_NAMES, SCHEMA, RELEASE_VARIANTS, ROLE_FORMULATIONS

ROOT = Path(__file__).resolve().parents[1]


def source_files(methods=None):
    # Every role pins the complete frozen V23 closure and all six new sources.
    # The finite solver, task bundle and numerics remain original dependencies.
    files = dict(frozen_v23_files(methods))
    files.update({'v24_canonical_task_fusion/'+n: ROOT/'v24_canonical_task_fusion'/n for n in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in source_files(methods).items()}


class ReleaseSession(FrozenSession):
    def __init__(self, *args, release_variant='canonical_joint', **kwargs):
        if release_variant not in RELEASE_VARIANTS:
            raise ValueError('explicit registered v23_joint/canonical_backend/canonical_joint role required')
        self.release_variant = release_variant
        self.numerical_sources = numerical_identity()
        super().__init__(*args, **kwargs)
        if self.tick != 0 or self.time != 0. or self.sensors.samples != 0:
            raise ValueError('bind task role before acquisition or physics')
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
            cls = FrozenV23Supervisor if self.release_variant == 'v23_joint' else ReleaseSupervisor
            self.v16_supervisor = cls(entry,
                open18=self.release_clearance_goal.reference18, observer=observer,
                robot_model=self.plant.model, limits5=self.controller.limits,
                budget=self.controller.budget, force_margin=self.controller.g.force_margin,
                formulation=ROLE_FORMULATIONS[self.release_variant])
            self.release_driver = self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        value = super().metadata()
        historical = {k: value.pop(k) for k in ('v19_release_design', 'v20_release_design') if k in value}
        active = getattr(self, 'v16_supervisor', None)
        frozen = self.release_variant == 'v23_joint'
        formulation = ROLE_FORMULATIONS[self.release_variant]
        value.update(schema=SCHEMA, benchmark_id='v24_canonical_task_contract_development',
            release_variant=self.release_variant, release_role=self.release_variant,
            task_contract='frozen_V23' if frozen else 'canonical_V24',
            research_source_sha256=self.research_sources,
            numerical_identity=self.numerical_sources,
            frozen_v21_source_sha256={k: self.research_sources[k] for k in frozen_v21_files([self.method])},
            frozen_v22_source_sha256={k: self.research_sources[k] for k in frozen_v22_files([self.method])},
            frozen_v23_source_sha256={k: self.research_sources[k] for k in frozen_v23_files([self.method])},
            historical_release_designs=historical,
            decision_algorithm={'v23_joint': 'frozen_V23_joint_supervisor_and_tasks',
                'canonical_backend': 'canonical_normal_budget_witness_with_hierarchy_tasks',
                'canonical_joint': 'canonical_normal_budget_witness_with_joint_tasks'}[self.release_variant],
            active_release_supervisor_class=None if active is None else type(active).__module__+'.'+type(active).__qualname__,
            controller_scope='Frozen V23 holding/devices/plant/authority/evaluation; registered release task contract only',
            v24_release_design=dict(same_information_all_variants=True,
                active_role=self.release_variant, canonical_task_contract=not frozen,
                declared_task_formulation=formulation,
                declared_coordinated_supervisor_is_frozen_v23=frozen,
                receiver_policy=self.v16_release_policy,
                coordinated_release_supervisor_bound=active is not None,
                actual_task_formulation=None if active is None else getattr(active, 'formulation', None),
                physical_model_and_evaluation_unchanged=True,
                full_hand_auxiliary_local_geometry_available=False,
                future_motion_or_force_prediction_available=False,
                mechanical_reference_bounds_are_not_contact_force_certification=True,
                placement_certified=False))
        return value
