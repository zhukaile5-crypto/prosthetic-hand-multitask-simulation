"""Zero-acquisition adapter replacement on the complete frozen V19 runtime.

All holding, clocks, governor, progress, normal/reference QP, native branch,
plant, actor, devices and RNG use the unchanged parent. This session is an
assigned package ablation, never a new optimized winning strategy.
"""
from pathlib import Path
import hashlib
import numpy as np
from v19_contact_reference.runtime import ReleaseSession as FullV19Session
from v19_contact_reference.runtime import METHODS,MOTION_MODES,source_files as parent_files
from .adapter import NoMotionAdapter,ABLATION_ROLE

ROOT=Path(__file__).resolve().parents[1]
SOURCE_NAMES=('__init__.py','adapter.py','runtime.py','run_development.py')


def source_files(methods=None):
    files=dict(parent_files(methods))
    files.update({'v19_motion_ablation/'+name:ROOT/'v19_motion_ablation'/name for name in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in source_files(methods).items()}


class ReleaseSession(FullV19Session):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.tick!=0 or self.time!=0. or self.sensors.samples!=0 or self.trace:
            raise ValueError('motion task ablation must bind before any sampling or integration')
        self.release_adapter=NoMotionAdapter(self.plant.model,
            open18=np.concatenate(self.plant.open_pose),receiver_mount=self.receiver_mount)
        self.research_sources=source_identity([self.method])

    def metadata(self):
        value=super().metadata()
        value.update(schema='v19_motion_task_ablation_development_session_v1',
            benchmark_id='v19_motion_task_ablation_development',
            research_source_sha256=self.research_sources,
            frozen_full_v19_source_sha256={k:self.research_sources[k] for k in parent_files([self.method])},
            ablation_role=ABLATION_ROLE,motion_tasks_enabled=False,
            controller_scope='Frozen FullV19 with assigned combined motion-effort task disabled; original observations and remaining task unchanged',
            v19_motion_ablation=dict(task_package='external_absolute_projection_and_current_robot_motion_effort',
                disabled_output='tau_motion25',observation_geometry_and_real_tau_diagnostics_preserved=True,
                same_reference_progress_normal_QP_governor_native_branch=True,
                new_gain_weight_or_parameter_selection=False,
                observation_sampling_physics_permissions_deadlines_metrics_unchanged=True,
                role='causal package contrast, not a separately selected improved algorithm'))
        return value
