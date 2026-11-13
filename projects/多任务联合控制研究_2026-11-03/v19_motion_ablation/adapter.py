"""Disable an assigned task, not the device or its velocity observations.

The full real V19 adapter still constructs every pressure/J, absolute external
projection, current robot motion and uncertainty/domain diagnostic. Only its
combined motion generalized-effort RHS is replaced by an explicit inactive0.
This ablates the external-plus-robot motion TASK PACKAGE, not either component
separately and not all frictional or motion-dependent physical effects.
"""
from copy import deepcopy
import numpy as np
from v19_contact_reference.adapter import ContactMotionAdapter

ABLATION_ROLE='same_V19_governor_progress_normal_reference_tasks_without_motion_effort_package'


class NoMotionAdapter(ContactMotionAdapter):
    def current_contact_intents(self,frame,*,desired_lift_velocity_m_s=0.):
        result=super().current_contact_intents(frame,
            desired_lift_velocity_m_s=desired_lift_velocity_m_s)
        original=np.asarray(result['tau_motion25'],dtype=float)
        if original.shape!=(25,) or not np.isfinite(original).all():
            raise ValueError('parent motion task output must be a known finite25-vector before assigned disabling')
        # This0 is explicitly the disabled TASK coefficient; no missing device
        # sample, invalid projection, zero speed or zero pressure is inferred.
        result['tau_motion25']=np.zeros(25,dtype=float)
        result['diagnostics']=deepcopy(result['diagnostics'])
        result['diagnostics'].update(motion_tasks_enabled=False,
            ablation_role=ABLATION_ROLE,
            original_motion_generalized_effort25=original.tolist(),
            assigned_disabled_generalized_effort25=[0.]*25,
            motion_task_disabled_is_not_missing_observation=True,
            zero_task_is_not_observed_zero_velocity=True,
            ablated_package='external_absolute_projection_plus_current_robot_motion_effort',
            normal_geometry_and_all_original_motion_diagnostics_preserved=True)
        return result
