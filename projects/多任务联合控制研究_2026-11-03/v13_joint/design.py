"""Declared controller design, fixed before this version's first physics run.

Reference rates are software design choices, not hardware velocity certificates.
Metric definitions/weights, physical model and native gains are not changed.
"""
import numpy as np
SCHEMA='joint_current_effort_controller_design_v1'
DECISION_PERIOD_S=.020
CONTROL_PERIOD_S=.005
HOLDING_REFERENCE_RATE21=(1.2,)*18+(.035,.12,.12)
RELEASE_REFERENCE_RATE21=(3.0,)*18+(.18,.12,.12)
NONTHUMB_CLEAR_TIME_S=1.4
THUMB_TUCK_TIME_S=.8
THUMB_TUCK_UNTIL_S=1.4
THUMB_UNFOLD_TIME_S=.8
LIFT_DESCENT_TIME_S=.6
QP_MAX_ITERATIONS=400
QP_ATOL=1e-8
QP_RTOL=1e-7
CONTROL_OBJECTIVE_MULTIPLIERS=dict(effort_residual=1.,force_prior=1.,effort_continuity=1.,conditional_load=1.)

def reference_rates(mode):
    if mode not in ('holding','release','monitor'):raise ValueError('Explicit joint execution mode required')
    return np.asarray(HOLDING_REFERENCE_RATE21 if mode=='holding' else RELEASE_REFERENCE_RATE21,float)

RELEASE_NONTHUMB_CLEARING_MONOTONE=True
