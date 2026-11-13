"""Explicit release-clearance endpoint, distinct from a preparation OPEN pose.

The frozen lifecycle supplies the endpoint, not a new fitted pose or a sensed
future state. Terminal query time is used only to extract a saturated task goal;
issued actuator references retain the current public clock and real authority.
"""
from dataclasses import dataclass
import numpy as np
from v13_a1.codec import finite_array
from v13_joint.lifecycle import ReleaseEntry,release_nominal
from v13_joint.design import (NONTHUMB_CLEAR_TIME_S, THUMB_TUCK_UNTIL_S,
    THUMB_UNFOLD_TIME_S, LIFT_DESCENT_TIME_S)


@dataclass(frozen=True,slots=True)
class ReleaseClearanceGoal:
    reference18:np.ndarray
    preparation_open18:np.ndarray
    extraction_query_s:float
    saturated_after_s:float

    def __post_init__(self):
        for key in ('reference18','preparation_open18'):
            object.__setattr__(self,key,finite_array(getattr(self,key),(18,),key))

    def to_public(self):
        return dict(role='release_clearance_terminal_reference_not_preparation_OPEN',
            source='frozen_v13_joint.lifecycle.release_nominal_saturated_endpoint',
            reference18=self.reference18.tolist(),preparation_open18=self.preparation_open18.tolist(),
            extraction_query_s=self.extraction_query_s,saturated_after_s=self.saturated_after_s,
            extraction_is_not_a_command_or_future_measurement=True,
            execution_authority_widened=False,original_release_deadline_changed=False)


def release_clearance_goal(entry,*,preparation_open18,mechanical_lower21,mechanical_upper21):
    if type(entry) is not ReleaseEntry:
        raise TypeError('Actual typed release entry required')
    opening=finite_array(preparation_open18,(18,),'preparation OPEN')
    lower=finite_array(mechanical_lower21,(21,),'mechanical lower')
    upper=finite_array(mechanical_upper21,(21,),'mechanical upper')
    if np.any(lower>upper):raise ValueError('Empty actual mechanical authority')
    terminal=max(NONTHUMB_CLEAR_TIME_S,THUMB_TUCK_UNTIL_S+THUMB_UNFOLD_TIME_S,LIFT_DESCENT_TIME_S)
    query=float(np.nextafter(entry.started_s+terminal,np.inf))
    if query-entry.started_s<terminal:
        raise FloatingPointError('Saturated lifecycle endpoint query is not representable')
    goal=release_nominal(entry,query,opening,lower,upper)
    return ReleaseClearanceGoal(goal.values[:18],opening,query,float(terminal))
