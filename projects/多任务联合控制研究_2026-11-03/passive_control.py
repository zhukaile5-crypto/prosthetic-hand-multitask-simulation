"""Hold-only ablation: retain prepared hand command, follow requested motion.

This tests whether a disturbance actually needs adaptive gripping. It receives
the same observations and mechanical release as the other holding policies.
"""
import numpy as np
import research_imports
from controller_adapter import HandControllerMixin
from vendor.frozen_controllers import CommonController


class PassiveHold(HandControllerMixin,CommonController):
    def _decide(self,frame,phase):
        self.label='passive_prepared_grasp'
        self.desired=self.state.normal.copy();self.predicted=self.state.normal.copy()
        self.debug=dict(passive_hold=True,prediction_available=False,
            prediction_unavailable_reason='no autonomous force prediction in the hold-only ablation')
        return np.zeros(5)


def create_passive(calibration,limits,budget):
    B=np.asarray(calibration['force_matrix'])
    controller=PassiveHold('fixed',np.asarray(limits),float(budget),dict(
        force_gain=tuple(np.maximum(.05,np.linalg.norm(B,axis=0))),
        force_tau=float(calibration['force_tau']),release_rate=.8))
    controller.name='passive'
    return controller
