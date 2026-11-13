"""Common real actuator decoder; legacy path is identical unless selected."""
import numpy as np
import research_imports
from hand_plant import HandPlant


class LifecyclePlant(HandPlant):
    def __init__(self,config):
        self.lifecycle_targets=None
        super().__init__(config)

    def apply_release_command(self,command):
        aids=np.concatenate(self.actuator_groups)
        target=np.asarray(command.target,float)
        limits=self.model.actuator_ctrlrange[aids]
        if target.shape!=aids.shape or not np.isfinite(target).all() or np.any(target<limits[:,0]-1e-8) or np.any(target>limits[:,1]+1e-8):
            raise ValueError('Invalid public lifecycle actuator command')
        self.lifecycle_targets=target.copy()
        self.lift_ref=float(command.lift_reference);self.wrist_ref=np.asarray(command.wrist_reference).copy()
        self._apply_ctrl()

    def _apply_ctrl(self):
        if self.lifecycle_targets is None or self.release_started is None:
            return super()._apply_ctrl()
        release_at=self.release_started
        held_clip_stat=self.max_hidden_clip
        try:
            self.release_started=None
            super()._apply_ctrl()
        finally:self.release_started=release_at
        self.data.ctrl[np.concatenate(self.actuator_groups)]=self.lifecycle_targets
        self.max_hidden_clip=held_clip_stat
        # A display/encoder-command summary only: released controls can leave
        # the one-dimensional held-grasp synergy, and are recorded in full ctrl.
        for i,indices in enumerate(self.actuator_groups):
            slope=self.closed_pose[i]-self.open_pose[i];changing=np.flatnonzero(abs(slope)>1e-10)
            if len(changing):
                k=int(changing[-1]);self.alpha[i]=(self.data.ctrl[indices[k]]-self.open_pose[i][k])/slope[k]
