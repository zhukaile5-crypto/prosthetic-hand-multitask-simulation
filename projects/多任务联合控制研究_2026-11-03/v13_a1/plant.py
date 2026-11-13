"""Behavior-preserving A1 sole robot21 writer; all original physics inherited."""
import numpy as np
import research_imports
from hand_plant import HandPlant
from .codec import ReferenceActor21
from .proposals import NativeProposalContext, NativeControlState, build_native_proposal

class A1Plant(HandPlant):
    """Only _apply_ctrl is overridden; initial reset already uses the new actor."""
    def _apply_ctrl(self):
        if not hasattr(self, 'a1_actor'):
            robot_ids = np.r_[np.concatenate(self.actuator_groups), self.lift_actuator,
                              self.aid('rh_A_WRJ2'), self.aid('rh_A_WRJ1')].astype(int)
            self.a1_actor = ReferenceActor21(self.model, robot_indices21=robot_ids)
            self.actor21 = self.a1_actor  # Stable root/independent audit handle.
            self.a1_context = NativeProposalContext(tuple(self.open_pose), tuple(self.closed_pose),
                tuple(self.model.actuator_ctrlrange[indices] for indices in self.actuator_groups), self.config.release_open_rate)
        state = NativeControlState(self.time, self.alpha, self.lift_ref, self.wrist_ref,
            self.max_hidden_clip, self.release_started, self.release_thumb, self.release_alpha, self.release_lift)
        proposal = build_native_proposal(self.a1_context, state)
        self.max_hidden_clip = proposal.max_hidden_clip_after
        if proposal.releasing:
            # Same original semantic mutation order, without intermediate writes.
            self.alpha[1:] = proposal.alpha_after5[1:]
            self.alpha[0] = proposal.alpha_after5[0]
            self.lift_ref = proposal.lift_ref_after
        self.a1_last_proposal = proposal
        self.a1_last_engine_port21 = self.a1_actor.apply(self.data, proposal.reference21)
        # The sole NON-robot motor retains its separate current-env semantics.
        self.data.ctrl[self.obstacle_motor] = float(np.clip(self.env.get('obstacle_x', 0.), -.4, .1))
