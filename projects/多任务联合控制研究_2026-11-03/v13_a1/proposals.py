"""Pure original holding/release reference formulas; no engine/data/plant writes."""
from dataclasses import dataclass
import numpy as np
from .codec import Reference21, finite_array, finite_scalar

GROUP_WIDTHS = (5, 3, 3, 3, 4)

@dataclass(frozen=True)
class NativeProposalContext:
    open_pose: tuple
    closed_pose: tuple
    ctrl_ranges5: tuple
    release_open_rate: float

    def __post_init__(self):
        if len(self.open_pose) != 5 or len(self.closed_pose) != 5 or len(self.ctrl_ranges5) != 5:
            raise ValueError('Five exact actuator groups required')
        for key, shape_suffix in (('open_pose', ()), ('closed_pose', ()), ('ctrl_ranges5', (2,))):
            value = tuple(finite_array(x, (width,)+shape_suffix, key) for x, width in zip(getattr(self, key), GROUP_WIDTHS))
            object.__setattr__(self, key, value)
        if any(np.any(r[:, 0] > r[:, 1]) for r in self.ctrl_ranges5):
            raise ValueError('Ordered native ctrl ranges required')
        rate = finite_scalar(self.release_open_rate, 'release_open_rate')
        if rate < 0: raise ValueError('Nonnegative original release rate required')
        object.__setattr__(self, 'release_open_rate', rate)

@dataclass(frozen=True)
class NativeControlState:
    time_s: float
    alpha5: np.ndarray
    lift_ref: float
    wrist_ref2: np.ndarray
    max_hidden_clip: float
    release_started: float | None = None
    release_thumb5: np.ndarray | None = None
    release_alpha5: np.ndarray | None = None
    release_lift: float = 0.

    def __post_init__(self):
        for key in ('time_s', 'lift_ref', 'max_hidden_clip', 'release_lift'):
            object.__setattr__(self, key, finite_scalar(getattr(self, key), key))
        if self.time_s < 0 or self.max_hidden_clip < 0:
            raise ValueError('Public time and accumulated hidden clip must be nonnegative')
        object.__setattr__(self, 'alpha5', finite_array(self.alpha5, (5,), 'alpha5'))
        object.__setattr__(self, 'wrist_ref2', finite_array(self.wrist_ref2, (2,), 'wrist_ref2'))
        if self.release_started is not None:
            object.__setattr__(self, 'release_started', finite_scalar(self.release_started, 'release_started'))
            if self.release_started < 0 or self.release_started > self.time_s:
                raise ValueError('release_started must be a nonnegative past/current public time')
            object.__setattr__(self, 'release_thumb5', finite_array(self.release_thumb5, (5,), 'release_thumb5'))
            object.__setattr__(self, 'release_alpha5', finite_array(self.release_alpha5, (5,), 'release_alpha5'))

@dataclass(frozen=True)
class NativeProposal:
    reference21: Reference21
    alpha_after5: np.ndarray
    lift_ref_after: float
    max_hidden_clip_after: float
    releasing: bool

def build_native_proposal(context, state):
    if not isinstance(context, NativeProposalContext) or not isinstance(state, NativeControlState):
        raise TypeError('Explicit native proposal context and public-control state required')
    alpha = state.alpha5.copy(); references = []; hidden = state.max_hidden_clip
    # Preserve old arithmetic and its pre-release hidden-clip accounting.
    for i in range(5):
        requested = context.open_pose[i] + alpha[i] * (context.closed_pose[i] - context.open_pose[i])
        actual = np.clip(requested, context.ctrl_ranges5[i][:, 0], context.ctrl_ranges5[i][:, 1])
        hidden = max(hidden, float(np.max(np.abs(requested - actual))))
        references.append(actual)
    lift = state.lift_ref
    releasing = state.release_started is not None
    if releasing:
        elapsed = state.time_s - state.release_started
        alpha[1:] = np.maximum(0., state.release_alpha5[1:] - context.release_open_rate * elapsed)
        for i in range(1, 5):
            # Original four-finger release formula does NOT reclip here.
            references[i] = context.open_pose[i] + alpha[i] * (context.closed_pose[i] - context.open_pose[i])
        tucked = np.array([-1.03, 1.21, .13, -.02, .985]); full_open = np.array([-1., .1, 0., 0., 0.])
        if elapsed < .8:
            thumb = state.release_thumb5 + np.clip(elapsed / .8, 0, 1) * (tucked - state.release_thumb5)
        elif elapsed < 1.4:
            thumb = tucked
        else:
            thumb = tucked + np.clip((elapsed - 1.4) / .8, 0, 1) * (full_open - tucked)
        references[0] = np.clip(thumb, context.ctrl_ranges5[0][:, 0], context.ctrl_ranges5[0][:, 1])
        # Intentionally use the original UNCLIPPED thumb for the alpha side effect.
        alpha[0] = (thumb[0] - context.open_pose[0][0]) / (context.closed_pose[0][0] - context.open_pose[0][0])
        lift = state.release_lift * (1. - np.clip(elapsed / .6, 0., 1.))
    values = np.r_[np.concatenate(references), lift, state.wrist_ref2]
    alpha.setflags(write=False)
    return NativeProposal(Reference21(values, state.time_s), alpha, lift, hidden, releasing)
