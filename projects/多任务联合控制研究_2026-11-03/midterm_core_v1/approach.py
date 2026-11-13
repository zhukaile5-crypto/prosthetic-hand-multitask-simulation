"""One memoryless distance-to-speed rule; no learned or scenario-specific gains."""
import numpy as np
import mujoco
from identification.v21_public_support_preview_v1 import (
    DeclaredShapeTemplate, StaticReceiverTop, SupportPreview)
from v14_coord.design import CONTROL_PERIOD_S, DECISION_PERIOD_S

MIN_FRACTION = CONTROL_PERIOD_S / DECISION_PERIOD_S
TEMPLATE = DeclaredShapeTemplate("capsule", .028, .047, "shared declared nominal capsule")

def receiver_top(model, mount):
    if mount.geom_name != "support":
        raise ValueError("Core pilot requires the existing support sensor")
    geom = model.geom(mount.geom_name)
    if (int(model.geom_bodyid[geom.id]) != 0
        or int(model.geom_type[geom.id]) != int(mujoco.mjtGeom.mjGEOM_BOX)
        or not np.array_equal(geom.quat, [1., 0., 0., 0.])
        or not np.array_equal(mount.origin_in_geom_m, [0., 0., geom.size[2]])):
        raise ValueError("Core pilot requires a declared horizontal fixed receiver")
    return StaticReceiverTop(tuple(geom.pos[:2]), tuple(geom.size[:2]),
                             float(geom.pos[2] + geom.size[2]))

def approach_request(original, envelope, now, preview, enabled=True):
    """Slow across one uncertainty-width before contact, with a fixed creep floor.
    Unknown geometry retains the original request and explicitly logs unavailable.
    This requests speed only; the common lift governor still owns actuator bounds.
    """
    if not np.isfinite([original, envelope, now]).all() or envelope < 0:
        raise ValueError("Finite original request, nonnegative speed envelope and clock required")
    record = dict(enabled=enabled, available=False, fraction=1.,
                  original_request_m_s=float(original), requested_velocity_m_s=float(original),
                  uses_state_memory=False, deadline_floor=False, response_prediction=False)
    if not enabled or original >= 0:
        return float(original), record
    try:
        if preview["available"] is not True or abs(preview["read_s"]-now)>1e-9:
            return float(original), record
        if abs(now-preview["acquisition_s"]-.010)>1e-9:
            return float(original), record
        lo, hi = np.asarray(preview["engineering_gap_m"], float)
        a, b = np.asarray(preview["conditional_plane_gap_m"], float)
        if not np.isfinite([lo,hi,a,b]).all() or not lo <= a <= b <= hi or hi <= lo:
            return float(original), record
        gap = .5*(a+b)
        fraction = float(np.clip(gap/(hi-lo), MIN_FRACTION, 1.))
        request = float(original) if fraction == 1. else -min(-original,envelope)*fraction
        record.update(available=True, fraction=fraction, gap_m=float(gap),
                      uncertainty_width_m=float(hi-lo), requested_velocity_m_s=request)
        return request, record
    except (KeyError, TypeError, ValueError):
        return float(original), record
