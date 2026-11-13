"""Conditional receiver-top clearance from known geometry and delayed skin.

No object identity, live truth, response model, or actuation is consulted.
The prototype estimates clearance at acquisition, never propagates a pose.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from v11.object_belief import DeclaredShapeTemplate, RelativeObjectBelief
from v14_coord.design import DECISION_PERIOD_S

READ_DELAY_S = .010
CLOCK_TOL = 1e-9


@dataclass(frozen=True)
class StaticReceiverTop:
    center_xy: tuple
    half_size_xy: tuple
    height_m: float

    def __post_init__(self):
        center = np.asarray(self.center_xy, dtype=float)
        half = np.asarray(self.half_size_xy, dtype=float)
        if (center.shape != (2,) or half.shape != (2,) or not np.isfinite(center).all()
                or not np.isfinite(half).all() or np.any(half <= 0.) or not math.isfinite(self.height_m)):
            raise ValueError("Known finite horizontal receiver top required")


def lowest_points(snapshot):
    """Conditional world points attaining min z across the fitted interval."""
    if snapshot.template.kind == "sphere":
        center = np.asarray(snapshot.center_reference, dtype=float)
        return (center - [0., 0., snapshot.template.radius_m])[None, :]
    axis = np.asarray(snapshot.axis_reference, dtype=float)
    origin = np.asarray(snapshot.center_line_origin_reference, dtype=float)
    interval = np.asarray(snapshot.axial_center_interval_m, dtype=float)
    if axis.shape != (3,) or origin.shape != (3,) or interval.shape != (2,):
        raise ValueError("Invalid conditional capsule hypothesis")
    if abs(axis[2]) <= 1e-12:
        # The whole generator has the same height. Use its midpoint as a
        # witness of possible contact inside the finite top footprint.
        down = 0.
    else:
        down = -math.copysign(snapshot.template.half_segment_m, axis[2])
    return origin[None, :] + (interval + down)[:, None] * axis - [0., 0., snapshot.template.radius_m]


class SupportPreview:
    """One causal episode of public acquisition-time geometry."""

    def __init__(self, mount, template):
        if type(mount) is not StaticReceiverTop or type(template) is not DeclaredShapeTemplate:
            raise TypeError("Typed static mount and declared object template required")
        self.mount = mount
        self.geometry = RelativeObjectBelief(template)

    def read(self, frame):
        now = float(frame["time"])
        stamp = frame.get("sample_time")
        if (not isinstance(stamp, Real) or isinstance(stamp, (bool, np.bool_))
                or not math.isfinite(stamp) or abs(now - stamp - READ_DELAY_S) > CLOCK_TOL):
            return dict(available=False, reason="wrong_acquisition_clock", read_s=now)
        estimate = self.geometry.update(frame)
        if not estimate.grouping_unambiguous:
            return dict(available=False, reason=estimate.status, read_s=now)
        acquisition = estimate.observation_time_s
        age = now - acquisition
        if abs(age - READ_DELAY_S) > CLOCK_TOL or abs(stamp - acquisition) > CLOCK_TOL:
            return dict(available=False, reason="wrong_acquisition_clock", read_s=now)
        position_sigma = estimate.diagnostics.get("public_position_3sigma_m")
        if position_sigma is None:
            return dict(available=False, reason="position_resolution_unknown", read_s=now)
        points = lowest_points(estimate)
        margin = float(estimate.diagnostics["tolerance_m"] + position_sigma)
        xy = np.max(abs(points[:, :2] - np.asarray(self.mount.center_xy)), axis=0)
        in_footprint = bool(np.all(xy + margin <= self.mount.half_size_xy))
        z = points[:, 2] - self.mount.height_m
        fit_low, fit_high = float(z.min()), float(z.max())
        if not in_footprint:
            return dict(available=False, reason="lowest_point_not_resolved_inside_receiver", read_s=now,
                        acquisition_s=acquisition, conditional_plane_gap_m=[fit_low, fit_high])
        velocity = frame.get("lift_velocity_m_s")
        if (not isinstance(velocity, Real) or isinstance(velocity, (bool, np.bool_))
                or not math.isfinite(velocity)):
            return dict(available=False, reason="robot_lift_velocity_unavailable", read_s=now)
        # Read-time robot speed sets an engineering approach horizon only.
        # It is not the object speed, a stopping distance, or a safety limit.
        distance = max(0., -velocity) * (age + DECISION_PERIOD_S)
        gap_low, gap_high = fit_low - margin, fit_high + margin
        return dict(available=True, read_s=now, acquisition_s=acquisition, sensor_age_s=age,
                    conditional_plane_gap_m=[fit_low, fit_high], engineering_gap_m=[gap_low, gap_high],
                    geometry_margin_m=margin, robot_lift_velocity_m_s=velocity,
                    preview_distance_m=distance,
                    cue_lower=bool(velocity < 0. and gap_low <= distance),
                    cue_center=bool(velocity < 0. and (fit_low + fit_high) / 2 <= distance),
                    cue_upper=bool(velocity < 0. and gap_high <= distance),
                    footprint_inside_for_fitted_low_points=True,
                    current_pose_predicted=False, stopping_safety_certified=False)
