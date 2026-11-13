"""Preview-shaped SEARCH velocity request for the existing V18 governor.

All values are reference intents. The downstream governor remains responsible
for the current mechanical interval, motion authority and deadline behavior.
"""

import math
from numbers import Real

from v14_coord.design import CONTROL_PERIOD_S, DECISION_PERIOD_S

READ_DELAY_S = .010
CLOCK_TOL = 1e-9
CREEP_FRACTION = CONTROL_PERIOD_S / DECISION_PERIOD_S


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def read_gap(now, preview):
    if not isinstance(preview, dict) or preview.get("available") is not True:
        return None
    stamp = preview.get("read_s")
    if not finite(stamp) or abs(stamp - now) > CLOCK_TOL:
        return None
    gap = preview.get("engineering_gap_m")
    center = preview.get("conditional_plane_gap_m")
    if not isinstance(gap, (tuple, list)) or not isinstance(center, (tuple, list)) or len(gap) != 2 or len(center) != 2:
        return None
    if not all(finite(x) for x in (*gap, *center)):
        return None
    lo, hi = gap
    mid = (center[0] + center[1]) / 2.
    return (lo, mid, hi) if lo < mid < hi and lo <= center[0] <= center[1] <= hi else None


def preview_speed(fast, gap, seconds_remaining):
    """Full approach at the lower-gap edge, creep by the central estimate."""
    lo, mid, hi = gap
    lookahead_m = fast * (READ_DELAY_S + DECISION_PERIOD_S)
    s = min(1., max(0., (mid - lookahead_m) / (mid - lo)))
    gradual = s*s*(3.-2.*s)
    creep = CREEP_FRACTION * fast
    # A schedule-pressure floor based on the upper engineering gap. This
    # influences intent only and cannot guarantee arriving by the deadline.
    deadline_floor = min(fast, max(0., hi) / seconds_remaining)
    return max(deadline_floor, creep + (fast-creep)*gradual), lookahead_m


class ApproachSchedule:
    """Causal speed shaping with bounded reacceleration within a SEARCH episode."""

    def __init__(self):
        self.tick = None
        self.payload = None
        self.result = None
        self.previous_speed = None

    def update(self, *, now, phase, preview, original_request_m_s, velocity_envelope_m_s, deadline_s):
        if not all(finite(x) for x in (now, original_request_m_s, velocity_envelope_m_s, deadline_s)):
            raise ValueError("Finite explicit approach clocks and speed authority required")
        tick = round(now / CONTROL_PERIOD_S)
        if now < 0. or abs(tick * CONTROL_PERIOD_S - now) > CLOCK_TOL or velocity_envelope_m_s < 0.:
            raise ValueError("Invalid public approach grid or velocity envelope")
        gap = read_gap(now, preview)
        payload = (phase, gap, original_request_m_s, velocity_envelope_m_s, deadline_s)
        if self.tick is not None and tick <= self.tick:
            if tick == self.tick and payload == self.payload:
                return self.result.copy()
            raise ValueError("Conflicting or rewound approach observation")
        elapsed = CONTROL_PERIOD_S if self.tick is None else (tick-self.tick)*CONTROL_PERIOD_S
        fast = min(max(0., -original_request_m_s), velocity_envelope_m_s)
        active = phase == "SEARCH" and now < deadline_s and fast > 0.
        distance = None
        if not active:
            speed, request, self.previous_speed = fast, original_request_m_s, None
            status = "original_phase_or_deadline"
        else:
            if gap is None:
                speed = fast if self.previous_speed is None else min(fast, self.previous_speed)
                status = "unknown_preview_no_acceleration"
            else:
                desired, distance = preview_speed(fast, gap, deadline_s-now)
                if self.previous_speed is None:
                    speed = desired
                else:
                    slew = (1.-CREEP_FRACTION) * fast * min(elapsed, DECISION_PERIOD_S) / DECISION_PERIOD_S
                    speed = min(desired, self.previous_speed + slew)
                status = "preview_slowed_search" if speed < fast else "original_far_search"
            self.previous_speed = speed
            request = original_request_m_s if speed == fast else -speed
        self.tick, self.payload = tick, payload
        self.result = dict(status=status, requested_velocity_m_s=request,
                           effective_request_magnitude_m_s=speed, original_envelope_m_s=fast,
                           fraction_of_original_envelope=None if fast == 0. else speed/fast,
                           approach_lookahead_m=distance, read_s=float(now),
                           prospective_physical_stop_certified=False)
        return self.result.copy()
