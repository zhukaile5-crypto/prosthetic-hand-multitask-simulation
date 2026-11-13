"""Causal tactile affiliation hypothesis in known robot link coordinates.

Persistent skin regions observed during the common warm-up are frozen at
handoff. New regions are *unassigned* external contacts, never automatically
credited as grasp support. This is a geometric hypothesis, not object identity:
two external surfaces producing the same local patch cannot be distinguished.
All finger pressure is retained for protection, including unassigned pressure.
No object pose, external geometry identity or future observation is read.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass
class GraspContactHypothesis:
    time: float
    sample_time: float
    persistent_normal: np.ndarray
    new_normal: np.ndarray
    all_normal: np.ndarray
    support_present: np.ndarray
    known_support: np.ndarray
    valid: np.ndarray
    fresh: np.ndarray
    geometry_available: np.ndarray
    confidence: np.ndarray
    status: np.ndarray
    anchor_count: np.ndarray
    frozen: bool

    def to_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in asdict(self).items()}


@dataclass
class _Anchor:
    digit: int
    body: int
    point: np.ndarray
    normal: np.ndarray
    last_sample: float
    duration: float = 0.
    samples: int = 1


class PersistentGraspContactObserver:
    """Warm-up-only enrollment with conservative, fixed local association.

    Defaults are explicit engineering association tolerances, not inferred
    material parameters. Confidence is the observed pressure fraction assigned
    to a persistent robot region; it is not a probability of object contact.
    Missing geometry/invalid sensors never provide positive support evidence.
    """
    def __init__(self, position_tolerance_m=.003, normal_tolerance_rad=math.pi/6,
                 enrollment_dwell_s=.08, maximum_age_s=.04,
                 maximum_sample_gap_s=.025, support_threshold_N=.15):
        values=[position_tolerance_m, normal_tolerance_rad, enrollment_dwell_s,
                maximum_age_s, maximum_sample_gap_s, support_threshold_N]
        if not np.isfinite(values).all() or any(x <= 0. for x in values):
            raise ValueError('Positive finite association tolerances required')
        if normal_tolerance_rad >= math.pi:
            raise ValueError('Normal tolerance must be below pi')
        self.position_tolerance=float(position_tolerance_m)
        self.normal_cosine=math.cos(float(normal_tolerance_rad))
        self.enrollment_dwell=float(enrollment_dwell_s)
        self.maximum_age=float(maximum_age_s)
        self.maximum_gap=float(maximum_sample_gap_s)
        self.support_threshold=float(support_threshold_N)
        self.anchors=[]
        self.frozen=False
        self.snapshot=None
        self.last_time=None
        self.last_sample=None

    @staticmethod
    def _vector(value, dtype, name):
        result=np.asarray(value, dtype=dtype)
        if result.shape != (5,):
            raise ValueError(f'{name} must contain five finger channels')
        return result.copy()

    def _match(self, digit, body, point, normal, eligible_only):
        candidates=[]
        for j,a in enumerate(self.anchors):
            if a.digit != digit or a.body != body:
                continue
            if eligible_only and a.duration+1e-10 < self.enrollment_dwell:
                continue
            distance=float(np.linalg.norm(point-a.point))
            if distance <= self.position_tolerance and float(normal@a.normal) >= self.normal_cosine:
                candidates.append((distance,j))
        return min(candidates)[1] if candidates else None

    def freeze_handoff(self):
        """Freeze only sustained, recently seen warm-up regions.

        Calling repeatedly is idempotent. This does not certify contact with a
        particular object or admit a local dynamics model.
        """
        if self.frozen:
            return
        if self.last_time is None:
            raise RuntimeError('Public tactile warm-up is required before handoff')
        self.anchors=[a for a in self.anchors
                      if a.duration+1e-10 >= self.enrollment_dwell
                      and self.last_time-a.last_sample <= self.maximum_age+1e-10]
        self.frozen=True
        if self.snapshot is not None:
            self.snapshot.frozen=True
            self.snapshot.anchor_count=np.bincount([a.digit for a in self.anchors], minlength=5)

    def update(self, frame, *, allow_enrollment=False):
        now=float(frame.time)
        if not np.isfinite(now):
            raise ValueError('Observation time must be finite')
        if self.last_time is not None and now < self.last_time-1e-10:
            raise ValueError('Contact observer cannot run backwards in time')
        self.last_time=now
        normal=self._vector(frame.normal,float,'normal')
        valid=self._vector(frame.valid,bool,'valid') & np.isfinite(normal)
        all_normal=np.where(valid,np.maximum(0.,normal),np.nan)
        sample=float(getattr(frame,'sample_time',now))
        sample_fresh=bool(np.isfinite(sample) and -1e-10 <= now-sample <= self.maximum_age+1e-10)
        if self.last_sample is not None and sample < self.last_sample-1e-10:
            sample_fresh=False
        if sample_fresh:
            self.last_sample=sample
        fresh=valid & sample_fresh
        support=fresh & (np.nan_to_num(all_normal,nan=0.)>self.support_threshold)
        geometry_available=np.zeros(5,bool)
        persistent=np.zeros(5)
        patch_total=np.zeros(5)
        hit_points={}
        enrollment=bool(allow_enrollment and not self.frozen)
        patches=getattr(frame,'skin_patches',None)
        if patches is not None:
            # An explicit empty list is a complete observation of no patches;
            # absence of the field is an unavailable geometry channel.
            geometry_available[fresh]=True
            for patch in patches:
                try:
                    digit=int(patch['digit']);body=int(patch['robot_body'])
                    point=np.asarray(patch['link_point'],float)
                    direction=np.asarray(patch['link_normal'],float)
                    pressure=float(patch['normal'])
                    acquired=float(patch.get('sample_time',sample))
                    shape_ok=point.shape==(3,) and direction.shape==(3,)
                    finite=shape_ok and np.isfinite(np.r_[point,direction,pressure,acquired]).all()
                except (KeyError,TypeError,ValueError,OverflowError):
                    # Malformed geometry cannot be used as support evidence;
                    # measured whole-finger pressure still remains unassigned.
                    continue
                if not 0 <= digit < 5 or body < 0 or not finite or pressure <= 0.:
                    continue
                if not fresh[digit] or not -1e-10 <= now-acquired <= self.maximum_age+1e-10:
                    continue
                if abs(acquired-sample)>1e-8:
                    continue
                length=float(np.linalg.norm(direction))
                if length <= 1e-12:
                    continue
                direction=direction/length
                patch_total[digit]+=pressure
                j=self._match(digit,body,point,direction,eligible_only=not enrollment)
                if j is None and enrollment:
                    # A public warm-up is short. Bound accidental malformed
                    # input growth without silently replacing old regions.
                    if sum(a.digit==digit for a in self.anchors) >= 64:
                        continue
                    self.anchors.append(_Anchor(digit,body,point.copy(),direction.copy(),acquired))
                    j=len(self.anchors)-1
                if j is not None:
                    a=self.anchors[j]
                    if enrollment:
                        hit_points.setdefault(j,[]).append((point,direction,acquired))
                    if a.duration+1e-10 >= self.enrollment_dwell:
                        persistent[digit]+=pressure
            for j,hits in hit_points.items():
                a=self.anchors[j]
                acquired=hits[0][2]
                elapsed=acquired-a.last_sample
                # Repeated delayed samples do not accumulate dwell. Gaps
                # restart continuity, rather than inventing unseen contact.
                if elapsed > 1e-10:
                    a.duration=a.duration+elapsed if elapsed<=self.maximum_gap+1e-10 else 0.
                    a.last_sample=acquired
                    point=np.mean([x[0] for x in hits],axis=0)
                    direction=np.mean([x[1] for x in hits],axis=0)
                    a.samples+=1
                    a.point+=(point-a.point)/a.samples
                    candidate=a.normal+(direction-a.normal)/a.samples
                    a.normal=candidate/max(1e-12,float(np.linalg.norm(candidate)))
        # Attribute no more than the measured pressure. Any missing/malformed
        # or unmatched patch share remains unassigned, never silently zeroed.
        measured=np.nan_to_num(all_normal,nan=0.)
        scale=np.minimum(1.,np.divide(measured,patch_total,out=np.ones(5),where=patch_total>0.))
        persistent*=scale
        persistent[~fresh]=0.
        new=np.maximum(0.,measured-persistent)
        new[~valid]=np.nan
        confidence=np.divide(persistent,measured,out=np.zeros(5),where=measured>0.)
        known=support & geometry_available & (persistent>self.support_threshold)
        status=np.full(5,'ABSENT',dtype='<U24')
        status[support]='NEW_UNKNOWN'
        status[known]='PERSISTENT'
        status[known & (new>self.support_threshold)]='MIXED_NEW'
        status[valid & ~fresh]='STALE'
        status[~valid]='UNOBSERVED'
        status[fresh & (patches is None)]='GEOMETRY_UNAVAILABLE'
        counts=np.bincount([a.digit for a in self.anchors if a.duration+1e-10>=self.enrollment_dwell],minlength=5)
        self.snapshot=GraspContactHypothesis(now,sample,persistent,new,all_normal,
            support,known,valid,fresh,geometry_available,confidence,status,counts,self.frozen)
        return self.snapshot
