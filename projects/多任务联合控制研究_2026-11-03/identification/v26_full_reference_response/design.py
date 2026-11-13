"""Fixed 43 training and four mixed holdout programs, not a response model."""
from dataclasses import dataclass
from numbers import Real
import numpy as np
from v13_a1.codec import REFERENCE_AXES21
from v25_response_identification.design import (CONTROL_PERIOD_S,PROBE_HOLD_S,
    AMPLITUDE_FRACTION,ORIGIN_TRIGGERS,TIMING_COLUMNS,DEBUG_TIMING_KEYS)

SCHEMA='v26_full_reference_response_v1'
SOURCE_NAMES=('__init__.py','design.py','coordinator.py','runtime.py','run_identification.py')
ACQUISITION_HORIZONS_S=(.005,.010,.020,.040,.100,.200)
SINGLE_ARMS=tuple('axis%02d_%s'%(i,sign) for i in range(21) for sign in ('plus','minus'))
# Literal names are validated against the actual native reference axis order.
# These are reference directions, not claims about OPEN, force or motion.
MIXED_COMPONENTS={
    'mixed_lift_wrj1':(('lift_reference_m',1),('WRJ1_reference_rad',-1)),
    'mixed_thumb_index':(('rh_A_THJ2',-1),('rh_A_FFJ0',-1)),
    'mixed_index_middle':(('rh_A_FFJ3',-1),('rh_A_MFJ3',-1)),
    'mixed_digits_lift':(('rh_A_THJ2',-1),('rh_A_FFJ0',-1),('rh_A_MFJ0',-1),
                         ('rh_A_RFJ0',-1),('rh_A_LFJ0',-1),('lift_reference_m',1)),
}
MIXED_ARMS=tuple(MIXED_COMPONENTS)
TRAINING_ARMS=('zero',)+SINGLE_ARMS
FULL_ARMS=TRAINING_ARMS+MIXED_ARMS

def arms_for_group(group):
    if group=='full':return FULL_ARMS
    if group=='mixed-holdout':return ('zero',)+MIXED_ARMS
    raise ValueError('Only fixed full47 or zero+four mixed holdout group is supported')

@dataclass(frozen=True)
class ReferenceProbeSpec:
    arm:str
    origin_s:float
    hold_s:float=PROBE_HOLD_S
    amplitude_fraction:float=AMPLITUDE_FRACTION

    def __post_init__(self):
        if self.arm not in FULL_ARMS:raise ValueError('Unknown full-reference arm')
        for key in ('origin_s','hold_s','amplitude_fraction'):
            value=getattr(self,key)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):raise ValueError('Finite real '+key+' required')
            object.__setattr__(self,key,float(value))
        if self.origin_s<0 or abs(self.origin_s/CONTROL_PERIOD_S-round(self.origin_s/CONTROL_PERIOD_S))>1e-8:raise ValueError('Existing5ms PRE_OBSERVE origin required')
        if self.hold_s!=PROBE_HOLD_S or self.amplitude_fraction!=AMPLITUDE_FRACTION:raise ValueError('Changed horizon/amplitude requires a different protocol')

    @property
    def role(self):return 'zero_control' if self.arm=='zero' else 'mixed_holdout' if self.arm in MIXED_ARMS else 'single_axis_training'

    @property
    def components(self):
        if self.arm=='zero':return ()
        if self.arm in MIXED_COMPONENTS:return tuple((REFERENCE_AXES21.index(name),sign) for name,sign in MIXED_COMPONENTS[self.arm])
        return ((int(self.arm[4:6]),1 if self.arm.endswith('_plus') else -1),)

def protocol():
    return dict(schema=SCHEMA,reference_axes21=list(REFERENCE_AXES21),training_arms=list(TRAINING_ARMS),
        mixed_holdout_arms={k:[dict(axis=REFERENCE_AXES21.index(n),name=n,sign=s) for n,s in v] for k,v in MIXED_COMPONENTS.items()},
        hold_s=PROBE_HOLD_S,amplitude_fraction=AMPLITUDE_FRACTION,control_period_s=CONTROL_PERIOD_S,
        acquisition_horizons_s=list(ACQUISITION_HORIZONS_S),
        amplitude_formula='0.5*min(original_release_rate*0.005,actual_origin_direction_margin)',
        single_unsupported_action='hold_same_zero_carrier_until_window_end; not adaptive nominal',
        partial_mixed_action='only originally declared supported components; explicitly partial and not complete mixed validation',
        flush='hold+declared_tactile_delay+one5ms_tick; original16s truncation retained',
        unload_trigger='first_TRANSFER_read_plus200ms_with_uninterrupted_TRANSFER; next_PRE_OBSERVE',
        response_model_admitted=False,performance_claim_allowed=False)
