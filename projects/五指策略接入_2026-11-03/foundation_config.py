"""Public physical parameters for the isolated articulated-hand benchmark."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class FoundationConfig:
    shape: str = 'capsule'
    size: float = 1.0
    mass: float = .12
    friction: float = .60
    solref: float = .008
    timestep: float = .0005
    control_dt: float = .005
    lift_target: float = .10
    object_enabled: bool = True
    seed: int = 510001
    object_offset_x: float | None = None
    object_offset_z: float = 0.
    hand_offset_z: float | None = None
    acquisition_force: tuple = (.8,1.8,1.8,1.8,1.8)
    holding_force: tuple = (3.,1.8,1.8,1.8,1.8)
    prelift_contact_force: float = .35
    prelift_dwell: float = .65
    lift_duration: float = 3.0
    force_transition_delay: float = .3
    force_transition_duration: float = 1.5
    collision_model: str = 'smooth_distal_capsules'
    contact_dimension: int = 3
    box_rounding_radius: float = .002
    release_open_rate: float = .8

    def __post_init__(self):
        if self.object_offset_x is None:object.__setattr__(self,'object_offset_x',.005 if self.shape=='box' else .007)
        if self.hand_offset_z is None:object.__setattr__(self,'hand_offset_z',.020 if self.shape=='box' else .010)
        object.__setattr__(self,'acquisition_force',tuple(self.acquisition_force))
        object.__setattr__(self,'holding_force',tuple(self.holding_force))
        if self.timestep<=0 or self.control_dt<=0 or self.mass<=0 or self.friction<=0:raise ValueError('positive physical parameters required')
        if not 0<self.lift_target<=.10:raise ValueError('public lift target must stay within 100 mm')

    def to_dict(self): return asdict(self)

BENCHMARK_ID = 'shadow_hand_strategy_integration_v1'
FINGER_NAMES = ['拇指','食指','中指','无名指','小指']
PREFIX = ['rh_th','rh_ff','rh_mf','rh_rf','rh_lf']
GROUPS = [['THJ5','THJ4','THJ3','THJ2','THJ1'],['FFJ4','FFJ3','FFJ0'],
          ['MFJ4','MFJ3','MFJ0'],['RFJ4','RFJ3','RFJ0'],['LFJ5','LFJ4','LFJ3','LFJ0']]
OPEN = [[.2,1.15,.1,.2,1.15],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[.1,0.,0.,0.]]
CLOSED = [[1.047,1.15,.1,.2,1.15],[0.,.6,2.],[0.,.6,2.],[0.,.6,2.],[.1,0.,.7,2.]]
