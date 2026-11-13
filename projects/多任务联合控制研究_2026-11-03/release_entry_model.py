"""Mode-specific release evidence interfaces, with no invented pressure model.

Schema/provenance and public-domain checks are not physical model admission.
No holding kernel, kinematic Jacobian or tactile linear extrapolation is used to
predict pressure. A future reviewed evaluator is a separate required artifact.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from hashlib import sha256
import json
from zipfile import BadZipFile
from release_material_contract import (COMMAND_AXES,COMMAND_UNITS,PRESSURE_AXES,
    RELEASE_MODES,ALL_SKIN_SCOPE,number,vector,hash_string,strict_payload)

MODEL_SCHEMA='release_pressure_mode_model_v1'

def validate_source_snapshot(path):
    path=Path(path);root=path.parent.resolve();payload=json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema')!='release_source_snapshot_v1' or not isinstance(payload.get('files'),dict) or not payload['files']:raise ValueError('Complete copied source snapshot manifest required')
    for item in payload['files'].values():
        if not isinstance(item,dict) or set(item)!={'snapshot_path','sha256'}:raise ValueError('Explicit snapshot path/hash required')
        expected=hash_string(item['sha256'],'snapshot.sha256');file=(root/item['snapshot_path']).resolve()
        if not file.is_relative_to(root) or not file.is_file() or sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Source snapshot file missing/changed/outside immutable copy')
    return len(payload['files'])

def validate_response_arrays(response_path,basis_path,manifest):
    """Inspect future numeric artifacts without executing an evaluator.

    These are shape/unit/provenance conditions, never evidence that the numbers
    describe physical pressure or a rigorous error bound.
    """
    import numpy as np
    with np.load(basis_path,allow_pickle=False) as source:
        required={'matrix_21_by_k','command_axes','command_units','coefficient_units'}
        if set(source.files)!=required:raise ValueError('Explicit release-basis array schema required')
        basis=np.asarray(source['matrix_21_by_k'],float)
        if basis.ndim!=2 or basis.shape[0]!=21 or not 1<=basis.shape[1]<=21 or not np.isfinite(basis).all() or np.linalg.matrix_rank(basis)!=basis.shape[1]:raise ValueError('Independent finite21-axis release directions required')
        if tuple(source['command_axes'].tolist())!=COMMAND_AXES or tuple(source['command_units'].tolist())!=COMMAND_UNITS or str(source['coefficient_units'])!='dimensionless':raise ValueError('Release basis physical units/order mismatch')
    with np.load(response_path,allow_pickle=False) as source:
        required={'time_s','kernel_pressure_N_per_basis_increment','empirical_error_N','raw_peak_error_N','raw_peak_window_s','measured_coefficient_lower','measured_coefficient_upper','pressure_axes','pressure_units'}
        if set(source.files)!=required:raise ValueError('Release pressure response schema required; holding kernels are incompatible')
        t=np.asarray(source['time_s'],float);k=np.asarray(source['kernel_pressure_N_per_basis_increment'],float);error=np.asarray(source['empirical_error_N'],float);peak=np.asarray(source['raw_peak_error_N'],float);lo=np.asarray(source['measured_coefficient_lower'],float);hi=np.asarray(source['measured_coefficient_upper'],float)
        if t.ndim!=1 or len(t)<2 or t[0]!=0. or not np.all(np.diff(t)>0.) or k.shape!=(len(t),6,basis.shape[1]) or error.shape!=(len(t),6) or peak.shape!=(6,):raise ValueError('Release response dimension/time mismatch')
        if lo.shape!=(basis.shape[1],) or hi.shape!=(basis.shape[1],):raise ValueError('Measured basis coefficient bounds required')
        if not all(np.isfinite(x).all() for x in [t,k,error,peak,lo,hi]) or np.any(error<0.) or np.any(peak<0.) or np.any(lo>0.) or np.any(hi<0.) or np.any(lo>hi):raise ValueError('Finite empirical uncertainty and measured input bounds required')
        if tuple(source['pressure_axes'].tolist())!=PRESSURE_AXES or tuple(source['pressure_units'].tolist())!=('N',)*6:raise ValueError('Pressure channels must use physical newtons')
        if tuple(source['raw_peak_window_s'].tolist())!=manifest.physical_peak_window_s:raise ValueError('Raw peak validation window mismatch')
        if any(not np.any(abs(t-h)<1e-9) for h in manifest.validated_horizons_s):raise ValueError('Claimed horizons absent from recorded response')
    return dict(basis_directions=int(basis.shape[1]),time_samples=int(len(t)),numeric_schema_verified=True,physical_model_admitted=False)

@dataclass(frozen=True)
class Artifact:
    path:str
    sha256:str
    role:str
    def __post_init__(self):
        if not self.path or not self.role:raise ValueError('Artifact path and role required')
        hash_string(self.sha256,'artifact.sha256')
    def verify(self,base_directory):
        p=Path(self.path);p=p if p.is_absolute() else Path(base_directory)/p
        if not p.is_file() or sha256(p.read_bytes()).hexdigest()!=self.sha256:raise ValueError(f'Artifact identity mismatch: {self.role}')
        return p.resolve()

@dataclass(frozen=True)
class PatchRegion:
    robot_link:str
    position_lower_m:tuple
    position_upper_m:tuple
    reference_normal:tuple
    minimum_normal_cosine:float
    def __post_init__(self):
        from math import sqrt
        if not self.robot_link.startswith('rh_'):raise ValueError('Robot skin link required')
        for k in ['position_lower_m','position_upper_m','reference_normal']:object.__setattr__(self,k,vector(getattr(self,k),3,k))
        if any(a>b for a,b in zip(self.position_lower_m,self.position_upper_m)) or abs(sqrt(sum(x*x for x in self.reference_normal))-1.)>1e-6:raise ValueError('Finite patch box and measured unit normal required')
        cosine=number(self.minimum_normal_cosine,'minimum_normal_cosine')
        if not -1.<=cosine<=1.:raise ValueError('Normal cosine outside[-1,1]')
        object.__setattr__(self,'minimum_normal_cosine',cosine)
    def contains(self,patch):
        return patch.robot_link==self.robot_link and all(a<=x<=b for a,x,b in zip(self.position_lower_m,patch.link_position_m,self.position_upper_m)) and sum(a*b for a,b in zip(self.reference_normal,patch.link_normal))>=self.minimum_normal_cosine

@dataclass(frozen=True)
class EntryDomain:
    """Observed feature coverage only; no hidden mass/friction/target identity."""
    joint_names:tuple
    joint_position_lower_rad:tuple
    joint_position_upper_rad:tuple
    joint_velocity_abs_max_rad_s:tuple
    actuator_lag_abs_max:tuple
    pressure_lower_N:tuple
    pressure_upper_N:tuple
    minimum_actual_history_s:float
    maximum_skin_age_s:float
    allowed_contact_link_sets:tuple=()
    patch_regions:tuple=()
    actual_history_period_s:float=.005
    def __post_init__(self):
        from release_material_contract import ROBOT_JOINT_NAMES
        names=tuple(self.joint_names)
        if len(set(names))!=24 or set(names)!=ROBOT_JOINT_NAMES:raise ValueError('Public robot24 joint domain required')
        object.__setattr__(self,'joint_names',names)
        for k in ['joint_position_lower_rad','joint_position_upper_rad','joint_velocity_abs_max_rad_s']:
            object.__setattr__(self,k,vector(getattr(self,k),24,k,k=='joint_velocity_abs_max_rad_s'))
        object.__setattr__(self,'actuator_lag_abs_max',vector(self.actuator_lag_abs_max,21,'actuator_lag_abs_max',True))
        for k in ['pressure_lower_N','pressure_upper_N']:object.__setattr__(self,k,vector(getattr(self,k),6,k,True))
        for k in ['minimum_actual_history_s','maximum_skin_age_s','actual_history_period_s']:object.__setattr__(self,k,number(getattr(self,k),k,True))
        if self.maximum_skin_age_s<=0. or self.actual_history_period_s<=0. or any(a>b for a,b in zip(self.joint_position_lower_rad,self.joint_position_upper_rad)) or any(a>b for a,b in zip(self.pressure_lower_N,self.pressure_upper_N)):raise ValueError('Nonempty observed domain required')
        modes=tuple(tuple(sorted(set(x))) for x in self.allowed_contact_link_sets)
        if any(not isinstance(n,str) or not n.startswith('rh_') for mode in modes for n in mode):raise ValueError('Public skin-link mode names only')
        regions=tuple(strict_payload(PatchRegion,x) if isinstance(x,dict) else x for x in self.patch_regions)
        if any(not isinstance(x,PatchRegion) for x in regions):raise ValueError('Measured patch regions required')
        object.__setattr__(self,'allowed_contact_link_sets',modes);object.__setattr__(self,'patch_regions',regions)

    def outside_reasons(self,state,contract):
        reasons=[]
        if tuple(state.joint_names)!=self.joint_names:return ('joint_order_mismatch',)
        if not state.pressure_available(contract):return ('incomplete_or_stale_all_skin_pressure',)
        if any(not a<=x<=b for a,x,b in zip(self.joint_position_lower_rad,state.joint_position_rad,self.joint_position_upper_rad)):reasons.append('joint_position_outside_measured_domain')
        if any(abs(x)>b for x,b in zip(state.joint_velocity_rad_s,self.joint_velocity_abs_max_rad_s)):reasons.append('joint_velocity_outside_measured_domain')
        if any(abs(q-u)>b for q,u,b in zip(state.actuator_coordinates,state.executed_command,self.actuator_lag_abs_max)):reasons.append('servo_lag_outside_measured_domain')
        if any(not a<=x<=b for a,x,b in zip(self.pressure_lower_N,state.pressure_vector(contract),self.pressure_upper_N)):reasons.append('pressure_outside_measured_domain')
        times=state.finger_sample_time_s+(state.auxiliary_sample_time_s,)
        if any(state.time_s-t>self.maximum_skin_age_s+1e-9 for t in times):reasons.append('skin_age_outside_model_domain')
        if not self.allowed_contact_link_sets:reasons.append('contact_geometry_domain_missing')
        elif tuple(sorted(set(p.robot_link for p in state.skin_patches))) not in self.allowed_contact_link_sets:reasons.append('skin_contact_mode_outside_measured_domain')
        for p in state.skin_patches:
            if state.time_s-p.sample_time_s>self.maximum_skin_age_s+1e-9 or not any(region.contains(p) for region in self.patch_regions):reasons.append('skin_patch_geometry_outside_measured_domain');break
        if self.minimum_actual_history_s>0.:
            h=state.executed_history
            start=state.time_s-self.minimum_actual_history_s
            tail=[x for x in h if x.time_s>=start-self.actual_history_period_s-1e-9]
            if not tail or tail[-1].time_s<state.time_s-1e-9 or tail[0].time_s>start+1e-9 or tail[-1].command!=state.executed_command or any(abs(b.time_s-a.time_s-self.actual_history_period_s)>1e-8 for a,b in zip(tail,tail[1:])):reasons.append('missing_actual_command_history')
        return tuple(reasons)

@dataclass(frozen=True)
class ModeModelManifest:
    model_id:str
    mode:str
    public_morphology:str
    robot_definition_sha256:str
    sensor_profile_sha256:str
    availability:str='missing'
    schema:str=MODEL_SCHEMA
    command_axes:tuple=COMMAND_AXES
    command_units:tuple=COMMAND_UNITS
    pressure_axes:tuple=PRESSURE_AXES
    pressure_units:tuple=('N',)*6
    response_artifact:Artifact|None=None
    basis_artifact:Artifact|None=None
    evaluator_artifact:Artifact|None=None
    calibration_plan_artifact:Artifact|None=None
    source_snapshot_artifact:Artifact|None=None
    independent_validation_artifact:Artifact|None=None
    entry_domain:EntryDomain|None=None
    validated_horizons_s:tuple=()
    physical_peak_window_s:tuple|None=None
    uncertainty_kind:str='missing'

    def __post_init__(self):
        if self.schema!=MODEL_SCHEMA or self.mode not in RELEASE_MODES:raise ValueError('A release-mode model is required; holding kernels are not accepted')
        if not self.model_id or not self.public_morphology:raise ValueError('Public mode/context identity required')
        hash_string(self.robot_definition_sha256,'robot_definition_sha256');hash_string(self.sensor_profile_sha256,'sensor_profile_sha256')
        if tuple(self.command_axes)!=COMMAND_AXES or tuple(self.command_units)!=COMMAND_UNITS or tuple(self.pressure_axes)!=PRESSURE_AXES or tuple(self.pressure_units)!=('N',)*6:raise ValueError('Explicit21-command and6-pressure units/order required')
        for k in ['command_axes','command_units','pressure_axes','pressure_units']:object.__setattr__(self,k,tuple(getattr(self,k)))
        for k in ['response_artifact','basis_artifact','evaluator_artifact','calibration_plan_artifact','source_snapshot_artifact','independent_validation_artifact']:
            a=getattr(self,k)
            if isinstance(a,dict):object.__setattr__(self,k,strict_payload(Artifact,a))
            elif a is not None and not isinstance(a,Artifact):raise ValueError('Explicit artifact identity required')
        if isinstance(self.entry_domain,dict):object.__setattr__(self,'entry_domain',strict_payload(EntryDomain,self.entry_domain))
        horizon=tuple(number(x,'validated_horizon',True) for x in self.validated_horizons_s)
        if any(a>=b for a,b in zip(horizon,horizon[1:])):raise ValueError('Ordered validation times required')
        object.__setattr__(self,'validated_horizons_s',horizon)
        artifacts=[getattr(self,k) for k in ['response_artifact','basis_artifact','evaluator_artifact','calibration_plan_artifact','source_snapshot_artifact','independent_validation_artifact']]
        if self.availability=='missing':
            if any(a is not None for a in artifacts) or self.entry_domain is not None or horizon or self.physical_peak_window_s is not None or self.uncertainty_kind!='missing':raise ValueError('Missing models cannot contain fabricated response, bounds or validation')
        elif self.availability=='artifact_bundle_ready_for_review':
            if any(a is None for a in artifacts) or self.entry_domain is None or not horizon or self.physical_peak_window_s is None:raise ValueError('Complete response/basis/evaluator/provenance/independent validation required')
            window=vector(self.physical_peak_window_s,2,'physical_peak_window_s',True)
            if window[1]<=window[0]:raise ValueError('Nonempty raw peak response window required')
            object.__setattr__(self,'physical_peak_window_s',window)
            if self.uncertainty_kind not in ['empirical_max_residual','statistical_predictive_interval']:raise ValueError('No deterministic force-bound claim is implemented')
        else:raise ValueError('Missing or reviewable artifact bundle only; global admission is not a JSON status')

    @classmethod
    def from_payload(cls,payload):return strict_payload(cls,payload)

    def as_payload(self):return asdict(self)

@dataclass(frozen=True)
class EntryAssessment:
    evidence_status:str
    public_feature_domain_covered:bool
    reasons:tuple
    release_request_may_be_delayed:bool=False
    holding_support_may_be_requested:bool=False
    pressure_prediction_available:bool=False
    material_certified:bool=False
    global_model_admitted:bool=False

class ReleaseEntryModel:
    """A provenance/domain gate, intentionally not a numerical force predictor.

An external reviewed evaluator remains necessary even for a schema-valid
bundle. Nothing in this module grants an arbitrary21D force forecast.
"""
    def __init__(self,manifest=None,base_directory=None):
        self.manifest=ModeModelManifest.from_payload(manifest) if isinstance(manifest,dict) else manifest
        if self.manifest is not None and not isinstance(self.manifest,ModeModelManifest):raise ValueError('Release model manifest or explicit missing required')
        self.base_directory=Path(base_directory or '.');self.verified_artifacts={}

    def assess_entry(self,state,contract,mode):
        if mode not in RELEASE_MODES:raise ValueError('Explicit release mode required')
        m=self.manifest
        if m is None or m.availability=='missing':return EntryAssessment('missing',False,('release_pressure_model_missing',))
        reasons=[]
        if mode!=m.mode:reasons.append('mode_not_validated')
        if contract.public_morphology!=m.public_morphology:reasons.append('public_morphology_not_validated')
        if state.robot_definition_sha256!=m.robot_definition_sha256:reasons.append('robot_definition_mismatch')
        if state.sensor_profile_sha256!=m.sensor_profile_sha256:reasons.append('sensor_profile_mismatch')
        if reasons:return EntryAssessment('unsupported_context',False,tuple(reasons))
        try:
            for k in ['response_artifact','basis_artifact','evaluator_artifact','calibration_plan_artifact','source_snapshot_artifact','independent_validation_artifact']:
                self.verified_artifacts[k]=str(getattr(m,k).verify(self.base_directory))
            report=json.loads(Path(self.verified_artifacts['independent_validation_artifact']).read_text(encoding='utf-8'))
            required={'schema','mode','public_morphology','robot_definition_sha256','sensor_profile_sha256','domain_sha256','validated_horizons_s','physical_peak_window_s','uncertainty_kind','response_sha256','basis_sha256','evaluator_sha256','plan_sha256','source_snapshot_sha256','planned','executed','unavailable','failed','raw_substeps_complete','all_failures_retained','split_grouping','scope'}
            if not required<=set(report) or report['schema']!='independent_release_pressure_validation_v1':raise ValueError('Independent raw validation schema missing')
            for key,artifact in [('response_sha256',m.response_artifact),('basis_sha256',m.basis_artifact),('evaluator_sha256',m.evaluator_artifact),('plan_sha256',m.calibration_plan_artifact),('source_snapshot_sha256',m.source_snapshot_artifact)]:
                if report[key]!=artifact.sha256:raise ValueError('Validation lineage mismatch')
            domain_hash=sha256(json.dumps(asdict(m.entry_domain),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if report['domain_sha256']!=domain_hash or report['public_morphology']!=m.public_morphology or report['robot_definition_sha256']!=m.robot_definition_sha256 or report['sensor_profile_sha256']!=m.sensor_profile_sha256 or tuple(report['validated_horizons_s'])!=m.validated_horizons_s or tuple(report['physical_peak_window_s'])!=m.physical_peak_window_s or report['uncertainty_kind']!=m.uncertainty_kind:raise ValueError('Unreviewed domain/context/horizon expansion')
            if any(type(report[k]) is not int or report[k]<0 for k in ['planned','executed','unavailable','failed']) or any(type(report[k]) is not bool for k in ['raw_substeps_complete','all_failures_retained']):raise ValueError('Explicit count/coverage types required')
            if report['mode']!=mode or not report['raw_substeps_complete'] or not report['all_failures_retained'] or report['split_grouping']!='complete_parent_entry' or report['planned']<=0 or report['planned']!=report['executed']+report['unavailable'] or report['failed']>report['executed']:raise ValueError('Incomplete validation accounting')
            validate_source_snapshot(self.verified_artifacts['source_snapshot_artifact'])
            validate_response_arrays(self.verified_artifacts['response_artifact'],self.verified_artifacts['basis_artifact'],m)
        except (OSError,ValueError,TypeError,KeyError,BadZipFile,EOFError) as exc:return EntryAssessment('evidence_invalid',False,(str(exc),))
        reasons=m.entry_domain.outside_reasons(state,contract)
        # Artifact schema and feature membership are never a material certificate.
        return EntryAssessment('artifact_schema_verified_only',not reasons,tuple(reasons)+('reviewed_numerical_pressure_evaluator_not_installed',))

    def forecast_pressure(self,*args,**kwargs):
        return dict(available=False,upper_pressure_N=None,kernel=None,reason='mode_specific_release_pressure_evaluator_missing',material_certified=False,global_model_admitted=False)
