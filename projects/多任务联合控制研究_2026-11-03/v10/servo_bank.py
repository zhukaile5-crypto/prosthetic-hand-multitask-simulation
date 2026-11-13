"""Empirical20ms robot servo anchor bank; no admission or mechanical authority.

One complete nearest reference anchor is selected, never a different anchor per
axis. K is the micro-trained finite20ms/four5ms response. qfree uses only current
actual robot q and velocity, candidateA of the separately saved offline study.
Unknown directions are not represented by invented zero physical gains.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Optional
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL_LEDGER_SHA='9f4fc2c229a745762f017614e49019432a85ddc0077bbcc6fde7c03487ad353f'
SHORT_LEDGER_SHA='f0c1cb091d5f15d59866cd2be2d9ad0c41080023dbc47028ddda6b86f681767f'
HELD_AXES8=('alpha_0','alpha_1','alpha_2','alpha_3','alpha_4','lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _read(path,expected=None):
    if expected is not None and _sha(path)!=expected:raise ValueError('source_fingerprint_mismatch:'+str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _array(value,shape,name):
    if value is None:raise ValueError(name+'_missing')
    x=np.asarray(value,float)
    if x.shape!=shape or not np.isfinite(x).all():raise ValueError(name+'_shape_or_finiteness')
    x=x.copy();x.setflags(write=False);return x


def _bool8(value,name):
    x=np.asarray(value)
    if x.shape!=(8,) or x.dtype.kind!='b':raise ValueError(name+'_requires_explicit_bool8')
    x=x.copy();x.setflags(write=False);return x


@dataclass(frozen=True)
class ServoSelection:
    status:str
    reasons:tuple=()
    free25:Optional[np.ndarray]=None
    response25x8:Optional[np.ndarray]=None
    error25:Optional[np.ndarray]=None
    known_plus8:Optional[np.ndarray]=None
    known_minus8:Optional[np.ndarray]=None
    proposal_lower8:Optional[np.ndarray]=None
    proposal_upper8:Optional[np.ndarray]=None
    debug:dict=field(default_factory=dict)
    global_model_admitted:bool=False

    def check_future_increment(self,delta8):
        try:u=_array(delta8,(8,),'declared_future_increment8')
        except ValueError as exc:return dict(status='UNSUPPORTED_FUTURE',reasons=[str(exc)],global_model_admitted=False)
        if self.known_plus8 is None:return dict(status='UNSUPPORTED_FUTURE',reasons=['servo_anchor_unavailable'],global_model_admitted=False)
        reasons=[]
        for i in range(8):
            if u[i]>1e-12 and not self.known_plus8[i]:reasons.append('unmeasured_positive_direction_'+str(i))
            if u[i]<-1e-12 and not self.known_minus8[i]:reasons.append('unmeasured_negative_direction_'+str(i))
            if u[i]<self.proposal_lower8[i]-1e-12 or u[i]>self.proposal_upper8[i]+1e-12:reasons.append('outside_actual_single_ramp_magnitude_'+str(i))
        return dict(status='UNSUPPORTED_FUTURE' if reasons else 'EMPIRICAL_COMPONENT_CANDIDATE',reasons=reasons,
            horizon_s=.02,finite_ramp='one20ms total increment executed as four equal5ms increments',
            sequence_support='component check only; arbitrary combination/history not certified',
            mechanical_permission='not determined by this empirical model check',global_model_admitted=False)


class Bank:
    def __init__(self,expected_axes25,*,research_root=None):
        self.axes25=tuple(expected_axes25);self.entries=[];self.load_errors=[];self.sources=[]
        if len(self.axes25)!=25 or len(set(self.axes25))!=25 or self.axes25[-1]!='transport_lift' or any(not x.startswith('rh_') for x in self.axes25[:24]):raise ValueError('Exact24 robot names+transport_lift expected')
        self.root=ROOT if research_root is None else Path(research_root)
        try:self._load()
        except (ValueError,OSError,KeyError,TypeError) as exc:
            self.entries=[];self.load_errors=[type(exc).__name__+':'+str(exc)]

    @classmethod
    def from_entries_for_test(cls,expected_axes25,entries):
        """Explicit synthetic fixture path; not a production provenance bypass."""
        obj=cls.__new__(cls);obj.axes25=tuple(expected_axes25);obj.entries=[];obj.sources=[];obj.load_errors=[];obj.root=None
        for entry in entries:
            e=obj._validate_entry(entry);e['synthetic_fixture']=True;obj.entries.append(e)
        return obj

    def _validate_entry(self,e):
        e=dict(e)
        if tuple(e['axes25'])!=self.axes25:raise ValueError('robot_axis_identity_mismatch')
        for key,shape in [('anchor_ref8',(8,)),('anchor_q25',(25,)),('anchor_velocity25',(25,)),('response25x8',(25,8)),('error25',(25,)),('micro_scale8',(8,)),('proposal_lower8',(8,)),('proposal_upper8',(8,)),('mixed_lower_ref8',(8,)),('mixed_upper_ref8',(8,))]:e[key]=_array(e[key],shape,key)
        for key in ('known_plus8','known_minus8'):e[key]=_bool8(e[key],key)
        e['mixed_reference_points8']=np.asarray(e['mixed_reference_points8'],float)
        if e['mixed_reference_points8'].ndim!=2 or e['mixed_reference_points8'].shape[1]!=8 or not np.isfinite(e['mixed_reference_points8']).all():raise ValueError('recorded_mixed_reference_points_required')
        if np.any(e['micro_scale8']<=0) or np.any(e['error25']<0) or np.any(e['proposal_lower8']>e['proposal_upper8']):raise ValueError('invalid_empirical_scope_or_error')
        if not isinstance(e['context_id'],str) or not e['context_id']:raise ValueError('complete_context_id_required')
        return e

    def _load(self):
        modelroot=self.root/'output/robot_servo_response_all15_v2';shortroot=self.root/'output/robot_servo_short_all15_v1'
        modelledger=_read(modelroot/'execution_complete.json',MODEL_LEDGER_SHA);shortledger=_read(shortroot/'execution_complete.json',SHORT_LEDGER_SHA)
        if not modelledger['complete'] or not shortledger['complete'] or not shortledger['dependencies_unchanged']:raise ValueError('incomplete_frozen_source_run')
        expected={f'{shape}/checkpoint_{i:02d}' for shape in ('capsule','box','sphere') for i in range(5)}
        models={row['context_id']:row for row in modelledger['contexts']};shorts={row['context_id']:row for row in shortledger['contexts']}
        if set(models)!=expected or set(shorts)!=expected or len(modelledger['contexts'])!=15 or len(shortledger['contexts'])!=15:raise ValueError('exact15_context_sources_required')
        self.sources=[dict(path=str(modelroot/'execution_complete.json'),sha256=MODEL_LEDGER_SHA),dict(path=str(shortroot/'execution_complete.json'),sha256=SHORT_LEDGER_SHA)]
        for cid in sorted(expected):
            mr=models[cid];sr=shorts[cid];mp=modelroot/cid/'model.json';sp=shortroot/cid/'comparison.json'
            if Path(mr['model_path']).resolve()!=mp.resolve() or Path(sr['path']).resolve()!=sp.resolve():raise ValueError('model_source_path_scope_mismatch')
            m=_read(mp,mr['model_sha256']);s=_read(sp,sr['sha256'])
            if m['global_model_admitted'] or s['global_model_admitted'] or m['context_id']!=cid or s['context_id']!=cid:raise ValueError('local_source_scope_mismatch')
            if s['model_sha256']!=mr['model_sha256'] or tuple(m['input_axes8'])!=HELD_AXES8 or m['ramp_period_s']!=.02 or m['execution_period_s']!=.005:raise ValueError('finite_ramp_source_mismatch')
            kp=Path(m['dense_kernel_npz_path']);fp=Path(m['actual_footprints_path'])
            if kp.resolve()!=(modelroot/cid/'robot_response_raw.npz').resolve() or fp.resolve()!=(modelroot/cid/'actual_input_footprints.json').resolve():raise ValueError('kernel_footprint_path_scope_mismatch')
            if _sha(kp)!=m['dense_kernel_npz_sha256']:raise ValueError('dense_kernel_source_mismatch')
            footprints=_read(fp,m['actual_footprints_sha256'])
            with np.load(kp,allow_pickle=False) as f:
                kt=f['response_kernel_times_s'];K=f['response25x8']
            if K.shape!=(101,25,8) or len(kt)!=101 or abs(kt[4]-.02)>1e-12:raise ValueError('exact20ms_measured_kernel_required')
            deltas=np.stack([row['actual_total_delta8'] for row in m['training']]);scale=np.max(abs(deltas),axis=0)
            lower=np.minimum(0.,np.min(deltas,axis=0));upper=np.maximum(0.,np.max(deltas,axis=0))
            if not np.array_equal(lower,np.asarray(m['measured_single_ramp_increment_lower8'])) or not np.array_equal(upper,np.asarray(m['measured_single_ramp_increment_upper8'])):raise ValueError('actual_training_increment_scope_mismatch')
            points=np.vstack([v['actual_reference'] for name,v in footprints.items() if name.startswith('mixed_')])
            e=self._validate_entry(dict(context_id=cid,shape=cid.split('/')[0],axes25=m['output_axes25'],anchor_ref8=m['base_reference8'],anchor_q25=m['robot_anchor25'],anchor_velocity25=m['robot_velocity_anchor25'],
                response25x8=K[4],error25=s['metrics_by_horizon']['0.02']['A']['max_abs_per_axis25'],
                known_plus8=m['known_plus8'],known_minus8=m['known_minus8'],micro_scale8=scale,proposal_lower8=lower,proposal_upper8=upper,
                mixed_lower_ref8=np.min(points,axis=0),mixed_upper_ref8=np.max(points,axis=0),mixed_reference_points8=points,
                source_fingerprints=dict(model=str(mp),model_sha256=mr['model_sha256'],short_comparison=str(sp),short_comparison_sha256=sr['sha256'],dense_kernel=str(kp),dense_kernel_sha256=m['dense_kernel_npz_sha256'],actual_footprints=str(fp),actual_footprints_sha256=m['actual_footprints_sha256']),synthetic_fixture=False))
            self.entries.append(e)
            for path,digest in ((mp,mr['model_sha256']),(sp,sr['sha256']),(kp,m['dense_kernel_npz_sha256']),(fp,m['actual_footprints_sha256'])):
                self.sources.append(dict(path=str(path),sha256=digest))

    def source_identity(self):
        """All62 directly consumed data files, for the caller's run closure."""
        return {row['path']:row['sha256'] for row in self.sources}

    def select(self,shape,q25,v25,actual_ref8,*,future_increment8=None):
        try:q=_array(q25,(25,),'current_robot_q25');v=_array(v25,(25,),'current_robot_velocity25');ref=_array(actual_ref8,(8,),'actual_robot_reference8')
        except ValueError as exc:return ServoSelection('UNAVAILABLE',(str(exc),),debug=dict(global_model_admitted=False))
        entries=[e for e in self.entries if e['shape']==shape]
        if not entries:return ServoSelection('UNAVAILABLE',tuple(self.load_errors or ['declared_morphology_model_missing']),debug=dict(shape=shape,global_model_admitted=False))
        # One full anchor, normalized only by that anchor's actual measured
        # micro amplitudes. No coordinatewise/contextwise mixing of columns.
        def distance(e):return float(np.linalg.norm((ref-e['anchor_ref8'])/e['micro_scale8']))
        e=min(entries,key=lambda x:(distance(x),x['context_id']));offset=ref-e['anchor_ref8'];state=q-e['anchor_q25'];velocity=v-e['anchor_velocity25']
        bounds=(offset>=e['proposal_lower8']-1e-12)&(offset<=e['proposal_upper8']+1e-12)
        mixed_box=(ref>=e['mixed_lower_ref8']-1e-12)&(ref<=e['mixed_upper_ref8']+1e-12)
        mixed_distance=float(np.min(np.linalg.norm((e['mixed_reference_points8']-ref)/e['micro_scale8'],axis=1)))
        debug=dict(context_id=e['context_id'],declared_morphology=shape,anchor_reference8=e['anchor_ref8'].tolist(),anchor_robot_q25=e['anchor_q25'].tolist(),
            anchor_robot_velocity25=e['anchor_velocity25'].tolist(),normalized_reference_distance=distance(e),reference_offset8=offset.tolist(),normalizing_actual_micro_amplitude8=e['micro_scale8'].tolist(),
            robot_state_distance=dict(angular_l2_rad=float(np.linalg.norm(state[:24])),angular_max_abs_rad=float(np.max(abs(state[:24]))),lift_abs_m=float(abs(state[24]))),
            robot_velocity_distance=dict(angular_l2_rad_s=float(np.linalg.norm(velocity[:24])),angular_max_abs_rad_s=float(np.max(abs(velocity[:24]))),lift_abs_m_s=float(abs(velocity[24]))),
            reference_in_single_axis_component_envelope8=bounds.tolist(),reference_outside_component_footprint=bool(not np.all(bounds)),
            reference_in_mixed_component_envelope8=mixed_box.tolist(),nearest_recorded_mixed_point_normalized_distance=mixed_distance,
            sequence_support='UNASSESSED_BY_POINT_SELECTION; component envelope or a matching point does not validate a sequence',
            mode_scope='holding calibration only; no novel contact/impact/release response identified',
            current_contact_mode_assessment='UNKNOWN; robot q/reference alone do not identify contact mode',horizon_s=.02,
            model_mode='EMPIRICAL_ANCHOR_LINEARIZATION',model_domain_transfer='experimental',
            actual_future_scope='NOT_CHECKED' if future_increment8 is None else 'CHECKED_COMPONENT_DIRECTIONS_ONLY',
            error_scope='A20 per-axis maximum observed mixed holdout error at this anchor; empirical, not guaranteed bound',
            past_memory='current actual velocity nuisance approximation, no fabricated20ms past ramp',
            mechanical_permission='separate robot execution bounds required',source_fingerprints=e.get('source_fingerprints',{}),
            synthetic_fixture=e.get('synthetic_fixture',False),global_model_admitted=False)
        selection=ServoSelection('EMPIRICAL_ANCHOR_LINEARIZATION',(),_array(q+.02*v,(25,),'free_robot25'),e['response25x8'],e['error25'],e['known_plus8'],e['known_minus8'],e['proposal_lower8'],e['proposal_upper8'],debug)
        if future_increment8 is not None:
            scope=selection.check_future_increment(future_increment8);debug['future_component_check']=scope
            if scope['status']=='UNSUPPORTED_FUTURE':return ServoSelection('UNSUPPORTED_FUTURE',tuple(scope['reasons']),selection.free25,None,selection.error25,selection.known_plus8,selection.known_minus8,selection.proposal_lower8,selection.proposal_upper8,debug)
        return selection
