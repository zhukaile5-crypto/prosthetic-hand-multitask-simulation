"""Offline robot-only identification from frozen phase3 branches; never steps physics.

Micro branches train a finite 20ms / four5ms-ramp response. Mixed branches are
held out. Pairing with a future zero trace is an OFFLINE differential diagnostic,
not a causal forecast. Causal scores use only initial/past held-out robot q.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT/'calibration'/'phase3_path_validation_v3'
INPUT_AXES = ['alpha_'+str(i) for i in range(5)]+['lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad']
HORIZONS = [.02,.04,.06,.1]
WINDOW = .12
CONTROL_KEYS = ('time','qpos','qvel','ctrl','alpha','input_reference','robot_lift_position_m','robot_wrist_position_rad')
PHYSICAL_KEYS = ('physical_time','physical_interval_start','physical_duration','physical_qpos','physical_qvel','physical_ctrl','physical_alpha','physical_input_reference','physical_robot_lift_position_m','physical_robot_wrist_position_rad')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def write_json(path,value):
    text=json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as f:f.write(text)


def source_binding(data):
    manifest=read_json(data/'source_snapshot_manifest.json'); entries=[]
    for label,item in manifest['files'].items():
        path=data/Path(item['snapshot_path'].replace('\\','/'))
        actual=sha(path)
        if actual!=item['snapshot_sha256'] or actual!=item['sha256']:
            raise ValueError('Frozen source identity mismatch: '+label)
        entries.append(dict(label=label,path=str(path),sha256=actual,bytes=path.stat().st_size))
    return dict(plan_path=str(data/'plan.json'),plan_sha256=sha(data/'plan.json'),
        execution_complete_path=str(data/'execution_complete.json'),execution_complete_sha256=sha(data/'execution_complete.json'),
        source_snapshot_manifest_path=str(data/'source_snapshot_manifest.json'),source_snapshot_manifest_sha256=sha(data/'source_snapshot_manifest.json'),
        verified_source_files=entries)


def mechanical_context(data,shape):
    """Only compile frozen model to obtain names/addresses/exact reference map."""
    frozen=data/'sources'/'foundation'
    sys.path.insert(0,str(frozen))
    import mujoco
    def forbidden(*_a,**_k):raise RuntimeError('Physics stepping forbidden in offline identification')
    for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
    import hand_plant
    import foundation_config
    if Path(hand_plant.__file__).resolve()!=frozen.joinpath('hand_plant.py').resolve():
        raise ValueError('Wrong imported foundation source')
    setup=read_json(data/shape/'setup.json')
    plant=hand_plant.HandPlant(foundation_config.FoundationConfig(**setup['config']))
    names=[mujoco.mj_id2name(plant.model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in plant.raw_joint_ids]
    if len(names)!=24 or any(not name.startswith('rh_') for name in names):raise ValueError('Robot24 name whitelist failed')
    qadr=np.r_[plant.raw_qadr,plant.lift_qadr].astype(int)
    if len(set(qadr))!=25 or plant.model.nq!=33:raise ValueError('Unexpected robot/free-object layout')
    object_ids=np.flatnonzero(plant.model.jnt_type==int(mujoco.mjtJoint.mjJNT_FREE))
    excluded=[]
    for j in object_ids:excluded.extend(range(int(plant.model.jnt_qposadr[j]),int(plant.model.jnt_qposadr[j])+7))
    if set(qadr)&set(excluded):raise ValueError('Free-object position entered robot whitelist')
    aids=np.r_[np.concatenate(plant.actuator_groups),plant.lift_actuator,plant.aid('rh_A_WRJ2'),plant.aid('rh_A_WRJ1')].astype(int)
    if len(aids)!=21 or len(set(aids))!=21:raise ValueError('Expected18finger+lift+2wrist mechanical references')
    return dict(plant=plant,axes=names+['transport_lift'],qadr=qadr,aids=aids,
        excluded_qpos_addresses=sorted(set(range(plant.model.nq))-set(qadr)),
        free_object_qpos_addresses=excluded,config=setup['config'],
        source_path=str(frozen/'hand_plant.py'),source_sha256=sha(frozen/'hand_plant.py'))


def decode_ctrl(context,reference):
    p=context['plant'];u=np.asarray(reference,float);out=np.empty((len(u),21));col=0
    for digit,ids in enumerate(p.actuator_groups):
        n=len(ids)
        out[:,col:col+n]=p.open_pose[digit]+u[:,digit,None]*(p.closed_pose[digit]-p.open_pose[digit])
        out[:,col:col+n]=np.clip(out[:,col:col+n],p.model.actuator_ctrlrange[ids,0],p.model.actuator_ctrlrange[ids,1]);col+=n
    out[:,18]=u[:,5]+p.model.body_subtreemass[p.hand]*9.81/1500.
    out[:,19:]=u[:,6:]
    return out


def load_branch(folder,context):
    meta=read_json(folder/'metadata.json')
    if not meta['available']:
        if meta.get('attempted_physics'):raise ValueError('Unavailable branch attempted physics')
        return dict(available=False,metadata=meta,metadata_path=str(folder/'metadata.json'),metadata_sha256=sha(folder/'metadata.json'))
    if not meta['complete']:raise ValueError('Incomplete branch cannot enter fit: '+str(folder))
    trace=folder/'trace.npz';before=sha(trace)
    if before!=meta['trace_sha256']:raise ValueError('Raw hash mismatch: '+str(trace))
    with np.load(trace,allow_pickle=False) as archive:
        a={k:archive[k].copy() for k in CONTROL_KEYS+PHYSICAL_KEYS}
    if sha(trace)!=before:raise ValueError('Input changed during read')
    qadr=context['qadr'];q=a.pop('qpos')[:,qadr].copy();pq=a.pop('physical_qpos')[:,qadr].copy()
    # qvel is only checked for finite recording, never fitted as another output.
    if any(not np.isfinite(v).all() for v in a.values()) or not np.isfinite(q).all() or not np.isfinite(pq).all():raise ValueError('Nonfinite selected robot recording')
    t=a['time'];pt=a['physical_time'];starts=a['physical_interval_start'];duration=a['physical_duration']
    if len(t)!=101 or len(pt)!=1000:raise ValueError('Expected complete.5s branch')
    if np.max(abs(np.diff(t)-.005))>1e-9 or np.max(abs(duration-.0005))>1e-12:raise ValueError('Clock period mismatch')
    if np.max(abs(pt-starts-duration))>1e-9:raise ValueError('Physical record is not end clock')
    end=np.arange(9,1000,10)
    if np.max(abs(t[1:]-pt[end]))>1e-9 or not np.array_equal(q[1:],pq[end]):raise ValueError('Control and physical q end-clock mismatch')
    for key in ('ctrl','alpha','input_reference'):
        if not np.array_equal(a[key][1:],a['physical_'+key][end]):raise ValueError('Physical/control actual command mismatch: '+key)
    # Each5ms actual command is set before its10 physical substeps; no future command is backdated.
    for key in ('ctrl','alpha','input_reference'):
        expected=np.repeat(a[key][1:],10,axis=0)
        if not np.array_equal(expected,a['physical_'+key]):raise ValueError('Command changed inside intended5ms hold: '+key)
    if not np.array_equal(a['alpha'],a['input_reference'][:,:5]):raise ValueError('Actual alpha vs reference mismatch')
    u=a['input_reference'];path=np.asarray(meta['requested_path'],float)
    ramp=np.vstack([path[0],np.stack([path[k]+j/4*(path[k+1]-path[k]) for k in range(25) for j in range(1,5)])])
    ramp_error=float(np.max(abs(u-ramp)))
    if ramp_error>1e-9:raise ValueError('Actual reference did not realize requested20ms finite ramp')
    expected=decode_ctrl(context,u);ctrl=a['ctrl'][:,context['aids']]
    max_ctrl=float(np.max(abs(ctrl-expected)))
    if max_ctrl>1e-12:raise ValueError('Actual21 mechanical control mismatch')
    if not np.array_equal(q[:,24],a['robot_lift_position_m']):raise ValueError('Lift whitelist mismatch')
    wrist=[context['axes'].index('rh_WRJ2'),context['axes'].index('rh_WRJ1')]
    if not np.array_equal(q[:,wrist],a['robot_wrist_position_rad']):raise ValueError('Wrist whitelist mismatch')
    if np.max(abs(q[0]-np.r_[read_json(folder.parent/'checkpoint.json')['actual_robot_joint_position'],read_json(folder.parent/'checkpoint.json')['actual_lift_position_m']]))>1e-12:raise ValueError('Checkpoint robot anchor mismatch')
    return dict(available=True,t=t-t[0],absolute_time=t,q=q,pq=pq,reference=u,ctrl=a['ctrl'],
        physical_reference=a['physical_input_reference'],physical_ctrl=a['physical_ctrl'],metadata=meta,
        source=dict(branch=folder.name,trace_path=str(trace),trace_sha256=before,metadata_path=str(folder/'metadata.json'),metadata_sha256=sha(folder/'metadata.json')),
        checks=dict(actual_ctrl_max_abs_error=max_ctrl,actual_ctrl_bit_exact=bool(np.array_equal(ctrl,expected)),actual_ramp_max_abs_error=ramp_error,
            q_end_clock_control_physical_exact=True,actual_input_hold_physical_exact=True,input_unchanged=True),
        footprint=dict(reference_base=u[0],reference_min=np.min(u,axis=0),reference_max=np.max(u,axis=0),
            cumulative_delta_min=np.min(u-u[0],axis=0),cumulative_delta_max=np.max(u-u[0],axis=0),
            interval_delta=np.diff(u[::4],axis=0),execution_delta=np.diff(u,axis=0),
            reference_times_s=t-t[0],actual_reference=u,command_start_times_s=t[:-1]-t[0],robot_anchor25=q[0]))


def fit_micro(branches,zero):
    K=np.empty((101,25,8));plus=[];minus=[];training=[];sides={};missing=[]
    for axis in range(8):
        numerator=np.zeros((101,25));denom=0.;side={}
        for sign,label in ((1,'plus'),(-1,'minus')):
            name=f'micro_{axis}_{label}';b=branches[name]
            if not b['available']:continue
            delta=b['reference'][4]-b['reference'][0]
            if abs(delta[axis])<1e-14 or np.count_nonzero(abs(delta)>1e-12)!=1 or sign*delta[axis]<=0:raise ValueError('Unexpected micro excitation')
            diff=b['q']-zero['q'];d=float(delta[axis]);numerator+=d*diff;denom+=d*d
            side[label]=diff/d
            training.append(dict(branch=name,input_axis=axis,actual_total_delta8=delta,denominator=d,normal_equation_weight=d*d))
        plus.append('plus' in side);minus.append('minus' in side)
        if not side:missing.append(axis);K[:,:,axis]=np.nan
        else:K[:,:,axis]=numerator/denom
        sides[str(axis)]=side
    return K,np.asarray(plus,bool),np.asarray(minus,bool),training,sides,missing


def effect(K,times,reference,query):
    """Finite-ramp interval convolution; only command intervals with start<query."""
    starts=times[:-1];delta=np.diff(reference,axis=0);take=starts<query-1e-10
    if not np.any(take):return np.zeros(25)
    age=query-starts[take]
    if np.max(age)>.5+1e-9:raise ValueError('Kernel memory beyond measured.5s is unavailable')
    # Recorded data.time accumulates binary roundoff; normalize ONLY the checked
    # endpoint tolerance, never extend the measured response memory.
    age=np.clip(age,0.,.5)
    grid=np.arange(101)*.005
    response=np.stack([np.interp(age,grid,K[:,o,i],left=0.,right=np.nan) for o in range(25) for i in range(8)],axis=-1).reshape(-1,25,8)
    return np.einsum('koi,ki->o',response,delta[take])


def residual_forecast(observed_t,residual,query):
    """Unweighted least squares over causal.12s window; no future zero data."""
    take=observed_t>=observed_t[-1]-WINDOW-1e-10;t=observed_t[take];r=residual[take]
    if len(t)==1:return r[0].copy(),np.array([1.])
    X=np.column_stack((np.ones(len(t)),t-t[-1]));weights=np.array([1.,query-t[-1]])@np.linalg.pinv(X)
    return weights@r,weights


def score(error):
    x=np.asarray(error,float)
    if x.ndim!=2 or x.shape[1]!=25 or not len(x):raise ValueError('Per-axis errors required')
    return dict(samples=len(x),rmse_per_axis25=np.sqrt(np.mean(x*x,axis=0)),p95_abs_per_axis25=np.quantile(abs(x),.95,axis=0),max_abs_per_axis25=np.max(abs(x),axis=0),
        angular_rad=dict(max_axis_rmse=float(np.max(np.sqrt(np.mean(x[:,:24]**2,axis=0)))),max_abs=float(np.max(abs(x[:,:24])))),
        lift_m=dict(rmse=float(np.sqrt(np.mean(x[:,24]**2))),p95_abs=float(np.quantile(abs(x[:,24]),.95)),max_abs=float(np.max(abs(x[:,24])))))


def validate_mixed(K,branches,zero):
    records=[];differential=[];causal={str(h):[] for h in HORIZONS};saved={}
    for name,b in branches.items():
        if not name.startswith('mixed_') or not b['available']:continue
        times=b['t'][::4];u=b['reference'][::4];q=b['q']
        all_effect=np.stack([effect(K,times,u,t) for t in b['t']])
        d=q-zero['q']-all_effect;differential.extend(d);d_by_h={}
        saved[name+'_paired_differential_error25']=d
        saved[name+'_modeled_controlled_increment25']=all_effect
        saved[name+'_actual_q25']=q
        saved[name+'_actual_reference8']=b['reference']
        residual=q-all_effect
        for h in HORIZONS:
            errors=[];origins=[];predictions=[];targets=[];weights=[]
            offset=round(h/.005)
            for origin in range(0,101-offset,4):
                end=origin+offset;bg,w=residual_forecast(b['t'][:origin+1],residual[:origin+1],b['t'][end])
                prediction=all_effect[end]+bg
                errors.append(q[end]-prediction);origins.append(b['t'][origin]);predictions.append(prediction);targets.append(q[end]);weights.append(w)
            causal[str(h)].extend(errors);d_by_h[str(h)]=score(errors)
            prefix=name+'_h'+str(round(h*1000))
            saved[prefix+'_origin_times_s']=np.asarray(origins)
            saved[prefix+'_prediction25']=np.asarray(predictions);saved[prefix+'_actual25']=np.asarray(targets)
            saved[prefix+'_error25']=np.asarray(errors)
            # Variable causal windows are stored as explicit padded coefficients; unused past slots=0 weights, not physical gains.
            W=np.zeros((len(weights),101))
            for row,(origin,w) in enumerate(zip(range(0,101-offset,4),weights)):W[row,origin+1-len(w):origin+1]=w
            saved[prefix+'_causal_past_q_weights']=W
        records.append(dict(branch=name,paired_zero_differential=score(d),causal_by_horizon=d_by_h))
    return dict(branches=records,paired_zero_differential=score(differential),causal_by_horizon={h:score(v) for h,v in causal.items()},
        paired_zero_uses_future_offline_only=True,causal_forecast_uses_no_zero_future=True,
        future_command_source='actual held-out robot reference schedule; treated as publicly known task input, no future q is used',
        residual_estimator=dict(kind='unweighted causal affine least squares',window_s=WINDOW,sample_period_s=.005,minimum_samples=1,one_sample_rule='constant residual',calibration_error_used_as_noise=False)),saved


def process_context(data,shape,index,out):
    name=f'checkpoint_{index:02d}';case=data/shape/name;target=out/shape/name;target.mkdir(parents=True)
    context=mechanical_context(data,shape);checkpoint=read_json(case/'checkpoint.json')
    branches={folder.name:load_branch(folder,context) for folder in sorted(case.iterdir()) if folder.is_dir() and (folder/'metadata.json').exists()}
    if len(branches)!=26:raise ValueError('Expected exact26 branch identities')
    zero=branches['zero'];replay=branches['zero_replay']
    replay_exact=all(np.array_equal(zero[k],replay[k]) for k in ('q','pq','reference','ctrl','physical_reference','physical_ctrl'))
    if not replay_exact:raise ValueError('Robot zero replay failed')
    if any(not np.array_equal(b['q'][0],zero['q'][0]) for b in branches.values() if b['available']):raise ValueError('Robot anchors differ across restored branches')
    K,plus,minus,training,sides,missing=fit_micro(branches,zero)
    if not np.array_equal(K[0],np.zeros((25,8)),equal_nan=False) and not missing:raise ValueError('Nonzero response at origin')
    sources={key:(b['source'] if b['available'] else dict(metadata_path=b['metadata_path'],metadata_sha256=b['metadata_sha256'],available=False,reason=b['metadata'].get('unavailable_reason'))) for key,b in branches.items()}
    binding=dict(context_id=shape+'/'+name,checkpoint_path=str(case/'checkpoint.json'),checkpoint_sha256=sha(case/'checkpoint.json'),branches=sources)
    write_json(target/'input_manifest.json',binding);binding_sha=sha(target/'input_manifest.json')
    arrays=dict(response_kernel_times_s=np.arange(101)*.005,robot_anchor25=zero['q'][0],zero_robot_q25=zero['q'],output_qpos_addresses25=context['qadr'])
    # Missing complete input columns remain NaN in diagnostic NPZ and JSONnull, never an invented zero response.
    arrays['response25x8']=K
    for axis,side in sides.items():
        for sign,value in side.items():arrays['one_sided_'+axis+'_'+sign]=value
    if missing:
        validation=dict(status='UNAVAILABLE',reasons=['entire_input_column_unmeasured_'+str(i) for i in missing]);diagnostics={}
    else:validation,diagnostics=validate_mixed(K,branches,zero)
    np.savez_compressed(target/'robot_response_raw.npz',**arrays,**diagnostics)
    footprints={key:b['footprint'] for key,b in branches.items() if b['available']}
    write_json(target/'actual_input_footprints.json',footprints)
    check_fields={key:b['checks'] for key,b in branches.items() if b['available']}
    write_json(target/'recording_checks.json',dict(zero_replay_robot_fields_exact=replay_exact,all_branch_anchor_robot25_exact=True,branches=check_fields,
        q_end_clock='control.time and physical_time are end times; physical_interval_start is start of finite integration',
        command_start_clock='row k+1 actual_reference applied at row k time and held for5ms; each4ticks realizes one20ms total increment'))
    hs=[round(h/.005) for h in HORIZONS]
    model=dict(schema='local_robot_servo_identification_v1',context_id=shape+'/'+name,output_axes25=context['axes'],input_axes8=INPUT_AXES,
        output_units=['rad']*24+['m'],output_qpos_addresses25=context['qadr'],excluded_qpos_addresses=context['excluded_qpos_addresses'],
        free_object_qpos_addresses_excluded=context['free_object_qpos_addresses'],robot_anchor25=zero['q'][0],base_reference8=zero['reference'][0],
        ramp_period_s=.02,execution_period_s=.005,diagnostic_horizons_s=HORIZONS,
        dense_kernel_npz_path=str(target/'robot_response_raw.npz'),dense_kernel_npz_sha256=sha(target/'robot_response_raw.npz'),
        response25x8_at_diagnostic_horizons=K[hs].tolist() if not missing else None,
        known_plus8=plus,known_minus8=minus,missing_entire_input_columns=missing,training=training,
        fit_definition='for every output/time/input: sum(actual_micro_delta * (q_micro-q_zero)) / sum(actual_micro_delta**2); through origin; only available micro branches',
        validation=validation,source_manifest_sha256=binding_sha,source_manifest_path=str(target/'input_manifest.json'),
        actual_footprints_path=str(target/'actual_input_footprints.json'),actual_footprints_sha256=sha(target/'actual_input_footprints.json'),
        adapter_export_status='WITHHELD_PENDING_SEQUENCE_SUPPORT_POLICY_AND_REGISTERED_ROBOT_ERROR_GATES',
        sequence_scope='only recorded micro ramps and held-out mixed sequences; componentwise ranges do not certify arbitrary8D combinations',
        time_scope='this measured initial robot state, contact configuration, and at most.5s response memory; no interpolation between checkpoint states',
        error_scope='observed per-axis holdout statistics only; not a hard bound, no new joint threshold borrowed from pressure gates',
        global_model_admitted=False,new_physics=False)
    write_json(target/'model.json',model)
    return dict(context_id=model['context_id'],available=not missing,known_plus8=plus,known_minus8=minus,
        executed_branches=sum(b['available'] for b in branches.values()),unavailable_branches=[n for n,b in branches.items() if not b['available']],
        zero_replay_exact=replay_exact,model_path=str(target/'model.json'),model_sha256=sha(target/'model.json'),
        causal_by_horizon=validation.get('causal_by_horizon'),paired_zero_differential=validation.get('paired_zero_differential'),
        global_model_admitted=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DEFAULT_DATA);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--shape',choices=['capsule','box','sphere'],default='capsule');parser.add_argument('--checkpoint',type=int,default=0);parser.add_argument('--all',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    ledger=dict(schema='offline_robot_servo_identification_run_v1',started_utc=datetime.now(timezone.utc).isoformat(),new_physics=False,complete=False,contexts=[],failures=[])
    try:
        binding=source_binding(args.data);write_json(args.out/'source_binding.json',binding)
        write_json(args.out/'analysis_definition.json',dict(schema='offline_robot_servo_definition_v1',training='micro only',validation='all available mixed only',horizons_s=HORIZONS,
            full_kernel_memory_s=.5,causal_residual_window_s=WINDOW,statistics='per axis rad/m; no pooled mixed-unit errors',
            gates='diagnostic only; no post-hoc robot error admission thresholds',execution_source=str(Path(__file__).resolve()),execution_source_sha256=sha(__file__),
            python=sys.version,numpy=np.__version__,platform=platform.platform(),source_binding_sha256=sha(args.out/'source_binding.json')))
        contexts=[(s,i) for s in ('capsule','box','sphere') for i in range(5)] if args.all else [(args.shape,args.checkpoint)]
        for shape,index in contexts:
            result=process_context(args.data,shape,index,args.out);ledger['contexts'].append(result)
            print(json.dumps(jsonable(result),ensure_ascii=False,allow_nan=False),flush=True)
        ledger['complete']=True
    except Exception as exc:
        ledger['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:
        ledger['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(args.out/'execution_complete.json',ledger)


if __name__=='__main__':main()
