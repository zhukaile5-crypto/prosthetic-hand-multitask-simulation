"""Public full-skin pressure/net-wrench calibration; no policy performance run.

Phase 1 identifies 11 outputs by five synergy inputs. Motion columns are NOT
filled with zeros: they remain unmeasured for a separate calibration protocol.
Every branch starts from the exact same complete MuJoCo data and public grasp.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_imports
from research_imports import FOUNDATION
from sensors import aggregate_patches


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x
def write_json(path,x):Path(path).write_text(json.dumps(serial(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def transform(w,R,p,R0,p0):
    f=R@w[:3];moment=R@w[3:]+np.cross(p-p0,f)
    return np.r_[R0.T@f,R0.T@moment]


def public_packet(plant,contacts,R0,p0):
    patches=contacts['patches'];measurement,diag=aggregate_patches(patches,support_axis=plant.data.xmat[plant.palm].reshape(3,3).T@np.array([0.,0.,1.]))
    pressure=np.asarray(measurement['normal']);normal=np.asarray(diag['normal_wrench_map'])@pressure
    tangent=np.asarray(diag['finger_non_normal_wrench']);aux=np.asarray(diag['raw_auxiliary_wrench']);wrench=normal+tangent+aux
    direct=np.zeros(6)
    for patch in patches:
        f=np.asarray(patch['force']);direct[:3]+=f;direct[3:]+=np.cross(patch['position'],f)+np.asarray(patch['torque'])
    if np.max(abs(wrench-direct))>1e-9:raise RuntimeError('Public distributed aggregation does not reconstruct all-patch wrench')
    R=plant.data.xmat[plant.palm].reshape(3,3).copy();p=plant.data.xpos[plant.palm].copy()
    wr=transform(wrench,R,p,R0,p0)
    return dict(time=float(plant.data.time),pressure=pressure,wrench_palm=wrench,wrench_reference=wr,
        normal_reference=transform(normal,R,p,R0,p0),tangent_reference=transform(tangent,R,p,R0,p0),
        auxiliary_reference=transform(aux,R,p,R0,p0),palm_rotation=R,palm_position=p,
        y=np.r_[pressure,wr],raw_patches=json.dumps(serial(patches),ensure_ascii=False),
        skin_pressure_sum=float(pressure.sum()),penetration=max([float(v['penetration']) for v in patches] or [0.]),
        table_support_N=float(contacts['support']),object_lift_m=float(plant.data.xpos[plant.object,2]-plant.initial_position[2]),
        qpos=plant.data.qpos.copy(),qvel=plant.data.qvel.copy(),ctrl=plant.data.ctrl.copy(),alpha=plant.alpha.copy())


SIDE=('wrist_ref','env','phase','alpha','lift_ref','release_started','release_thumb','release_lift','release_alpha','release_ctrl',
      'control_steps','last_requested','last_executed','raw_requested_delta','max_hidden_clip','force_peak','force_peak_contacts')
def capture_state(plant):
    import mujoco
    data=mujoco.MjData(plant.model);mujoco.mj_copyData(data,plant.model,plant.data)
    return dict(api=plant.clone_state(),data=data,side={k:copy.deepcopy(getattr(plant,k)) for k in SIDE if hasattr(plant,k)})
def restore_state(plant,state):
    import mujoco
    plant.restore_state(state['api'])
    for key,value in state['side'].items():setattr(plant,key,copy.deepcopy(value))
    # Last, restore ALL integration and derived solver/contact state, including
    # warmstart, to avoid an extra mj_forward changing the copied branch start.
    mujoco.mj_copyData(plant.data,plant.model,state['data']);plant.substep_records=[];plant._cache=None


class ProbeStop(RuntimeError):pass
def guard(row,plan):
    g=plan['stop_gates'];p=row['pressure'];reason=[]
    if not np.isfinite(row['y']).all():reason.append('nonfinite_public_measurement')
    if np.any(p>g['per_finger_pressure_N']):reason.append('finger_pressure')
    if p.sum()>g['sum_finger_pressure_N']:reason.append('total_finger_pressure')
    if row['penetration']>g['penetration_m']:reason.append('penetration')
    if row['table_support_N']>g['table_support_N']:reason.append('table_support')
    if row['object_lift_m']<g['minimum_object_lift_m']:reason.append('object_lift')
    if np.count_nonzero(p>.2)<3 or p[0]<=.2:reason.append('public_skin_contact_support')
    if reason:raise ProbeStop(','.join(reason))


def arrays(rows,prefix=''):
    if not rows:return {}
    return {prefix+k:np.asarray([r[k] for r in rows]) for k in rows[0]}


def branch(plant,state,R0,p0,delta,lower,upper,plan,folder):
    restore_state(plant,state);base=plant.alpha.copy();requested=base+delta
    folder.mkdir(parents=True,exist_ok=False)
    if np.any(requested<lower-1e-12) or np.any(requested>upper+1e-12):
        meta=dict(requested_delta=delta,executed_delta=np.zeros(5),base_alpha=base,lower=lower,upper=upper,
            complete=False,available=False,stop_reason=None,unavailable_reason='requested_direction_outside_admitted_intersection',
            attempted_physics=False,valid_input_delta_lower=lower-base,valid_input_delta_upper=upper-base)
        write_json(folder/'metadata.json',meta);return [],[],meta
    original=plant._contacts;physical=[];control=[];last_time=-np.inf;failure=None;t0=float(plant.data.time)
    def hooked(patches=True):
        nonlocal last_time
        result=original(patches)
        if float(plant.data.time)>last_time+1e-12:
            full=result if patches else original(True);row=public_packet(plant,full,R0,p0)
            row['interval_start']=float(plant.data.time-plant.model.opt.timestep);row['duration']=float(plant.model.opt.timestep)
            physical.append(row);last_time=float(plant.data.time);guard(row,plan)
        return result
    start=public_packet(plant,original(True),R0,p0);control.append(start)
    try:
        guard(start,plan);plant._contacts=hooked
        for tick in range(round(plan['duration_s']/.005)):
            desired=base+delta*min(1.,(tick+1)/4.)
            plant.execute_delta(desired-plant.alpha)
            if np.any(plant.alpha<lower-1e-10) or np.any(plant.alpha>upper+1e-10):raise ProbeStop('actual_coordinate_outside_intersection')
            plant.advance(.005)
            row=public_packet(plant,original(True),R0,p0);control.append(row);guard(row,plan)
    except ProbeStop as exc:failure=str(exc)
    finally:plant._contacts=original
    packed={**arrays(control),**arrays(physical,'physical_')}
    meta=dict(requested_delta=delta,executed_delta=plant.alpha-base,base_alpha=base,lower=lower,upper=upper,
        t0=t0,complete=failure is None,available=True,stop_reason=failure,all_external_patches=True,target_identity_filter=False,
        maximum_pressure_N=np.max([r['pressure'] for r in physical or control],axis=0),
        maximum_penetration_m=max(r['penetration'] for r in physical or control),
        reference_origin_world_m=p0,reference_rotation_world=R0)
    packed['metadata']=json.dumps(serial(meta),ensure_ascii=False)
    np.savez_compressed(folder/'trace.npz',**packed);write_json(folder/'metadata.json',dict(**meta,trace_sha256=sha(folder/'trace.npz')))
    return control,physical,meta


def fit_model(branches,history,base_packet,cal,plan):
    times=np.array(plan['horizons_s']);zero=branches['zero'][0];zero_y=np.stack([zero[round(h/.005)]['y'] for h in times])
    J=np.zeros((len(times),11,5));train=plan['training_amplitude'];hold=plan['holdout_amplitude'];availability=[]
    for axis in range(5):
        pos=branches[f'train_{axis}_plus'];neg=branches[f'train_{axis}_minus'];directions=[]
        if pos[2]['available']:directions.append('plus')
        if neg[2]['available']:directions.append('minus')
        availability.append(dict(axis=axis,training_directions=directions,
            input_delta_lower=float(neg[2]['lower'][axis]-neg[2]['base_alpha'][axis]),
            input_delta_upper=float(pos[2]['upper'][axis]-pos[2]['base_alpha'][axis])))
        if directions==['plus','minus']:
            yp=np.stack([pos[0][round(h/.005)]['y'] for h in times]);yn=np.stack([neg[0][round(h/.005)]['y'] for h in times]);J[:,:,axis]=(yp-yn)/(2.*train)
        elif directions:
            chosen=pos if directions[0]=='plus' else neg;amplitude=float(chosen[2]['executed_delta'][axis])
            yy=np.stack([chosen[0][round(h/.005)]['y'] for h in times]);J[:,:,axis]=(yy-zero_y)/amplitude
    past=[x for x in history if x['time']>=base_packet['time']-.25-1e-10]
    tt=np.array([x['time'] for x in past]);yy=np.stack([x['y'] for x in past]);center=tt-tt.mean();slope=center@(yy-yy.mean(axis=0))/(center@center)
    causal_baseline=base_packet['y'][None,:]+times[:,None]*slope[None,:]
    validations=[]
    for axis in range(5):
        for sign,label in ((1,'plus'),(-1,'minus')):
            chosen=branches[f'hold_{axis}_{label}']
            if not chosen[2]['available']:
                validations.append(dict(axis=axis,sign=sign,available=False,reason=chosen[2]['unavailable_reason']));continue
            actual=np.stack([chosen[0][round(h/.005)]['y'] for h in times]);response=np.einsum('hoi,i->ho',J,np.asarray(chosen[2]['executed_delta']))
            validations.append(dict(axis=axis,sign=sign,available=True,response_difference_error=actual-zero_y-response,
                complete_causal_prediction_error=actual-causal_baseline-response,
                baseline_kind='Only start public output plus past .25s linear slope; no holdout-future zero output is used in complete prediction.'))
    exported=J.tolist()
    for axis,row in enumerate(availability):
        if not row['training_directions']:
            for horizon in exported:
                for output in horizon:output[axis]=None
    valid=[i for i,row in enumerate(availability) if row['training_directions']]
    return dict(horizons_s=times,jacobian_11x5=exported,input_availability=availability,
        identified_input_rank_by_horizon=[int(np.linalg.matrix_rank(x[:,valid])) if valid else 0 for x in J],units_outputs=['N']*8+['Nm']*3,
        input_units=['dimensionless_synergy']*5,causal_history_slope=slope,holdout=validations,
        missing_motion_columns=['lift_m','wrist1_rad','wrist2_rad'],full_11x8_identified=False,
        valid_amplitudes=dict(training=train,holdout=hold),seen_calibration_not_performance=True)


def self_test():
    import mujoco
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .2"><freejoint/><geom type="sphere" size=".05" mass=".1"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m)
    for _ in range(20):mujoco.mj_step(m,d)
    saved=mujoco.MjData(m);mujoco.mj_copyData(saved,m,d)
    for _ in range(10):mujoco.mj_step(m,d)
    expected=d.qpos.copy();expected_v=d.qvel.copy();mujoco.mj_copyData(d,m,saved)
    for _ in range(10):mujoco.mj_step(m,d)
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);w=np.array([1.,2.,3.,.1,.2,.3]);p=np.array([.2,-.1,.4])
    out=transform(w,R,p,np.eye(3),np.zeros(3));f=R@w[:3]
    checks=dict(full_data_future_qpos_bitwise=bool(np.array_equal(expected,d.qpos)),full_data_future_qvel_bitwise=bool(np.array_equal(expected_v,d.qvel)),
       fixed_force_rotation=bool(np.allclose(out[:3],f,atol=1e-12)),fixed_origin_moment=bool(np.allclose(out[3:],R@w[3:]+np.cross(p,f),atol=1e-12)))
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks,synthetic_only=True)


def make_plan():
    return dict(schema='public_synergy_wrench_calibration_phase1_v1',status='frozen_calibration_design',
        shapes=['capsule','box','sphere'],mass_setup_kg=.12,seed=630101,clone_time_s=7.5,duration_s=.5,
        training_amplitude=.004,holdout_amplitude=.008,horizons_s=[.02,.04,.06,.1,.2,.5],
        stop_gates=dict(per_finger_pressure_N=5.02,sum_finger_pressure_N=18.02,penetration_m=.002,table_support_N=.05,minimum_object_lift_m=.07),
        source_sha256={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):sha(p) for p in [Path(__file__).resolve(),ROOT/'runtime.py',ROOT/'research_imports.py',FOUNDATION/'hand_plant.py',FOUNDATION/'sensors.py',FOUNDATION/'contracts.py']},
        scope='Public model identification, not policy performance. Only first 5 inputs identified. Motion explicitly deferred; .100m lift has no positive authority.',
        complete_holdout_prediction='Report independently from zero-paired response-kernel error; only initial output and causal past can predict baseline.',
        stop_entire_sweep_on_first_violation=True)


def run(plan,out):
    import mujoco
    from runtime import ResearchSession
    out.mkdir(parents=True,exist_ok=False);write_json(out/'plan.json',plan);summary=dict(complete=False,shapes=[])
    for shape in plan['shapes']:
        session=ResearchSession(method='ratio',shape=shape,scenario='nominal',seed=plan['seed'],mode='scheduled',split='demonstration')
        plant=session.plant;past=[]
        while session.time<7.5-1e-9:
            session.step_one_sensor_tick()
            R=plant.data.xmat[plant.palm].reshape(3,3).copy();p=plant.data.xpos[plant.palm].copy()
            past.append(dict(time=float(plant.data.time),contacts=plant._contacts(True),R=R,p=p))
        if session.handoff_attempted:raise RuntimeError('A strategy acted before calibration cloning')
        if abs(plant.config.mass-.12)>1e-12:raise RuntimeError('Standard calibration must not inherit a heavy-box case')
        R0=plant.data.xmat[plant.palm].reshape(3,3).copy();p0=plant.data.xpos[plant.palm].copy();state=capture_state(plant)
        history=[]
        for item in past[-60:]:
            measurement,d=aggregate_patches(item['contacts']['patches']);n=np.asarray(measurement['normal']);w=np.asarray(d['normal_wrench_map'])@n+np.asarray(d['finger_non_normal_wrench'])+np.asarray(d['raw_auxiliary_wrench'])
            history.append(dict(time=item['time'],y=np.r_[n,transform(w,item['R'],item['p'],R0,p0)]))
        cal=session.calibration;radius=float(cal['selected_radius']);ref=np.asarray(cal['reference_alpha']);base=plant.alpha.copy()
        lower=np.maximum.reduce([plant.closure_min,ref-radius,base-radius]);upper=np.minimum.reduce([plant.closure_max,ref+radius,base+radius])
        if np.any(base<lower-1e-9) or np.any(base>upper+1e-9):raise RuntimeError('Common preparation outside admitted coordinate intersection')
        case=out/shape;case.mkdir();write_json(case/'base_metadata.json',dict(shape=shape,config=session.metadata()['config'],calibration=cal,reference_origin=p0,reference_rotation=R0,lower=lower,upper=upper,base_alpha=base,history=history))
        labels=[('zero',np.zeros(5)),('zero_replay',np.zeros(5))]
        for split,amp in [('train',plan['training_amplitude']),('hold',plan['holdout_amplitude'])]:
            for axis in range(5):
                for sign,label in [(1,'plus'),(-1,'minus')]:labels.append((f'{split}_{axis}_{label}',np.eye(5)[axis]*sign*amp))
        results={}
        for label,delta in labels:
            result=branch(plant,state,R0,p0,delta,lower,upper,plan,case/label);results[label]=result
            print(shape,label,'complete',result[2]['complete'],'available',result[2]['available'],flush=True)
            if result[2]['available'] and not result[2]['complete']:
                summary.update(stopped=dict(shape=shape,branch=label,reason=result[2]['stop_reason']));write_json(out/'summary.json',summary);return
            if label=='zero_replay':
                a,b=results['zero'][1],results['zero_replay'][1]
                identical=len(a)==len(b) and all(np.array_equal(x['y'],y['y']) and np.array_equal(x['qpos'],y['qpos']) for x,y in zip(a,b))
                write_json(case/'restore_replay_check.json',dict(passed=identical,physical_samples=len(a)))
                if not identical:raise RuntimeError('Complete restored zero trajectories differ')
        model=fit_model(results,history,results['zero'][0][0],cal,plan);write_json(case/'kernel_model.json',model)
        summary['shapes'].append(dict(shape=shape,model_sha256=sha(case/'kernel_model.json'),probes=len(results)));write_json(out/'summary.json',summary)
    summary['complete']=True;write_json(out/'summary.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--prepare-plan',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'calibration/phase1_synergy_v1');a=p.parse_args()
    planpath=ROOT/'calibration/plan_phase1_v1.json'
    if a.self_test:
        result=self_test();write_json(ROOT/'calibration/self_test_v1.json',result);print(json.dumps(result));return
    if a.prepare_plan:
        if planpath.exists():raise FileExistsError(planpath)
        write_json(planpath,make_plan());print(planpath);return
    if not a.run:p.error('Choose --self-test, --prepare-plan or --run')
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    for key,value in plan['source_sha256'].items():
        if sha(Path(key) if Path(key).is_absolute() else ROOT/key)!=value:raise RuntimeError('Calibration source identity changed: '+key)
    out=a.out.resolve()
    if not out.is_relative_to(ROOT/'calibration'):raise ValueError('Write only under new calibration directory')
    run(plan,out)


if __name__=='__main__':main()
