"""Independent phase-2 public MIMO calibration, not a performance experiment.

Identifies pressure/net-wrench response to 5 synergies, lift and 2 wrists.
Holdouts are eight prescribed smooth combined-input sequences per shape. The
online-error calculation uses only starting/prior measured outputs and known
commands; future zero-branch output is never used as its baseline.
"""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from public_synergy_calibration import (sha,serial,write_json,transform,public_packet,
    capture_state,restore_state,guard,ProbeStop,arrays,self_test)
import research_imports
from research_imports import FOUNDATION


def references(plant):return np.r_[plant.alpha,plant.lift_ref,plant.wrist_ref]
def make_hadamard():
    H=np.ones((1,1))
    for _ in range(3):H=np.block([[H,H],[H,-H]])
    return H


def design_holdout(index,base,lo,hi,plan):
    steps=round(plan['duration_s']/.02);t=np.arange(steps+1)/steps;H=make_hadamard()
    columns=[1,2,3,4,5,0,6,7];amp=np.asarray(plan['holdout_amplitude'])
    offsets=np.zeros((steps+1,8));directions=[]
    for j in range(8):
        sign=float(H[index,columns[j]])
        if base[j]+amp[j]>hi[j]+1e-12:sign=-1.
        if base[j]-amp[j]<lo[j]-1e-12:sign=1.
        wave=np.sin(np.pi*t)**2*(.8+.2*np.cos(2*np.pi*(1+j%3)*t+index*.37+j*.23))
        offsets[:,j]=sign*amp[j]*wave;directions.append(sign)
    return base[None,:]+offsets,directions


def probe(plant,state,R0,p0,path,lo,hi,plan,folder):
    restore_state(plant,state);base=references(plant);folder.mkdir(parents=True,exist_ok=False)
    unavailable=np.flatnonzero(np.any((path<lo[None,:]-1e-12)|(path>hi[None,:]+1e-12),axis=0)).tolist()
    if unavailable:
        meta=dict(available=False,complete=False,attempted_physics=False,unavailable_inputs=unavailable,
            requested_path=path,executed_path=None,valid_lower=lo,valid_upper=hi,base=base)
        write_json(folder/'metadata.json',meta);return None,meta
    original=plant._contacts;physical=[];control=[];last=-np.inf;failure=None
    def packet(contacts):
        row=public_packet(plant,contacts,R0,p0);row['input_reference']=references(plant);row['y_palm']=np.r_[row['pressure'],row['wrench_palm']]
        return row
    def hooked(patches=True):
        nonlocal last
        result=original(patches)
        if float(plant.data.time)>last+1e-12:
            row=packet(result if patches else original(True));row['interval_start']=float(plant.data.time-plant.model.opt.timestep);row['duration']=float(plant.model.opt.timestep)
            physical.append(row);last=float(plant.data.time);guard(row,plan)
        return result
    control.append(packet(original(True)))
    try:
        guard(control[0],plan);plant._contacts=hooked
        for interval in range(len(path)-1):
            start=references(plant)
            for sub in range(4):
                target=path[interval]+(sub+1)/4.*(path[interval+1]-path[interval])
                plant.execute_delta(target[:5]-plant.alpha);plant.set_motion(float(target[5]),target[6:])
                actual=references(plant)
                if np.max(abs(actual-target))>1e-9:raise ProbeStop('requested_vs_actual_reference_mismatch')
                if np.any(actual<lo-1e-10) or np.any(actual>hi+1e-10):raise ProbeStop('actual_input_outside_public_intersection')
                plant.advance(.005);control.append(packet(original(True)));guard(control[-1],plan)
    except ProbeStop as exc:failure=str(exc)
    finally:plant._contacts=original
    meta=dict(available=True,complete=failure is None,stop_reason=failure,requested_path=path,
        executed_path=np.stack([r['input_reference'] for r in control]),base=base,valid_lower=lo,valid_upper=hi,
        maximum_pressure_N=np.max([r['pressure'] for r in physical or control],axis=0),
        maximum_penetration_m=max(r['penetration'] for r in physical or control),
        all_external_patch_measurement=True,target_identity_selection=False)
    packed={**arrays(control),**arrays(physical,'physical_')};packed['metadata']=json.dumps(serial(meta),ensure_ascii=False)
    np.savez_compressed(folder/'trace.npz',**packed);meta['trace_sha256']=sha(folder/'trace.npz');write_json(folder/'metadata.json',meta)
    return control,meta


def convolution(kernel,path):
    """20ms interval increments, each implemented by four equal 5ms ramps."""
    increments=np.diff(path,axis=0);n=len(path);result=np.zeros((n,11))
    for k in range(1,n):
        for j in range(k):result[k]+=kernel[k-j]@increments[j]
    return result


def error_summary(errors,plan):
    e=np.asarray(errors);rmse=np.sqrt(np.mean(e*e,axis=0));p95=np.quantile(abs(e),.95,axis=0)
    threshold=np.array([plan['model_gates']['pressure_rmse_N']]*5+[plan['model_gates']['force_axis_rmse_N']]*3+[plan['model_gates']['moment_axis_rmse_Nm']]*3)
    passed=rmse<=threshold;passed[:5]&=p95[:5]<=plan['model_gates']['pressure_p95_N']
    return dict(rmse_per_output=rmse,p95_abs_per_output=p95,max_abs_per_output=np.max(abs(e),axis=0),
                engineering_gate_per_output=passed,all_engineering_gates_passed=bool(np.all(passed)),vectors=len(e),
                units=['N']*8+['Nm']*3,not_a_statistical_error_bound=True)


def fit_case(results,history,base,plan):
    nt=round(plan['duration_s']/.02)+1;zero=results['zero'][0];zidx=np.arange(nt)*4
    exports={};availability=[]
    for axis in range(8):availability.append(dict(axis=axis,plus=results[f'train_{axis}_plus'][1]['available'],minus=results[f'train_{axis}_minus'][1]['available']))
    for frame,key in [('initial_fixed_reference','y'),('current_palm','y_palm')]:
        Y0=np.stack([zero[i][key] for i in zidx]);K=np.zeros((nt,11,8));identified=np.zeros(8,bool)
        for axis in range(8):
            valid=[sign for sign in ('plus','minus') if results[f'train_{axis}_{sign}'][1]['available']]
            if not valid:continue
            identified[axis]=True
            if len(valid)==2:
                rp,mp=results[f'train_{axis}_plus'];rn,mn=results[f'train_{axis}_minus'];yp=np.stack([rp[i][key] for i in zidx]);yn=np.stack([rn[i][key] for i in zidx]);dp=mp['requested_path'][-1,axis]-base[axis];dn=mn['requested_path'][-1,axis]-base[axis];K[:,:,axis]=(yp-yn)/(dp-dn)
            else:
                rows,meta=results[f'train_{axis}_{valid[0]}'];y=np.stack([rows[i][key] for i in zidx]);amplitude=meta['requested_path'][-1,axis]-base[axis];K[:,:,axis]=(y-Y0)/amplitude
        absolute=[];response=[];by_horizon={str(h):[] for h in plan['forecast_horizons_s']};details=[]
        hist_t=np.array([r['time'] for r in history]);hist_y=np.stack([r[key] for r in history]);hist_t=hist_t-hist_t[-1]
        for q in range(plan['holdout_sequences']):
            rows,meta=results[f'holdout_{q:02d}']
            if not meta['available']:details.append(dict(sequence=q,available=False));continue
            path=np.asarray(meta['requested_path']);measured=np.stack([rows[i][key] for i in zidx]);modeled=convolution(K,path)
            # Differential kernel errors may use future zero ONLY as a separate
            # offline response-identification diagnostic, never online baseline.
            response.append(measured-Y0-modeled)
            residual=measured-modeled;sequence_errors=[]
            for origin in range(nt):
                past_t=np.r_[hist_t,np.arange(1,origin+1)*.02];past_y=np.vstack((hist_y,residual[1:origin+1]))
                now=origin*.02;select=past_t>=now-plan['causal_window_s']-1e-10;tt=past_t[select];yy=past_y[select]
                if len(tt)>=4:
                    x=tt-tt.mean();slope=x@(yy-yy.mean(axis=0))/max(1e-12,float(x@x));level=yy.mean(axis=0)+slope*(now-tt.mean())
                else:slope=np.zeros(11);level=yy[-1]
                for h in plan['forecast_horizons_s']:
                    ahead=round(h/.02);target=origin+ahead
                    if target>=nt:continue
                    prediction=level+h*slope+modeled[target]
                    error=prediction-measured[target];by_horizon[str(h)].append(error);sequence_errors.append(dict(origin_s=now,horizon_s=h,error=error))
            details.append(dict(sequence=q,available=True,complete_causal_errors=sequence_errors,
                input_min=path.min(axis=0)-base,input_max=path.max(axis=0)-base))
        exported=K.tolist()
        for axis in np.flatnonzero(~identified):
            for block in exported:
                for row in block:row[axis]=None
        exports[frame]=dict(kernel_response_to_20ms_interval_increment=exported,
            interval_s=.02,kernel_duration_s=plan['duration_s'],identified_inputs=identified,
            differential_response_error=error_summary(np.concatenate(response),plan),
            complete_causal_by_horizon={k:error_summary(v,plan) for k,v in by_horizon.items()},holdout=details)
    return dict(schema='public_response_phase2_model_v1',coordinates=exports,input_availability=availability,
        input_units=['dimensionless_synergy']*5+['m','rad','rad'],output_units=['N']*8+['Nm']*3,
        model_admitted=False,admission_scope='Engineering gates only; independent audit and explicit measured-context admission required.',
        no_future_zero_in_complete_prediction=True,calibration_not_performance=True)


def plan_design():
    files=[Path(__file__),ROOT/'calibration/public_synergy_calibration.py',ROOT/'runtime.py',ROOT/'research_imports.py',FOUNDATION/'hand_plant.py',FOUNDATION/'sensors.py',FOUNDATION/'contracts.py']
    return dict(schema='public_response_phase2_plan_v1',status='frozen_before_phase2_data',shapes=['capsule','box','sphere'],mass_setup_kg=.12,seed=630201,
        clone_time_s=7.5,duration_s=.5,interval_s=.02,causal_window_s=.12,holdout_sequences=8,
        training_amplitude=[.004]*5+[.0005,.002,.002],holdout_amplitude=[.006]*5+[.0008,.003,.003],
        forecast_horizons_s=[.02,.04,.06,.1],
        model_gates=dict(pressure_rmse_N=.35,pressure_p95_N=.75,force_axis_rmse_N=.05,moment_axis_rmse_Nm=.005),
        stop_gates=dict(per_finger_pressure_N=5.02,sum_finger_pressure_N=18.02,penetration_m=.002,table_support_N=.05,minimum_object_lift_m=.07),
        source_sha256={str(p.resolve()):sha(p) for p in files},
        interpretation='Public calibration, not policy performance; unexecutable positive lift and sphere-index directions remain explicit. Model validation gates are engineering applicability tests, not statistical or physical safety guarantees.',
        holdout_definition='Eight disjoint prescribed smooth multi-input sequences per shape. No holdout sequence is fitted. Past public residual only, known command convolution, and no future zero baseline.',
        frame_definition=dict(current_palm='Each sample own palm origin/axes.',initial_fixed_reference='Permanently at clone palm origin/axes, using hand pose only. Runtime use requires public robot FK, never object pose.'),
        input_extent='Only measured relative paths; no assertion of full old .02/.04 action domain or bidirectional positive lift above .100m.',
        stop_entire_sweep_on_first_safety_violation=True)


def run(plan,out):
    from runtime import ResearchSession
    out.mkdir(parents=True,exist_ok=False);write_json(out/'plan.json',plan);summary=dict(execution_complete=False,model_admitted=False,shapes=[])
    for shape in plan['shapes']:
        s=ResearchSession(method='ratio',shape=shape,scenario='nominal',seed=plan['seed'],mode='scheduled',split='demonstration');plant=s.plant;past=[]
        while s.time<7.5-1e-9:
            s.step_one_sensor_tick();R=plant.data.xmat[plant.palm].reshape(3,3).copy();p=plant.data.xpos[plant.palm].copy();c=plant._contacts(True)
            from sensors import aggregate_patches
            m,d=aggregate_patches(c['patches']);n=np.asarray(m['normal']);w=np.asarray(d['normal_wrench_map'])@n+np.asarray(d['finger_non_normal_wrench'])+np.asarray(d['raw_auxiliary_wrench'])
            past.append(dict(time=float(plant.data.time),pressure=n,wrench=w,R=R,p=p))
        if s.handoff_attempted or abs(plant.config.mass-.12)>1e-12:raise RuntimeError('Invalid public preparation/setup')
        R0=plant.data.xmat[plant.palm].reshape(3,3).copy();p0=plant.data.xpos[plant.palm].copy();state=capture_state(plant);base=references(plant)
        cal=s.calibration;rad=float(cal['selected_radius']);ref=np.asarray(cal['reference_alpha']);lo5=np.maximum.reduce([plant.closure_min,ref-rad,base[:5]-rad]);hi5=np.minimum.reduce([plant.closure_max,ref+rad,base[:5]+rad])
        wrist_ids=[plant.aid(k) for k in ('rh_A_WRJ2','rh_A_WRJ1')];wr=plant.model.actuator_ctrlrange[wrist_ids];lo=np.r_[lo5,0.,wr[:,0]];hi=np.r_[hi5,plant.config.lift_target,wr[:,1]]
        history=[dict(time=x['time']-plant.time,y=np.r_[x['pressure'],transform(x['wrench'],x['R'],x['p'],R0,p0)],y_palm=np.r_[x['pressure'],x['wrench']]) for x in past[-60:]]
        folder=out/shape;folder.mkdir();write_json(folder/'base_metadata.json',dict(config=s.metadata()['config'],calibration=cal,base=base,lower=lo,upper=hi,reference_origin=p0,reference_rotation=R0,history=history))
        nt=round(plan['duration_s']/.02)+1;zero=np.tile(base,(nt,1));designs=[('zero',zero),('zero_replay',zero.copy())]
        for axis,amplitude in enumerate(plan['training_amplitude']):
            for sign,label in ((1,'plus'),(-1,'minus')):
                path=zero.copy();path[1:,axis]+=sign*amplitude;designs.append((f'train_{axis}_{label}',path))
        for q in range(plan['holdout_sequences']):designs.append((f'holdout_{q:02d}',design_holdout(q,base,lo,hi,plan)[0]))
        results={}
        for label,path in designs:
            result=probe(plant,state,R0,p0,path,lo,hi,plan,folder/label);results[label]=result;print(shape,label,'available',result[1]['available'],'complete',result[1]['complete'],flush=True)
            if result[1]['available'] and not result[1]['complete']:
                summary['stopped']=dict(shape=shape,branch=label,reason=result[1]['stop_reason']);write_json(out/'summary.json',summary);return
            if label=='zero_replay':
                a,b=results['zero'][0],result[0];equal=all(np.array_equal(x['y'],y['y']) and np.array_equal(x['qpos'],y['qpos']) for x,y in zip(a,b));write_json(folder/'restore_check.json',dict(passed=equal,samples=len(a)))
                if not equal:raise RuntimeError('Restore replay differs')
        model=fit_case(results,history,base,plan);write_json(folder/'model.json',model)
        summary['shapes'].append(dict(shape=shape,model_sha256=sha(folder/'model.json'),planned=len(designs),available=sum(x[1]['available'] for x in results.values()),unavailable=[k for k,v in results.items() if not v[1]['available']]));write_json(out/'summary.json',summary)
    summary['execution_complete']=True;write_json(out/'summary.json',summary)


def test():
    base=self_test();n=12;K=np.zeros((n,11,8));K[1:,:8,:]=np.eye(8)[None,:,:];path=np.zeros((n,8));path[1:,0]=.004;result=convolution(K,path)
    checks=dict(base['checks']);checks['interval_kernel_step_superposition']=bool(np.allclose(result[1:,0],.004) and np.allclose(result[:,1:],0.))
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--prepare-plan',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'calibration/phase2_public_v1');args=p.parse_args();planfile=ROOT/'calibration/plan_phase2_v1.json'
    if args.self_test:result=test();write_json(ROOT/'calibration/self_test_phase2_v1.json',result);print(json.dumps(result));return
    if args.prepare_plan:
        if planfile.exists():raise FileExistsError(planfile)
        write_json(planfile,plan_design());print(planfile);return
    if not args.run:p.error('Choose --self-test/--prepare-plan/--run')
    plan=json.loads(planfile.read_text(encoding='utf-8'))
    if any(sha(Path(k))!=v for k,v in plan['source_sha256'].items()):raise RuntimeError('Frozen phase-2 source identity changed')
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'calibration'):raise ValueError('Write only new calibration root')
    run(plan,out)


if __name__=='__main__':main()
