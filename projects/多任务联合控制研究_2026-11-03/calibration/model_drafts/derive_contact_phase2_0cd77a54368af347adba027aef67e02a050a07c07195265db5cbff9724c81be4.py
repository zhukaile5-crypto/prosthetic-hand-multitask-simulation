"""Offline 16-output contact extension of existing phase-2 raw records.

No physics/model/controller execution. Existing phase-2 plan/model/raw files
are never modified. Five shear magnitudes use the SAME declared aggregate
virtual-tactile definition, not online per-patch traction or material friction.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x


def shear(patches):
    total=np.zeros(5)
    for p in patches:
        digit=int(p['digit'])
        if 0<=digit<5:
            tang=np.asarray(p['force'])-float(p['normal'])*np.asarray(p['normal_dir'])
            total[digit]+=float(np.linalg.norm(tang))
    return total


def extend(y,tau):return np.c_[y[:,:5],tau,y[:,5:]]
def response(kernel,path):
    change=np.diff(path,axis=0);pred=np.zeros((len(path),16))
    for k in range(1,len(path)):
        for j in range(k):pred[k]+=kernel[k-j]@change[j]
    return pred


def summarize(error):
    e=np.asarray(error);rmse=np.sqrt(np.mean(e**2,axis=0));p95=np.quantile(abs(e),.95,axis=0)
    groups={}
    for name,sl,unit in [('normal',slice(0,5),'N'),('shear_magnitude',slice(5,10),'N'),('net_force',slice(10,13),'N'),('net_moment',slice(13,16),'Nm')]:
        groups[name]=dict(rmse_per_output=rmse[sl],p95_abs_per_output=p95[sl],maximum_abs_per_output=np.max(abs(e[:,sl]),axis=0),unit=unit)
    return dict(groups=groups,rmse_per_output=rmse,p95_abs_per_output=p95,vectors=len(e),
        empirical_error_covariance=np.cov(e,rowvar=False),correlated_overlapping_forecasts=True,
        engineering_reference_pass=dict(normal=bool(np.all(rmse[:5]<=.35)&np.all(p95[:5]<=.75)),
            shear_exploratory=bool(np.all(rmse[5:10]<=.35)&np.all(p95[5:10]<=.75)),
            net_force=bool(np.all(rmse[10:13]<=.05)),net_moment=bool(np.all(rmse[13:]<=.005))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'calibration/phase2_public_v1');p.add_argument('--out',type=Path,default=ROOT/'calibration/public_contact_model_phase2_v1.json');args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    if not out.is_relative_to(ROOT/'calibration'):raise ValueError('Write only new derived calibration artifact')
    plan=read(args.root/'plan.json');existing=ROOT/'calibration/public_model_phase2_v1.json';base_document=read(existing);cases=[];manifest=[]
    for case in base_document['shapes']:
        shape=case['shape'];folder=args.root/shape;base=read(folder/'base_metadata.json');data={};meta={};idx=np.arange(26)*4
        for path in sorted(folder.glob('*/metadata.json')):
            label=path.parent.name;m=read(path);meta[label]=m
            if not m['available']:continue
            with np.load(path.with_name('trace.npz'),allow_pickle=False) as z:
                raw=z['raw_patches'];tau=np.stack([shear(json.loads(str(x))) for x in raw]);y=z['y'];yp=z['y_palm'];r=z['input_reference']
            data[label]=dict(initial_fixed_reference=extend(y,tau),current_palm=extend(yp,tau),inputs=r)
            manifest.append(dict(path=str(path.with_name('trace.npz').resolve()),archived_sha256=m['trace_sha256']))
        frames={}
        for frame in ('current_palm','initial_fixed_reference'):
            yzero=data['zero'][frame];dense=np.zeros((101,16,8));identified=np.zeros(8,bool)
            for axis in range(8):
                valid=[s for s in ('plus','minus') if meta[f'train_{axis}_{s}']['available']]
                if not valid:continue
                identified[axis]=True
                if len(valid)==2:
                    lp,lm=f'train_{axis}_plus',f'train_{axis}_minus';amplitude=data[lp]['inputs'][-1,axis]-data[lm]['inputs'][-1,axis];dense[:,:,axis]=(data[lp][frame]-data[lm][frame])/amplitude
                else:
                    label=f'train_{axis}_{valid[0]}';amplitude=data[label]['inputs'][-1,axis]-data[label]['inputs'][0,axis];dense[:,:,axis]=(data[label][frame]-yzero)/amplitude
            kernel=dense[idx];errors={str(h):[] for h in plan['forecast_horizons_s']};differential=[];details=[]
            old_history=np.array([x['y_palm' if frame=='current_palm' else 'y'] for x in base['history']]);ht=np.array([x['time'] for x in base['history']]);ht-=ht[-1]
            for q in range(8):
                label=f'holdout_{q:02d}'
                if not meta[label]['available']:details.append(dict(sequence=q,available=False));continue
                measured=data[label][frame][idx];path=data[label]['inputs'][idx];modeled=response(kernel,path);beta=measured-modeled;differential.append(measured-yzero[idx]-modeled)
                # Negative-time phase2 history did not save tau. Do NOT infer it
                # from future zero. Start tau from this holdout's acquired t=0,
                # and use only its subsequently available past residuals.
                history=np.full((len(ht),16),np.nan);history[:,:5]=old_history[:,:5];history[:,10:]=old_history[:,5:];history[-1,5:10]=measured[0,5:10]
                sequence=[]
                for origin in range(len(path)):
                    now=origin*.02;tt=np.r_[ht,np.arange(1,origin+1)*.02];yy=np.vstack((history,beta[1:origin+1]));take=tt>=now-plan['causal_window_s']-1e-10;tt=tt[take];yy=yy[take]
                    slope=np.zeros(16);level=np.zeros(16)
                    for output in range(16):
                        valid=np.isfinite(yy[:,output]);ts=tt[valid];ys=yy[valid,output]
                        if len(ts)>=4:
                            centered=ts-ts.mean();slope[output]=centered@(ys-ys.mean())/max(1e-12,float(centered@centered));level[output]=ys.mean()+slope[output]*(now-ts.mean())
                        else:level[output]=ys[-1]
                    for h in plan['forecast_horizons_s']:
                        target=origin+round(h/.02)
                        if target>=len(path):continue
                        pred=level+h*slope+modeled[target];err=pred-measured[target];errors[str(h)].append(err);sequence.append(dict(origin_s=now,horizon_s=h,error=err))
                details.append(dict(sequence=q,available=True,complete_causal_errors=sequence))
            reference_old=np.array(case['coordinates'][frame]['response_to_20ms_interval_increment']);subset=np.r_[np.arange(5),np.arange(10,16)]
            mismatch=float(np.max(abs(dense[:,subset]-reference_old)))
            frames[frame]=dict(dense_response_times_s=np.arange(101)*.005,dense_response_to_20ms_interval_increment=dense,
                interval_response_times_s=np.arange(26)*.02,kernel_response_to_20ms_interval_increment=kernel,
                identified_inputs=identified,input_availability=case['input_availability'],original_11_output_kernel_max_difference=mismatch,
                differential_response_error=summarize(np.concatenate(differential)),
                complete_causal_by_horizon={h:summarize(e) for h,e in errors.items()},individual_holdout=details)
        cases.append(dict(shape=shape,input_availability=case['input_availability'],input_order=case['input_order'],input_units=case['input_units'],
            measured_input_delta_min=case['measured_input_delta_min'],measured_input_delta_max=case['measured_input_delta_max'],
            coordinate_reference=case['coordinate_reference'],planned_branches=case['planned_branches'],executed_branches=case['executed_branches'],unavailable_branches=case['unavailable_branches'],coordinates=frames))
        print(shape,'contact extension complete',flush=True)
    result=dict(schema='public_contact_response_kernel_16x8_phase2_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        model_admitted=False,derived_offline_no_new_physics=True,calibration_not_performance=True,
        source_script_sha256=sha(__file__),source_public_11_model_path=str(existing),source_public_11_model_sha256=sha(existing),
        source_phase2_plan_sha256=sha(args.root/'plan.json'),raw_archive_manifest=manifest,
        output_order=['normal_thumb','normal_index','normal_middle','normal_ring','normal_little','shear_mag_thumb','shear_mag_index','shear_mag_middle','shear_mag_ring','shear_mag_little','net_Fx','net_Fy','net_Fz','net_Mx','net_My','net_Mz'],
        output_units=['N']*13+['Nm']*3,input_order=base_document['shapes'][0]['input_order'],input_units=base_document['shapes'][0]['input_units'],
        shear_definition='Per digit sum of norms of all external patch forces minus patch normal component, identical to declared aggregate virtual tactile shear_magnitude. Rotation invariant, no target-ID selection.',
        new_shear_engineering_reference=dict(rmse_N=.35,p95_N=.75,status='Exploratory additional-output engineering reference declared after physical data collection, before deriving these shear kernels; not a precollection admission gate.'),
        causal_history_note='Old phase2 negative-time history lacks shear. Complete shear forecasts initialize from this holdout start only, then past residuals; no future zero is used to invent shear history.',
        limitations=['Aggregate shear is not exact per-patch traction or a material-friction identification.',
            'Only existing measured micro-input paths and prepared contexts. Missing directions remain missing.',
            '5ms dense responses are stored, while forecast summaries cover only20/40/60/100ms.',
            'No safety or policy-performance claim and no change to original phase2 model/plan/raw.'],shapes=cases)
    out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':main()
