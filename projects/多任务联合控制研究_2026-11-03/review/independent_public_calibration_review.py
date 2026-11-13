"""Read-only independent phase1 raw/protocol audit; never imports a plant."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,io
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'calibration/phase1_synergy_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def near(a,b,tol=1e-9):return bool(np.allclose(a,b,atol=tol,rtol=0,equal_nan=True))


def aggregate(patches,R,p,R0,p0):
    n=np.zeros(5);w=np.zeros(6);normal=np.zeros((6,5));tangent=np.zeros(6);aux=np.zeros(6)
    world_force=np.zeros(3);world_moment=np.zeros(3)
    for patch in patches:
        pressure=float(patch['normal'])
        if pressure<=1e-10:continue
        f=np.asarray(patch['force']);x=np.asarray(patch['position']);torque=np.asarray(patch['torque'])
        v=np.r_[f,np.cross(x,f)+torque];w+=v
        force_world=R@f;point_world=p+R@x
        world_force+=force_world;world_moment+=np.cross(point_world-p0,force_world)+R@torque
        digit=int(patch['digit'])
        if 0<=digit<5:
            n[digit]+=pressure;direction=np.asarray(patch['normal_dir']);direction/=np.linalg.norm(direction)
            fn=pressure*direction;wn=np.r_[fn,np.cross(x,fn)]
            normal[:,digit]+=wn;tangent+=v-wn
        else:aux+=v
    wr=np.r_[R0.T@world_force,R0.T@world_moment]
    def transform(v):return np.r_[R0.T@(R@v[:3]),R0.T@(R@v[3:]+np.cross(p-p0,R@v[:3]))]
    norm_ref=transform(normal.sum(axis=1));tan_ref=transform(tangent);aux_ref=transform(aux)
    W=np.zeros((6,5))
    for i in range(5):
        if n[i]>1e-10:W[:,i]=transform(normal[:,i])/n[i]
    return n,w,wr,norm_ref,tan_ref,aux_ref,W


def group_error(values):
    x=np.asarray(values)
    return dict(pressure_RMSE_N=float(np.sqrt(np.mean(x[:,:,:5]**2))),
        force_RMSE_N=float(np.sqrt(np.mean(x[:,:,5:8]**2))),
        moment_RMSE_Nm=float(np.sqrt(np.mean(x[:,:,8:11]**2))),
        maximum_pressure_error_N=float(np.max(abs(x[:,:,:5]))),
        maximum_force_error_N=float(np.max(abs(x[:,:,5:8]))),maximum_moment_error_Nm=float(np.max(abs(x[:,:,8:11]))))


def main():
    plan=read(FOLDER/'plan.json');summary=read(FOLDER/'summary.json')
    if not summary.get('complete'):raise RuntimeError('Wait for the unique calibration run to complete')
    checks=dict(summary_complete=summary['complete'],exact_three_shapes=[r['shape'] for r in summary['shapes']]==plan['shapes'],
        plan_copy_identical=sha(FOLDER/'plan.json')==sha(ROOT/'calibration/plan_phase1_v1.json'))
    inputs={str((FOLDER/'plan.json').relative_to(ROOT)):sha(FOLDER/'plan.json'),str((FOLDER/'summary.json').relative_to(ROOT)):sha(FOLDER/'summary.json')}
    for name,expected in plan['source_sha256'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        checks['source/'+name]=path.is_file() and sha(path)==expected
    cases=[];planned_count=actual_count=packet_count=0
    for shape in plan['shapes']:
        case=FOLDER/shape;base=read(case/'base_metadata.json');model=read(case/'kernel_model.json');rows={};metas={};local={}
        local['model_summary_hash']=sha(case/'kernel_model.json')==next(r['model_sha256'] for r in summary['shapes'] if r['shape']==shape)
        inputs[str((case/'base_metadata.json').relative_to(ROOT))]=sha(case/'base_metadata.json');inputs[str((case/'kernel_model.json').relative_to(ROOT))]=sha(case/'kernel_model.json')
        R0=np.asarray(base['reference_rotation']);p0=np.asarray(base['reference_origin']);zero_state=None;max_packet_error=0.
        unavailable=[];limits=plan['stop_gates'];max_pressure=0.;max_penetration=0.
        expected_labels=['zero','zero_replay']+[f'{split}_{axis}_{sign}' for split in ('train','hold') for axis in range(5) for sign in ('plus','minus')]
        local['exact_branch_inventory']={x.name for x in case.iterdir() if x.is_dir()}==set(expected_labels)
        for label in expected_labels:
            planned_count+=1;directory=case/label;m=read(directory/'metadata.json');metas[label]=m
            inputs[str((directory/'metadata.json').relative_to(ROOT))]=sha(directory/'metadata.json')
            if not m['available']:
                unavailable.append(label);local[label+'/unavailable_not_faked_zero']=not m['complete'] and not m['attempted_physics'] and not(directory/'trace.npz').exists()
                continue
            actual_count+=1;path=directory/'trace.npz';blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
            with np.load(io.BytesIO(blob),allow_pickle=False) as a:z={k:a[k] for k in a.files}
            del blob;inputs[str(path.relative_to(ROOT))]=digest
            local[label+'/trace_hash']=digest==m['trace_sha256']
            t0=m['t0'];local[label+'/clock_coverage']=len(z['time'])==101 and len(z['physical_time'])==1000 and near(z['time'],t0+np.arange(101)*.005,1e-8) and near(z['physical_interval_start'],t0+np.arange(1000)*.0005,1e-8) and near(z['physical_time'],t0+(np.arange(1000)+1)*.0005,1e-8) and near(z['physical_duration'],.0005)
            local[label+'/actual_requested_delta']=near(m['executed_delta'],m['requested_delta'],1e-12) and near(z['alpha'][-1]-z['alpha'][0],m['executed_delta'],1e-12)
            local[label+'/actual_4tick_ramp']=all(near(z['alpha'][i]-z['alpha'][0],np.asarray(m['requested_delta'])*min(1.,i/4.),1e-12) for i in range(101))
            local[label+'/coordinate_domain']=bool(np.all(z['physical_alpha']>=np.asarray(m['lower'])-1e-10) and np.all(z['physical_alpha']<=np.asarray(m['upper'])+1e-10))
            start={k:z[k][0] for k in ('qpos','qvel','ctrl','alpha','pressure','y','palm_rotation','palm_position')}
            if zero_state is None:zero_state=start
            local[label+'/identical_restored_start']=all(np.array_equal(start[k],zero_state[k]) for k in start)
            packet_ok=True;gate_ok=True
            for prefix in ('','physical_'):
                n=z[prefix+'pressure'];w=z[prefix+'wrench_reference']
                gate_ok &= bool(np.isfinite(z[prefix+'y']).all() and (n<=limits['per_finger_pressure_N']).all() and (n.sum(axis=1)<=limits['sum_finger_pressure_N']).all() and
                    (z[prefix+'penetration']<=limits['penetration_m']).all() and (z[prefix+'table_support_N']<=limits['table_support_N']).all() and
                    (z[prefix+'object_lift_m']>=limits['minimum_object_lift_m']).all() and ((n>.2).sum(axis=1)>=3).all() and (n[:,0]>.2).all())
                max_pressure=max(max_pressure,float(n.max()));max_penetration=max(max_penetration,float(z[prefix+'penetration'].max()))
                for i,raw in enumerate(z[prefix+'raw_patches']):
                    values=aggregate(json.loads(str(raw)),z[prefix+'palm_rotation'][i],z[prefix+'palm_position'][i],R0,p0)
                    expected=(z[prefix+'pressure'][i],z[prefix+'wrench_palm'][i],z[prefix+'wrench_reference'][i],z[prefix+'normal_reference'][i],z[prefix+'tangent_reference'][i],z[prefix+'auxiliary_reference'][i])
                    err=max(float(np.max(abs(a-b))) for a,b in zip(values[:6],expected));max_packet_error=max(max_packet_error,err);packet_count+=1
                    packet_ok &= err<1e-9
                packet_ok &= near(z[prefix+'y'],np.c_[n,w],1e-9)
            local[label+'/all_patch_and_fixed_origin_arithmetic']=packet_ok
            local[label+'/all_raw_and_control_stop_gates']=gate_ok and m['complete'] and m['stop_reason'] is None
            rows[label]={k:v for k,v in z.items() if not k.endswith('raw_patches')}
            print(shape,label,'reviewed',flush=True)
        local['zero_replay_all_saved_physical_fields_bitwise']=all(np.array_equal(rows['zero'][k],rows['zero_replay'][k]) for k in rows['zero'] if k.startswith('physical_'))
        # Refit using actual executed deltas, keeping amplitude holdout out of J.
        horizons=np.asarray(plan['horizons_s']);idx=np.round(horizons/.005).astype(int);zero_y=rows['zero']['y'][idx]
        J=np.zeros((len(idx),11,5));available=[]
        for axis in range(5):
            plus=f'train_{axis}_plus';minus=f'train_{axis}_minus';hasplus=metas[plus]['available'];hasminus=metas[minus]['available']
            if hasplus and hasminus:
                denominator=metas[plus]['executed_delta'][axis]-metas[minus]['executed_delta'][axis]
                J[:,:,axis]=(rows[plus]['y'][idx]-rows[minus]['y'][idx])/denominator
            elif hasplus or hasminus:
                chosen=plus if hasplus else minus;J[:,:,axis]=(rows[chosen]['y'][idx]-zero_y)/metas[chosen]['executed_delta'][axis]
            available.append(hasplus or hasminus)
        local['independent_actual_delta_refit']=near(J,np.asarray(model['jacobian_11x5'],float),1e-7)
        ranks=[int(np.linalg.matrix_rank(v[:,available])) for v in J]
        local['identified_rank_recomputed']=ranks==model['identified_input_rank_by_horizon']
        local['motion_not_zero_filled_or_claimed']=model['full_11x8_identified'] is False and model['missing_motion_columns']==['lift_m','wrist1_rad','wrist2_rad']
        local['output_units']=model['units_outputs']==['N']*8+['Nm']*3
        history=base['history'];ht=np.asarray([r['time'] for r in history]);hy=np.asarray([r['y'] for r in history]);t0=metas['zero']['t0']
        past=ht>=t0-.25-1e-10;hx=ht[past]-ht[past].mean();slope=hx@(hy[past]-hy[past].mean(axis=0))/(hx@hx)
        local['only_past_baseline_samples']=bool(ht.max()<=t0+1e-8)
        local['causal_slope_recomputed']=near(slope,model['causal_history_slope'],1e-8)
        baseline=rows['zero']['y'][0]+horizons[:,None]*slope
        response_errors=[];causal_errors=[];per_horizon=[]
        for val in model['holdout']:
            label=f"hold_{val['axis']}_{'plus' if val['sign']>0 else 'minus'}"
            if not val['available']:
                local[label+'/unavailable_in_model_matches_raw']=not metas[label]['available'];continue
            actual=rows[label]['y'][idx];pred=np.einsum('hoi,i->ho',J,np.asarray(metas[label]['executed_delta']))
            er=actual-zero_y-pred;ec=actual-baseline-pred
            local[label+'/separate_holdout_errors_recomputed']=near(er,val['response_difference_error'],1e-8) and near(ec,val['complete_causal_prediction_error'],1e-8)
            response_errors.append(er);causal_errors.append(ec)
        for j,h in enumerate(horizons):per_horizon.append(dict(horizon_s=float(h),response_difference=group_error(np.asarray(response_errors)[:,j:j+1,:]),complete_causal=group_error(np.asarray(causal_errors)[:,j:j+1,:])))
        # Quantify both the empirical normal-only map omission and the old WB
        # model, with its four execution-tick response aligned to each horizon.
        first=rows['zero'];raw=read(case/'zero/metadata.json')
        with np.load(case/'zero/trace.npz',allow_pickle=False) as z:patches=json.loads(str(z['raw_patches'][0]))
        W0=aggregate(patches,first['palm_rotation'][0],first['palm_position'][0],R0,p0)[6]
        B=np.asarray(base['calibration']['force_matrix']);tau=base['calibration']['force_tau'];a=np.exp(-.005/tau)
        mismatch=[];S=np.r_[np.ones(3),np.full(3,20.)]
        for j,h in enumerate(horizons):
            latent=0.
            for tick in range(1,int(round(h/.005))+1):latent=a*latent+(1.-a)*min(1.,tick/4.)
            empirical=J[j,5:];normal_only=W0@J[j,:5];old_WB=W0@B*latent
            def stats(approx):
                delta=empirical-approx
                return dict(force_matrix_error_N_per_alpha=float(np.linalg.norm(delta[:3])),moment_matrix_error_Nm_per_alpha=float(np.linalg.norm(delta[3:])),
                    scaled_relative_error=float(np.linalg.norm(S[:,None]*delta)/max(.1,float(np.linalg.norm(S[:,None]*empirical)))))
            scaled=S[:,None]*empirical;singular=np.linalg.svd(scaled,compute_uv=False)
            mismatch.append(dict(horizon_s=float(h),four_tick_latent_gain=float(latent),
                empirical_normal_only_omission=stats(normal_only),old_normal_WB_aligned=stats(old_WB),
                full_wrench_scaled_singular_values=singular.tolist()))
        checks.update({shape+'/'+k:bool(v) for k,v in local.items()})
        cases.append(dict(shape=shape,checks=local,planned_branches=22,executed_branches=22-len(unavailable),unavailable=unavailable,
            maximum_actual_pressure_N=max_pressure,maximum_actual_penetration_m=max_penetration,max_packet_reconstruction_error=max_packet_error,
            identified_rank=ranks,holdout_direction_count=len(causal_errors),holdout_per_horizon=per_horizon,
            normal_WB_comparisons=mismatch,causal_history_latest_time=float(ht.max()),clone_time=float(t0)))
    all_checks=all(checks.values())
    result=dict(raw_design_audit_passed=all_checks,checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),planned_branches=planned_count,executed_branches=actual_count,packets_recomputed=packet_count,
        cases=cases,input_sha256=inputs,reviewer_sha256=sha(__file__),
        model_admitted=False,full_11x8_identified=False,
        admission_missing=['Three motion input columns remain unmeasured.',
            'Holdout is only same-clone single-axis amplitude extrapolation, not unseen physical seeds or mixed-input trajectories.',
            'No predeclared numerical per-output/per-horizon predictive-error admission thresholds or downstream robust-controller contract exists.',
            'Rank alone is not sufficient identification quality or control authority; signs, singular scales, geometry/contact changes and error must remain explicit.',
            'Physical pressure/contact/penetration engineering stop gates passed; this is not a hardware safety certificate. MuJoCo warning counters are not recorded in this probe packet.'],
        scope='Independent saved-data audit only. No hand/solver/strategy imported or run, no source/trace modification, no old preservation rehash.')
    out=ROOT/('review/independent_public_calibration_phase1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(raw_design_audit_passed=all_checks,checks=len(checks),failed=result['failed'],planned=planned_count,executed=actual_count,packets=packet_count,model_admitted=False,path=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':main()
