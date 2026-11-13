"""Refit public local models and remeasure holdout errors from saved physics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,ast
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    protocol_path=ROOT/'acceptance_protocol_local_domains_v8.json'
    calibration_path=ROOT/'review/foundation_calibration_final.json'
    protocol=read(protocol_path);calibration=read(calibration_path)
    checks={'active_protocol_chain':calibration['protocol_sha256']==sha(protocol_path),
            'raw_protocol_lineage':calibration['foundation_protocol_sha256']==protocol['trust_domain_revision']['parent_sha256']}
    rows={}
    for shape,declared in protocol['local_execution_model']['selected_local_domains'].items():
        raw=ROOT/declared['response_raw'];summary=ROOT/declared['response_summary'];r=read(summary)
        with np.load(raw,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        curves=a['curves'];base=a['baseline'];delta=a['fit_deltas'];times=curves[0,:,0]
        selected=times<=.080+1e-9;Y=(curves[:,:,1:]-base[None,:,1:])[:,selected,:].reshape(-1,5)
        fits=[]
        for tau in np.asarray(r['fitted_time_constants']):
            X=(delta[:,None,:]*(1.-np.exp(-times[selected]/tau))[None,:,None]).reshape(-1,5)
            B=np.linalg.lstsq(X,Y,rcond=1e-10)[0].T
            fits.append((float(np.mean((X@B.T-Y)**2)),float(tau),B))
        mse,tau,B=min(fits,key=lambda item:item[0])
        checks[shape+'_fit_recomputed']=abs(tau-r['force_tau'])<1e-12 and np.allclose(B,r['force_matrix'],atol=1e-10,rtol=0)
        hold=a['holdout'];hdelta=a['holdout_deltas'];horizons=np.asarray(r['holdout_horizons'])
        indices=np.array([int(np.argmin(abs(hold[0,:,0]-h))) for h in horizons])
        pred=base[0,1:][None,None,:]+(hdelta@B.T)[:,None,:]*(1.-np.exp(-horizons[None,:,None]/tau))
        errors=hold[:,indices,1:]-pred
        checks[shape+'_stored_holdout_error_correct']=np.allclose(errors,a['holdout_errors'],atol=1e-10,rtol=0)
        results=[]
        for i,radius in enumerate((.02,.04)):
            error=errors[8*i:8*(i+1)];rmse=np.sqrt(np.mean(error**2,axis=(0,1)));p95=np.quantile(abs(error),.95,axis=(0,1))
            results.append({'radius':radius,'rmse_per_digit_N':rmse.tolist(),'p95_per_digit_N':p95.tolist(),
                'pooled_rmse_N':float(np.sqrt(np.mean(error**2))),'pooled_p95_N':float(np.quantile(abs(error),.95)),
                'proxy':np.maximum(.05,p95).tolist(),
                'passed':bool(np.max(rmse)<=.35 and np.max(p95)<=.75)})
        chosen=max((x for x in results if x['passed']),key=lambda item:item['radius'])
        c=calibration['shapes'][shape];c={**c,**c.get('response',{})}
        checks[shape+'_selected_radius_correct']=chosen['radius']==c['trust_radius']==declared['selected_radius']
        checks[shape+'_selected_error_proxy']=np.allclose(c['response_error_bound'],chosen['proxy'],atol=1e-10,rtol=0)
        checks[shape+'_calibration_matrix_and_tau']=np.allclose(c['force_matrix'],B,atol=1e-10,rtol=0) and abs(c['force_tau']-tau)<1e-12
        checks[shape+'_source_raw_summary_identity']=sha(raw)==declared['response_raw_sha256']==c['selected_holdout']['source_sha256'] and sha(summary)==declared['response_summary_sha256']==c['selected_holdout']['summary_sha256']
        sources=c['source_sha256']
        checks[shape+'_current_physical_source_identity']=all(sha(ROOT/name)==sources[name] for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py'))
        probe=summary.parent/'raw_probe.npz'
        with np.load(probe,allow_pickle=False) as z:physical=json.loads(str(z['metadata']))
        profile=np.asarray(physical['config']['holding_force'],float);profile/=np.mean(profile)
        checks[shape+'_public_holding_profile']=np.allclose(c['nominal_force_weights'],profile,atol=1e-12,rtol=0)
        rows[shape]={'model_fit_mse':mse,'effective_tau_s':tau,'candidate_domains':results,'selected_radius':chosen['radius'],
                     'raw_sha256':sha(raw),'response_summary_sha256':sha(summary)}
    bare=Path(calibration['bare_hand_source_path'])
    checks['bare_hand_source_exists_and_bound']=bare.is_file() and sha(bare)==calibration['bare_hand_source_sha256']
    # Read-only source checks; the public returned frame must not expose hidden
    # object/environment state. Diagnostic target grouping stays outside it.
    tree=ast.parse((ROOT/'contracts.py').read_text(encoding='utf-8'))
    sensor_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SensorFrame')
    fields=[n.target.id for n in sensor_cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
    forbidden={'object_mass','true_mass','friction','true_friction','object_position','object_pose','future_disturbance','stable_marker'}
    checks['no_forbidden_object_oracle_in_sensor_frame']=not(forbidden&set(fields))
    # Target grouping is permitted in evaluation diagnostics. Flip every target
    # label and verify every public measurement/geometry input stays identical.
    import sys,copy
    sys.path.insert(0,str(ROOT))
    from sensors import aggregate_patches
    patches=[dict(digit=0,normal=2.,force=[0.,0.,2.],torque=[0.,0.,0.],
                  position=[.03,0.,0.],normal_dir=[0.,0.,1.],slip_velocity=[0.,0.,0.],is_target=True),
             dict(digit=-1,normal=1.,force=[0.,0.,1.],torque=[0.,0.,0.],
                  position=[0.,0.,0.],normal_dir=[0.,0.,1.],slip_velocity=[0.,0.,0.],is_target=True)]
    changed=copy.deepcopy(patches)
    for patch in changed:patch['is_target']=False
    a,da=aggregate_patches(patches);b,db=aggregate_patches(changed)
    public=('raw_auxiliary_wrench','auxiliary_pressure','normal_wrench_map','finger_non_normal_by_finger',
            'shear_magnitude','slip_magnitude','normal_support_coeff','tangent_support_coeff','contact_present',
            'distributed_valid','distributed_error')
    checks['sensor_aggregation_no_target_identity_filter']=all(np.array_equal(a[k],b[k],equal_nan=True) for k in a) and all(np.array_equal(np.asarray(da[k]),np.asarray(db[k]),equal_nan=True) for k in public)
    checks['target_identity_only_changes_evaluation_ledger']=da['target_normal_for_evaluation']!=db['target_normal_for_evaluation']
    result={'schema':'independent-physical-calibration-v1','checked_utc':datetime.now(timezone.utc).isoformat(),
            'benchmark_id':'shadow_hand_strategy_integration_v1','passed':all(checks.values()),'checks':checks,
            'protocol_sha256':sha(protocol_path),'calibration_sha256':sha(calibration_path),
            'source_sha256':{name:sha(ROOT/name) for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py','contracts.py','sensors.py')},
            'models':rows,'limits':'No dynamics or policy was run. Finite physical perturbation validation is not a worst-case force or safety guarantee.'}
    output=ROOT/'review/independent_calibration_final_v2.json'
    if output.exists():raise FileExistsError('Refusing to overwrite independent calibration evidence.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'failed':[k for k,v in checks.items() if not v],'checks':len(checks),'output':str(output)},ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
