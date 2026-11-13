"""Known six-trajectory author regression; no additional scientific data."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'review/continuous_diagnostics_v1.py'
spec=importlib.util.spec_from_file_location('frozen_aux_measurements',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'review/continuous_diagnostics_author_regression_v1.json'
    if out.exists():raise FileExistsError(out)
    oldpath=ROOT/'review/v15_six_unique_holding_physical_diagnostics_v1.json';old=json.loads(oldpath.read_text(encoding='utf-8'));checks=[];details=[];source_before=sha(SOURCE)
    def ck(name,v,detail=None):checks.append(dict(name=name,passed=bool(v),detail=detail))
    def common_equal(reference,actual,path=''):
        if isinstance(reference,dict):return isinstance(actual,dict) and all(k in actual and common_equal(v,actual[k],path+'/'+k) for k,v in reference.items())
        if isinstance(reference,list):return isinstance(actual,list) and len(reference)==len(actual) and all(common_equal(x,y,path) for x,y in zip(reference,actual))
        if isinstance(reference,float):return isinstance(actual,(int,float)) and np.float64(reference).tobytes()==np.float64(actual).tobytes()
        return reference==actual
    for row in old['records']:
        batch='v15_passive_blind_bridge_v1' if row['cell']=='legacy_blind' else 'v15_holding_task_ablation_v1'
        path=ROOT/'output'/batch/'runs'/f"nominal__{row['method']}__617901";now=module.read_run(path,record_label=row['cell'])
        tag=row['cell']+'/'+row['method']
        # Only prior fixed-window measures are regression targets. Additive
        # CV/contact/end-state fields are not retroactive old measures.
        for name in module.WINDOWS:
            prior=row['measurements'][name];current=now['measurements'][name]
            ck(tag+'/'+name+'/all_prior_force_fields_IEEE_exact',common_equal(prior['unselected_force'],current['unselected_force']))
            ck(tag+'/'+name+'/all_prior_physical_fields_IEEE_exact',common_equal(prior['physical_object'],current['physical_object']))
            f=current['unselected_force'];mean=f['mean_total_force_N'];ck(tag+'/'+name+'/new_total_CV_definition',f['total_force_CV']==f['total_force_SD_N']/mean)
        for name in ('before_release_0_to12','policy_holding_7p5_to12'):ck(tag+'/'+name+'/prior_floor_diagnostics_IEEE_exact',common_equal(row['floor'][name],now['floor'][name]))
        saved=json.loads((path/'result.json').read_text(encoding='utf-8'));ck(tag+'/original_result_intact',now['original_result']==saved)
        ck(tag+'/no_new_success_or_score_key',not any(k in now for k in ('success','new_success','weighted_score','total_score')))
        details.append(dict(cell=row['cell'],method=row['method'],trace_sha256=now['input_sha256']['trace'],delivery_sha256=now['input_sha256']['delivery'],measurement_status=now['status']))
    # Minimal software boundaries for the NEW CV and fixed-window availability
    # contract only, not a new independent optimization/physics audit.
    t=np.arange(141)*.005+8.2;zero=module.force_window(t,np.zeros((141,5)),8.2,8.9)
    ck('new_CV_zero_mean_is_undefined_not_zero',zero['available'] and zero['total_force_CV'] is None and zero['per_finger_CV']==[None]*5)
    missing=module.force_window(t[:-1],np.zeros((140,5)),8.2,8.9);ck('missing_fixed_tail_unavailable_not_zero',not missing['available'] and missing['metrics'] is None)
    ck('frozen_reader_source_unchanged',sha(SOURCE)==source_before)
    ck('no_controller_model_or_physics_import',not any(k in sys.modules for k in ('mujoco','v15_support.runtime','v15_support.controller')))
    report=dict(schema='continuous_diagnostics_known_six_author_regression_v1',passed=all(c['passed'] for c in checks),checks=checks,records=details,source_sha256=source_before,
        source_reference_json=dict(path=str(oldpath),sha256=sha(oldpath)),checker_sha256=sha(__file__),physics_steps=0,
        scope='Six already-seen unique trajectories only. Exact regression of every old fixed-window force/body field and old floor diagnostics; new CV formula/zero-mean/one-missing-tail software boundary. No new data, score, threshold, scientific inference or independent mathematical audit.')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=len(checks),failures=[x['name'] for x in checks if not x['passed']],source_sha256=source_before,out=str(out))))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
