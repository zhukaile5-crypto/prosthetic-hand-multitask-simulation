"""Record sample-validated common local model domains before policy runs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol_rounded_box_v7.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v8-selected-local-domains'
    result['status']='sample-validated-common-local-domains-before-any-policy-performance-data'
    rows={};rmse_limit=original['local_execution_model']['validated_active_digit_prediction_rmse_max_N']
    p95_limit=original['local_execution_model']['validated_active_digit_prediction_abs_error_p95_max_N']
    for shape in ('capsule','box','sphere'):
        folder=ROOT/f'review/foundation_matrix_rounded_final/{shape}_s0.008_dt0.0005'
        summary=folder/'response.json';raw=folder/'response_raw.npz'
        description=json.loads(summary.read_text(encoding='utf-8'))
        with np.load(raw,allow_pickle=False) as z:
            curves=z['holdout'];deltas=z['holdout_deltas'];initial=z['baseline'][0,1:]
        B=np.asarray(description['force_matrix']);tau=float(description['force_tau'])
        horizons=np.asarray(description['holdout_horizons']);times=curves[0,:,0]
        indices=np.array([int(np.argmin(abs(times-h))) for h in horizons])
        predicted=initial[None,None,:]+(deltas@B.T)[:,None,:]*(1.-np.exp(-horizons[None,:,None]/tau))
        errors=curves[:,indices,1:]-predicted
        candidates=[]
        for i,radius in enumerate((.02,.04)):
            error=errors[8*i:8*(i+1)]
            per_rmse=np.sqrt(np.mean(error**2,axis=(0,1)))
            per_p95=np.quantile(abs(error),.95,axis=(0,1))
            pooled_rmse=float(np.sqrt(np.mean(error**2)));pooled_p95=float(np.quantile(abs(error),.95))
            passed=bool(pooled_rmse<=rmse_limit and pooled_p95<=p95_limit and np.max(per_rmse)<=rmse_limit and np.max(per_p95)<=p95_limit)
            candidates.append(dict(radius=radius,passed=passed,pooled_rmse_N=pooled_rmse,pooled_p95_N=pooled_p95,
                per_digit_rmse_N=per_rmse.tolist(),per_digit_p95_N=per_p95.tolist(),
                p95_error_proxy_per_digit_N=np.maximum(.05,per_p95).tolist()))
        acceptable=[candidate for candidate in candidates if candidate['passed']]
        if not acceptable:raise ValueError(f'No independently passing local model domain for {shape}')
        selected=max(acceptable,key=lambda candidate:candidate['radius'])
        rows[shape]={'selected_radius':selected['radius'],'selected':selected,'candidates':candidates,
                     'response_raw':str(raw.relative_to(ROOT)),'response_raw_sha256':sha(raw),
                     'response_summary':str(summary.relative_to(ROOT)),'response_summary_sha256':sha(summary)}
    result['local_execution_model']['selected_local_domains']=rows
    result['local_execution_model']['domain_selection_rule']='Largest of the tested 0.02/0.04 normalized-closure radii passing the unchanged 0.35 N RMSE and 0.75 N absolute p95 criteria, both pooled and per digit, using independently recomputed 20/40/60 ms prediction errors. Same domain for both methods.'
    result['local_execution_model']['domain_interpretation']='Finite prespecified multi-finger physical perturbation patterns establish an engineering sample-validated region, not a mathematical worst-case bound for every point in the cube. Error proxies retain their own selected-radius provenance.'
    result['local_execution_model']['failed_larger_domain_retained']=True
    result['trust_domain_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'physical_matrix_executed_under_parent':True,
        'physical_matrix_basis_protocol_sha256':sha(parent),
        'physical_gates_and_model_design_unchanged':True,
        'calibration_chain':'The exported calibration may bind this active protocol while retaining foundation_protocol_sha256=v7. This protocol binds immutable response raw/summary files, not the aggregate calibration envelope, avoiding a circular hash.',
        'no_policy_performance_data_used':True}
    assert result['physical_foundation']==original['physical_foundation']
    out=ROOT/'acceptance_protocol_local_domains_v8.json'
    if out.exists():raise SystemExit('Refusing to overwrite selected-domain evidence.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'selected_radii':{k:v['selected_radius'] for k,v in rows.items()},'physical_gates_unchanged':True},ensure_ascii=False))
if __name__=='__main__':main()
