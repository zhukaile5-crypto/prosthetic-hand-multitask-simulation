"""Read-only first-failure mechanism timeline; no policy/physics execution."""
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def load(folder,method):
    d=ROOT/'output'/folder/'runs'/f'nominal__{method}__617901'
    fields=('time','normal','skin_normal','support','all_hand_object','palm','position','relative','relative_velocity','angular_velocity','lift','lift_ref','qpos','qvel','ctrl','ctrl_before','requested_reference21','executed_reference21','reference_increment21','observed_normal','validity','observed_age','estimated_load','estimated_load_available','observer_uncertainty','observed_contact','estimated_normal','posterior_mean_time_s','tactile_acquisition_time')
    with np.load(d/'trace.npz',allow_pickle=False) as z:
        a={k:z[k] for k in fields};raw=z['observation_json'];t=a['time'];indices=np.flatnonzero((t>=7.3-1e-9)&(t<=8.4+1e-9));obs={int(i):json.loads(str(raw[i])) for i in indices}
    return d,a,obs,read(d/'metadata.json'),read(d/'result.json')
def first(t,mask):return None if not mask.any() else float(t[np.flatnonzero(mask)[0]])
def first_difference(t,a,b):
    axes=tuple(range(1,a.ndim));same=a==b
    if axes:same=same.all(axis=axes)
    return first(t,~same)
def cost_parts(e):
    p=e['qp_problem'];q=e['qp'];d=np.asarray(q['d21']);z=np.asarray(q['zP']);P=len(z);A=np.asarray(p['A21x25']);J=np.asarray(p['Jpx25']).reshape(P,25);y=np.asarray(p['y_p']);sc=p['scales']
    support=float(np.asarray(p['support_coeff_p'])@z);load=p['load_proxy_N'];short=max(0.,load-support)
    parts=dict(effort_intent_residual=float(np.square(np.asarray(sc['residual25'])*(A.T@d-J.T@(z-y))).sum()),base_force_economy=float(np.square(np.asarray(sc['intentP'])*(z-np.asarray(p['z_intent_p']))).sum()),effort_regularization=float(np.square(np.asarray(sc['delta21'])*d).sum()),load_shortfall=float((sc['load']*short)**2))
    E=p['extra_task_rows'];labels=e['task_labels'];rowcost={}
    if E is not None:
        residual=np.asarray(E['matrix'])@np.r_[d,z]-np.asarray(E['target'])
        for label,value in zip(labels,residual**2):
            group='conditional_continuity' if label.startswith('conditional_digit_force') else 'conditional_balance' if label.startswith('conditional_') else 'hand_manifold' if label.startswith('grasp_shape') else 'model_effort_history' if label.startswith('native_effort_history') else 'other'
            rowcost[group]=rowcost.get(group,0.)+float(value)
    parts.update(rowcost)
    return dict(cost=parts,sum_cost=sum(parts.values()),reported_objective=q['objective'],load_demand_N=load,intent_support_N=support,soft_shortfall_N=short,
        observed_patch_normal_N=y.tolist(),z_intent_P=z.tolist(),base_prior_P=p['z_intent_p'],Sf_P=sc['intentP'],support_coeff_P=p['support_coeff_p'],patch_digits=p['patch_digit_p'],
        normal_caps5=p['caps5'],total_budget=p['total_budget'],z_digit_total=np.bincount(p['patch_digit_p'],weights=z,minlength=5).tolist(),
        conditional_active=e['task_activation']['conditional_load'],intent_group_debug=e['force_intent_debug'],task_labels=labels,
        qp_status=q['status'],proposal_acceptance=e['proposal_acceptance'],native_status=None if e['native'] is None else e['native']['status'],native_summary=None if e['native'] is None else dict(scaled_realization_linf=e['native']['diagnostics'].get('scaled_realization_linf'),reason=e['native']['reason']),
        accepted_effort_delta21=q['d21'],p0_effort21=p['p0_21'],native_tasks=e['native_tasks'])
def main():
    out=ROOT/'review/v15_holding_failure_timeline_v1.json'
    if out.exists():raise FileExistsError(out)
    folder='v15_holding_task_ablation_v1';cases={};summary={};timeline=[];checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    for m in ('fusion_effort','ratio_effort','fixed_effort'):
        d,a,o,meta,result=load(folder,m);cases[m]=(a,o,meta,result);t=a['time'];hold=(t>=7.5)&(t<12.)
        contacts=a['normal']>.2;support=contacts[:,0]&(contacts.sum(axis=1)>=3);raised=a['position'][:,2]>=np.asarray(meta['initial_position'])[2]+.025;drift=np.linalg.norm(a['relative']-meta['reference_relative'],axis=1)
        summary[m]=dict(first_missing_thumb_or_three=first(t,hold&~support),first_below_lift_height=first(t,hold&~raised),first_retention_drift_above80mm=first(t,hold&(drift>.08)),
            physical_failure=result['research_metrics']['failure_reason'],failure_at=result['research_metrics']['failure_at'],metadata_failure=meta['failure_reason'],task_success=result['task_success'],
            static_quality=result['research_metrics']['static_quality'],dynamic_quality=result['research_metrics']['dynamic_quality'],release=result['research_metrics']['release'],safety=result['research_metrics']['safety'])
        for tt in (7.5,7.6,7.7,7.8,7.9,8.,8.04,8.06,8.08,8.085,8.1,8.12,8.16,8.18,8.185,8.2,8.24,8.3,8.4,8.5,8.9,9.3,11.7,12.):
            i=int(np.argmin(abs(t-tt)));obs=o.get(i,{})
            timeline.append(dict(method=m,time_s=float(t[i]),target_normal_N=a['normal'][i],skin_all_external_N=a['skin_normal'][i],observed_normal_N=a['observed_normal'][i],valid5=a['validity'][i],
                pressure_age_s=float(a['observed_age'][i]),estimated_load_N=float(a['estimated_load'][i]),load_available=bool(a['estimated_load_available'][i]),
                object_position_m=a['position'][i],relative_displacement_m=drift[i],object_support_N=float(a['support'][i]),raised=bool(raised[i]),thumb_plus_three=bool(support[i]),
                actual_robot_lift_m=float(a['lift'][i]),target_lift_m=float(a['lift_ref'][i]),requested_ref21=a['requested_reference21'][i],actual_ref21=a['executed_reference21'][i],
                raw_observed_shear=obs.get('shear_magnitude'),raw_observed_slip=obs.get('slip_magnitude'),aux_wrench=obs.get('auxiliary_wrench'),load_uncertainty=obs.get('load_uncertainty')))
        # Only early relevant decisions, not every5ms duplicated log snapshot.
        decisions=[]
        with gzip.open(d/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                e=json.loads(line)
                if e['time_s']>8.4+1e-9:break
                if e['time_s']<7.5-1e-9:continue
                decisions.append(dict(time_s=e['time_s'],**cost_parts(e)))
        summary[m]['early_decisions']=decisions
        ck(m+'/early_cost_expansion',all(abs(x['sum_cost']-x['reported_objective'])<=1e-10*max(1.,abs(x['reported_objective'])) for x in decisions))
        summary[m]['metadata_sha256']=sha(d/'metadata.json');summary[m]['trace_sha256']=sha(d/'trace.npz')
    # Fresh blind is exact originalV14 baseline and its own sensor RNG sequence.
    bd,b,bo,bmeta,bresult=load('v15_passive_blind_bridge_v1','fusion_effort');a=cases['fusion_effort'][0];t=a['time']
    for k in ('requested_reference21','executed_reference21','ctrl','ctrl_before','qpos','qvel','normal','skin_normal','position','support','observed_normal','estimated_normal','estimated_load'):
        summary['first_difference_from_blind_fusion/'+k]=first_difference(t,a[k],b[k])
    ck('common_pre7.5_actual_physics_exact',all(np.array_equal(a[k][t<7.5],b[k][t<7.5]) for k in ('qpos','qvel','normal','skin_normal','ctrl','ctrl_before')))
    ck('7.5_preaction_physics_exact',all(np.array_equal(a[k][1500],b[k][1500]) for k in ('qpos','qvel','normal','skin_normal','ctrl_before','observed_normal')))
    summary['baseline_fusion_original_result']=bresult
    report=dict(schema='v15_holding_failed_case_readonly_mechanism_v1',passed=all(x['passed'] for x in checks),checks=checks,summary=summary,timeline=plain(timeline),
        qualifier='Observed sequence and same-run objective decomposition only. z is intention, not predicted/actual pressure. No factor within the new holding task package has been causally ablated; no parameter/source changes or new physics.',physics_steps=0,checker_sha256=sha(__file__))
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(out),passed=report['passed'],checks=checks,first_divergences={k:v for k,v in summary.items() if k.startswith('first_difference')},first_failure=summary['fusion_effort']['failure_at']),ensure_ascii=False))
    print('fusion truth timeline',json.dumps([x for x in plain(timeline) if x['method']=='fusion_effort' and x['time_s']<=8.3],ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
