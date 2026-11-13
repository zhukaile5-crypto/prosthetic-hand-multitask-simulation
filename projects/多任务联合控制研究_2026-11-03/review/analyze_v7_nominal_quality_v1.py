"""Read-only, saved-public-stream algebra. No plant, controls, or physics step.

Only fresh review outputs are written. Inputs and declared execution sources
are hashed before and after. QP diagnostics reconstruct selected recorded
decisions; this is not a physical counterfactual or a new performance run.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v7.controller import create_joint
import v7.feedback as feedback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return serial(v.tolist())
    if isinstance(v,np.generic):return serial(v.item())
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def run():
    base=ROOT/'output/development_v7_counterexamples_v1/runs'
    keys=('time','normal','skin_normal','alpha_before','alpha','estimated_normal','posterior_mean_time_s',
          'observer_uncertainty','prediction_at','prediction_available','predicted_normal','debug_json',
          'observed_normal','observation_json','task_request_json','executed_motion_reference')
    inputs=[];runs={};summary={}
    for method in ('ratio','passive','joint'):
        folder=base/f'nominal__{method}__610212'
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        if not result['record_complete']:raise ValueError('Only completed runs are diagnosed')
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        with np.load(folder/'trace.npz',allow_pickle=False) as archive:
            data={k:archive[k] for k in keys} # each compressed column read exactly once
        inputs.extend(folder/name for name in ('trace.npz','result.json','metadata.json'))
        t=data['time'];static=(t>=8.2-1e-9)&(t<=8.9+1e-9)
        dec=(t>=7.5-1e-9)&(t<12.-1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7)
        js={int(i):json.loads(str(data['debug_json'][i])) for i in np.flatnonzero(dec)}
        alpha=data['alpha_before'][static];raw=data['skin_normal'][static]
        summary[method]=dict(research_quality={k:result['research_metrics'][k] for k in ('static_quality','dynamic_quality','entire_holding_quality')},
            alpha_range=np.ptp(alpha,axis=0),alpha_TV=abs(np.diff(alpha,axis=0)).sum(axis=0),
            raw_force_TV_per_finger=abs(np.diff(raw,axis=0)).sum(axis=0),raw_force_net_per_finger=abs(raw[-1]-raw[0]),
            posterior_current_RMSE=np.sqrt(np.mean((data['estimated_normal'][static]-raw)**2,axis=0)),
            posterior_current_max_abs_error=abs(data['estimated_normal'][static]-raw).max(axis=0))
        runs[method]=(data,meta,js)
    frozen=[ROOT/'v7'/n for n in ('belief.py','controller.py','feedback.py')]
    frozen+=[ROOT/'calibration/public_contact_model_phase2_v1.json']
    before={str(p):sha(p) for p in inputs+frozen}
    d,meta,js=runs['joint'];t=d['time'];static=(t>=8.2-1e-9)&(t<=8.9+1e-9)
    jlist=list(js.values());sd=[j for i,j in js.items() if static[i]]
    summary['joint'].update(decisions=len(js),model_available=sum(j['model_usable'] for j in jlist),
        static_decisions=len(sd),static_model_available=sum(j['model_usable'] for j in sd),
        unknown_reasons=Counter(j['prediction_invalid_reason'] for j in jlist if not j['model_usable']),
        stage_status={s:Counter(next(x['status'] for x in j['stage_results'] if x['stage']==s) for j in jlist)
            for s in [x['stage'] for x in jlist[0]['stage_results']]},
        pressure_slack_max=max(max(j['material_slack_N']) for j in jlist),budget_slack_max=max(j['budget_slack_N'] for j in jlist),
        contact_slack_max=max(max(j['contact_slack_N']) for j in jlist),
        request_range=[min(j['requested_total_force_N'] for j in jlist),max(j['requested_total_force_N'] for j in jlist)],
        request_max_jump=max(abs(np.diff([j['requested_total_force_N'] for j in jlist]))))
    # Match only distinct available decision predictions with their actual
    # declared future clock. Future data is for evaluation, never fed to belief.
    errors=[];predrows=[]
    for i,j in js.items():
        if not d['prediction_available'][i]:continue
        future=float(d['prediction_at'][i]);k=int(np.argmin(abs(t-future)))
        if abs(t[k]-future)>1e-9:raise ValueError('Prediction has no exact saved physical evaluation clock')
        e=d['predicted_normal'][i]-d['skin_normal'][k]
        errors.append(e);predrows.append(dict(time=float(t[i]),prediction_at=future,error=e))
    error=np.asarray(errors)
    prediction=dict(available_decision_count=len(error),horizon_s=.02,comparison='saved predicted normal at prediction_at versus actual all-skin normal at exact timestamp',
        RMSE=np.sqrt(np.mean(error**2,axis=0)),max_abs=abs(error).max(axis=0),rows=predrows)
    # Reconstruct a single posterior from the exact saved public stream. This
    # does not advance physics and does not feed future observations to belief.
    ctl=create_joint(meta['calibration'],meta['limits'],meta['budget'],gains=meta['actual_ratio_task_gains'])
    chosen={round(x,3) for x in (7.5,8.2,8.4,8.5,8.9,9.3,11.7)}
    recon=[];modecheck=[];qpchecks=[]
    original_qp=feedback._qp;calls=[]
    def traced_qp(Q,q,A,b,seed,*args,**kwargs):
        result,info=original_qp(Q,q,A,b,seed,*args,**kwargs)
        quadratic=lambda x:float(.5*x@Q@x+q@x)
        calls.append(dict(seed=np.array(seed,copy=True),solution=result.copy(),info=dict(info),
            objective_change=quadratic(result)-quadratic(seed),seed_to_solution_inf=float(np.max(abs(result-seed))),
            Q_rank=int(np.linalg.matrix_rank(Q)),Q_eigenvalues=np.linalg.eigvalsh((Q+Q.T)*.5)))
        return result,info
    feedback._qp=traced_qp
    try:
        for i in np.flatnonzero((t>=7.-1e-9)&(t<12.-1e-9)):
            obj=json.loads(str(d['observation_json'][i]));frame=SimpleNamespace(**{
                k:np.asarray(v) if isinstance(v,list) and k!='skin_patches' else v for k,v in obj.items()})
            ctl.state.update(frame,.005)
            if i not in js:continue
            request=json.loads(str(d['task_request_json'][i]));ctl.set_task_request(request)
            if ctl.execution_anchor is None:
                ctl.grasp.freeze_handoff();ctl.execution_anchor=ctl.state.alpha.copy();ctl.motion_anchor=ctl.executed_motion.copy()
                ctl.required_group=ctl.grasp.snapshot.anchor_count>0
            hyp=ctl.grasp.snapshot;new=hyp.new_normal>max(.04,3.*request['sensor_characteristics']['force_sigma_N'])
            ctl.belief.set_mode_evidence(not new.any(),('new_unaffiliated_contact',) if new.any() else ())
            f=ctl.belief.forecast();b=ctl.belief.snapshot;record=js[i]
            modecheck.append(dict(time=float(t[i]),recorded=record['model_usable'],reconstructed=f.model_usable,
                posterior_max_difference=float(np.max(abs(b.mean16[:5]-d['estimated_normal'][i])))))
            if round(float(t[i]),3) not in chosen:continue
            ctl.reference.reconcile=lambda *args:None
            ctl.reference.propose=lambda *args, rec=record:(rec['reference_task']['reference_scalar'],rec['reference_task'])
            ctl.last_plan_time=None if abs(t[i]-7.5)<1e-9 else float(t[i]-.02)
            if ctl.last_plan_time is not None:
                previous=js[i-4];ctl.interval_action=np.asarray(previous['applied_input_increment_8'],float);ctl.last_debug=previous
            captured={};original_solve=ctl.planner.solve
            def solve_capture(**kwargs):
                result=original_solve(**kwargs);captured.update(kwargs=kwargs,result=result);return result
            ctl.planner.solve=solve_capture;calls.clear();ctl._decision();ctl.planner.solve=original_solve
            plan=captured['result'];kw=captured['kwargs'];h=len(kw['user_targetsH3'])
            lo=kw['step_lower8'];hi=kw['step_upper8'];scale=np.maximum(np.maximum(abs(lo),abs(hi)),1e-8)
            flo=np.tile(lo,h);fhi=np.tile(hi,h);fs=np.tile(scale,h);fixed=abs(fhi-flo)<=1e-12
            offset=np.where(fixed,flo,0.);free=np.flatnonzero(~fixed)
            def full_from_x(x):
                u=offset.copy();u[free]+=fs[free]*x[:len(free)];return u
            rows=[]
            for stage,c in zip(plan.debug['stage_results'],calls):
                before_u=full_from_x(c['seed']);after_u=full_from_x(c['solution'])
                before_p=(kw['pressure_free5H']+kw['pressure_map']@before_u).reshape(h,5)
                after_p=(kw['pressure_free5H']+kw['pressure_map']@after_u).reshape(h,5)
                before_dp=np.diff(np.vstack((kw['pressure_now5'],before_p)),axis=0)
                after_dp=np.diff(np.vstack((kw['pressure_now5'],after_p)),axis=0)
                rows.append(dict(stage=stage['stage'],status=c['info']['status'],objective_before=stage['objective']-c['objective_change'],
                    objective_after=stage['objective'],objective_change=c['objective_change'],seed_to_solution_inf=c['seed_to_solution_inf'],
                    first_action_before=before_u[:8],first_action_after=after_u[:8],pressure_DV_before=before_dp,pressure_DV_after=after_dp,
                    unweighted_DV_squared_before=float((before_dp**2).sum()),unweighted_DV_squared_after=float((after_dp**2).sum())))
            current=kw['pressure_now5'];future_index=int(np.argmin(abs(t-(t[i]+.02))))
            all_pressure_error=plan.predicted_pressure[0]-d['skin_normal'][future_index]
            rec=dict(time=float(t[i]),model_available=plan.debug['model_usable'],recorded_model_available=record['model_usable'],
                action_reconstruction_error=float(np.max(abs(plan.first_action8-np.asarray(record['applied_input_increment_8'])))),
                pressure_now=current,pressure_now_covariance=kw['pressure_covariance5x5'],pressure_free=kw['pressure_free5H'].reshape(h,5),
                no_action_future_minus_now=kw['pressure_free5H'].reshape(h,5)-current,
                calibration_error=kw['calibration_error5H'].reshape(h,5),posterior_fit=b.fit_error16,
                posterior_slope=b.drift_rate16,actual_force_now=d['skin_normal'][i],actual_force_at_plus20=d['skin_normal'][future_index],
                plan=plan.plan,predicted_pressure=plan.predicted_pressure,matched20_error=all_pressure_error,
                model_action_response=(kw['pressure_map']@plan.plan.ravel()).reshape(h,5),stage_diagnostics=rows)
            recon.append(rec)
            print('selected_decision_reconstructed',t[i],'action_error',rec['action_reconstruction_error'],flush=True)
    finally:feedback._qp=original_qp
    after={str(p):sha(p) for p in inputs+frozen}
    if before!=after:raise AssertionError('Read-only inputs changed')
    return serial(dict(schema='v7_nominal_quality_mechanism_v1',scope='one completed development seed; saved public posterior/QP algebra only; no physics or controller-source modifications',
        passed_readonly=True,input_and_source_sha256=before,inputs_unchanged=before==after,summary=summary,prediction=prediction,
        posterior_reconstruction=dict(decisions=len(modecheck),mode_mismatches=sum(x['recorded']!=x['reconstructed'] for x in modecheck),
            maximum_pressure_difference=max(x['posterior_max_difference'] for x in modecheck)),selected_decisions=recon))


if __name__=='__main__':
    target=ROOT/'review/v7_nominal_quality_mechanism_v1.json'
    if target.exists():raise FileExistsError(target)
    report=run()
    with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print('WROTE',target)
    print(json.dumps(dict(prediction=report['prediction']|{'rows':'omitted in stdout'},posterior_reconstruction=report['posterior_reconstruction']),ensure_ascii=False))
