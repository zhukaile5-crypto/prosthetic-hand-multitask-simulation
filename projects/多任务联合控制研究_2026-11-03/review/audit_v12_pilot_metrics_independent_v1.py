"""Independent pilot3 metric arithmetic from completed NPZs, no evaluator import.

No plant, physics, source mutation or production evaluator execution. All
filters, windows, force integrals and dwell arithmetic below are local reviewer
implementations of the already declared V3 metric definitions.
"""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/development_v12_pilot_v1'
REPORT=ROOT/'report/v12_pilot_v1/metric_readout.json'
OUT=ROOT/'review/v12_pilot_metrics_independent_v1.json'
EPS=1e-9

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def variation(t,x):
    d=t[-1]-t[0];tv=np.sum(np.abs(np.diff(x,axis=0)),axis=0);drift=np.abs(x[-1]-x[0]);total=x.sum(1)
    rev=np.maximum(0.,(tv-drift)/2);ttv=np.abs(np.diff(total)).sum();td=abs(total[-1]-total[0])
    return dict(duration_s=float(d),samples=len(t),finger_TV_N=float(tv.sum()),finger_net_drift_N=float(drift.sum()),
        finger_reversal_N=float(rev.sum()),finger_TV_rate_N_s=float(tv.sum()/d),finger_net_drift_rate_N_s=float(drift.sum()/d),
        finger_reversal_rate_N_s=float(rev.sum()/d),total_force_TV_N=float(ttv),total_force_net_drift_N=float(td),
        total_force_reversal_N=float(max(0.,(ttv-td)/2)),total_force_TV_rate_N_s=float(ttv/d),
        finger_SD_mean_N=float(np.mean(np.std(x,axis=0))),total_force_SD_N=float(np.std(total)),
        path_identity_residual_N=float(abs(tv.sum()-drift.sum()-2*rev.sum())))

def dwell(t,flag,start,end,length):
    times=t[(t>=start-EPS)&(t<=end+EPS)];yes=np.asarray(flag,bool)[(t>=start-EPS)&(t<=end+EPS)]
    first=None;begin=None
    for now,valid in zip(times,yes):
        if not valid:begin=None
        elif begin is None:begin=float(now)
        if begin is not None and now-begin>=length-EPS and first is None:first=float(now)
    terminal=None
    if len(times) and yes[-1]:
        k=len(times)-1
        while k and yes[k-1]:k-=1
        candidates=times[k:][times[k:]-times[k]>=length-EPS]
        if len(candidates):terminal=float(candidates[0])
    return dict(first_confirmation_s=first,first_latency_s=None if first is None else first-start,
        terminal_confirmation_s=terminal,terminal_latency_s=None if terminal is None else terminal-start,
        any_recontact_after_first=bool(first is not None and np.any((times>first+EPS)&~yes)))

def main():
    terminal=json.loads((RUN/'execution_complete.json').read_text(encoding='utf-8'))
    assert terminal['execution_complete'] and terminal['planned']==terminal['trials']==3 and not terminal['runtime_failures']
    report=json.loads(REPORT.read_text(encoding='utf-8'));reported={x['method']:x for x in report['records']}
    checks=[];rows=[];masks=[];input_sha={str(REPORT.relative_to(ROOT)):sha(REPORT),str((RUN/'execution_complete.json').relative_to(ROOT)):sha(RUN/'execution_complete.json')}
    def check(name,actual,expected):
        if isinstance(expected,(bool,str)) or expected is None:ok=actual==expected
        else:ok=np.asarray(actual).shape==np.asarray(expected).shape and bool(np.allclose(actual,expected,atol=1e-9,rtol=2e-10,equal_nan=True))
        checks.append(dict(name=name,passed=bool(ok),actual=clean(actual),expected=clean(expected)))
    def numeric_dict(name,actual,expected):
        for k,v in actual.items():
            if k in expected and isinstance(v,(int,float)):check(name+'.'+k,v,expected[k])
    paths=sorted(RUN.glob('runs/*/result.json'));assert len(paths)==3 and set(reported)=={'ratio_effort','fixed_effort','fusion_effort'}
    for path in paths:
        result=json.loads(path.read_text(encoding='utf-8'));m=json.loads((path.parent/'metadata.json').read_text(encoding='utf-8'))
        with np.load(path.parent/'trace.npz',allow_pickle=False) as archive:z={k:archive[k] for k in archive.files if not k.endswith('_json') and k!='metadata'}
        method=result['method'];r=result['research_metrics'];rep=reported[method]
        for p in (path,path.parent/'trace.npz',path.parent/'metadata.json'):input_sha[str(p.relative_to(ROOT))]=sha(p)
        check(method+'.result_hash',sha(path),rep['result_sha256']);check(method+'.trace_hash',sha(path.parent/'trace.npz'),rep['trace_sha256'])
        t=z['time'];st=z['substep_time'];dt=z['substep_duration'];end=st+dt;n=z['normal'];skin=z['substep_skin_normal']
        complete=bool(len(t)==3201 and len(st)==32000 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0) and
            np.allclose(st,np.arange(32000)*.0005,atol=1e-7,rtol=0) and np.allclose(dt,.0005,atol=1e-9,rtol=0) and
            abs(dt.sum()-16.)<1e-8 and np.isfinite(skin).all() and np.isfinite(z['substep_normal']).all())
        check(method+'.complete_raw_clock',complete,True);check(method+'.report_record_complete',complete,rep['record_complete'])
        th=(n>.2);support=th[:,0]&(th.sum(1)>=3)
        initial=np.asarray(m.get('initial_position',z['position'][0]));raised=z['position'][:,2]>=initial[2]+.025
        exposed=z['controller_active'];tablefree=z['support']<=.05
        # Local independent implementation of V3 stage-separated 50Hz FIR.
        grid=np.r_[0.,end];physical=np.vstack((n[0],z['substep_normal']));half=50
        kt=np.arange(-half,half+1)*.0005;kernel=.05*np.sinc(100*kt)*np.hanning(101);kernel/=kernel.sum()
        filtered=np.full_like(n,np.nan)
        release=float(m['release_at'])
        for start,finish in zip((0.,7.5,release),(7.5,release,16.)):
            ix=(grid>=start-EPS)&(grid<=finish+EPS);coarse=(t>=start-EPS)&(t<=finish+EPS)
            if start==release and start>0:coarse&=t>start+EPS
            for i in range(5):
                f=np.convolve(np.pad(physical[ix,i],(50,50),mode='reflect'),kernel,mode='valid')
                filtered[coarse,i]=np.interp(t[coarse],grid[ix],f)
        qualities={}
        for name,start,finish in (('static',8.2,8.9),('holding',7.5,12.)):
            ix=(t>=start-EPS)&(t<=finish+EPS);rx=(end>=start-EPS)&(end<=finish+EPS)
            q=r['static_quality' if name=='static' else 'entire_holding_quality']
            eligible=bool(complete and ix.sum()==round((finish-start)/.005)+1 and np.all(support[ix]&raised[ix]&tablefree[ix]&exposed[ix]))
            numeric_dict(method+'.'+name+'.coverage',dict(coverage=float(min(1.,(t[ix][-1]-t[ix][0])/(finish-start))),
                physical_support_fraction=float(support[ix].mean()),policy_exposure_fraction=float(exposed[ix].mean())),q)
            check(method+'.'+name+'.eligible',eligible,q['eligible']);check(method+'.'+name+'.report_eligible',eligible,rep[name+'_eligible' if name=='static' else 'entire_holding_eligible'])
            force=float(n[ix].sum(1).mean());v=variation(t[ix],n[ix]);band=variation(t[ix],filtered[ix]);raw=variation(end[rx],z['substep_normal'][rx])
            check(method+'.'+name+'.force',force,q['mean_actual_finger_force_N']);check(method+'.'+name+'.report_force',force,rep['static_force_N' if name=='static' else 'entire_holding_force_N'])
            numeric_dict(method+'.'+name+'.raw200',v,q['raw_200Hz']);numeric_dict(method+'.'+name+'.bandlimited',band,q['bandlimited'])
            numeric_dict(method+'.'+name+'.report_raw200',v,rep['static_raw200Hz' if name=='static' else 'entire_holding_raw200Hz'])
            numeric_dict(method+'.'+name+'.report_raw2000',raw,rep['extra_raw_integration_diagnostics'][name])
            qualities[name]=dict(eligible=eligible,samples=int(ix.sum()),raw_end_samples=int(rx.sum()),support_fraction=float(support[ix].mean()),
                raised_fraction=float(raised[ix].mean()),tablefree_fraction=float(tablefree[ix].mean()),policy_exposure_fraction=float(exposed[ix].mean()),
                total_force_mean_N=force,raw200=v,bandlimited=band,raw2000_end=raw)
            if name=='static':masks.append(dict(method=method,times=t[ix].tolist(),all_eligible=eligible))
        limits=np.asarray(m['limits']);budget=float(m['budget']);excess=np.maximum(skin-limits,0.);totalexcess=np.maximum(skin.sum(1)-budget,0.)
        peak=np.maximum(skin.max(0),z['skin_normal'][-1]);guard=bool(complete and np.all(skin<=limits+.02) and np.all(skin.sum(1)<=budget+.02)
            and np.all(z['skin_normal'][-1]<=limits+.02) and z['skin_normal'][-1].sum()<=budget+.02)
        safety=dict(guard_pass=guard,finger_peak_N=peak.tolist(),local_excess_integral_Ns=float((excess*dt[:,None]).sum()),total_excess_integral_Ns=float(totalexcess@dt))
        for k,v in safety.items():check(method+'.safety.'+k,v,r['safety'][k])
        check(method+'.report_guard',guard,rep['force_guard']);check(method+'.report_peak',float(peak.max()),rep['global_max_finger_force_N'])
        check(method+'.report_Ns',safety['local_excess_integral_Ns'],rep['local_excess_integral_Ns'])
        stages={}
        for name,ix in [('shared_preparation',st<7.5-EPS),('policy_holding_transport',(st>=7.5-EPS)&(st<release-EPS)),('mechanical_release_and_after',st>=release-EPS)]:
            stage=dict(duration_s=float(dt[ix].sum()),finger_peak_N=skin[ix].max(0).tolist(),local_excess_integral_Ns=float((excess[ix]*dt[ix,None]).sum()),total_excess_integral_Ns=float(totalexcess[ix]@dt[ix]))
            for k,v in stage.items():check(method+'.stage.'+name+'.'+k,v,r['safety']['stages'][name][k]);check(method+'.report_stage.'+name+'.'+k,v,rep['safety_by_stage'][name][k])
            stages[name]=stage
        coarse_clear=dwell(t,z['all_hand_object']<=.2,release,15.5,.2)
        raw_clear=dwell(grid,np.r_[z['all_hand_object'][0],z['substep_all_hand_object']]<=.2,release,15.5,.2)
        raw_full=dwell(grid,np.r_[z['all_hand_object'][0],z['substep_all_hand_object']]<=.2,release,16.,.2)
        check(method+'.coarse_release_latency',coarse_clear['first_latency_s'],r['release']['physical_clearance_latency_s'])
        check(method+'.report_release_latency',coarse_clear['first_latency_s'],rep['release']['physical_clearance_latency_s'])
        hold=(t>=7.5-EPS)&(t<release-EPS);rel=np.linalg.norm(z['relative']-np.asarray(m['reference_relative']),axis=1)
        lost=dwell(t,~support|~raised|(rel>.08),7.5,release-.005,.1)['first_confirmation_s']
        lift_failed=dwell(t,np.abs(z['lift']-z['lift_ref'])>.01,7.5,release-.005,.3)['first_confirmation_s']
        wrist_failed=dwell(t,np.any(np.abs(z['wrist_position']-z['requested_wrist_reference'])>.015,axis=1),7.5,release-.005,.3)['first_confirmation_s']
        j=int(round(release/.005));release_exposure=bool(support[j] and raised[j] and tablefree[j] and z['all_hand_object'][j]>.2)
        mission=bool(complete and m['handoff_passed'] and lost is None and lift_failed is None and release_exposure and coarse_clear['first_confirmation_s'] is not None and z['warning_count'].max()==0)
        check(method+'.mission',mission,r['mission_success']);check(method+'.report_mission',mission,rep['mission_success'])
        check(method+'.full_motion_mission',bool(mission and wrist_failed is None),rep['full_motion_mission_success'])
        rows.append(dict(method=method,condition=dict(scenario=result['scenario'],seed=result['seed'],motion_mode=result['motion_mode']),
            qualities=qualities,safety=safety,stages=stages,release_command_at_s=release,release_coarse200Hz=coarse_clear,
            release_raw2000Hz_END_by_deadline=raw_clear,release_raw2000Hz_END_through16s=raw_full,
            raw_force_guard_exceeding_substeps=int(np.any(skin>limits+.02,axis=1).sum()),mission_success=mission,
            full_motion_mission_success=bool(mission and wrist_failed is None)))
    by={x['method']:x for x in rows};f=by['fusion_effort'];b=by['ratio_effort']
    changes={}
    for baseline in ('ratio_effort','fixed_effort'):
        base=by[baseline]
        changes[baseline]=dict(static_mean_force_percent=100*(f['qualities']['static']['total_force_mean_N']/base['qualities']['static']['total_force_mean_N']-1),
            static_raw200_TV_rate_percent=100*(f['qualities']['static']['raw200']['finger_TV_rate_N_s']/base['qualities']['static']['raw200']['finger_TV_rate_N_s']-1),
            static_raw2000_END_TV_rate_percent=100*(f['qualities']['static']['raw2000_end']['finger_TV_rate_N_s']/base['qualities']['static']['raw2000_end']['finger_TV_rate_N_s']-1),
            full_holding_raw200_total_SD_percent=100*(f['qualities']['holding']['raw200']['total_force_SD_N']/base['qualities']['holding']['raw200']['total_force_SD_N']-1))
    check('root_static_force_rounded4',round(changes['ratio_effort']['static_mean_force_percent'],4),-52.8050)
    check('root_static_TV200_rounded2',round(changes['ratio_effort']['static_raw200_TV_rate_percent'],2),-54.05)
    check('root_static_TV2000_rounded2',round(changes['ratio_effort']['static_raw2000_END_TV_rate_percent'],2),-68.28)
    common=all(x['all_eligible'] and x['times']==masks[0]['times'] for x in masks)
    check('all_three_same_fixed_static_window_jointly_eligible',common,True)
    output=dict(schema='v12_pilot3_independent_metric_audit_v1',role='independent_metric_reviewer',physics_steps=0,
        planned_trials=3,observed_trials=3,independent_physical_conditions=1,statistical_scope='one nominal/hold/617901 development condition; time samples are not independent trials',
        inputs_sha256=input_sha,author_reader_sha256=sha(ROOT/'review/read_v12_metrics_v1.py'),reviewer_source_sha256=sha(Path(__file__)),
        common_static_window=dict(window_s=[8.2,8.9],jointly_eligible=common,coarse_samples_each=141,raw_END_samples_each=1401),
        rows=rows,fusion_percent_change_vs_baseline=changes,checks=len(checks),passed_checks=sum(c['passed'] for c in checks),
        issues=[c for c in checks if not c['passed']],full_check_log=checks,
        conclusions=dict(report_arithmetic_and_checked_metrics_match=all(c['passed'] for c in checks),all_three_whole_force_guards_fail=all(not x['safety']['guard_pass'] for x in rows),
            all_three_holding_excess_is_zero=all(x['stages']['policy_holding_transport']['local_excess_integral_Ns']==0 for x in rows),
            full_holding_total_SD_increases_vs_ratio=changes['ratio_effort']['full_holding_raw200_total_SD_percent']>0,
            performance_superiority_proven=False),
        limitations=['Physical-prefix/observer/source79 independent audit belongs to parent; this audit hashes the specific report/results/NPZ/metadata inputs and checks complete recorded clocks.',
            'Raw200Hz truth is an extra force evaluation and is not decimation of raw integration force; raw uses substep START+duration=END.',
            'Release lowforce dwell is target-contact force clearance, not geometric separation or a pressure predictor.',
            'Every assigned pilot is retained; all three task/fullmotion success and all three forceguard failure are separate outcomes.',
            'Lower static TV cannot imply every force-quality metric improves; full-holding total SD includes intended redistribution/drift.',
            'No new weighting, failure threshold, safety certificate or population/confirmation claim.'])
    OUT.write_text(json.dumps(clean(output),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT.with_suffix('.md')).write_text('V12 pilot3 independent metric audit\n\n'+
        f"Checks {output['passed_checks']}/{output['checks']}; common [8.2,8.9] window eligible for all three.\n\n"+
        'Fusion vs ratio static mean force / 200Hz TV rate / 2000Hz END TV rate (%): '+str(changes['ratio_effort'])+'\n\n'+
        'All three whole-task guards fail; all holding-stage excess integrals are zero and all recorded excess occurs in native release/after. '+
        'One development condition only; no superiority or geometric-clearance claim. Full raw-END and coarse release dwell clocks are separate in JSON.\n',encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),checks=output['checks'],passed=output['passed_checks'],issues=len(output['issues']),changes=changes,
        raw_end_release={x['method']:x['release_raw2000Hz_END_by_deadline'] for x in rows}),ensure_ascii=False))
    if output['issues']:raise SystemExit(1)

if __name__=='__main__':main()
