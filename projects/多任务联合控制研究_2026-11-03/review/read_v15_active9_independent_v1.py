"""Terminal-gated active9 source/physical-interface/metric audit. No replay.

Uses all three registered active cells and the completed passive-blind cell.
All assigned methods/failures remain denominators. No new thresholds/weights.
One static robot model may be compiled for port names/bias; no MjData, plant,
forward/FK/contact solve or integration is permitted. --self-test reads no runs.
"""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import argparse,gzip,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
METHODS=('ratio_effort','fixed_effort','fusion_effort')
PREP=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','penetration','world_penetration','position','relative','rotation','relative_velocity','angular_velocity','lift','qpos','qvel','wrist_position','ctrl_before','ctrl','alpha_before','alpha','phase','observed_normal','validity','observed_age','observed_auxiliary_wrench','observed_auxiliary_valid','raw_patches_json','legacy_common_sensor_json','environment_json')
BOUNDARY=tuple(k for k in PREP if k not in ('ctrl','alpha','phase'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def array_sha(a):
    a=np.ascontiguousarray(a)
    if a.dtype.kind in 'US':return hashlib.sha256(json.dumps(a.tolist(),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def summary(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return dict(n=len(a),median=None if not len(a) else float(np.median(a)),p95=None if not len(a) else float(np.quantile(a,.95)),max=None if not len(a) else float(a.max()))
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
class CachedArchive:
    def __init__(self,d):self.d=d;self.files=list(d)
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def __getitem__(self,k):return self.d[k]
def terminal_gate(stages,reader=read):
    states=[]
    for s in stages:
        p=ROOT/s['output']/'execution_complete.json'
        try:d=reader(p)
        except FileNotFoundError:states.append(dict(stage=s['name'],state='pending'));continue
        terminal=d.get('execution_complete') is True or bool(d.get('ended_utc')) or bool(d.get('runtime_failures'))
        states.append(dict(stage=s['name'],state='terminal' if terminal else 'pending',completion=d))
    return dict(ready=all(x['state']=='terminal' for x in states),states=states,pending_is_failure=False,branch_results_read=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--out',default='review/v15_active9_independent_v1.json');args=ap.parse_args()
    if args.self_test:
        opened=[]
        def missing(p):opened.append(str(p));raise FileNotFoundError(p)
        gate=terminal_gate([dict(name='a',output='output/DO_NOT_READ_BRANCH')],missing)
        assert not gate['ready'] and not gate['pending_is_failure'] and len(opened)==1 and opened[0].endswith('execution_complete.json')
        print(json.dumps(dict(passed=True,checks=1,branch_results_read=False,physics_steps=0)));return 0
    if not args.execute:raise ValueError('Explicit --execute required; preparation is not result verification')
    registration=ROOT/'review/v15_runtime_and_pilot_plan_v1.json';plan=read(registration);stages=plan['sequential_stages'];gate=terminal_gate(stages)
    if not gate['ready']:print(json.dumps(dict(status='PENDING',gate=gate)));return 2
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    checks=[];issues=[];rows=[];source_cache={};baseline_prefix=None;baseline_boundary=None;identity0=None;env0=None;physical0=None;actor=None;forbidden=[]
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:issues.append(dict(name=name,detail=detail))
    def close(name,a,b,tol=1e-10):
        a=np.asarray(a,float);b=np.asarray(b,float);error=float(np.max(abs(a-b),initial=0.)) if a.shape==b.shape else float('inf')
        ck(name,a.shape==b.shape and error<=tol,dict(max_abs_error=error,tolerance=tol))
    def cached_sha(p):
        p=Path(p).resolve()
        if p not in source_cache:source_cache[p]=sha(p)
        return source_cache[p]
    import research_imports,mujoco,hand_plant
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            def deny(*a,_n=name,**k):forbidden.append(_n);raise AssertionError('No physical pipeline '+_n)
            setattr(mujoco,name,deny)
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    def recalc(path,d,result,label):
        def load(p,*a,**k):
            if Path(p).resolve()==path.resolve():return CachedArchive(d)
            raise AssertionError('Unexpected NPZ reload '+str(p))
        with patch.object(np,'load',load):old=evaluate_trace(path);new=evaluate_research(path)
        ck(label+'/canonical_old_evaluator_all_fields',all(k in result and equal(v,result[k]) for k,v in old.items()))
        ck(label+'/research_V3_evaluator_all_fields',equal(new,result['research_metrics']))
    try:
      for stage in stages:
        folder=ROOT/stage['output'];label=stage['name'];done=read(folder/'execution_complete.json');manifest=read(folder/'manifest.json');p=read(folder/'plan.json');trials=read(folder/'trials.json');failures=read(folder/'runtime_failures.json');assignment=read(folder/'assignment_ledger.json')
        ck(label+'/terminal_complete_all3_no_hidden_failures',done['execution_complete'] and done['sources_unchanged'] and not failures and done['planned']==done['trials']==3)
        for k in ('trials','assignment_ledger','plan','manifest'):ck(label+'/'+k+'_hash',done[k+'_sha256']==sha(folder/(k+'.json')))
        expected={('nominal',m,617901) for m in METHODS};key=lambda r:(r['scenario'],r['method'],int(r['seed']))
        ck(label+'/all_assignments',len(p['jobs'])==len(assignment)==3 and {key(j) for j in p['jobs']}==expected and {key(x['job']) for x in assignment}==expected)
        ck(label+'/all_trial_denominators',len(trials)==3 and {key(j) for j in trials}==expected and all(x['status']=='evaluated' for x in assignment))
        ident=manifest['research_source_sha256'];snap=read(folder/'source_snapshot_manifest.json');env=read(folder/'environment.json')['pinned_environment']
        ck(label+'/114sources',len(ident)==114 and set(ident)==set(snap['files']))
        ck(label+'/snapshot_manifest_identity',manifest['source_snapshot_manifest_sha256']==sha(folder/'source_snapshot_manifest.json'))
        bad=[]
        for name,h in ident.items():
            record=snap['files'][name];sp=(folder/record['snapshot_path']).resolve();lp=Path(record['original_path']).resolve()
            if not sp.is_relative_to(folder.resolve()) or not lp.is_relative_to(ROOT.parent.resolve()):raise ValueError('Source path escaped project')
            if cached_sha(sp)!=h or cached_sha(lp)!=h or record['sha256']!=h:bad.append(name)
        ck(label+'/all_source_snapshot_and_current_bytes',not bad,bad[:10])
        if identity0 is None:identity0=ident;env0=env
        ck(label+'/same_source_and_environment_all_cells',ident==identity0 and env==env0)
        binaries=env['native_binaries_sha256'];ck(label+'/42_native_binaries_pinned',len(binaries)==42 and all(cached_sha(Path(k))==v for k,v in binaries.items()) and cached_sha(Path(env['executable']))==env['executable_sha256'])
        trialmap={key(t):t for t in trials}
        for method in METHODS:
          dpath=folder/'runs'/f'nominal__{method}__617901';tag=label+'/'+method
          if not (dpath/'result.json').exists():ck(tag+'/result_present',False);rows.append(dict(cell=label,method=method,missing=True));continue
          result=read(dpath/'result.json');meta=read(dpath/'metadata.json');trace=dpath/'trace.npz';a=load_npz(trace);t=a['time'];active=a['controller_active'].astype(bool)
          ck(tag+'/result_trial_exact',equal(result,trialmap[('nominal',method,617901)]))
          ck(tag+'/source_trace_policy',meta['research_source_sha256']==ident and result['trace_sha256']==meta['trace_sha256']==sha(trace) and meta['task_policy']==stage['task_policy'] and meta['receiver_policy']==stage['receiver_policy'])
          ck(tag+'/full16_counts',meta['completed'] and len(t)==3201 and len(a['substep_time'])==32000)
          close(tag+'/200Hz_clock',t,np.arange(3201)*.005,1e-12);close(tag+'/raw_clock_end',a['substep_time']+a['substep_duration'],np.arange(1,32001)*.0005,1e-8)
          phys={k:meta.get(k) for k in ('shape','scenario','seed','split','config','initial_position','limits','budget','noise','model_sha256','motion_mode','observer_warmup_s')}
          if physical0 is None:physical0=phys
          ck(tag+'/same_physical_sensor_context',equal(phys,physical0))
          prep={k:array_sha(a[k][t<7.5-1e-9]) for k in PREP};boundary={k:array_sha(a[k][abs(t-7.5)<1e-9]) for k in BOUNDARY}
          if baseline_prefix is None:baseline_prefix=prep;baseline_boundary=boundary
          ck(tag+'/common_prep_all_columns_bitexact',prep==baseline_prefix);ck(tag+'/handoff_preaction_only_bitexact',boundary==baseline_boundary)
          recalc(trace,a,result,tag)
          if actor is None:
              model,_=hand_plant._model(FoundationConfig(**meta['config']));actor=ReferenceActor21(model)
          ids=actor.indices21;port=a['executed_reference21'].copy();port[:,18]+=actor.lift_bias_m
          ck(tag+'/actual_sole21_engine_encoding',port.tobytes()==a['ctrl'][:,ids].tobytes())
          ck(tag+'/public_before_engine_snapshot',a['robot_actuator_target21'].tobytes()==a['ctrl_before'][:,ids].tobytes())
          debug=[json.loads(str(s)) for s in a['debug_json']];graphs=[json.loads(str(s)) for s in a['receiver_release_tasks_json']];receivers=[json.loads(str(s)) for s in a['receiver_measurement_json']]
          mismatch=[];rate_error=0.
          for i in np.flatnonzero(active):
              excess=abs(a['reference_increment21'][i])-reference_rates(debug[i]['mode'])*.005;rate_error=max(rate_error,float(excess.max()))
              changed=a['requested_reference21'][i].tobytes()!=a['executed_reference21'][i].tobytes()
              if changed!=bool(debug[i].get('execution_changed_plan')) or int(debug[i]['mode_generation'])!=int(a['mode_generation'][i]):mismatch.append(int(i))
          ck(tag+'/5ms_actual_rates_and_generation_receipts',rate_error<1e-12 and not mismatch,dict(max_rate_excess=rate_error,mismatches=mismatch[:8]))
          jm=read(dpath/'joint_decision_manifest.json');ck(tag+'/decision_log_hash',sha(dpath/jm['file'])==jm['sha256']);counts=Counter();decision_bad=[];di=[];timings=[];by_mode={}
          with gzip.open(dpath/jm['file'],'rt',encoding='utf-8') as f:
            for k,line in enumerate(f):
              r=json.loads(line);i=int(round(r['time_s']/.005));di.append(i);mode=r['mode'];counts['decisions']+=1;counts[mode+'_decisions']+=1
              acceptance=r.get('proposal_acceptance') or {};accepted=bool(acceptance.get('accepted'));q=r.get('qp');n=r.get('native')
              status='not_called' if q is None else q['status'];counts['qp:'+status]+=1
              category=('certified' if acceptance.get('optimality_certified') else 'provisional') if accepted else 'no_accepted_goal';counts['goal:'+category]+=1
              by_mode.setdefault(mode,Counter())[status]+=1;timings.append(debug[i].get('solver_timing_ms',{}))
              if r['decision_id']!=k or debug[i].get('decision_id')!=k or abs(t[i]-r['time_s'])>1e-9 or r['generation']!=int(a['mode_generation'][i]):decision_bad.append((k,'record_clock_generation'))
              if acceptance.get('different_force_feedback_problem_used'):decision_bad.append((k,'hidden_other_problem'))
              if accepted and (not acceptance.get('objective_bytes_equal') or acceptance['primal_violation']>r['qp_problem']['atol'] or acceptance['original_objective']>acceptance['anchor_objective']+acceptance['objective_roundoff_allowance']):decision_bad.append((k,'accepted_primal_merit'))
              if mode=='release' and (r.get('task_labels') or (r.get('qp_problem') or {}).get('extra_task_rows') or r.get('native_tasks',{}).get('rows',0)!=0):decision_bad.append((k,'holding_task_during_release'))
              if n is not None:
                counts['native:'+n['status']]+=1
                if n.get('reference21') is not None:
                  full=n['diagnostics'].get('prepared_value_final_full_recheck',{});counts['native_final_full_pass']+=bool(full.get('passed'))
                  if not accepted or not full.get('passed') or n['status']!='KNOWN_NATIVE_FINITE_GOAL_REALIZATION':decision_bad.append((k,'native_qualification'))
                  if np.asarray(n['p0_effort21']).tobytes()!=np.asarray(r['domain']['p0_effort21']).tobytes() or np.asarray(n['requested_delta_effort21']).tobytes()!=np.asarray(q['d21']).tobytes():decision_bad.append((k,'goal_retargeted'))
                  real=np.asarray(n['native_effort21'])-np.asarray(n['p0_effort21']);err=real-np.asarray(n['requested_delta_effort21'])
                  if real.tobytes()!=np.asarray(n['actual_delta_effort21']).tobytes() or err.tobytes()!=np.asarray(n['diagnostics']['actual_effort_minus_requested_delta21']).tobytes():decision_bad.append((k,'realization_diagnostic'))
              elif accepted:decision_bad.append((k,'native_missing_for_accepted_goal'))
          ck(tag+'/native_and_QP_record_consistency_no_hidden_substitution',not decision_bad and len(di)==jm['records'],decision_bad[:10])
          g_bad=[];phases=Counter();transitions=[];lastg=None
          for i,g in enumerate(graphs):
            if g is None:continue
            phases[g['phase']]+=1;cur=g['current_reference21'][18];phase=g['phase'];ex=float(a['executed_reference21'][i,18])
            if abs(g['current_time_s']-t[i])>1e-9:g_bad.append((i,'graph_clock'))
            if phase=='APPROACH':lo=0.;hi=max(cur,g['entry_reference21'][18],g['descent_start_lift_m'])
            elif phase in ('UNKNOWN_HOLD','GUARDED_CLEAR'):lo=cur;hi=max(cur,g['hold_anchor_m'])
            else:lo=cur;hi=max(cur,g['withdraw_goal_m'])
            if not lo-1e-12<=ex<=hi+1e-12:g_bad.append((i,'actual_lift_outside_graph_authority'))
            if g['receiver'].get('release_evidence_eligible') and g['receiver']['acquisition_time_s']<g['started_s']-1e-9:g_bad.append((i,'pre_receiver_evidence'))
            if g.get('hand_low_since_acquisition_s') is not None and g['hand_low_since_acquisition_s']<g['started_s']-1e-9:g_bad.append((i,'pre_hand_dwell'))
            if lastg is None or g['version']!=lastg['version']:
              transitions.append(dict(time_s=float(t[i]),phase=phase,graph_version=g['version'],mode_generation=int(a['mode_generation'][i]),receiver=g['receiver'],hand=g['hand']))
              if debug[i].get('decision_time_s')!=t[i]:g_bad.append((i,'generation_change_not_effective_same_5ms'))
              if lastg is not None and int(a['mode_generation'][i])==int(a['mode_generation'][i-1]):g_bad.append((i,'graph_did_not_invalidate_generation'))
            if phase=='WITHDRAW' and (lastg is None or lastg['phase']!='WITHDRAW'):
              if not g['hand_low_confirmed'] or not g['terminal_reference_matched'] or t[i]-g['started_s']<2.2-1e-9:g_bad.append((i,'withdraw_gate'))
            lastg=g
          ck(tag+'/guarded_graph_clock_and_actual_lift_authority',not g_bad and (bool(phases) if stage['receiver_policy']=='guarded' else not phases),g_bad[:10])
          dm=read(dpath/'delivery_manifest.json');ck(tag+'/offline_delivery_hashes',dm['privileged_offline_only'] and all(sha(dpath/dm[k]['file'])==dm[k]['sha256'] for k in ('arrays','summary')))
          ledger=load_npz(dpath/dm['arrays']['file']);ls=read(dpath/dm['summary']['file']);ends=ledger['interval_end_s'];starts=ledger['interval_start_s'];dt=np.maximum(0.,np.minimum(ends,16.)-np.maximum(starts,12.));mask=dt>0;valid=ledger['contact_valid'].astype(bool)
          ck(tag+'/ledger32k_exact_raw_start',len(ends)==32000 and starts.tobytes()==a['substep_time'].tobytes())
          close(tag+'/ledger_continuity',starts[1:],ends[:-1],0.);close(tag+'/ledger_raw_end',ends,a['substep_time']+a['substep_duration'],1e-12)
          ledger_bad=[]
          for name,s in ls['target_contact_classes'].items():
            force=ledger['target_'+name+'_normal_N'];complete=ls['window_coverage_complete'] and bool(valid[mask].all())
            if s['available']!=complete:ledger_bad.append((name,'availability'))
            if complete:
              values=[float(np.max(force[mask])),float(np.sum(force[mask]*dt[mask])),float(dt[mask&(force>0)].sum())]
              if not np.allclose(values,[s['peak_N'],s['impulse_Ns'],s['positive_contact_duration_s']],atol=1e-10,rtol=1e-10):ledger_bad.append((name,'statistics'))
            elif any(s[k] is not None for k in ('peak_N','impulse_Ns','positive_contact_duration_s')):ledger_bad.append((name,'unknown_filled'))
          ck(tag+'/ledger_window_statistics_and_unknown_semantics',not ledger_bad and ls['evaluation_only'] and not ls['controller_input_permitted'],ledger_bad)
          support=sum(ledger['target_'+k+'_normal_N'] for k in ('receiver_top','receiver_otherface','floor','other_nonhand_world','other_nonworld'))
          close(tag+'/ledger_reconstructs_original_raw_support',support[valid],a['substep_support'][valid]);close(tag+'/ledger_reconstructs_original_raw_hand',ledger['target_hand_normal_N'][valid],a['substep_all_hand_object'][valid])
          endidx=np.rint(ends/.005).astype(int);which=np.flatnonzero((endidx>=1)&(endidx<=3200)&(abs(ends-endidx*.005)<1e-9));qa=ls['object_qpos_address'];va=ls['object_qvel_address'];stateok=ledger['object_state_valid'][which].astype(bool)
          close(tag+'/ledger_qend_matches_trace',ledger['object_free_qpos7_end'][which[stateok]],a['qpos'][endidx[which[stateok]],qa:qa+7],0.);close(tag+'/ledger_vend_matches_trace',ledger['object_free_qvel6_end'][which[stateok]],a['qvel'][endidx[which[stateok]],va:va+6],0.)
          # Descriptive mechanism at latch: receiver boundary sample and nearby
          # cached intervals are shown separately; no target/actor inference online.
          latch=next((x for x in transitions if x['receiver'].get('consecutive_fresh_acquisitions',0)>=2),None);context=None
          if latch is not None:
            acq=latch['receiver']['acquisition_time_s'];idx=np.flatnonzero(abs(ends-acq)<=.00050001);j=int(round(acq/.005));patches=json.loads(str(a['raw_patches_json'][j]));receiverid=int(model.geom('support').id)
            # All recorded external IDs stay explicit; this does not guess which
            # ones are receiver top without an exact geometry/phase match.
            context=dict(delivery_time_s=latch['time_s'],acquisition_time_s=acq,measurement=receivers[int(round(latch['time_s']/.005))],
                cached_intervals=[dict(start_s=float(starts[k]),end_s=float(ends[k]),target_top_N=float(ledger['target_receiver_top_normal_N'][k]),anyactor_top_N=float(ledger['receiver_any_actor_top_normal_N'][k]),floor_N=float(ledger['target_floor_normal_N'][k])) for k in idx],
                boundary_hand_receiver_contacts=[dict(external_geom=p.get('external_geom'),hand_geom=p.get('hand_geom'),normal_N=p.get('normal'),world_position=p.get('world_position')) for p in patches if p.get('external_geom')==receiverid],
                phase_warning='Boundary forward public packet and cached integration contact solve are distinct; actor provenance cannot be inferred from anyactor minus target alone.')
          top=ledger['target_receiver_top_normal_N'];anytop=ledger['receiver_any_actor_top_normal_N'];floor=ledger['target_floor_normal_N'];first=lambda f:None if not np.any(mask&(f>0)) else [float(starts[np.flatnonzero(mask&(f>0))[0]]),float(ends[np.flatnonzero(mask&(f>0))[0]])]
          ti=np.flatnonzero(mask&(top>0));fi=np.flatnonzero(mask&(floor>0));event_indices={}
          if len(ti):event_indices['first_target_top']=int(ti[0])
          if len(fi):
            event_indices['first_floor']=int(fi[0]);earlier=ti[ti<fi[0]]
            if len(earlier):event_indices['last_target_top_before_floor']=int(earlier[-1])
          event_states={name:dict(interval_start_s=float(starts[k]),interval_end_s=float(ends[k]),target_top_N=float(top[k]),floor_N=float(floor[k]),object_qpos7_end=ledger['object_free_qpos7_end'][k].tolist(),object_qvel6_end=ledger['object_free_qvel6_end'][k].tolist(),end_state_valid=bool(ledger['object_state_valid'][k])) for name,k in event_indices.items()}
          decmask=np.zeros(len(t),bool);decmask[di]=True;timing={k:dict(active_5ms=summary(a[k][active]),decision_records=summary(a[k][decmask]),between_decisions=summary(a[k][active&~decmask])) for k in ('control_ms','full_control_loop_ms','observation_ms','actor_callback_ms')}
          timing.update({'solver_'+k:summary([x[k] for x in timings if k in x]) for k in ('qp','native_inverse','whole_decision')})
          rows.append(dict(cell=label,method=method,result=result,trace_sha256=result['trace_sha256'],decision_log_sha256=jm['sha256'],technical_counts=dict(counts),qp_by_mode={k:dict(v) for k,v in by_mode.items()},graph_phases_5ms=dict(phases),graph_transitions=transitions,latch_context=context,
            delivery_summary=ls,delivery_diagnostics=dict(first_target_top_interval_s=first(top),first_floor_interval_s=first(floor),anyactor_minus_target_top_max_N=float(np.nanmax(abs(anytop-top))),anyactor_target_top_exact=anytop.tobytes()==top.tobytes(),contact_valid_rows=int(valid.sum()),object_state_valid_rows=int(ledger['object_state_valid'].sum()),event_states=event_states),timing_ms=timing,
            preparation_prefix_sha256=prep,handoff_preaction_sha256=boundary))
      ck('no_physics_pipeline_calls',not forbidden,forbidden)
    except Exception as exc:issues.append(dict(name='reader_exception',error=repr(exc),traceback=traceback.format_exc()))
    # Descriptive same-method differences; NULL requires common eligible windows.
    comparisons=[];by={(x['cell'],x['method']):x for x in rows if not x.get('missing')}
    for method in METHODS:
      base=by.get(('passive_blind',method))
      if base is None:continue
      for s in stages[1:]:
        new=by.get((s['name'],method))
        if new is None:continue
        r0=base['result']['research_metrics'];r1=new['result']['research_metrics'];quality={}
        for key in ('static_quality','dynamic_quality','entire_holding_quality'):
          a0=r0[key];a1=r1[key];common=a0['eligible'] and a1['eligible'];diff=None
          if common:
            diff=dict(mean_N=a1['mean_actual_finger_force_N']-a0['mean_actual_finger_force_N'],
                raw200_TV_rate_N_s=a1['raw_200Hz']['finger_TV_rate_N_s']-a0['raw_200Hz']['finger_TV_rate_N_s'],
                raw200_total_SD_N=a1['raw_200Hz']['total_force_SD_N']-a0['raw_200Hz']['total_force_SD_N'],
                bandlimited_TV_rate_N_s=a1['bandlimited']['finger_TV_rate_N_s']-a0['bandlimited']['finger_TV_rate_N_s'])
          quality[key]=dict(common_eligible=common,baseline_eligible=a0['eligible'],candidate_eligible=a1['eligible'],absolute_delta_candidate_minus_baseline=diff)
        comparisons.append(dict(cell=s['name'],method=method,baseline='passive_blind_same_method',quality=quality,task_success=[base['result']['task_success'],new['result']['task_success']],full_motion=[r0['full_motion_mission_success'],r1['full_motion_mission_success']],guard=[r0['safety']['guard_pass'],r1['safety']['guard_pass']]))
    report=dict(schema='v15_active9_independent_v1',passed=not issues,role='independent_result_source_and_interface_audit',checks=checks,issues=issues,rows=rows,comparisons=comparisons,registration_sha256=sha(registration),
        terminal_gate=gate,source_identity=identity0,environment=env0,denominators=dict(active_assigned=9,blind_comparators=3,methods=3,physical_contexts=1,physical_seed=617901,motion='hold',scenario='nominal'),
        scope=dict(all_records='12 result/trace/source/ledger records, including3 completed blind comparators;9active are primary cohort',source_files_per_cell=114,native_binaries=42,static_model_compilations=0 if actor is None else 1,physics_steps=0,plant_constructions=0,
            native_evidence='Saved fixed-goal/actual-delta/final-full-check consistency and actual engine21/reference; no rerun of F067 or mathematical solvers.',common_prep=dict(columns=len(PREP),interval='to7.5 exclusive;7.5 preaction only',scope='Numeric bytes and exact text contents; Unicode archive padding width excluded')),
        limitations=['Seen single context/physical seed; cells/methods are paired interventions, not independent population samples.','Original success labels remain; low hand force, force guard and receiver contact do not certify delivery or placed.',
            'Quality NULL/ineligible preserved; no success-only selection, no aggregate utility or posthoc weighting.','Cached integration contact and boundary-forward measurement phases differ. No online target truth is introduced.'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# V15 active9 独立终态审计','',f"技术审计：{'PASS' if report['passed'] else 'FAIL'}；检查{len(checks)}项，问题{len(issues)}项。单一已见nominal/hold/617901，9条active与3条blind对照分列，不是12个独立样本。",'', '|cell|method|task|full motion|guard|floor event|末态z / m|末速 / m/s|','|---|---|---|---|---|---|---|---|']
    for r in rows:
      if r.get('missing'):continue
      v=r['result'];q=v['research_metrics'];s=r['delivery_summary'];state=s['window_terminal_object_state'];pos=state['position_m']
      lines.append(f"|{r['cell']}|{r['method']}|{v['task_success']}|{q['full_motion_mission_success']}|{q['safety']['guard_pass']}|{s['floor_event']}|{None if pos is None else pos[2]}|{state['linear_speed_m_s']}|")
    lines+=['','完整静态、动态与全持握质量向量、共同可评绝对差、QP/兑现/阶段/账本/计算时间见同名JSON。不可评为NULL，不补0。','', '技术通过只说明记录与冻结合同一致，不说明策略更优。触台不等于稳定承载；floor事件和终点运动代价与原清手/mission标签并列保留。无物理回放或阈值改写。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),issues=issues[:8],rows=len(rows),path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
