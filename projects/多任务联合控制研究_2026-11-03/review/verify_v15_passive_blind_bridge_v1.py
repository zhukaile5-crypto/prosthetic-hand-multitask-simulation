"""Read-only blind instrumentation bridge: baseline preparation or final audit.

No runtime/controller/plant construction, rendering or physical replay. NPZ
columns are loaded once per trial, then reused for comparison and evaluation.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,gzip,hashlib,json,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
METHODS=('fixed_effort','ratio_effort','fusion_effort')
TIMING_COLUMNS={'control_ms','observation_ms','full_control_loop_ms','physics_advance_ms','actor_callback_ms'}
NEW_TRACE={'receiver_measurement_json','receiver_release_tasks_json'}
NEW_OBS={'receiver_surface','receiver_release_tasks'}
DEBUG_TIMING_LEAVES=('qp','native_inverse','whole_decision')
METADATA_EQUAL=('method','shape','scenario','seed','mode','split','config','duration','sensor_period','handoff_time','physics_timestep',
    'initial_position','limits','budget','gains','noise','calibration','admission_sha256','source_sha256','model_sha256',
    'handoff_passed','handoff_alpha','preparation_failure','reference_relative','reference_kind','release_at','release_supported_at_command',
    'mechanical_release_at','max_hidden_ctrl_clip','failure_reason','failure_at','completed','shared_execution','commands',
    'motion_mode','observer_warmup_s','actual_ratio_task_gains','legacy_model_domain_supported','legacy_model_bounds_diagnostic')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)) and np.isnan(a) and np.isnan(b):return True
    return a==b
def same_array(a,b):
    return a.shape==b.shape and (a.dtype==b.dtype and a.tobytes()==b.tobytes() if a.dtype.kind in 'biufc' else np.array_equal(a,b))
def clean_debug(d):
    d=dict(d)
    if 'solver_timing_ms' in d:
        t=dict(d['solver_timing_ms'])
        for k in DEBUG_TIMING_LEAVES:t.pop(k,None)
        d['solver_timing_ms']=t
    return d
def metric_identity_stripped(d):
    if isinstance(d,dict):return {k:metric_identity_stripped(v) for k,v in d.items() if k not in ('run_id','trace_path','trace_sha256','wall_seconds')}
    if isinstance(d,list):return [metric_identity_stripped(x) for x in d]
    return d
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
class CachedArchive:
    def __init__(self,a):self.a=a;self.files=list(a)
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def __getitem__(self,k):return self.a[k]

def main():
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--baseline-only',action='store_true');mode.add_argument('--execute',action='store_true')
    ap.add_argument('--baseline',default='output/development_v14_coherent_pilot_v1');ap.add_argument('--candidate',default='output/v15_passive_blind_bridge_v1');ap.add_argument('--out',required=True);args=ap.parse_args()
    baseline=(ROOT/args.baseline).resolve();candidate=(ROOT/args.candidate).resolve();out=(ROOT/args.out).resolve()
    if out.exists():raise FileExistsError(out)
    checks=[];issues=[];records=[];identity_cache={};physics_forbidden=[];ready=False
    def ck(name,yes,detail=None):
        checks.append(dict(name=name,passed=bool(yes),detail=detail))
        if not yes:issues.append(name)
    def close(name,a,b,tolerance=1e-10):
        aa=np.asarray(a,float);bb=np.asarray(b,float);err=float(np.max(abs(aa-bb),initial=0.)) if aa.shape==bb.shape else float('inf')
        ck(name,aa.shape==bb.shape and err<=tolerance,dict(max_error=err,tolerance=tolerance))
    def source_chain(folder,expected_count=None):
        manifest=read(folder/'manifest.json');snapshot=read(folder/'source_snapshot_manifest.json');identity=manifest['research_source_sha256'];files=snapshot['files']
        ck(str(folder.name)+'/source_set',set(files)==set(identity) and (expected_count is None or len(identity)==expected_count))
        ck(str(folder.name)+'/snapshot_manifest_hash',manifest['source_snapshot_manifest_sha256']==sha(folder/'source_snapshot_manifest.json'))
        bad=[]
        for name,expected in identity.items():
            rec=files[name];sp=(folder/rec['snapshot_path']).resolve();original=Path(rec['original_path']).resolve()
            if not sp.is_relative_to(folder) or not original.is_relative_to(ROOT.parent):raise ValueError('Evidence source path outside project')
            for p in (sp,original):
                if p not in identity_cache:identity_cache[p]=sha(p)
                if identity_cache[p]!=expected:bad.append(str(p))
            if rec['sha256']!=expected:bad.append(name+'/manifest')
        ck(str(folder.name)+'/all_source_snapshot_and_live_hashes',not bad,bad[:5])
        finish=read(folder/'execution_complete.json')
        for field,file in [('trials_sha256','trials.json'),('assignment_ledger_sha256','assignment_ledger.json'),('plan_sha256','plan.json'),('manifest_sha256','manifest.json')]:ck(folder.name+'/'+field,finish[field]==sha(folder/file))
        ck(folder.name+'/terminal_sources',finish['execution_complete'] and finish['sources_unchanged'] and not finish['runtime_failures'] and finish['trials']==finish['planned']==3)
        return manifest,identity
    def load_trial(folder,method):
        d=folder/'runs'/f'nominal__{method}__617901';a=load_npz(d/'trace.npz');meta=read(d/'metadata.json');result=read(d/'result.json')
        ck(folder.name+'/'+method+'/trace_identity',sha(d/'trace.npz')==meta['trace_sha256']==result['trace_sha256'])
        ck(folder.name+'/'+method+'/whole_record',a['time'].shape==(3201,) and a['substep_time'].shape==(32000,) and meta['completed'])
        close(folder.name+'/'+method+'/sensor_clock',a['time'],np.arange(3201)*.005,1e-12)
        close(folder.name+'/'+method+'/raw_interval_continuity',a['substep_time'][1:],a['substep_time'][:-1]+a['substep_duration'][:-1],1e-10)
        close(folder.name+'/'+method+'/raw_full16',[a['substep_time'][0],a['substep_time'][-1]+a['substep_duration'][-1]],[0.,16.],1e-9)
        for k in ('qpos','qvel','ctrl','ctrl_before','normal','skin_normal','substep_normal','substep_skin_normal'):ck(folder.name+'/'+method+'/finite/'+k,np.isfinite(a[k]).all())
        return d,a,meta,result
    def recompute(d,a,result):
        import research_imports
        import mujoco
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):
                def deny(*x,_n=name,**kw):physics_forbidden.append(_n);raise AssertionError('No physics in reader:'+_n)
                setattr(mujoco,name,deny)
        from evaluate_runs import evaluate_trace
        from evaluation import evaluate_research
        original=np.load
        def cached(path,*x,**kw):
            if Path(path).resolve()==(d/'trace.npz').resolve():return CachedArchive(a)
            raise AssertionError('Unexpected NPZ second load '+str(path))
        with patch.object(np,'load',cached):old=evaluate_trace(d/'trace.npz');research=evaluate_research(d/'trace.npz')
        ck(d.parent.parent.name+'/'+d.name+'/original_evaluator_recomputed',all(k in result and equal(v,result[k]) for k,v in old.items()))
        ck(d.parent.parent.name+'/'+d.name+'/research_evaluator_recomputed',equal(research,result['research_metrics']))
    def receiver_and_ledger(d,a,meta):
        instrument=meta['receiver_instrument'];valid=0;unknown=0;bad=[];graphs=[]
        allowed={'schema','time_s','acquisition_time_s','age_s','valid','wrench6','normal_load_N','sigma6','uncertainty6','uncertainty_kind','cop_xy_m','cop_valid','sampling_phase','invalid_reason','role','frame','wrench_order','normal_load_role','cop_role'}
        for i,(t,s) in enumerate(zip(a['time'],a['receiver_measurement_json'])):
            x=json.loads(str(s));graphs.append(json.loads(str(a['receiver_release_tasks_json'][i])))
            if set(x)!=allowed or abs(x['time_s']-t)>1e-10:bad.append((i,'schema_clock'))
            if i<2:
                if x['valid'] or x['acquisition_time_s'] is not None or x['invalid_reason']!='delay_buffer_warming':bad.append((i,'unknown_head'))
            else:
                if x['acquisition_time_s'] is None or abs(x['acquisition_time_s']-(t-.01))>1e-9 or abs(x['age_s']-.01)>1e-9 or x['sampling_phase']!='boundary_after_forward':bad.append((i,'delayed_clock_phase'))
            if not equal(x['sigma6'],instrument['sigma6']) or not np.array_equal(np.asarray(x['uncertainty6']),3.*np.asarray(x['sigma6'])):bad.append((i,'sigma_units'))
            if x['valid']:
                valid+=1
                if len(x['wrench6'])!=6 or not np.isfinite(x['wrench6']).all() or x['normal_load_N']!=max(0.,-x['wrench6'][2]):bad.append((i,'measured_resultant'))
            else:
                unknown+=1
                if any(x[k] is not None for k in ('wrench6','normal_load_N','cop_xy_m')):bad.append((i,'unknown_filled'))
        ck(d.name+'/receiver_200Hz10ms_schema',not bad,bad[:5]);ck(d.name+'/blind_release_graph_inactive',all(g is None for g in graphs))
        ck(d.name+'/receiver_fixed_independent_draws',instrument['acquisitions']==3201 and instrument['gaussian_draws']==3201*6 and instrument['uniform_draws']==3201 and instrument['seed']==617901 and instrument['independent_stream_id']==0x56313552)
        ck(d.name+'/receiver_head_only_unknown',valid==3199 and unknown==2,dict(valid=valid,unknown=unknown))
        dm=read(d/'delivery_manifest.json');ck(d.name+'/sidecar_manifest_offline',dm['privileged_offline_only'] and dm['original_success_labels_unchanged'])
        for key in ('arrays','summary'):ck(d.name+'/delivery_'+key+'_hash',sha(d/dm[key]['file'])==dm[key]['sha256'])
        ledger=load_npz(d/dm['arrays']['file']);summary=read(d/dm['summary']['file']);n=len(ledger['interval_start_s'])
        ck(d.name+'/ledger32k_all_valid',n==32000 and ledger['contact_valid'].shape==(32000,) and ledger['contact_valid'].all() and ledger['object_state_valid'].all())
        ck(d.name+'/ledger_raw_start_exact',same_array(ledger['interval_start_s'],a['substep_time']))
        close(d.name+'/ledger_raw_end',ledger['interval_end_s'],a['substep_time']+a['substep_duration'],1e-12)
        close(d.name+'/ledger_continuous_end_to_next_start',ledger['interval_end_s'][:-1],ledger['interval_start_s'][1:],1e-12)
        support=sum(ledger['target_'+k+'_normal_N'] for k in ('receiver_top','receiver_otherface','floor','other_nonhand_world','other_nonworld'))
        close(d.name+'/ledger_cached_total_support_reconstruction',support,a['substep_support'],1e-10)
        close(d.name+'/ledger_cached_hand_reconstruction',ledger['target_hand_normal_N'],a['substep_all_hand_object'],1e-10)
        ck(d.name+'/ledger_offline_clock_contract',summary['evaluation_only'] and not summary['controller_input_permitted'] and summary['capture_phase']=='cached_contact_solve_at_integration_end_state' and summary['diagnostic_window_s']==[12.,16.] and summary['window_coverage_complete'])
        qa=summary['object_qpos_address'];va=summary['object_qvel_address'];ends=ledger['interval_end_s'];indices=np.rint(ends/.005).astype(int);which=np.flatnonzero((indices>=1)&(indices<=3200)&(abs(ends-indices*.005)<1e-9))
        ck(d.name+'/ledger200Hz_state_matches_count',len(which)==3200)
        close(d.name+'/ledger_end_object_qpos_vs_actual_trace',ledger['object_free_qpos7_end'][which],a['qpos'][indices[which],qa:qa+7],0.)
        close(d.name+'/ledger_end_object_qvel_vs_actual_trace',ledger['object_free_qvel6_end'][which],a['qvel'][indices[which],va:va+6],0.)
        dt=np.maximum(0.,np.minimum(ledger['interval_end_s'],16.)-np.maximum(ledger['interval_start_s'],12.));mask=dt>0
        for k,st in summary['target_contact_classes'].items():
            f=ledger['target_'+k+'_normal_N'];ck(d.name+'/ledger_summary_available/'+k,st['available'])
            close(d.name+'/ledger_window_peak/'+k,st['peak_N'],np.max(f[mask]),1e-12)
            close(d.name+'/ledger_window_impulse/'+k,st['impulse_Ns'],np.sum(f*dt),1e-10)
            close(d.name+'/ledger_window_contact_duration/'+k,st['positive_contact_duration_s'],dt[f>0].sum(),1e-10)
        return dict(receiver_valid=valid,receiver_unknown=unknown,receiver_mount=instrument,delivery_summary=summary,sidecar_sha256=dm)
    report=None
    try:
        base_manifest,base_identity=source_chain(baseline,105)
        base_trials=read(baseline/'trials.json');ck('baseline_three_unique_assignments',len(base_trials)==3 and {x['method'] for x in base_trials}==set(METHODS))
        if args.execute:
            finish_path=candidate/'execution_complete.json'
            if not finish_path.is_file() or not read(finish_path).get('execution_complete',False):
                report=dict(status='pending_candidate_not_terminal',passed=None,checks=checks,issues=issues,records=records)
            else:
                new_manifest,new_identity=source_chain(candidate);plan=read(candidate/'plan.json');reg=read(plan['registration']);new_trials=read(candidate/'trials.json')
                ck('new_blind_legacy_registration',plan['task_policy']=='legacy' and plan['receiver_policy']=='blind' and plan['motion_mode']=='hold' and plan['seeds']==[617901] and plan['scenarios']==['nominal'])
                ck('registration_source_and_assignment_pin',reg['source_sha256']==new_identity and plan['registration_sha256']==sha(plan['registration']) and reg['assignment']['task_policy']=='legacy' and reg['assignment']['receiver_policy']=='blind')
                ck('pinned_native_environment_matches',equal(read(baseline/'environment.json')['pinned_environment'],read(candidate/'environment.json')['pinned_environment']) and equal(reg['environment'],read(candidate/'environment.json')['pinned_environment']))
                ck('new_three_unique_assignments',len(new_trials)==3 and {x['method'] for x in new_trials}==set(METHODS))
                ready=True
        if args.baseline_only or ready:
            for method in METHODS:
                bd,b,bm,br=load_trial(baseline,method);recompute(bd,b,br)
                row=dict(method=method,baseline_fields=sorted(b),baseline_trace_sha256=sha(bd/'trace.npz'),baseline_metadata_sha256=sha(bd/'metadata.json'),baseline_metric_summary=dict(task_success=br['task_success'],research_metrics=br['research_metrics']))
                ck('baseline/'+method+'/trials_result_equal',equal(next(v for v in base_trials if v['method']==method),br))
                if ready:
                    nd,n,nm,nr=load_trial(candidate,method);recompute(nd,n,nr)
                    ck(method+'/candidate_result_assignment_exact',equal(next(v for v in new_trials if v['method']==method),nr))
                    ck(method+'/all_old_columns_preserved_exact_new_set',set(n)-set(b)==NEW_TRACE and not(set(b)-set(n)))
                    mismatch=[];exact=[]
                    for key in b:
                        if key in TIMING_COLUMNS or key in ('metadata','observation_json','debug_json'):continue
                        (exact if same_array(b[key],n[key]) else mismatch).append(key)
                    ck(method+'/all_actual_old_physical_observation_policy_columns_exact',not mismatch,dict(exact=exact,mismatch=mismatch))
                    obs_bad=[];debug_bad=[];carrier_bad=[]
                    for i in range(3201):
                        ob=json.loads(str(b['observation_json'][i]));on=json.loads(str(n['observation_json'][i]))
                        if set(on)-set(ob)!=NEW_OBS or set(ob)-set(on):obs_bad.append((i,'field_set'));continue
                        for key in NEW_OBS:on.pop(key)
                        if not equal(ob,on):obs_bad.append((i,'old_value_changed'))
                        if not equal(json.loads(str(n['observation_json'][i]))['receiver_surface'],json.loads(str(n['receiver_measurement_json'][i]))):carrier_bad.append(i)
                        if not equal(clean_debug(json.loads(str(b['debug_json'][i]))),clean_debug(json.loads(str(n['debug_json'][i])))):debug_bad.append(i)
                    ck(method+'/public_observation_only_registered_additions',not obs_bad,obs_bad[:5]);ck(method+'/debug_only_three_timing_leaves_differ',not debug_bad,debug_bad[:5]);ck(method+'/receiver_carrier_consistent',not carrier_bad,carrier_bad[:5])
                    md=[key for key in METADATA_EQUAL if key not in bm or key not in nm or not equal(bm[key],nm[key])];ck(method+'/physical_environment_and_task_metadata_exact',not md,md)
                    ck(method+'/sidecar_metadata_blind_legacy',nm['task_policy']=='legacy' and nm['receiver_policy']=='blind')
                    ck(method+'/all_original_metric_values_unchanged',equal(metric_identity_stripped(br),metric_identity_stripped(nr)))
                    with gzip.open(bd/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as f:bj=[json.loads(x) for x in f]
                    with gzip.open(nd/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as f:nj=[json.loads(x) for x in f]
                    ck(method+'/joint_decision_content_exact_not_gzip_header',equal(bj,nj))
                    sidecar=receiver_and_ledger(nd,n,nm)
                    row.update(candidate_trace_sha256=sha(nd/'trace.npz'),exact_old_columns=exact,failed_old_columns=mismatch,sidecar=sidecar)
                    del n,nj,bj
                records.append(row);del b
            report=dict(status='completed_blind_bridge_audit' if ready else 'baseline_prepared_waiting_candidate',passed=not issues,checks=checks,issues=issues,records=records)
    except Exception:issues.append('exception');report=dict(status='reader_failure',passed=False,checks=checks,issues=issues,records=records,traceback=traceback.format_exc())
    report.update(schema='v15_passive_blind_bridge_independent_v1',baseline=str(baseline),candidate=str(candidate),checker_sha256=sha(__file__),
        exact_comparison_exclusions=dict(trace_columns=sorted(TIMING_COLUMNS|{'metadata'}),json_special=['observation_json: remove only receiver_surface/receiver_release_tasks from candidate','debug_json: remove only solver_timing_ms.qp/native_inverse/whole_decision leaves']),
        source_snapshot_and_live_files_hashed=len(identity_cache),physics_calls=physics_forbidden,physics_steps=0,
        rng_scope='Fresh original legacy_common_sensor_json and all public observed values must be exact for3201 ticks, with unchanged original noise/config and fixed independent receiver draws. Original final RNG state was not saved; no invented final-state equality claim.',
        evidence_scope='No copied source_* policy fields are counted as new physical evidence. New receiver/delivery diagnostics have no old per-surface counterpart and are never backfilled as historical zero. Passing blind equality is instrumentation noninterference, not algorithm performance or safe delivery.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],passed=report['passed'],checks=len(checks),issues=issues,records=len(records),out=str(out)),ensure_ascii=False))
    return 1 if report['passed'] is False else 0
if __name__=='__main__':raise SystemExit(main())
