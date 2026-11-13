"""Narrow parent-authorized status update; immutable sources/history untouched."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def canonical(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    progress=ROOT/'研究进度.json';backup=ROOT/'review/progress_before_v12_current_status_v1.json';proof=ROOT/'review/progress_v12_current_status_update_v1.json'
    if backup.exists() or proof.exists():raise FileExistsError('Status update already recorded; no overwrite')
    before_bytes=progress.read_bytes();before_sha=sha(progress);old=json.loads(before_bytes);new=deepcopy(old)
    run=ROOT/'output/release_primitive_prospective_v2';execution=js(run/'execution_complete.json');manifest=js(run/'manifest.json');summary=js(run/'summary.json')
    audit_path=ROOT/'review/release_primitive_prospective_v2_independent_v1.json';audit=js(audit_path)
    cross_path=ROOT/'review/release_primitive_prospective_v2_author_crosscheck_v1.json';cross=js(cross_path)
    if not execution['execution_complete'] or execution['planned']!=32 or execution['completed']!=32 or execution['runtime_failures'] or not execution['sources_unchanged']:raise ValueError('Verified complete32 required')
    if sha(run/'results.json')!=execution['results_sha256'] or sha(run/'summary.json')!=execution['summary_sha256']:raise ValueError('Terminal data binding failed')
    if not audit['passed'] or audit['issues'] or not cross['passed']:raise ValueError('Independent/crosscheck pass required')
    if any(row['primitive_offset_enabled'] for row in summary['results']):raise ValueError('This status snapshot expects verified0primitive exposures; re-evaluate instead of guessing')
    if cross['pre_release_failure_reasons']!={'handoff_outside_calibrated_local_domain':32}:raise ValueError('Preparation status changed')
    stamp=datetime.now(timezone.utc).isoformat()
    new['current_physics_processes']=[]
    new['current_work']='Fresh32 release-component assignments completed full16s with316sources unchanged and0runtime failures. All32 shared old handoff-domain refusals;0actual primitive exposures, so prospective effect remains unobserved. V12 independent observation/intent/known-effort architecture is in software integration:39zero-integration tracker checks plus9numeric-comparison regressions; noV12physics or superiority/safety admission.'
    new['current_sources_status']='HISTORICAL_V9_77_SOURCE_DICTIONARY_RETAINED_VERBATIM: current_sources is an olderV9-era dictionary, not a V12 execution closure. FrozenV10/142 and release/244/316 source manifests remain authoritative for their own unchanged historical results. V12 candidate has separate software-only source/evidence; no physical admission.'
    new['current_physics']=dict(status='idle_after_complete32; V12 software-only, no current physics job',status_updated_utc=stamp,
        last_completed=dict(path=str(run.relative_to(ROOT)),session=34715,planned=32,completed=32,paired_contexts=16,physical_seed_clusters=2,
            physical_seeds=[617901,617902],runtime_failures=0,sources_unchanged=True,source_count=len(manifest['sources']),
            dependency_lock_sha256=manifest['dependency_lock_sha256'],execution_sha256=sha(run/'execution_complete.json'),
            results_sha256=execution['results_sha256'],summary_sha256=execution['summary_sha256'],
            old_handoff_domain_refusals=32,actual_primitive_exposures=0,
            interpretation='All32attempts retained;16zero/slow pairs physically identical through fallback. No valid release-primitive prospective effect estimate; not32observed intervention failures or a benefit confirmation.',
            independent_audit_path=str(audit_path.relative_to(ROOT)),independent_audit_sha256=sha(audit_path),independent_checks=len(audit['checks'])),
        v12=dict(status='software integration; no new physics',effort_tracker_sha256=sha(ROOT/'v12/effort_tracker.py'),
            current_state_effort_QA_checks=39,numeric_comparison_regressions=9,
            author_QA='review/v12_effort_tracker_author_v2.json',
            numeric_regression='review/v12_native_loss_comparison_regression_v2.json',
            previous88bb_candidate='review/frozen_candidates/v12_effort_tracker_author_v1.py',
            boundaries=['No empirical pressureB inverse or calibration local-domain dependency in effort tracker',
                'Known current-state actuation matching is not future pressure prediction, contact-force guarantee or safety certificate',
                'Real native reference/rate bounds, delayed feedback timing and actual-execution antiwindup remain required',
                'Controller integration and any V12 physical effectiveness still unverified']),
        source_dictionary_scope='Older current_sources77 dictionary intentionally unchanged; version annotation in current_sources_status',
        model_admitted=False,material_safety_certified=False,superiority_claim_added=False)
    advantages=new['local_advantages_observed']
    v2=advantages['release_v2_lift_clock_delay']
    v2.update(clearance_delay_increase_s=.011,clearance_delay_increase_clock='full0.5ms integration records, END labels',
        old_200Hz_frame_confirmation_change_s=.065,raw_end_zero_confirmation_latency_s=.884,raw_end_lift_confirmation_latency_s=.895,
        raw_end_confirmation_change_s=.011,timing_resolution_audit_status='complete21/21;257checks',
        timing_resolution_evidence='review/release_clearance_resolution_independent_v2.json',
        timing_interpretation='Old200Hz frame +65ms and rawEND +11ms are different measurement records.200Hz includes extra boundarymj_forward; not simple raw decimation. Low-force dwell is not geometric clearance or continuous-time safety certificate.')
    v3=advantages['release_v3_lift_delay_at_350ms']
    old_continuous=v3.pop('continuous_clearance_confirmation_change_2000Hz_s',None)
    v3.update(raw_end_zero_confirmation_latency_s=.884,raw_end_lift_confirmation_latency_s=.897,
        raw_end_clearance_confirmation_change_s=.013,
        superseded_timing_field=dict(name='continuous_clearance_confirmation_change_2000Hz_s',value=old_continuous,reason='recorded integration-force dwell is not a mathematical continuous-contact certificate'),
        timing_resolution_audit_status='completed21/21;257checks',timing_resolution_evidence='review/release_clearance_resolution_independent_v2.json',
        timing_interpretation='Frame200Hz apparent105ms speedup does not survive full0.5ms integration-force dwell: rawEND .897vs.884,13ms later. Extra boundarymj_forward means records differ, not simple downsampling.29.9298% excess-integral reduction remains saved raw outcome; low-force dwell is not geometric or safety certification.')
    allowed={'current_work','current_physics_processes','current_sources_status','current_physics','local_advantages_observed'}
    unchanged=[k for k in old if k not in allowed]
    if not all(new[k]==old[k] for k in unchanged):raise AssertionError('Unrelated historical field changed')
    if new['current_sources']!=old['current_sources'] or new['goal_status']!=old['goal_status']:raise AssertionError('Source dictionary or goal status changed')
    if sha(progress)!=before_sha:raise RuntimeError('Concurrent progress edit; no commit')
    with backup.open('xb') as f:f.write(before_bytes)
    temp=ROOT/'review/progress_v12_current_status_commit.tmp.json'
    if temp.exists():raise FileExistsError('Unexpected temporary status path')
    temp.write_text(json.dumps(new,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(progress)
    after=js(progress)
    result=dict(schema='narrow_v12_current_status_update_v1',before_sha256=before_sha,after_sha256=sha(progress),
        backup_path=str(backup),backup_sha256=sha(backup),changed_top_level_keys=sorted(k for k in new if new.get(k)!=old.get(k)),
        all_other_top_level_fields_unchanged=all(after[k]==old[k] for k in unchanged),
        historical_current_sources77_canonical_sha256_before=canonical(old['current_sources']),historical_current_sources77_canonical_sha256_after=canonical(after['current_sources']),
        goal_status_before=old['goal_status'],goal_status_after=after['goal_status'],
        updated_utc_header_intentionally_not_rewritten=True,input_terminal_execution_sha256=sha(run/'execution_complete.json'),
        source_sha256=sha(__file__),new_physics=False)
    with proof.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
