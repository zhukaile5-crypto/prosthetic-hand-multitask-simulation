"""Read-only validation of retained16 + continued64 completion receipts.

Does not import execution code, parse result outcomes or load NPZ arrays.
LocalStore hashes files; synthetic stores are for bounded parser tests only.
"""
from pathlib import Path
import hashlib,json

CONTINUE_SHA='d2229449a1e3ec075ff6066cd3ec8219486f9f545ac73ea263634e6f24932cea'
HELPER_SHA='577778f855111c8d80b5c3d5c85ea083e52cfc173b5fff7cef7cf2a46c74357b'
PLAN_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
REL_PLAN='review/v17b_prospective80_plan_v1.json'
REL_STOP='review/v17b_prospective80_progress_v1.json'
REL_RESUME='review/v17b_prospective80_resume_plan_v1.json'
REL_PROGRESS='review/v17b_prospective80_resume_progress_v1.json'
REL_SCHEDULER='review/continue_v17b_prospective80_v1.py'
REL_HELPER='review/v17b_same_instrument_prefix_v1.py'
PREFIX_COLUMNS=tuple('time normal skin_normal shear palm hand_other all_hand_object support penetration world_penetration position relative rotation relative_velocity angular_velocity lift qpos qvel wrist_position ctrl_before ctrl alpha_before alpha robot_actuator_target21 executed_reference21 phase command observed_normal validity observed_age observed_auxiliary_wrench observed_auxiliary_valid legacy_common_sensor_json receiver_measurement_json task_request_json environment_json raw_patches_json'.split())

class LocalStore:
    def __init__(self):self.cache={}
    def exists(self,path):return Path(path).is_file()
    def read(self,path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    def sha(self,path):
        path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        if key not in self.cache:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            self.cache[key]=h.hexdigest()
        return self.cache[key]
    def retained_paths(self,folders):
        return [p.resolve() for folder in folders for p in Path(folder).rglob('*')
                if p.is_file() and 'sources' not in p.relative_to(folder).parts]

def require(value,message):
    if not value:raise ValueError(message)
def inside(root,path):
    root=Path(root).resolve();p=(root/path).resolve()
    require(p.is_relative_to(root),'Receipt path outside research root')
    return p
def equal_file(store,path,pin,label):
    require(store.exists(path),label+' file missing')
    require(store.sha(path)==pin,label+' hash mismatch')
def job_tuple(job):
    return (job['method'],job['scenario'],int(job['seed']),job['motion_mode'],job['task_policy'],job['receiver_policy'])
def expected_jobs(stage,root):
    return {(method,scene,int(seed),stage['motion_mode'],stage['task_policy'],stage['receiver_policy']):
        inside(root,Path(stage['output'])/'runs'/f'{scene}__{method}__{seed}')
        for seed in stage['seeds'] for scene in stage['scenarios'] for method in stage['methods']}

def validate_prefix_set(rows,motion,master,root,store):
    full=next(s for s in master['stages'] if s['arm']=='full_V17' and s['motion_mode']==motion)
    hold=next(s for s in master['stages'] if s['arm']=='holding_A' and s['motion_mode']==motion)
    expected={str(inside(root,Path(full['output'])/'runs'/f'{scene}__ratio_effort__{seed}')):
        str(inside(root,Path(hold['output'])/'runs'/f'{scene}__ratio_effort__{seed}'))
        for seed in master['seeds'] for scene in master['scenarios']}
    require(len(rows)==8,'Exactly8 strict prefixes required per motion')
    seen=set()
    for row in rows:
        f=str(inside(root,row['full']));h=str(inside(root,row['holding']))
        require(f not in seen and expected.get(f)==h,'Prefix duplicate/unassigned pair')
        seen.add(f)
        require(row.get('passed') is True and row.get('no_fields_removed') is True
            and row.get('additional_domain_metadata_exact') is True and row.get('full16_clocks_complete') is True
            and row.get('scope')=='strictly_before12' and row.get('columns')==37
            and row.get('metadata_packets_compared')==2400,'Prefix evidence failed/incomplete/projected')
        checks=row.get('checks',{})
        require(set(checks)==set(PREFIX_COLUMNS) and all(v.get('equal') is True for v in checks.values()),'All37 prefix fields must match')
        equal_file(store,Path(f)/'trace.npz',row['full_trace_sha256'],'full prefix trace')
        equal_file(store,Path(h)/'trace.npz',row['holding_trace_sha256'],'holding prefix trace')
    require(seen==set(expected),'Prefix contexts not complete')

def validate_chain(master,chain_path,*,root,store=None):
    store=store or LocalStore();root=Path(root).resolve();chain_path=inside(root,chain_path)
    progress_path=inside(root,REL_PROGRESS)
    progress=store.read(progress_path) if store.exists(progress_path) else None
    if progress is not None and progress.get('physics_active') is True:
        return dict(status='pending',ready=False,reason='resume_physics_active',outcomes_read=False)
    if not store.exists(chain_path):
        return dict(status='pending',ready=False,reason='completion_chain_missing',outcomes_read=False)
    require(progress is not None,'Completion chain without actual resume progress')
    chain=store.read(chain_path)
    require(chain.get('schema')=='v17b_prospective80_completion_chain_v1' and chain.get('complete') is True,'Invalid/incomplete completion chain schema')
    require(progress.get('schema')=='v17b_same_cohort_resume_progress_v1' and progress.get('complete') is True
        and progress.get('physics_active') is False and progress.get('current') is None,'Resume progress not fully terminal')
    equal_file(store,inside(root,REL_PLAN),PLAN_SHA,'Original master')
    require(chain.get('original_plan_sha256')==PLAN_SHA,'Chain master identity mismatch')
    resume_path=inside(root,REL_RESUME);stop_path=inside(root,REL_STOP)
    equal_file(store,resume_path,chain['resume_plan_sha256'],'resume plan')
    equal_file(store,progress_path,chain['resume_progress_sha256'],'resume progress')
    equal_file(store,stop_path,chain['stopped_progress_sha256'],'retained stopped progress')
    resume=store.read(resume_path);stop=store.read(stop_path)
    require(resume.get('schema')=='v17b_same_cohort_resume_remaining64_v1' and resume.get('original_plan_sha256')==PLAN_SHA
        and resume.get('stopped_progress_sha256')==chain['stopped_progress_sha256'],'Resume origin differs')
    require(resume.get('retained_stages')==master['stages'][:2] and resume.get('remaining_stages')==master['stages'][2:],'Resume changes/reorders original assignments')
    for obj in (chain,resume):
        require(obj.get('source_sha256')==master['source_sha256'] and obj.get('environment')==master['environment'],'Source136/environment differs from master')
    require((resume.get('retained_trials'),resume.get('remaining_trials'),resume.get('total_planned'))==(16,64,80)
        and (chain.get('retained_trials'),chain.get('continued_trials'),chain.get('total_completed'))==(16,64,80),'Retained/continued denominator changed')
    require(resume.get('old16_repeated') is False and resume.get('algorithm_model_gain_measurements_metrics_thresholds_assignments_changed') is False
        and chain.get('old16_repeated') is False and chain.get('control_or_metric_changes') is False
        and chain.get('retained_data_unchanged') is True and chain.get('original_stopped_record_unchanged') is True,'Forbidden source/data/metric repetition/change flag')
    require(stop.get('master_plan_sha256')==PLAN_SHA and stop.get('complete') is False and stop.get('physics_active') is False
        and stop.get('stopped_on_prefix_contract_failure') is True,'Original stop was not preserved literally')
    require(progress.get('master_plan_sha256')==PLAN_SHA and progress.get('resume_plan_sha256')==chain['resume_plan_sha256']
        and (progress.get('retained'),progress.get('remaining_planned'),progress.get('continued'),progress.get('total_completed'))==(16,64,64,80),'Resume progress count/identity mismatch')
    require(len(stop.get('records',[]))==2 and len(progress.get('records',[]))==4,'Expected2 retained and4 continued stage receipts')
    equal_file(store,inside(root,REL_SCHEDULER),CONTINUE_SHA,'Reviewed continuation scheduler')
    equal_file(store,inside(root,REL_HELPER),HELPER_SHA,'Reviewed literal prefix helper')
    require(resume.get('scheduler_sha256')==CONTINUE_SHA and resume.get('corrected_reader_sha256')==HELPER_SHA
        and chain.get('corrected_reader_sha256')==HELPER_SHA,'Resume helper/scheduler pin mismatch')
    equal_file(store,inside(root,master['scheduler']),master['scheduler_sha256'],'Original scheduler')
    for p,pin in master['prefix_helper_sha256'].items():equal_file(store,inside(root,p),pin,'Original prefix helper')
    for role in ('incident_gate','reader_gate'):
        gate_path=inside(root,resume[role]);pin=resume[role+'_sha256']
        require(chain[role+'_sha256']==pin,'Chain correction-gate pin mismatch')
        equal_file(store,gate_path,pin,role);gate=store.read(gate_path)
        require(gate.get('passed') is True,'Correction gate failed')
        if role=='incident_gate':require(gate.get('plan_sha256')==PLAN_SHA and gate.get('source_sha256')==master['source_sha256'],'Incident proof covers different source')
        else:require(gate.get('helper_sha256')==HELPER_SHA,'Independent reader gate covers different helper')
    registration_pins=resume.get('original_registration_sha256',{})
    require(set(registration_pins)=={s['registration'] for s in master['stages']},'All6 original registration pins required')
    marker_pins=chain.get('stage_completion_sha256',{});ledger_pins=chain.get('stage_assignment_ledger_sha256',{})
    names={s['name'] for s in master['stages']}
    require(set(marker_pins)==set(ledger_pins)==names,'All6 marker/ledger receipts required')
    stage_ends={};source_files=0;first_snapshot=None
    for stage in master['stages']:
        name=stage['name'];folder=inside(root,stage['output']);marker=folder/'execution_complete.json';ledger_path=folder/'assignment_ledger.json'
        equal_file(store,marker,marker_pins[name],'stage marker');equal_file(store,ledger_path,ledger_pins[name],'stage ledger')
        e=store.read(marker);stage_ends[name]=e
        require(e.get('execution_complete') is True and e.get('sources_unchanged') is True and e.get('runtime_failures')==[]
            and e.get('planned')==stage['planned'] and e.get('trials')==stage['planned'],'Stage not complete/unchanged')
        for file,key in (('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')):
            equal_file(store,folder/file,e[key],file)
        manifest=store.read(folder/'manifest.json');sp=store.read(folder/'plan.json');ledger=store.read(ledger_path)
        require(manifest.get('research_source_sha256')==master['source_sha256'],'Stage source declaration mismatch')
        expected=expected_jobs(stage,root)
        for jobs in (sp['jobs'],manifest['jobs'],[r['job'] for r in ledger]):
            require(len(jobs)==len(expected) and {job_tuple(j) for j in jobs}==set(expected),'Stage job set/duplicate differs')
            require(all(inside(root,j['destination'])==expected[job_tuple(j)] and j.get('split')=='robustness' for j in jobs),'Job destination/randomization differs')
        require(all(r.get('status')=='evaluated' for r in ledger),'Complete stage has unevaluated assignments')
        # Identity/count only, never task-success or quality selection.
        trials=store.read(folder/'trials.json')
        require(len(trials)==len(expected) and {(t['method'],t['scenario'],int(t['seed'])) for t in trials}
            =={key[:3] for key in expected},'All assigned result identities/count required')
        equal_file(store,inside(root,stage['registration']),registration_pins[stage['registration']],'Original registration')
        registration=store.read(inside(root,stage['registration']))
        wanted={k:stage[k] for k in ('methods','scenarios','seeds','motion_mode','physical_randomization','task_policy','receiver_policy')}
        require(registration.get('source_sha256')==master['source_sha256'] and registration.get('environment')==master['environment']
            and registration.get('assignment')==wanted,'Registered source/environment/assignment differs')
        environment=store.read(folder/'environment.json')
        require(environment.get('pinned_environment')==master['environment'],'Actual stage environment differs')
        snapshot_path=folder/'source_snapshot_manifest.json'
        equal_file(store,snapshot_path,manifest['source_snapshot_manifest_sha256'],'Source snapshot manifest')
        files=store.read(snapshot_path)['files']
        require(set(files)==set(master['source_sha256']),'Source snapshot coverage differs')
        for key,pin in master['source_sha256'].items():
            item=files[key];require(item.get('sha256')==pin,'Source snapshot pin differs')
            snapshot=(folder/item['snapshot_path']).resolve();require(snapshot.is_relative_to(folder),'Snapshot path escapes stage')
            equal_file(store,snapshot,pin,'Snapshot source');source_files+=1
        if first_snapshot is None:first_snapshot=files
    for receipt,stage in zip(stop['records'],master['stages'][:2]):
        require(receipt['stage']==stage['name'] and receipt.get('exit_code')==0 and receipt.get('terminal')==stage_ends[stage['name']],'Retained marker changed')
    for receipt,stage in zip(progress['records'],master['stages'][2:]):
        require(receipt['stage']==stage['name'] and receipt.get('exit_code')==0 and receipt.get('terminal')==stage_ends[stage['name']],'Continued marker differs')
    retained=resume.get('retained_files_sha256',{})
    folders=[inside(root,s['output']) for s in master['stages'][:2]]
    actual={str(p.relative_to(root)):p for p in store.retained_paths(folders)}
    normalized={str(Path(p)):pin for p,pin in retained.items()}
    require(len(normalized)==len(retained) and set(actual)==set(normalized),'Retained16 complete file inventory changed')
    for relative,p in actual.items():equal_file(store,p,normalized[relative],'Retained16 non-source data')
    for key,pin in master['source_sha256'].items():equal_file(store,Path(first_snapshot[key]['original_path']),pin,'Original frozen execution source')
    env=master['environment'];binaries=env['native_binaries_sha256']
    for p,pin in binaries.items():equal_file(store,Path(p),pin,'Frozen native environment')
    equal_file(store,Path(env['executable']),env['executable_sha256'],'Frozen Python executable')
    retained_prefix=resume['retained_prefix_comparisons'];continued_prefix=[p for r in progress['records'] for p in r['prefix_comparisons']]
    require(chain.get('retained_prefix_comparisons')==retained_prefix and chain.get('continued_prefix_comparisons')==continued_prefix,'Prefix receipt chain changed')
    validate_prefix_set(retained_prefix,'hold',master,root,store);validate_prefix_set(continued_prefix,'transport',master,root,store)
    return dict(status='ready_complete_via_receipt_chain',ready=True,chain_sha256=store.sha(chain_path),
        resume_plan_sha256=chain['resume_plan_sha256'],resume_progress_sha256=chain['resume_progress_sha256'],
        original_stopped_progress_sha256=chain['stopped_progress_sha256'],retained_trials=16,continued_trials=64,total=80,
        stage_count=6,prefix_pairs=16,no_fields_removed=True,retained_files_hashed=len(actual),snapshot_source_files_hashed=source_files,
        original_execution_sources_hashed=len(master['source_sha256']),native_binary_files_hashed=len(binaries),
        old_progress_not_rewritten=True,outcomes_read=False)
