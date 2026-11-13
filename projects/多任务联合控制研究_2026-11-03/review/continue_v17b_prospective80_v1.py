"""Retain the stopped16, correct only the prefix reader, execute original64.

The original master, scheduler, failed progress, jobs, model, code136, metrics
and every completed file remain immutable. This creates a separate receipt
chain instead of rewriting the stopped attempt as a successful execution.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MASTER=ROOT/'review/v17b_prospective80_plan_v1.json'
STOP=ROOT/'review/v17b_prospective80_progress_v1.json'
RESUME=ROOT/'review/v17b_prospective80_resume_plan_v1.json'
PROGRESS=ROOT/'review/v17b_prospective80_resume_progress_v1.json'
CHAIN=ROOT/'review/v17b_prospective80_completion_chain_v1.json'
HELPER=ROOT/'review/v17b_same_instrument_prefix_v1.py'
MASTER_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)
def now():return datetime.now(timezone.utc).isoformat()


def prefix_checked(full,holding):
    import numpy as np
    from review.v17b_same_instrument_prefix_v1 import compare
    result=compare(full,holding);clocks=[]
    for folder in (full,holding):
        with np.load(Path(folder)/'trace.npz',allow_pickle=False) as z:t=z['time']
        clocks.append(bool(t.shape==(3201,) and np.allclose(t,np.arange(3201)*.005,rtol=0,atol=1e-8)
            and int(np.count_nonzero(t<12.-1e-9))==2400))
    result.update(full16_clocks_complete=all(clocks),passed=result['passed'] and all(clocks))
    return result


def invariant(master):
    from v17b_full_evidence.runtime import source_identity,METHODS
    from v13_a1.run_bridge import environment
    if sha(MASTER)!=MASTER_SHA or source_identity(METHODS)!=master['source_sha256'] or environment()!=master['environment']:
        raise RuntimeError('Original master/source136/environment42 changed')
    pins={master['scheduler']:master['scheduler_sha256'],**master['prefix_helper_sha256'],
        master['auxiliary_metrics']['reader']:master['auxiliary_metrics']['reader_sha256'],
        master['auxiliary_metrics']['contract']:master['auxiliary_metrics']['contract_sha256']}
    if any(sha(ROOT/p)!=h for p,h in pins.items()):raise RuntimeError('Original registered code or metric changed')


def retained_files(master):
    paths=[]
    for stage in master['stages'][:2]:
        folder=ROOT/stage['output']
        for p in folder.rglob('*'):
            if p.is_file() and 'sources' not in p.relative_to(folder).parts:
                if not p.resolve().is_relative_to(ROOT):raise ValueError('Retained path outside workspace')
                paths.append(p)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def validate_resume_plan(master,plan):
    if (plan.get('schema')!='v17b_same_cohort_resume_remaining64_v1'
        or plan.get('original_plan_sha256')!=MASTER_SHA
        or plan.get('source_sha256')!=master['source_sha256']
        or plan.get('environment')!=master['environment']
        or plan.get('retained_trials')!=16 or plan.get('remaining_trials')!=64 or plan.get('total_planned')!=80
        or plan.get('retained_stages')!=master['stages'][:2]
        or plan.get('remaining_stages')!=master['stages'][2:]
        or plan.get('old16_repeated') is not False
        or plan.get('algorithm_model_gain_measurements_metrics_thresholds_assignments_changed') is not False):
        raise ValueError('Continuation must retain exactly original16 and execute exactly original64, with original source/environment')


def resume_invariant(master,plan,resume_sha):
    invariant(master);validate_resume_plan(master,plan)
    if (sha(RESUME)!=resume_sha or sha(STOP)!=plan['stopped_progress_sha256']
        or sha(Path(__file__))!=plan['scheduler_sha256'] or sha(HELPER)!=plan['corrected_reader_sha256']):
        raise RuntimeError('Continuation code/receipt or retained stopped progress changed')
    for path,pin in plan['original_registration_sha256'].items():
        if sha(ROOT/path)!=pin:raise RuntimeError('Original assignment registration changed')
    for path,pin in ((plan['incident_gate'],plan['incident_gate_sha256']),(plan['reader_gate'],plan['reader_gate_sha256'])):
        if sha(ROOT/path)!=pin or read(ROOT/path).get('passed') is not True:
            raise RuntimeError('Independent correction gate changed')


def prepare(incident_gate,reader_gate):
    if any(p.exists() for p in (RESUME,PROGRESS,CHAIN)):raise FileExistsError('Never overwrite a retained continuation')
    master=read(MASTER);invariant(master);stop=read(STOP)
    if (stop.get('complete') or stop.get('physics_active') or not stop.get('stopped_on_prefix_contract_failure')
        or len(stop['records'])!=2 or stop['master_plan_sha256']!=MASTER_SHA):
        raise RuntimeError('Only the specific terminal16 prefix-reader stop can continue here')
    for s,r in zip(master['stages'][:2],stop['records']):
        e=read(ROOT/s['output']/'execution_complete.json')
        if r['stage']!=s['name'] or r['exit_code']!=0 or e!=r['terminal'] or not e['execution_complete'] or e['trials']!=8 or not e['sources_unchanged']:
            raise RuntimeError('Retained original16 are not exact complete batches')
    incident=read(ROOT/incident_gate);gate=read(ROOT/reader_gate)
    if (incident.get('passed') is not True or incident.get('plan_sha256')!=MASTER_SHA
        or incident.get('source_sha256')!=master['source_sha256']):raise RuntimeError('Independent incident gate differs')
    if gate.get('passed') is not True or sha(HELPER) not in json.dumps(gate):
        raise RuntimeError('Independent corrected-reader gate is missing/stale')
    checks=[];full,holding=master['stages'][1],master['stages'][0]
    for seed in master['seeds']:
        for scene in master['scenarios']:
            run=Path('runs')/(scene+'__ratio_effort__'+str(seed))
            checks.append(prefix_checked(ROOT/full['output']/run,ROOT/holding['output']/run))
    if len(checks)!=8 or not all(c['passed'] and c['columns']==37 and c['no_fields_removed'] for c in checks):
        raise RuntimeError('Retained prefixes not literally equal')
    remaining=master['stages'][2:]
    if sum(s['planned'] for s in remaining)!=64:raise RuntimeError('Not exactly original remaining64')
    if any((ROOT/s['output']).exists() for s in remaining):raise FileExistsError('A remaining batch already exists')
    plan=dict(schema='v17b_same_cohort_resume_remaining64_v1',created_utc=now(),original_plan=str(MASTER.relative_to(ROOT)),
        original_plan_sha256=MASTER_SHA,stopped_progress=str(STOP.relative_to(ROOT)),stopped_progress_sha256=sha(STOP),
        source_sha256=master['source_sha256'],environment=master['environment'],retained_trials=16,remaining_trials=64,total_planned=80,
        retained_stages=master['stages'][:2],remaining_stages=remaining,retained_files_sha256=retained_files(master),
        original_registration_sha256={s['registration']:sha(ROOT/s['registration']) for s in master['stages']},
        incident_gate=incident_gate,incident_gate_sha256=sha(ROOT/incident_gate),
        reader_gate=reader_gate,reader_gate_sha256=sha(ROOT/reader_gate),
        corrected_reader=str(HELPER.relative_to(ROOT)),corrected_reader_sha256=sha(HELPER),
        scheduler_sha256=sha(Path(__file__)),retained_prefix_comparisons=checks,
        algorithm_model_gain_measurements_metrics_thresholds_assignments_changed=False,
        original_stopped_attempt_retained=True,old16_repeated=False,
        finding='Old comparator removed6 metadata fields on only one side; same-instrument JSON already exact, now compare complete UTF8 without field removal',
        no_partial_outcome_selection=True,no_performance_claim=True)
    write(RESUME,plan)
    print(json.dumps(dict(prepared=True,retained=16,remaining=64,resume_plan_sha256=sha(RESUME))),flush=True)


def execute():
    if PROGRESS.exists() or CHAIN.exists():raise FileExistsError('Never restart an existing continuation implicitly')
    master=read(MASTER);plan=read(RESUME);resume_sha=sha(RESUME)
    resume_invariant(master,plan,resume_sha)
    if retained_files(master)!=plan['retained_files_sha256']:raise RuntimeError('Original16 files changed')
    for s in plan['remaining_stages']:
        if (ROOT/s['output']).exists():raise FileExistsError(ROOT/s['output'])
    state=dict(schema='v17b_same_cohort_resume_progress_v1',created_utc=now(),resume_plan_sha256=sha(RESUME),
        master_plan_sha256=MASTER_SHA,retained=16,remaining_planned=64,records=[],current=None,complete=False,physics_active=False)
    write(PROGRESS,state);os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    try:
        for s in plan['remaining_stages']:
            resume_invariant(master,plan,resume_sha);state.update(current=s['name'],physics_active=True,current_started_utc=now());write(PROGRESS,state)
            cmd=[sys.executable,'-B','-m','v17b_full_evidence.run_development','--out',s['output'],
                '--registration',s['registration'],'--methods',*s['methods'],'--scenarios',*s['scenarios'],
                '--seeds',*[str(seed) for seed in s['seeds']],'--motion-mode',s['motion_mode'],
                '--task-policy',s['task_policy'],'--receiver-policy',s['receiver_policy'],'--physical-randomization','--workers',str(s['workers'])]
            print(json.dumps(dict(starting=s['name'],assigned=s['planned'],retained=16)),flush=True)
            result=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy());p=ROOT/s['output']/'execution_complete.json'
            end=read(p) if p.exists() else None
            record=dict(stage=s['name'],exit_code=result.returncode,terminal=end,prefix_comparisons=[]);state['records'].append(record)
            if result.returncode or end is None or not end.get('execution_complete'):
                state.update(current=None,physics_active=False,ended_utc=now(),stopped_on_runtime_failure=True);write(PROGRESS,state);return 1
            resume_invariant(master,plan,resume_sha)
            if s['arm']=='full_V17':
                holding=next(x for x in master['stages'] if x['arm']=='holding_A' and x['motion_mode']==s['motion_mode'])
                for seed in s['seeds']:
                    for scene in s['scenarios']:
                        rel=Path('runs')/(scene+'__ratio_effort__'+str(seed))
                        record['prefix_comparisons'].append(prefix_checked(ROOT/s['output']/rel,ROOT/holding['output']/rel))
                if not all(c['passed'] for c in record['prefix_comparisons']):
                    state.update(current=None,physics_active=False,ended_utc=now(),stopped_on_prefix_contract_failure=True);write(PROGRESS,state);return 1
            write(PROGRESS,state)
        resume_invariant(master,plan,resume_sha)
        if retained_files(master)!=plan['retained_files_sha256']:raise RuntimeError('Retained16 changed while executing remaining64')
        stages={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in master['stages']}
        if not all(e.get('execution_complete') and e.get('sources_unchanged') and e['trials']==s['planned']
            for s in master['stages'] for e in (stages[s['name']],)):raise RuntimeError('All original80 not complete')
        state.update(current=None,physics_active=False,complete=True,ended_utc=now(),continued=64,total_completed=80);write(PROGRESS,state)
        chain=dict(schema='v17b_prospective80_completion_chain_v1',complete=True,created_utc=now(),
            original_plan_sha256=MASTER_SHA,stopped_progress_sha256=sha(STOP),resume_plan_sha256=sha(RESUME),
            resume_progress_sha256=sha(PROGRESS),source_sha256=master['source_sha256'],environment=master['environment'],
            retained_trials=16,continued_trials=64,total_completed=80,retained_data_unchanged=True,
            retained_prefix_comparisons=plan['retained_prefix_comparisons'],
            continued_prefix_comparisons=[c for r in state['records'] for c in r['prefix_comparisons']],
            stage_completion_sha256={s['name']:sha(ROOT/s['output']/'execution_complete.json') for s in master['stages']},
            stage_assignment_ledger_sha256={s['name']:sha(ROOT/s['output']/'assignment_ledger.json') for s in master['stages']},
            corrected_reader_sha256=sha(HELPER),reader_gate_sha256=plan['reader_gate_sha256'],incident_gate_sha256=plan['incident_gate_sha256'],
            old16_repeated=False,original_stopped_record_unchanged=True,control_or_metric_changes=False)
        write(CHAIN,chain);print(json.dumps(dict(complete=True,retained=16,continued=64,total=80,completion_chain_sha256=sha(CHAIN))),flush=True)
        return 0
    except Exception as exc:
        state.update(current=None,physics_active=False,complete=False,ended_utc=now(),stopped_on_contract_exception=type(exc).__name__+': '+str(exc));write(PROGRESS,state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--incident-gate');p.add_argument('--reader-gate');a=p.parse_args()
    if a.prepare:
        if not a.incident_gate or not a.reader_gate:p.error('Both independent gates required before preparation')
        prepare(a.incident_gate,a.reader_gate)
    else:raise SystemExit(execute())
