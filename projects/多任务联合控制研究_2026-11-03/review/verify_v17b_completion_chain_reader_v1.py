"""Synthetic receipt/parser gates only; no real16/64 outcomes or files read."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import read_v17b_delivery_prospective80_v2 as base
import v17b_completion_chain_reader_v1 as subject

class MemoryStore:
    def __init__(self):self.files={};self.overrides={};self.reads=[]
    def key(self,p):return str(Path(p).resolve())
    def put(self,p,value,pin=None):
        k=self.key(p);self.files[k]=deepcopy(value)
        if pin is not None:self.overrides[k]=pin
        else:self.overrides.pop(k,None)
    def exists(self,p):return self.key(p) in self.files
    def read(self,p):self.reads.append(self.key(p));return deepcopy(self.files[self.key(p)])
    def sha(self,p):
        k=self.key(p)
        if k in self.overrides:return self.overrides[k]
        value=self.files[k];b=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(b).hexdigest()
    def retained_paths(self,folders):
        return [Path(p) for p in self.files for folder in folders if Path(p).is_relative_to(folder)
                and 'sources' not in Path(p).relative_to(folder).parts]

def fixture():
    # Production CLI validates real immutable master via base.pinned_plan.
    # This pure fixture substitutes source/env bytes; only the known master,
    # helper and scheduler hash anchors are declared mock trust roots.
    m=deepcopy(base.pinned_plan());store=MemoryStore();P=lambda x:subject.inside(ROOT,x)
    core=P('review/_synthetic/core.py');device=P('review/_synthetic/device.py')
    store.put(core,b'known core');store.put(device,b'known device')
    m['source_sha256']={'core.py':store.sha(core),'device.py':store.sha(device)}
    exe=P('review/_synthetic/python.exe');binary=P('review/_synthetic/native.dll')
    store.put(exe,b'known executable');store.put(binary,b'known binary')
    m['environment']={'executable':str(exe),'executable_sha256':store.sha(exe),'native_binaries_sha256':{str(binary):store.sha(binary)}}
    store.put(P(subject.REL_PLAN),m,pin=subject.PLAN_SHA)
    store.put(P(subject.REL_SCHEDULER),b'synthetic reviewed scheduler',pin=subject.CONTINUE_SHA)
    store.put(P(subject.REL_HELPER),b'synthetic reviewed helper',pin=subject.HELPER_SHA)
    store.put(P(m['scheduler']),b'original scheduler',pin=m['scheduler_sha256'])
    for path,pin in m['prefix_helper_sha256'].items():store.put(P(path),b'original prefix helper',pin=pin)
    ends={};prefix={}
    for stage in m['stages']:
        folder=P(stage['output']);expected=subject.expected_jobs(stage,ROOT);jobs=[]
        for k,destination in expected.items():
            method,scene,seed,motion,tp,rp=k
            jobs.append(dict(method=method,scenario=scene,seed=seed,motion_mode=motion,task_policy=tp,receiver_policy=rp,
                split='robustness',destination=str(destination)))
            store.put(destination/'trace.npz',('synthetic opaque trace '+str(destination)).encode())
        registration=dict(source_sha256=m['source_sha256'],environment=m['environment'],assignment={k:stage[k]
            for k in ('methods','scenarios','seeds','motion_mode','physical_randomization','task_policy','receiver_policy')})
        store.put(P(stage['registration']),registration)
        store.put(folder/'plan.json',{'jobs':jobs})
        store.put(folder/'trials.json',[{'method':j['method'],'scenario':j['scenario'],'seed':j['seed']} for j in jobs])
        store.put(folder/'assignment_ledger.json',[dict(job=j,status='evaluated') for j in jobs])
        files={}
        for key,pin in m['source_sha256'].items():
            orig=core if key=='core.py' else device;snapshot='sources/'+key
            store.put(folder/snapshot,store.files[store.key(orig)])
            files[key]=dict(original_path=str(orig),snapshot_path=snapshot,sha256=pin)
        store.put(folder/'source_snapshot_manifest.json',{'files':files})
        store.put(folder/'environment.json',{'pinned_environment':m['environment']})
        manifest=dict(jobs=jobs,research_source_sha256=m['source_sha256'],source_snapshot_manifest_sha256=store.sha(folder/'source_snapshot_manifest.json'))
        store.put(folder/'manifest.json',manifest)
        e=dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],planned=stage['planned'],trials=stage['planned'],
            trials_sha256=store.sha(folder/'trials.json'),assignment_ledger_sha256=store.sha(folder/'assignment_ledger.json'),
            plan_sha256=store.sha(folder/'plan.json'),manifest_sha256=store.sha(folder/'manifest.json'))
        store.put(folder/'execution_complete.json',e);ends[stage['name']]=e
    def prefixes(motion):
        full=next(s for s in m['stages'] if s['arm']=='full_V17' and s['motion_mode']==motion)
        holding=next(s for s in m['stages'] if s['arm']=='holding_A' and s['motion_mode']==motion)
        rows=[]
        for seed in m['seeds']:
            for scene in m['scenarios']:
                f=P(Path(full['output'])/'runs'/f'{scene}__ratio_effort__{seed}');h=P(Path(holding['output'])/'runs'/f'{scene}__ratio_effort__{seed}')
                rows.append(dict(full=str(f),holding=str(h),passed=True,no_fields_removed=True,columns=37,
                    additional_domain_metadata_exact=True,full16_clocks_complete=True,scope='strictly_before12',metadata_packets_compared=2400,
                    full_trace_sha256=store.sha(f/'trace.npz'),holding_trace_sha256=store.sha(h/'trace.npz'),
                    checks={k:{'equal':True} for k in subject.PREFIX_COLUMNS}))
        return rows
    retained_prefix=prefixes('hold');continued_prefix=prefixes('transport')
    stop=dict(master_plan_sha256=subject.PLAN_SHA,complete=False,physics_active=False,stopped_on_prefix_contract_failure=True,
        records=[dict(stage=s['name'],exit_code=0,terminal=ends[s['name']],prefix_comparisons=[]) for s in m['stages'][:2]])
    store.put(P(subject.REL_STOP),stop)
    incident=P('review/_synthetic/incident.json');gate=P('review/_synthetic/reader_gate.json')
    store.put(incident,dict(passed=True,plan_sha256=subject.PLAN_SHA,source_sha256=m['source_sha256']))
    store.put(gate,dict(passed=True,helper_sha256=subject.HELPER_SHA))
    folders=[P(s['output']) for s in m['stages'][:2]]
    retained={str(p.relative_to(ROOT)):store.sha(p) for p in store.retained_paths(folders)}
    resume=dict(schema='v17b_same_cohort_resume_remaining64_v1',original_plan_sha256=subject.PLAN_SHA,stopped_progress_sha256=store.sha(P(subject.REL_STOP)),
        retained_stages=m['stages'][:2],remaining_stages=m['stages'][2:],source_sha256=m['source_sha256'],environment=m['environment'],
        retained_trials=16,remaining_trials=64,total_planned=80,old16_repeated=False,
        algorithm_model_gain_measurements_metrics_thresholds_assignments_changed=False,scheduler_sha256=subject.CONTINUE_SHA,
        corrected_reader_sha256=subject.HELPER_SHA,incident_gate=str(incident),reader_gate=str(gate),
        incident_gate_sha256=store.sha(incident),reader_gate_sha256=store.sha(gate),
        retained_files_sha256=retained,retained_prefix_comparisons=retained_prefix,
        original_registration_sha256={s['registration']:store.sha(P(s['registration'])) for s in m['stages']})
    store.put(P(subject.REL_RESUME),resume)
    progress=dict(schema='v17b_same_cohort_resume_progress_v1',master_plan_sha256=subject.PLAN_SHA,
        resume_plan_sha256=store.sha(P(subject.REL_RESUME)),complete=True,physics_active=False,current=None,
        retained=16,remaining_planned=64,continued=64,total_completed=80,records=[dict(stage=s['name'],exit_code=0,terminal=ends[s['name']],
            prefix_comparisons=continued_prefix if s['arm']=='full_V17' else []) for s in m['stages'][2:]])
    store.put(P(subject.REL_PROGRESS),progress)
    chain=dict(schema='v17b_prospective80_completion_chain_v1',complete=True,original_plan_sha256=subject.PLAN_SHA,
        stopped_progress_sha256=store.sha(P(subject.REL_STOP)),resume_plan_sha256=store.sha(P(subject.REL_RESUME)),resume_progress_sha256=store.sha(P(subject.REL_PROGRESS)),
        source_sha256=m['source_sha256'],environment=m['environment'],retained_trials=16,continued_trials=64,total_completed=80,
        old16_repeated=False,control_or_metric_changes=False,retained_data_unchanged=True,original_stopped_record_unchanged=True,
        corrected_reader_sha256=subject.HELPER_SHA,incident_gate_sha256=store.sha(incident),reader_gate_sha256=store.sha(gate),
        stage_completion_sha256={s['name']:store.sha(P(s['output'])/'execution_complete.json') for s in m['stages']},
        stage_assignment_ledger_sha256={s['name']:store.sha(P(s['output'])/'assignment_ledger.json') for s in m['stages']},
        retained_prefix_comparisons=retained_prefix,continued_prefix_comparisons=continued_prefix)
    cp=P('review/_synthetic/completion_chain.json');store.put(cp,chain)
    return m,store,cp,core,binary

def relink(store,cp):
    r=store.read(ROOT/subject.REL_RESUME);p=store.read(ROOT/subject.REL_PROGRESS);c=store.read(cp)
    p['resume_plan_sha256']=store.sha(ROOT/subject.REL_RESUME);store.put(ROOT/subject.REL_PROGRESS,p)
    c.update(resume_plan_sha256=store.sha(ROOT/subject.REL_RESUME),resume_progress_sha256=store.sha(ROOT/subject.REL_PROGRESS),
        retained_prefix_comparisons=r['retained_prefix_comparisons'],continued_prefix_comparisons=[x for rr in p['records'] for x in rr['prefix_comparisons']])
    store.put(cp,c)

def main(out):
    checks=[]
    def check(name,fn,expect_reject=False):
        try:value=fn();passed=not expect_reject and bool(value)
        except (ValueError,KeyError,OSError):passed=expect_reject
        checks.append(dict(name=name,passed=passed))
    m,s,c,core,binary=fixture();r=subject.validate_chain(m,c,root=ROOT,store=s)
    check('valid synthetic16+64 chain verifies without rewriting old stop',lambda:r['ready'] and r['total']==80 and r['prefix_pairs']==16 and s.read(ROOT/subject.REL_STOP)['complete'] is False)
    m,s,c,_,_=fixture();s.files.pop(s.key(c))
    check('missing chain is pending and never opens stage receipts',lambda:subject.validate_chain(m,c,root=ROOT,store=s)['status']=='pending' and not any('trials.json' in x for x in s.reads))
    m,s,c,_,_=fixture();p=s.read(ROOT/subject.REL_PROGRESS);p['physics_active']=True;s.put(ROOT/subject.REL_PROGRESS,p)
    check('active progress blocks even stale completechain',lambda:subject.validate_chain(m,c,root=ROOT,store=s)['status']=='pending')
    m,s,c,_,_=fixture();s.files.pop(s.key(ROOT/m['stages'][-1]['output']/'execution_complete.json'))
    check('missing laststage marker rejects complete80',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    m,s,c,_,_=fixture();ch=s.read(c);ch['resume_plan_sha256']='0'*64;s.put(c,ch)
    check('receipt hash tamper rejected',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    m,s,c,_,_=fixture();p=s.read(ROOT/subject.REL_RESUME);p['remaining_stages']=list(reversed(p['remaining_stages']));s.put(ROOT/subject.REL_RESUME,p);relink(s,c)
    check('changed assignment semantics rejected even after honest rehash',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    m,s,c,_,_=fixture();p=next(p for p in s.retained_paths([ROOT/m['stages'][0]['output']]) if p.name=='trace.npz');s.put(p,b'changed retained trace')
    check('changed retained16 bytes rejected',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    m,s,c,core,_=fixture();s.put(core,b'changed actual execution code')
    check('actual source file mutation rejected not just map compared',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    m,s,c,_,binary=fixture();s.put(binary,b'changed native binary')
    check('actual native environment mutation rejected',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    m,s,c,_,_=fixture();p=s.read(ROOT/subject.REL_PROGRESS)
    full=next(r for r in p['records'] if r['prefix_comparisons']);full['prefix_comparisons'][0]['checks'].pop('legacy_common_sensor_json')
    s.put(ROOT/subject.REL_PROGRESS,p);relink(s,c)
    check('prefix missing original JSON field rejected despite passedboolean and valid hashes',lambda:subject.validate_chain(m,c,root=ROOT,store=s),True)
    doc=dict(schema='v17b_completion_chain_reader_software_gate_v1',passed=all(x['passed'] for x in checks),checks=checks,count=len(checks),
        validator_sha256=hashlib.sha256(Path(subject.__file__).read_bytes()).hexdigest(),
        scope=dict(synthetic_only=True,no_actual16_or64_data_read=True,new80_performance_values_read=False,physics_steps=0,model_constructions=0,
            synthetic_hash_trust_roots='In-memory source/environment and known master/helper/scheduler anchors; production CLI separately authenticates original real master+frozen metrics'))
    p=ROOT/out
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(p),passed=doc['passed'],count=len(checks),failed=[x['name'] for x in checks if not x['passed']],sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
    return 0 if doc['passed'] else 1
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
