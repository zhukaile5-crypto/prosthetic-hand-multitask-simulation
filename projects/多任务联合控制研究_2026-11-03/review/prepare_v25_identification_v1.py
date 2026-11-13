"""Register V25 bridge, then five probes; never dispatch or construct a model."""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse, ast, hashlib, json, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
CONTRACT='review/V25公开接触投影响应识别首轮固定实验合同_v1.md'
SELF_GATE='review/v25_preparer_software_v1.json'
CONTEXT=('nominal','transport',619741)
ARMS=('zero','lift_plus','lift_minus','wrj1_plus','wrj1_minus')
HORIZONS=(.005,.010,.020,.040,.100,.200)
V25='v25_response_identification/'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def pin(path):
    p=Path(path).resolve()
    return dict(path=p.relative_to(ROOT.resolve()).as_posix(),sha256=sha(p))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def local_path(value,parent,*,fresh=False):
    p=(ROOT/value).resolve();base=(ROOT/parent).resolve()
    require(p!=base and p.is_relative_to(base),'Strict ROOT/'+parent+' descendant required')
    if fresh:require(not p.exists(),'Preserve earlier attempts; fresh path required: '+str(p))
    return p


def assignment(mode):
    require(mode in ('bridge-fork','probe-five'),'Unsupported identification mode')
    return dict(mode=mode,method='ratio_effort',scenario=CONTEXT[0],seed=CONTEXT[2],
        motion_mode=CONTEXT[1],split='robustness',origin_trigger='receiver_transfer')


def experiment_contract():
    return dict(context_key=list(CONTEXT),task_policy='integrated',receiver_policy='coordinated',
        release_variant='canonical_joint',trigger='receiver_transfer',
        origin_boundary='next_PRE_OBSERVE_after_first_public_TRANSFER_receiver_fresh_streak_at_least2',
        truth_or_outcome_origin_selection=False,probe_arms=list(ARMS),
        target_hold_s=.200,actual_control_period_s=.005,amplitude_fraction=.5,
        amplitude_formula='0.5*min(original_release_reference_rate*0.005, actual_origin_direction_margin)',
        perturbed_axes=dict(lift=18,WRJ1=20),only_two_input_columns_supported=True,full21_model_admitted=False,
        acquisition_horizons_s=list(HORIZONS),
        flush_rule='origin+0.200+declared_tactile_delay_s+0.005; retain original16s truncation',
        no_future_zero_online=True,labels_wait_for_public_availability=True,
        all_unsupported_missing_cancelled_truncated_retained=True,
        full_task_performance_claim_allowed=False,independent_physical_seed_clusters=1)


def validate_gate(gate,identity,required_sources,*,full=False,min_checks=1):
    require(gate.get('passed') is True,'An actual passed gate is required')
    require(type(gate.get('physics_steps')) is int and gate['physics_steps']==0,'Pure/zero gate must state physics_steps=0')
    maps=[]
    for key in ('source_sha256','production_source_sha256','parent192_source_sha256'):
        value=gate.get(key)
        if value is not None:
            require(isinstance(value,dict) and value,'Nonempty source map required: '+key)
            require(all(identity.get(k)==v for k,v in value.items()),'Gate source changed: '+key)
            maps.append(value)
    union={k:v for m in maps for k,v in m.items()}
    require(set(required_sources)<=set(union),'Gate does not cover its assigned component')
    if full:require(any(m==identity and len(m)==198 for m in maps),'Actual complete198 identity gate required')
    if not full:
        checks=gate.get('checks')
        require(isinstance(checks,list) and len(checks)>=min_checks,'Actual finite checks absent')
        require(all(isinstance(c,dict) and c.get('passed') is True for c in checks),'A declared check did not pass')
    require(not gate.get('issues') and not gate.get('error'),'Gate has unresolved issues/error')
    if 'complete' in gate:require(gate['complete'] is True,'Independent gate is not complete')
    for name in ('source_unchanged','parent192_unchanged'):
        if name in gate:require(gate[name] is True,'Gate reports changed sources: '+name)
    return union


def validate_completion(value,identity,numeric,environment,mode,planned):
    require(value.get('execution_complete') is True and value.get('passed') is True,'Actual execution not complete/passed')
    require(value.get('mode')==mode and value.get('planned')==planned and value.get('completed')==planned,'Wrong actual execution role/count')
    require(value.get('all_assigned_accounted') is True and len(value.get('rows',[]))==planned,'All assignments must be retained')
    require(value.get('runtime_failures')==[] and value.get('sources_unchanged') is True and value.get('numerical_unchanged') is True,'Actual failure/source/numerical gate failed')
    require(value.get('source_sha256')==identity and value.get('numerical_identity')==numeric and value.get('environment')==environment,'Execution identities differ')


def validate_bridge_payload(bridge,identity,numeric):
    require(bridge.get('schema')=='v25_full_remaining_fork_bridge_v1' and bridge.get('passed') is True,'Actual full fork bridge not passed')
    require(bridge.get('full_remaining_episode_compared') is True and bridge.get('opaque_state_never_exposed_to_controller') is True,'Complete bridge / opaque-state boundary absent')
    require(bridge.get('source_sha256')==identity and bridge.get('numerical_identity')==numeric,'Bridge source/numerical identity differs')
    for name in ('fork_trace_bridge','frozen_V24_trace_bridge'):
        b=bridge.get(name,{})
        require(b.get('passed') is True and b.get('same_column_keys') is True and b.get('full16') is True,'Full16 trace bridge failed: '+name)
        require(isinstance(b.get('checks'),dict) and b['checks'] and all(v is True for v in b['checks'].values()),'A full trace field failed: '+name)
        require(b.get('no_algorithm_or_public_measurement_fields_removed') is True,'Bridge broadened comparison exclusions')
    public=bridge.get('clone_public_logs',{})
    require(public.get('passed') is True,'Cloned public observation/receipt logs differ')
    expected={'response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz'}
    require(set(public.get('streams',{}))==expected,'Public log stream missing')
    require(all(v.get('passed') is True and type(v.get('records')) is int and v['records']>0 for v in public['streams'].values()),'Actual public stream comparison absent/failed')


def frozen_reference(identity,numeric,environment):
    planpath=ROOT/'review/v24_small_plan_v1.json';reportpath=ROOT/'review/v24_small_readout_v1.json'
    p=read(planpath);r=read(reportpath)
    parent=p['source_sha256']
    require(len(parent)==192 and all(identity.get(k)==v for k,v in parent.items()),'Frozen parent192 changed')
    require(p['environment']==environment and len(environment['native_binaries_sha256'])==42,'Original42 physical environment changed')
    require(p['numerical_identity']==numeric,'Frozen numerical environment changed')
    require(r.get('complete') is True and r.get('issues')==[] and r.get('assigned')==9 and r.get('plan_sha256')==sha(planpath),'FrozenV24 cohort not completed/identical')
    stages=[s for s in p['stages'] if s['variant']=='canonical_joint' and s['context_key']==list(CONTEXT)]
    require(len(stages)==1,'Exactly one original canonical_joint context required')
    stage=stages[0];run=ROOT/stage['output']/'runs'/'nominal__ratio_effort__619741'
    m=read(run/'metadata.json');scene=m.get('scenario');scene=scene.get('name') if isinstance(scene,dict) else scene
    require((scene,m.get('motion_mode'),m.get('seed'))==CONTEXT,'Frozen reference actual metadata context differs')
    require(m.get('method')=='ratio_effort' and m.get('release_variant')=='canonical_joint' and m.get('task_policy')=='integrated' and m.get('receiver_policy')=='coordinated','Frozen reference actual policy differs')
    require(m.get('completed') is True and m.get('research_source_sha256')==parent and m.get('numerical_identity')==numeric,'Frozen original full16 identity absent')
    require(m.get('trace_sha256')==sha(run/'trace.npz'),'Original trace hash changed')
    files=('trace.npz','metadata.json','result.json','release_decisions.jsonl.gz','delivery_intervals.npz')
    reference=dict(run=str(run.resolve()),input_sha256={n:sha(run/n) for n in files})
    return reference,{x:sha(ROOT/x) for x in ('review/v24_small_plan_v1.json','review/v24_small_readout_v1.json')}


def check_actual_bridge(path,identity,numeric,environment,reference):
    path=local_path(path,'output');folder=path.parent
    require(path.name=='bridge_complete.json','Explicit bridge_complete.json required')
    bridge=read(path);validate_bridge_payload(bridge,identity,numeric)
    complete=read(folder/'execution_complete.json');validate_completion(complete,identity,numeric,environment,'bridge-fork',2)
    require({r['key'] for r in complete['rows']}=={'donor','clone'} and all(r['status']=='completed_full16' for r in complete['rows']),'Bridge donor/clone not both full16')
    p=read(folder/'plan.json')
    require(p['assignment']==assignment('bridge-fork') and p['mode']=='bridge-fork','Bridge context or origin trigger differs')
    require(p['source_sha256']==identity and p['numerical_identity']==numeric and p['environment']==environment,'Bridge plan identities differ')
    regpath=Path(p['registration']);reg=read(regpath)
    require(sha(regpath)==p['registration_sha256'] and reg['assignment']==assignment('bridge-fork') and reg['frozen_v24_reference']==reference,'Bridge registration/input reference changed')
    manifest=read(folder/'manifest.json');snap=read(folder/'source_snapshot_manifest.json')
    require(complete['plan_sha256']==sha(folder/'plan.json')==manifest['plan_sha256'],'Bridge plan chain changed')
    require(complete['manifest_sha256']==sha(folder/'manifest.json') and complete['assignment_ledger_sha256']==sha(folder/'assignment_ledger.json'),'Bridge completion chain changed')
    require(manifest['source_sha256']==identity and manifest['numerical_identity']==numeric and manifest['source_snapshot_manifest_sha256']==sha(folder/'source_snapshot_manifest.json'),'Bridge manifest changed')
    require(set(snap['files'])==set(identity),'Bridge source snapshots incomplete')
    for k,row in snap['files'].items():
        f=(folder/row['snapshot_path']).resolve()
        require(f.is_relative_to(folder.resolve()) and row['sha256']==identity[k] and sha(f)==identity[k],'Bridge source snapshot differs: '+k)
    files=['bridge_complete.json','execution_complete.json','assignment_ledger.json','runtime_failures.json','plan.json','manifest.json','source_snapshot_manifest.json','origin_public.json','opaque_fork_metadata.json']
    for role in ('donor','clone'):
        files.extend(role+'/'+n for n in ('trace.npz','metadata.json','result.json','response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz'))
    require(read(folder/'runtime_failures.json')==[],'Bridge retained runtime errors')
    origin=read(folder/'origin_public.json')
    require(origin.get('available') is True and origin.get('trigger')=='receiver_transfer' and origin.get('next_PRE_OBSERVE_boundary') is True,'Bridge did not expose fixed public origin')
    return pin(path),{str((folder/n).relative_to(ROOT)).replace('\\','/'):sha(folder/n) for n in files}


def prepare(args):
    from v25_response_identification.runtime import source_identity
    from v25_response_identification.design import PROBE_ARMS,PROBE_HOLD_S,AMPLITUDE_FRACTION,CONTROL_PERIOD_S
    from v22_joint_task_fusion.numerics import numerical_identity
    from v13_a1.run_bridge import environment
    identity=source_identity();numeric=numerical_identity();env=environment()
    require(len(identity)==198,'Complete parent192+six V25 source closure required')
    require(tuple(PROBE_ARMS)==ARMS and (PROBE_HOLD_S,AMPLITUDE_FRACTION,CONTROL_PERIOD_S)==(.2,.5,.005),'Fixed five-probe contract changed')
    reference,deps=frozen_reference(identity,numeric,env)
    selfgate=read(ROOT/args.self_gate)
    require(selfgate.get('passed') is True and selfgate.get('physics_steps')==0 and selfgate.get('preparer_sha256')==sha(__file__) and selfgate.get('contract_sha256')==sha(ROOT/CONTRACT),'Current preparer software gate absent')
    specs=[('observer_author',args.observer_author_gate,{V25+'observer.py'},38),
           ('observer_independent',args.observer_independent_gate,{V25+'observer.py'},1),
           ('fork_author',args.fork_author_gate,{V25+'forks.py'},1),
           ('fork_independent',args.fork_independent_gate,{V25+'forks.py'},1),
           ('runtime',args.runtime_gate,{V25+n for n in ('__init__.py','design.py','runtime.py','run_identification.py')},31)]
    gates={};used=[]
    for role,relative,required,n in specs:
        p=local_path(relative,'review');g=read(p);pins=validate_gate(g,identity,required,min_checks=n)
        if 'numerical_identity' in g:require(g['numerical_identity']==numeric,'Gate numerical identity differs: '+role)
        gates[role]=dict(**pin(p),checked_sources=pins);used.append(p)
    require(len(set(used))==len(used),'Independent and author gates must be separate artifacts')
    zero_arg=(ROOT/args.zero_gate).resolve();zero_review=None
    if zero_arg.is_relative_to((ROOT/'review').resolve()):
        zero_review=read(zero_arg);validate_gate(zero_review,identity,set(identity),full=True)
        require(zero_review.get('numerical_identity')==numeric and zero_review.get('environment')==env,'Zero review numerical/environment identity differs')
        require(len(zero_review.get('checks',[]))>=10 and all(c.get('passed') is True for c in zero_review['checks']),'Actual zero binding checks absent')
        for name,digest in zero_review.get('input_sha256',{}).items():
            p=(ROOT/name).resolve()
            require(p.is_relative_to(ROOT.resolve()) and sha(p)==digest,'Actual zero input receipt changed')
        zero_path=local_path(str(Path(zero_review['actual_output'])/'capability_zero.json'),'output')
    else:zero_path=local_path(args.zero_gate,'output')
    z=read(zero_path)
    validate_gate(z,identity,set(identity),full=True)
    require(z.get('source_count')==198 and z.get('numerical_identity')==numeric,'Actual constructor full198/numerical gate absent')
    require(type(z.get('initialization_forward_calls')) is int and z['initialization_forward_calls']>=0,'Actual zero constructor phase absent')
    zcomplete=read(zero_path.parent/'execution_complete.json')
    validate_completion(zcomplete,identity,numeric,env,'capability-zero',1)
    require(zcomplete.get('physics_steps')==0,'Zero mode advanced physics')
    gates['actual198_zero']=dict(**pin(zero_path),completion=pin(zero_path.parent/'execution_complete.json'))
    if zero_review is not None:gates['actual198_zero']['independent_binding_gate']=pin(zero_arg)
    for relative in (CONTRACT,args.self_gate,'review/v18_passive_trace_comparison_v3.py'):
        deps[relative]=sha(ROOT/relative)
    registration=local_path(args.registration,'review',fresh=True);planpath=local_path(args.plan,'review',fresh=True)
    output=local_path(args.experiment_output,'output',fresh=True)
    require(registration!=planpath,'Plan and registration require separate fresh files')
    reg=dict(schema='v25_fixed_public_identification_registration_v1',assignment=assignment(args.mode),
        source_sha256=identity,environment=env,numerical_identity=numeric,software_gate_passed=True,
        software_gates=gates,dependencies=deps,preparer_sha256=sha(__file__),
        identification_contract=experiment_contract(),frozen_v24_reference=reference,
        seen_development=True,performance_claim_allowed=False)
    if args.mode=='probe-five':
        require(args.bridge_complete is not None,'Actual full198 physical fork bridge required before probe registration')
        reg['required_bridge_gate'],reg['required_bridge_artifact_sha256']=check_actual_bridge(args.bridge_complete,identity,numeric,env,reference)
    else:require(args.bridge_complete is None,'Bridge mode cannot consume a different origin/bridge')
    command=['-B','-m','v25_response_identification.run_identification','--'+args.mode,
        '--registration',registration.relative_to(ROOT).as_posix(),'--out',output.relative_to(ROOT).as_posix()]
    plan=dict(schema='v25_fixed_public_identification_plan_v1',mode=args.mode,
        planned=2 if args.mode=='bridge-fork' else 5,
        assigned=['donor','clone'] if args.mode=='bridge-fork' else list(ARMS),
        assignment=reg['assignment'],source_sha256=identity,environment=env,numerical_identity=numeric,
        contract=experiment_contract(),software_gates=gates,dependencies=deps,
        frozen_v24_reference=reference,registration=registration.relative_to(ROOT).as_posix(),
        output=output.relative_to(ROOT).as_posix(),python_arguments=command,
        preparer_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat(),
        physics_dispatched=False,expansion_requires_new_registration=True)
    # Re-read identities after all inputs, before either registration is created.
    require(source_identity()==identity and numerical_identity()==numeric and environment()==env,'Source/environment changed while preparing')
    write(registration,reg);plan['registration_sha256']=sha(registration);write(planpath,plan)
    print(json.dumps(dict(plan=str(planpath),registration=str(registration),planned=plan['planned'],source_count=len(identity),python_arguments=command,physics_dispatched=False),ensure_ascii=False))


def self_test(path):
    path=local_path(path,'review',fresh=True);checks=[]
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    def rejects(fn):
        try:fn()
        except (ValueError,KeyError,TypeError):return True
        return False
    identity={f's{i}':'a'*64 for i in range(198)};num={'fixture':'numeric'}
    gate=dict(passed=True,physics_steps=0,production_source_sha256={'s0':'a'*64},checks=[dict(passed=True)])
    ck('subset gate is explicit not complete198',bool(validate_gate(gate,identity,{'s0'})) and rejects(lambda:validate_gate(gate,identity,{'s0'},full=True)))
    ck('failed or mismatched-source gate rejected',rejects(lambda:validate_gate(dict(gate,passed=False),identity,{'s0'})) and rejects(lambda:validate_gate(dict(gate,production_source_sha256={'s0':'b'*64}),identity,{'s0'})))
    ck('missing owned source and false individual check rejected',rejects(lambda:validate_gate(gate,identity,{'s1'})) and rejects(lambda:validate_gate(dict(gate,checks=[dict(passed=False)]),identity,{'s0'})))
    ck('only fixed seen context and next-public trigger assignment exposed',assignment('bridge-fork')==dict(mode='bridge-fork',method='ratio_effort',scenario='nominal',seed=619741,motion_mode='transport',split='robustness',origin_trigger='receiver_transfer') and 'origin_time_s' not in assignment('probe-five'))
    b=dict(schema='v25_full_remaining_fork_bridge_v1',passed=True,source_sha256=identity,numerical_identity=num,full_remaining_episode_compared=True,opaque_state_never_exposed_to_controller=True)
    sub=dict(passed=True,same_column_keys=True,full16=True,checks={'actual_force':True,'public_UTF8':True},no_algorithm_or_public_measurement_fields_removed=True)
    b.update(fork_trace_bridge=deepcopy(sub),frozen_V24_trace_bridge=deepcopy(sub),clone_public_logs=dict(passed=True,streams={n:dict(passed=True,records=3) for n in ('response_observations.jsonl.gz','response_receipts.jsonl.gz','response_input_intervals.jsonl.gz')}))
    validate_bridge_payload(b,identity,num)
    bad=deepcopy(b);bad['fork_trace_bridge']['checks']['actual_force']=False
    ck('top-level passed cannot hide actual-field bridge failure',rejects(lambda:validate_bridge_payload(bad,identity,num)))
    bad=deepcopy(b);bad['clone_public_logs']['streams'].pop('response_receipts.jsonl.gz')
    ck('missing public receipt stream blocks probes',rejects(lambda:validate_bridge_payload(bad,identity,num)))
    ck('different bridge source or numeric blocks probes',rejects(lambda:validate_bridge_payload(b,dict(identity,extra='a'*64),num)) and rejects(lambda:validate_bridge_payload(b,identity,{})))
    c=experiment_contract()
    ck('five held directions support only two columns and exact acquisition horizons',c['probe_arms']==list(ARMS) and c['only_two_input_columns_supported'] and not c['full21_model_admitted'] and c['acquisition_horizons_s']==list(HORIZONS) and c['target_hold_s']==.2)
    source=Path(__file__).read_text(encoding='utf-8');tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    ck('no dispatch/subprocess/Session/fork/physics call in preparer',not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('step_one_sensor_tick','advance','Popen','run','fork_session','mj_step','mj_forward') for n in calls))
    runtext=(ROOT/'v25_response_identification/run_identification.py').read_text(encoding='utf-8')
    ck('actual runner CLI modes match generated arguments',all("'--"+s+"'" in runtext for s in ('bridge-fork','probe-five','registration','out')))
    result=dict(schema='v25_preparer_software_v1',passed=all(c['passed'] for c in checks),checks=checks,
        physics_steps=0,registrations_created=0,preparer_sha256=sha(__file__),contract_sha256=sha(ROOT/CONTRACT),
        scope='Finite fake gate/source/bridge rejection fixtures and actual CLI AST; no real registration, model or outcomes read.')
    write(path,result);print(json.dumps(dict(passed=result['passed'],checks=len(checks),path=str(path)),ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--self-test',action='store_true')
    ap.add_argument('--out',default=SELF_GATE,help='Self-test evidence only')
    ap.add_argument('--mode',choices=('bridge-fork','probe-five'))
    for key in ('registration','plan','experiment-output','observer-independent-gate','fork-independent-gate','zero-gate','bridge-complete'):
        ap.add_argument('--'+key)
    ap.add_argument('--observer-author-gate',default='review/v25_observer_author_final_v1.json')
    ap.add_argument('--fork-author-gate',default='review/v25_forks_author_final_v5.json')
    ap.add_argument('--runtime-gate',default='review/v25_runtime_author_final_v1.json')
    ap.add_argument('--self-gate',default=SELF_GATE)
    args=ap.parse_args()
    if args.self_test:raise SystemExit(self_test(args.out))
    for key in ('mode','registration','plan','experiment_output','observer_independent_gate','fork_independent_gate','zero_gate'):
        if getattr(args,key) is None:ap.error('--'+key.replace('_','-')+' is required for registration')
    prepare(args)
