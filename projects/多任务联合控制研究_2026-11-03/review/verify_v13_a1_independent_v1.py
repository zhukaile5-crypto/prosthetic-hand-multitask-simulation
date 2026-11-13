"""Independent A1 archived-command bridge and typed fault reader, zero dynamics.

Execution requires explicit final candidate hashes. This file does not alter
candidate code, old outputs, source80/source88, or any physical state trajectory.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,hashlib,json,sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
METHODS=('ratio_effort','fixed_effort','fusion_effort')
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def cls(path,name):return next(n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name==name)
def fn(c,name):return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)


def static_contract(check):
    base=cls(FOUNDATION/'session.py','TrialSession');boot=cls(ROOT/'v13_a1/bootstrap.py','A1Bootstrap')
    class FactoryOnly(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id=='A1Plant':n.id='HandPlant'
            return n
    check('bootstrap_only_factory_change',dump(FactoryOnly().visit(copy.deepcopy(fn(boot,'__init__'))))==dump(fn(base,'__init__')))
    bootcalls=[n for n in ast.walk(fn(boot,'__init__')) if isinstance(n,ast.Call)]
    check('bootstrap_no_old_plant_or_super_constructor',not any((isinstance(n.func,ast.Name) and n.func.id=='HandPlant') or
        (isinstance(n.func,ast.Attribute) and n.func.attr=='__init__') for n in bootcalls))
    new=cls(ROOT/'v13_a1/runtime.py','V13A1Session');old=cls(ROOT/'v12_native/runtime.py','V12NativeSession')
    check('runtime_base_is_A1Bootstrap',[ast.unparse(n) for n in new.bases]==['A1Bootstrap'])
    for name in ('__init__','_handoff','_public_task','_lifecycle_step','step_one_sensor_tick'):
        check('runtime_body_exact/'+name,dump(fn(new,name))==dump(fn(old,name)))
    plant=cls(ROOT/'v13_a1/plant.py','A1Plant');calls=[n for n in ast.walk(fn(plant,'_apply_ctrl')) if isinstance(n,ast.Call)]
    check('A1_override_does_not_call_old_writer',not any(isinstance(n.func,ast.Attribute) and n.func.attr=='_apply_ctrl' for n in calls))
    writes=[]
    for node in ast.walk(fn(plant,'_apply_ctrl')):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Subscript) and ast.unparse(target.value)=='self.data.ctrl':writes.append(ast.unparse(target.slice))
    check('A1_direct_nonrobot_write_only',writes==['self.obstacle_motor'])
    proposals=ast.parse((ROOT/'v13_a1/proposals.py').read_text(encoding='utf-8'))
    check('proposal_has_no_engine_buffer_reference',not any(isinstance(n,ast.Attribute) and n.attr in ('data','ctrl') for n in ast.walk(proposals)))


def jobs_for(cohort):
    if cohort=='old':
        reg=read(ROOT/'review/v12_multiscenario_development_registration_v1.json')
        batches=[ROOT/'output/development_v12_pilot_v1']+[ROOT/g['output'] for g in reg['groups']]
    else:
        reg=read(ROOT/'review/v12_native_development_registration_v1.json');batches=[ROOT/g['output'] for g in reg['groups']]
    jobs=[]
    for batch in dict.fromkeys(batches):
        final=read(batch/'execution_complete.json');manifest=read(batch/'manifest.json')
        if final['execution_complete'] is not True or final.get('sources_unchanged') is not True:raise ValueError('Nonterminal/changed source archive')
        if sha(batch/'trials.json')!=final['trials_sha256']:raise ValueError('Archive terminal ledger changed')
        jobs.extend((batch,j) for j in manifest['jobs'])
    if len(jobs)!=48 or len({(j['scenario'],j['motion_mode'],j['seed'],j['method']) for b,j in jobs})!=48:raise ValueError('Exact48 archive required')
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--codec-sha',required=True);p.add_argument('--proposals-sha',required=True);p.add_argument('--plant-sha',required=True)
    p.add_argument('--out',type=Path,default=ROOT/'review/v13_a1_independent_v1.json');args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    checks=[];issues=[];rows=[];start=time.perf_counter();integration_calls=[]
    watch=[ROOT/'v13_a1'/n for n in ('codec.py','proposals.py','plant.py','bootstrap.py','runtime.py')]
    before={str(p):sha(p) for p in watch}
    def check(name,ok,detail=None):
        item=dict(name=name,passed=bool(ok),detail=detail);checks.append(item)
        if not ok:issues.append(item)
    def reject(name,call):
        try:call()
        except (ValueError,TypeError):check(name,True)
        else:check(name,False,'Invalid input was accepted')
    report=dict(schema='v13_a1_independent_archived_commands_and_faults_v1',checks=checks,issues=issues,rows=rows,
        source_sha256=before,new_physics_steps=0,closed_loop_equivalence_proven=False)
    try:
        root_zero=ROOT/'review/v13_a1_runtime_zero_root_v1.json';zero=read(root_zero);source87=zero['source_sha256']
        def location(key):return FOUNDATION/key.removeprefix('foundation/') if key.startswith('foundation/') else ROOT/key
        check('root_frozen87_identity',zero['passed'] is True and len(source87)==87 and
            all(sha(location(k))==v for k,v in source87.items()))
        report.update(execution_source87_sha256=source87,source87_root_evidence_path=str(root_zero),source87_root_evidence_sha256=sha(root_zero))
        for name,expected in [('codec.py',args.codec_sha),('proposals.py',args.proposals_sha),('plant.py',args.plant_sha)]:
            check('final_candidate/'+name,sha(ROOT/'v13_a1'/name)==expected)
        if issues:raise ValueError('Expected final source identity required before candidate import')
        static_contract(check)
        import research_imports,mujoco
        for name in ('mj_step','mj_step1','mj_step2','mj_forward'):
            def forbidden(*a,_name=name,**kw):integration_calls.append(_name);raise AssertionError('No dynamics/forward allowed '+_name)
            setattr(mujoco,name,forbidden)
        from foundation_config import FoundationConfig,GROUPS,OPEN,CLOSED
        import hand_plant
        hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No plant construction in archived codec check'))
        from v13_a1.codec import Reference21,EnginePort21,ReferenceActor21,REFERENCE_AXES21,ACTUATORS21
        from v13_a1.proposals import NativeProposalContext,NativeControlState,build_native_proposal
        model,_=hand_plant._model(FoundationConfig())
        actor=ReferenceActor21(model)
        robot_ids=np.asarray([model.actuator_ctrladr[model.actuator(n).id] for n in ACTUATORS21],int)
        groups=[np.asarray([model.actuator_ctrladr[model.actuator('rh_A_'+n).id] for n in g],int) for g in GROUPS]
        context=NativeProposalContext(OPEN,CLOSED,tuple(model.actuator_ctrlrange[g] for g in groups),.8)
        bias=float(model.body_subtreemass[model.body('rh_forearm').id]*9.81/1500.)
        value=np.zeros(21);ref=Reference21(value,.1);value[0]=1.
        check('typed_reference_copies_caller_array',ref.values[0]==0. and not ref.values.flags.writeable)
        encoded=actor.encode(ref);check('lift_bias_applied_once',encoded.values[18]==bias and actor.lift_bias_m==bias)
        reject('engine_cannot_be_encoded_again',lambda:actor.encode(encoded))
        reject('array_not_an_explicit_reference',lambda:actor.encode(np.zeros(21)))
        reject('reference_wrong_axes',lambda:Reference21(np.zeros(21),.1,axes=ACTUATORS21))
        reject('engine_wrong_axes',lambda:EnginePort21(np.zeros(21),.1,bias,axes=REFERENCE_AXES21))
        reject('unknown_control_address_order',lambda:ReferenceActor21(model,robot_indices21=robot_ids[::-1]))
        reject('wrong_lift_bias_rejected',lambda:ReferenceActor21(model,lift_bias_m=2*bias))
        reject('negative_reference_clock',lambda:Reference21(np.zeros(21),-.005))
        reject('negative_engine_clock',lambda:EnginePort21(np.zeros(21),-.005,bias))
        reject('NaN_reference',lambda:Reference21(np.full(21,np.nan),.1))
        reject('numeric_strings_rejected',lambda:Reference21(['0']*21,.1))
        reject('numpy_bool_rejected',lambda:Reference21([np.bool_(False)]+[0.]*20,.1))
        reject('q25_not_a_reference21',lambda:Reference21(np.zeros(25),.1))
        reject('negative_proposal_clock',lambda:NativeControlState(-.005,np.zeros(5),0.,np.zeros(2),0.))
        reject('negative_running_max_clip',lambda:NativeControlState(.1,np.zeros(5),0.,np.zeros(2),-.1))
        reject('release_origin_in_future',lambda:NativeControlState(.1,np.zeros(5),0.,np.zeros(2),0.,.2,np.zeros(5),np.zeros(5),0.))
        raw=np.zeros(21);raw[0]=100.;port=actor.encode(Reference21(raw,.1));data=SimpleNamespace(ctrl=np.full(22,-999.))
        actor.apply(data,Reference21(raw,.1))
        check('actor_does_not_add_new_clip',port.values[0]==100. and data.ctrl[robot_ids[0]]==100. and actor.last_diagnostics['raw_command_outside_native_ctrlrange21'][0])
        nonrobot=sorted(set(range(22))-set(robot_ids));check('actor_does_not_write_obstacle',len(nonrobot)==1 and data.ctrl[nonrobot[0]]==-999.)
        # Full actual streams use original snapshots/commands as the independent
        # oracle. No policy, observer, force model or object state is invoked.
        for cohort in ('old','native'):
            for batch,job in jobs_for(cohort):
                folder=Path(job['destination']);path=folder/'trace.npz';meta=read(folder/'metadata.json')
                check(cohort+'/'+folder.name+'/trace_unchanged',sha(path)==meta['trace_sha256'])
                with np.load(path,allow_pickle=False) as z:
                    a={k:z[k] for k in ('time','alpha_before','alpha','ctrl_before','ctrl','executed_motion_reference','robot_actuator_target21','phase')}
                release=float(meta['mechanical_release_at']);entry=int(np.argmin(abs(a['time']-release)))
                bad=[];checked=0;first=None
                check(cohort+'/'+folder.name+'/3201_complete_clock',exact(a['time'],np.arange(3201,dtype=float)*.005))
                check(cohort+'/'+folder.name+'/actual21_named_order',exact(a['ctrl_before'][:,robot_ids],a['robot_actuator_target21']))
                check(cohort+'/'+folder.name+'/same_public_release_rate',meta['config']['release_open_rate']==.8)
                for i,t in enumerate(a['time']):
                    active=i>=entry
                    state=NativeControlState(float(t),a['alpha'][i],float(a['executed_motion_reference'][i,0]),a['executed_motion_reference'][i,1:],0.,
                        release if active else None,a['ctrl_before'][entry,groups[0]] if active else None,
                        a['alpha_before'][entry] if active else None,float(a['executed_motion_reference'][entry,0]))
                    proposal=build_native_proposal(context,state);actual=actor.encode(proposal.reference21)
                    ok=(exact(actual.values,a['ctrl'][i,robot_ids]) and exact(proposal.alpha_after5,a['alpha'][i]) and
                        proposal.lift_ref_after==a['executed_motion_reference'][i,0] and proposal.reference21.time_s==float(t))
                    if not ok:
                        bad.append(i)
                        if first is None:first=dict(index=i,time_s=float(t),phase=str(a['phase'][i]),
                            max_port_error=float(np.max(abs(actual.values-a['ctrl'][i,robot_ids]))),actual=actual.values.tolist(),expected=a['ctrl'][i,robot_ids].tolist())
                    checked+=1
                check(cohort+'/'+folder.name+'/full21_command_and_native_sideeffects_bytes',not bad,first)
                rows.append(dict(cohort=cohort,job=job,trace_sha256=meta['trace_sha256'],frames=checked,
                    release_frames=len(a['time'])-entry,mismatched_indices=bad,first_mismatch=first))
                if len(rows)%6==0:print(json.dumps(dict(records_checked=len(rows),planned=96,issues=len(issues))),flush=True)
                del a
        check('full96_streams_and_frames',len(rows)==96 and sum(r['frames'] for r in rows)==96*3201)
    except Exception as exc:
        check('completed_without_exception',False,str(exc));report['exception']=traceback.format_exc()
    check('zero_dynamics_or_forward_calls',not integration_calls,integration_calls)
    check('candidate_sources_unchanged',before=={str(p):sha(p) for p in watch})
    if 'execution_source87_sha256' in report:
        check('all_frozen87_sources_unchanged',all(sha(location(k))==v for k,v in report['execution_source87_sha256'].items()))
    report.update(passed=not issues,wall_seconds=time.perf_counter()-start,check_count=len(checks),
        complete_records=len(rows),limitations=['Archived command/typed software bridge only; root nine-run physical bridge remains necessary.',
            'Actual public references and native release semantics are replayed, not a new force strategy.',
            'Only named robot controls are supplied to the actor; no object q/pose/mass/friction enters these codec calls.'])
    with args.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=not issues,checks=len(checks),issues=len(issues),output=str(args.out),sha256=sha(args.out),new_physics_steps=0)))
    return 0 if not issues else 2


if __name__=='__main__':raise SystemExit(main())
