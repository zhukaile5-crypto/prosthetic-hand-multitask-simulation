"""Independent masks/dispatch/source audit; never launches physics or pins."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'identification/v13_release_factorial_v1'
EXPECTED='c669f8107f5e1d4c5759a186ec6472719115b947ee7475f7aa2ce5ec9f0b6943'
OUT=ROOT/'review/v13_release_factorial_independent_v1.json'
MD=ROOT/'review/v13_release_factorial_independent_v1.md'
ARTIFACTS=ROOT/'review/v13_release_factorial_independent_fixtures_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def exact(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def main():
    if OUT.exists() or MD.exists() or ARTIFACTS.exists():raise FileExistsError('Independent attempts remain immutable')
    if sha(PACKAGE/'run.py')!=EXPECTED:raise ValueError('Only frozenc669 is audited')
    ARTIFACTS.mkdir();checks=[];cases=[];before={}
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    sys.path.insert(0,str(ROOT))
    from identification.v13_release_factorial_v1.inputs import load,reference_program
    from identification.v13_release_factorial_v1.pinning import identity
    sources=identity();old=read(ROOT/'review/v13_release_factorial_author_zero_attempt1.json')['sources'];expected=old.copy();expected['identification/v13_release_factorial_v1/run.py']=EXPECTED
    check('frozen108_only_declared_run_fix',sources==expected and len(sources)==108)
    parent=read(ROOT/'output/development_v13_joint_pilot_v1/manifest.json')['research_source_sha256']
    check('frozen_parent99_unchanged',len(parent)==99 and all(sources[k]==h for k,h in parent.items()))
    # All current original sources are read/hash-checked without changing locks.
    from identification.v13_release_factorial_v1.pinning import files
    for k,p in files().items():before[str(p)]=sha(p);check('source_'+k,before[str(p)]==sources[k])
    lock=PACKAGE/'source_lock.json';check('auditor_did_not_create_lock',not lock.exists())
    r=load();a=r['arrays'];k=2400;entry=np.asarray(json.loads(str(a['observation_json'][k]))['robot_reference21'],float)
    check('entry_at_actual12_incoming',r['release_index']==k and a['time'][k]==12. and exact(r['entry_reference21'],entry))
    check('entry_is_previous_executed21',exact(entry,a['executed_reference21'][k-1]))
    check('entry_is_not_post_first_quarter',not exact(entry,a['executed_reference21'][k]))
    check('incoming_port_is_actual_preaction21',exact(r['entry_engine21'],a['robot_actuator_target21'][k]))
    variants={'both':(True,True),'lift_only':(True,False),'fingers_only':(False,True),'neither':(False,False)}
    for name,(lift,fingers) in variants.items():
        actual=reference_program(r,name);wanted=a['executed_reference21'].copy()
        if not lift:wanted[k:,18]=entry[18]
        if not fingers:wanted[k:,:18]=entry[:18]
        check(name+'_independent_full_program_bits',exact(actual,wanted))
        check(name+'_full16s_finite',actual.shape==(3201,21) and np.isfinite(actual).all())
        check(name+'_common_prefix_all21',exact(actual[:k],a['executed_reference21'][:k]))
        check(name+'_wrist_source_unchanged',exact(actual[:,19:],a['executed_reference21'][:,19:]))
    check('complete_fixed_environment',len(r['environments'])==3201 and all(set(x)=={'force','torque','squeeze','obstacle_x','friction'} for x in r['environments']))
    check('recorded_actual_physics_config',r['metadata']['config']['timestep']==.0005 and r['metadata']['sensor_period']==.005 and r['metadata']['duration']==16.)
    before[str(r['path'])]=sha(r['path']);before[str(r['folder']/'metadata.json')]=sha(r['folder']/'metadata.json')
    env=read(ROOT/'output/development_v13_joint_pilot_v1/environment.json')['pinned_environment']
    check('same_python',str(Path(sys.executable).resolve())==str(Path(env['executable']).resolve()) and sha(sys.executable)==env['executable_sha256'])
    check('42_native_binaries_unchanged',len(env['native_binaries_sha256'])==42 and all(sha(p)==h for p,h in env['native_binaries_sha256'].items()))
    source=PACKAGE/'run.py';tree=ast.parse(source.read_text(encoding='utf-8-sig'));part=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('execute','write')],type_ignores=[])
    for name in ('snapshot_failure','pool_creation_failure','partial_submit_failure','worker_failure','normal_bridge','normal_factorial','failed_bridge_pass_flag','failed_bridge_rng_flag'):
        fixture_root=ARTIFACTS/name;fixture_root.mkdir();target=fixture_root/'output/attempt'
        bridge=dict(passed=True,execution_complete=True,sources_unchanged=True,identical_noise_rng_final=True,both_bridge_all_recomputed_exact=True,source_lock_sha256='LOCK',results=[dict(noise_rng_final='SAME')])
        if name=='failed_bridge_pass_flag':bridge['passed']=False
        if name=='failed_bridge_rng_flag':bridge['identical_noise_rng_final']=False
        class Future:
            def __init__(self,variant):self.variant=variant
            def result(self):
                if name=='worker_failure':raise RuntimeError('ORIGINAL_WORKER_FAILURE')
                return dict(variant=self.variant,complete=True,noise_rng_final='SAME',bridge_all_recomputed_exact=True)
        class Pool:
            def __init__(self,*args,**kw):
                self.submitted=0
                if name=='pool_creation_failure':raise OSError('ORIGINAL_POOL_FAILURE')
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def submit(self,worker,variant,destination):
                self.submitted+=1
                if name=='partial_submit_failure' and self.submitted==2:raise OSError('ORIGINAL_SUBMIT_FAILURE')
                return Future(variant)
        def snapshot(*args):
            if name=='snapshot_failure':raise OSError('ORIGINAL_SNAPSHOT_FAILURE')
        ns=dict(Path=Path,ROOT=fixture_root,load=lambda:{},require=lambda r:{'input':{}},identity=lambda:{'fixture':'UNCHANGED'},sha=lambda p:'LOCK',LOCK=Path('lock'),snapshot=snapshot,
            ProcessPoolExecutor=Pool,as_completed=lambda fs:list(fs),worker=lambda *x:None,read=lambda p:bridge,time=time,json=json,traceback=traceback)
        exec(compile(part,str(source),'exec'),ns)
        factorial=name in ('partial_submit_failure','normal_factorial','failed_bridge_pass_flag','failed_bridge_rng_flag')
        try:result=ns['execute'](SimpleNamespace(phase='factorial' if factorial else 'bridge',bridge=str(fixture_root/'bridge'),out=str(target),workers=1));exception=None
        except Exception as e:result=None;exception=dict(type=type(e).__name__,error=str(e))
        ledger=read(target/'assignment_ledger.json') if (target/'assignment_ledger.json').exists() else None
        if name.startswith('failed_bridge_'):
            check(name+'_refused_before_dispatch',exception is not None and exception['type']=='ValueError' and not target.exists())
        else:
            check(name+'_completion_and_assigned_ledger_retained',exception is None and (target/'execution_complete.json').is_file() and ledger is not None and len(ledger)==(3 if factorial else 1))
            if name in ('snapshot_failure','pool_creation_failure','partial_submit_failure','worker_failure'):
                check(name+'_false_completion_and_pass',not result['execution_complete'] and not result['passed'])
                if name=='worker_failure':check(name+'_original_worker_error_retained','ORIGINAL_WORKER_FAILURE' in ledger[0]['error'] and ledger[0]['status']=='runtime_failed')
                elif name=='partial_submit_failure':check(name+'_submitted_success_and_unexecuted_retained',ledger[0]['status']=='completed' and all(x['status']=='not_executed_due_to_snapshot_or_dispatch_failure' for x in ledger[1:]) and len(result['results'])==1)
                else:check(name+'_unexecuted_not_fabricated',not result['results'] and ledger[0]['status']=='not_executed_due_to_snapshot_or_dispatch_failure')
            else:check(name+'_good_normal_pass',result['passed'] and result['execution_complete'])
        cases.append(dict(name=name,result=result,exception=exception,assignment_ledger=ledger))
    replay=ast.parse((PACKAGE/'replay.py').read_text(encoding='utf-8-sig'));text=(PACKAGE/'replay.py').read_text(encoding='utf-8-sig')
    check('recorder_has_no_forward_or_step_calls',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip') for n in ast.walk(replay)))
    check('run_job_keeps_partial_save_error',"target/'partial_trace.npz'" in text and "result['partial_save_error']" in text and 'traceback.format_exc()' in text)
    check('source_provenance_excluded_from_exact',"values['source_'+key]=a[key]" in text and "required=set(TRUTH_FIELDS+SENSOR_FIELDS+COMMAND_FIELDS)" in text and "'source_'" not in text.split('required=set',1)[1].split('\n',1)[0])
    check('zero_physics_no_simulator_import','mujoco' not in sys.modules)
    for path,h in before.items():check('unchanged_'+path,sha(path)==h)
    check('final_run_identity',sha(source)==EXPECTED)
    result=dict(schema='v13_release_factorial_independent_prephysics_v1',passed=all(x['passed'] for x in checks),check_count=len(checks),failures=[x for x in checks if not x['passed']],checks=checks,cases=cases,
        source_sha256=sources,checker_sha256=sha(__file__),input_sha256=before,source_entry_reference21=entry.tolist(),
        old_failure_evidence=str(ROOT/'review/v13_release_factorial_dispatch_old18168_v1/audit.json'),old_failure_sha256=sha(ROOT/'review/v13_release_factorial_dispatch_old18168_v1/audit.json'),
        integration_calls=0,workers_spawned=0,plant_constructions=0,source_lock_created=False,
        classification=dict(independent_outputs='fresh q/v, positions/rotation, raw and public all-skin forces/sensors, actual encoded21/22 and clocks; bridge must verify these plus every original integration force column',
            input_bookkeeping='alpha/alpha_before/phase/command and fixed environment labels are replayed instrument/input memory; exact matching is an input check, not independently recomputed policy',
            unrecomputed_policy='source_* desired/observer/decision/timing/outcome provenance excluded from exact certification'),
        boundary='Prephysics software/source review only. Reset does not restore unsaved original warmstart; root must run exact both bridge, and failed bridge blocks remaining3. No claim of numerical convergence/contact causation/performance from these tests. Source final RNG absent: source3201 public packet exact plus new4 branch final RNG agreement required.')
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V13释放2×2固定输入诊断独立预检','',f"冻结run c669：{len(checks)}项，失败{len(result['failures'])}项。只读来源、纯掩码与隔离派发器故障注入；实际plant构造0、worker启动0、积分0，未自行pin。",'',
        '108来源含原99与新包9，除登记的run故障修复外与作者首次108映射一致。固定源为单个已见fusion nominal/hold/617901，不是新算法或多独立条件。',
        '', '四种程序在12s之前21参考完全相同；held finger/lift使用12s incoming公共robot_reference21，等于11.995s最后执行值，确实不同于12s首quarter输出。18finger与lift18独立掩码，wrist19:21始终源程序。全16s/3201控制时刻和同物参/完整环境/设备噪声合同不变。',
        '', '旧18168的snapshot/pool失败无终态账本及未严核bridge passed/RNG已用三个纯见证保存。c669新增snapshot、pool、部分submit、worker失败均完整留分配与原错误；已经提交的成功不会因后续submit失败丢失，也不补造未执行结果。passed或RNG为false的bridge在派发前拒绝；正常bridge/factorial的纯分支均通过。',
        '', '录制_contacts只读取既存solver/contact数据，不增加forward或step；raw力START+duration为END。仍继承原5ms boundary forward，因此200Hz力不是raw的简单下采样。物理是否完全不受观测读取影响必须由root的both全输入桥验证，纯静态检查不能代替它。',
        '', 'exact分类必须明确：alpha/alpha_before/phase/command是源程序/仪器记账，environment是固定输入，不能计作独立新策略重算；真正q/v/受力/公共传感/端口/clock和原substep是mandatory。source_*策略/observer/计时/任务标签明确不计exact，不使用复制标签伪造全列通过。',
        '', '原source最终RNG未保存，不能承诺其内部状态逐字节同；要求source3201公共packet exact和四个新分支最终RNG一致。当前尚未物理bridge，未给接触分离、力安全或因果收益结论。包8其他源保持不动；root应在相同108/42binary锁下先both，通过后才另3。']
    if result['failures']:lines+=['','## 未通过项','',*['- '+x['name']+': '+str(x['detail']) for x in result['failures']]]
    MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=result['failures'],evidence=str(OUT)),ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
