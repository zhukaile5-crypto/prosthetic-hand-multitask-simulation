"""Independent exact-public-domain audit, no goal solve or physics.

Compares all public fields, including model_signature and signed zeros, to the
unchanged5b PreparedValue domain. No exclusions and no author helper imported.
"""
from pathlib import Path
from copy import copy
import ast,hashlib,json,struct,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='254b4a95dbaa7557bd7e767b5db7692b26287c1dfcf898341df655dbf91d5060'
BASE='5b95ec9f1614415ec75b91ecb02629f4316d844960b6111932ffb8d1c76f81ac'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bits(v):
    if isinstance(v,np.ndarray):return b'A'+str(v.dtype).encode()+repr(v.shape).encode()+v.tobytes()
    if isinstance(v,(float,np.floating)):return b'F'+struct.pack('<d',float(v))
    if isinstance(v,(bool,np.bool_)):return b'B'+bytes([bool(v)])
    if isinstance(v,(int,np.integer)):return b'I'+str(int(v)).encode()+b';'
    if v is None:return b'N'
    if isinstance(v,str):return b'S'+str(len(v.encode())).encode()+b':'+v.encode()
    if isinstance(v,dict):return b'D'+b''.join(bits(k)+bits(x) for k,x in v.items())
    if isinstance(v,(tuple,list)):return b'L'+b''.join(bits(x) for x in v)
    raise TypeError(type(v))
def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v
def main():
    out=ROOT/'review/v14_fast_domain_independent_v1.json';md=ROOT/'review/v14_fast_domain_independent_v1.md'
    if out.exists() or md.exists():raise FileExistsError('Fresh independent evidence required')
    source=ROOT/'v14_coord/effort_domain.py';base_source=ROOT/'v13_joint/prepared_value_effort21.py'
    if sha(source)!=EXPECTED or sha(base_source)!=BASE:raise ValueError('Frozen candidate or baseline identity mismatch')
    before={str(p):sha(p) for p in [source,base_source,ROOT/'v14_coord/effort_realization.py',ROOT/'v13_joint/native_effort21.py',ROOT/'v13_joint/prepared_native_effort21.py',ROOT/'v11/robot_servo_effort.py',ROOT/'v13_a1/codec.py']}
    checks=[];records=[];calls={};constructors=0;solves=0;public_snapshots=[]
    def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=clean(detail)))
    def domain_ast(path,cls):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
        fn=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='build_domain')
        body=next(x for x in fn.body if isinstance(x,ast.Try)).body
        def assigns_name(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        a=next(i for i,x in enumerate(body) if assigns_name(x,'anchor'));b=next(i for i,x in enumerate(body) if assigns_name(x,'context'))
        body[a:b]=[]
        return ast.dump(fn,include_attributes=False)
    check('all_checks_except_endpoint_strategy_AST_identical',domain_ast(source,'FastEffortGoal21')==domain_ast(ROOT/'v13_joint/native_effort21.py','NativeEffort21'))
    sys.path.insert(0,str(ROOT));import research_imports,mujoco,hand_plant
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            calls[name]=0
            def forbidden(*args,_name=name,**kw):calls[_name]+=1;raise AssertionError('Forbidden '+_name)
            setattr(mujoco,name,forbidden)
    def reject_plant(*args,**kw):
        nonlocal constructors
        constructors+=1;raise AssertionError('No actual plant construction')
    hand_plant.HandPlant.__init__=reject_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import Reference21,EnginePort21
    from v13_joint.prepared_value_effort21 import PreparedValueEffort21
    from v14_coord.effort_domain import FastEffortGoal21
    model,_=hand_plant._model(FoundationConfig())
    base=PreparedValueEffort21(model);fast=FastEffortGoal21(model)
    def never_solve(*args,**kwargs):
        nonlocal solves
        solves+=1;raise AssertionError('Domain-only audit never calls goal/inverse solve')
    base.solve=fast.solve=never_solve
    jobs=read(ROOT/'output/v13_a1_native_behavior_bridge_v1/manifest.json')['jobs']
    done=read(ROOT/'output/v13_a1_native_behavior_bridge_v1/execution_complete.json')
    check('complete9_public_archives',done['execution_complete'] and done['all9_exact'] and len(jobs)==9)
    packets=[]
    for index,job in enumerate(jobs):
        p=Path(job['destination'])/'trace.npz';before[str(p)]=sha(p);meta=read(p.with_name('metadata.json'))
        check('public_packet_trace_identity_'+str(index),before[str(p)]==meta['trace_sha256'] and meta['completed'])
        with np.load(p,allow_pickle=False) as z:
            ts=z['time'].copy();q=z['qpos'][:,base.servo.qadr].copy();v=z['qvel'][:,base.servo.dadr].copy();control=z['robot_actuator_target21'].copy()
        when=(7.5,9.435,12.21)[index%3];i=int(np.argmin(abs(ts-when)))
        ref=base.actor.decode(EnginePort21(control[i],float(ts[i]),base.actor.lift_bias_m))
        packets.append(dict(q=q[i],v=v[i],ref=ref,source=str(p),source_sha256=before[str(p)],identity={k:job[k] for k in ('scenario','method','seed','motion_mode')}))
    def build(core,p,lower,upper,nominal=None,extra=None):
        current=p['ref'];kw=dict(q25=p['q'],v25=p['v'],robot_axes25=core.robot_axes25,time_s=current.time_s,current_reference=current,
            nominal_reference=current if nominal is None else nominal,actual_engine=core.actor.encode(current),reference_lower21=lower,reference_upper21=upper)
        if extra:kw.update(extra)
        return core.build_domain(**kw)
    def compare(name,b,f,p,lower,upper,nominal=None,extra=None,expected_available=None):
        begin=time.perf_counter_ns();db=build(b,p,lower,upper,nominal,extra);tb=(time.perf_counter_ns()-begin)/1e6
        begin=time.perf_counter_ns();df=build(f,p,lower,upper,nominal,extra);tf=(time.perf_counter_ns()-begin)/1e6
        pb=db.to_dict();pf=df.to_dict()
        check(name+'_same_public_field_set',list(pb)==list(pf))
        for k in pb:check(name+'_field_'+k,bits(pb[k])==bits(pf[k]))
        check(name+'_all_public_bytes_NO_EXCLUSIONS',bits(pb)==bits(pf))
        if expected_available is not None:check(name+'_expected_availability',db.available==df.available==expected_available,dict(base=db.reason,fast=df.reason))
        check(name+'_private_owner_separate',db._owner is b._owner and df._owner is f._owner and b._owner is not f._owner)
        if db.available and df.available:
            check(name+'_protected_input_checksum_equal',db._signature==df._signature)
            for k in ('q','v','current','nominal','actual','anchor','supports'):check(name+'_context_'+k,bits(db._context[k])==bits(df._context[k]))
            check(name+'_immutable_public_numeric_arrays',all(not getattr(df,k).flags.writeable for k in ('p0_effort21','effort_lower21','effort_upper21','reference_lower21','reference_upper21','transmission21x25')))
        records.append(dict(name=name,status=db.status,reason=db.reason,available=db.available,public_digest=hashlib.sha256(bits(pb)).hexdigest(),
            baseline_build_ms=tb,fast_build_ms=tf,source=p.get('source'),identity=p.get('identity'),time_s=p['ref'].time_s))
        return db,df
    for index,p in enumerate(packets):
        delta=np.r_[np.full(18,.005),.00003,.00015,.00015];r=p['ref'].values
        compare('public_'+str(index),base,fast,p,r-delta,r+delta,expected_available=True)
    p=packets[0];r=p['ref'].values
    nominal=r.copy();nominal[0]+=.3
    db,df=compare('raw_nominal_outside_proposal',base,fast,p,r-.001,r+.001,Reference21(nominal,p['ref'].time_s),expected_available=True)
    check('raw_nominal_not_retargeted_to_permission',df.p0_effort21[0]>df.effort_upper21[0] or df.p0_effort21[0]<df.effort_lower21[0])
    compare('step_and_absolute_intersection',base,fast,p,r-.01,r+.01,extra=dict(step_lower21=-np.ones(21)*.0003,step_upper21=np.ones(21)*.0004),expected_available=True)
    compare('step_only_domain',base,fast,p,None,None,extra=dict(step_lower21=-np.ones(21)*.0003,step_upper21=np.ones(21)*.0004),expected_available=True)
    compare('singleton_bounds',base,fast,p,r.copy(),r.copy(),expected_available=True)
    for mode in ('negative_gain','zero_gain','force_clip','signed_zero_efforts'):
        altered=copy(model);aid=base.servo.aids[0];outadr=base.servo.outadr[0]
        values=r.copy();lower=values-.001;upper=values+.001
        if mode in ('negative_gain','zero_gain'):
            altered.actuator_gainprm[aid,0]=-1. if mode=='negative_gain' else 0.;altered.actuator_biasprm[aid,0]=.07
        elif mode=='force_clip':altered.actuator_forcerange[outadr]=[-.0001,.0001]
        else:
            altered.actuator_gainprm[aid,:3]=[-0.,0.,0.];altered.actuator_biasprm[aid,:3]=[-0.,-0.,-0.]
        b=PreparedValueEffort21(altered);f=FastEffortGoal21(altered)
        pp=dict(p)
        if mode=='signed_zero_efforts':
            pp['q']=np.zeros(25);pp['v']=np.zeros(25);values=np.ones(21)*.01;values[19]=np.float64(-0.);values[20]=0.;pp['ref']=Reference21(values,p['ref'].time_s);lower=values-.001;upper=values+.001
        x,y=compare(mode,b,f,pp,lower,upper,expected_available=True)
        if mode=='negative_gain':check('negative_gain_interval_reordered',f.servo.evaluate(pp['q'],pp['v'],f.actor.encode(Reference21(lower,pp['ref'].time_s)).values).predicted_effort21[0]>=f.servo.evaluate(pp['q'],pp['v'],f.actor.encode(Reference21(upper,pp['ref'].time_s)).values).predicted_effort21[0])
        if mode=='zero_gain':check('flat_interval_present',y.effort_lower21[0]==y.effort_upper21[0])
        if mode=='force_clip':check('actual_force_plateau_tested',y.effort_lower21[0]==y.effort_upper21[0] and y.diagnostics['current_native_clipped21'][0])
        if mode=='signed_zero_efforts':check('negative_zero_endpoint_bits_actually_tested',np.signbit(y.effort_lower21[0]) and np.signbit(y.effort_upper21[0]) and y.effort_lower21[0]==0.)
    invalid=[('q_unknown',dict(q25=None)),('v_nonfinite',dict(v25=np.full(25,np.nan))),('q_boolean',dict(q25=np.zeros(25,bool))),
        ('time_negative',dict(time_s=-1.)),('time_boolean',dict(time_s=True)),('clock_mismatch',dict(time_s=p['ref'].time_s+.005)),
        ('reference_unknown',dict(current_reference=None)),('raw_reference_not_typed',dict(current_reference=r)),('engine_not_typed',dict(actual_engine=np.ones(21))),
        ('reordered_axes',dict(robot_axes25=base.robot_axes25[::-1])),('missing_upper',dict(reference_upper21=None)),
        ('no_explicit_domain',dict(reference_lower21=None,reference_upper21=None)),('empty_steps',dict(step_lower21=np.ones(21),step_upper21=-np.ones(21))),
        ('coupled_domain_unknown',dict(coupled_reference_constraints={'unsupported':True}))]
    for name,extra in invalid:compare(name,base,fast,p,r-.001,r+.001,extra=extra,expected_available=False)
    compare('empty_intersection',base,fast,p,r+.1,r-.1,expected_available=False)
    bad=copy(model);bad.actuator_dyntype[base.servo.aids[0]]=mujoco.mjtDyn.mjDYN_FILTER
    compare('initial_unsupported_activation',PreparedValueEffort21(bad),FastEffortGoal21(bad),p,r-.001,r+.001,expected_available=False)
    shared=copy(model);shared.actuator_trnid[base.servo.aids[1]]=shared.actuator_trnid[base.servo.aids[0]]
    compare('overlapping_generalized_supports',PreparedValueEffort21(shared),FastEffortGoal21(shared),p,r-.001,r+.001,expected_available=False)
    changed=copy(model);changed.opt.disableactuator=1<<5;b=PreparedValueEffort21(changed);f=FastEffortGoal21(changed)
    compare('before_group_mutation',b,f,p,r-.001,r+.001,expected_available=True)
    b.servo.model.actuator_group[b.servo.aids[0]]=5;f.servo.model.actuator_group[f.servo.aids[0]]=5
    compare('after_group_mutation_rejected',b,f,p,r-.001,r+.001,expected_available=False)
    check('no_forbidden_pipeline_calls',all(v==0 for v in calls.values()),calls);check('no_actual_plant',constructors==0);check('no_solve_calls',solves==0)
    check('no_robot_writer',base.actor.write_count==fast.actor.write_count==0)
    for path,h in before.items():check('unchanged_'+path,sha(path)==h)
    trows=[r for r in records if r['name'].startswith('public_')]
    def timing(key):
        a=np.asarray([r[key] for r in trows]);return dict(n=len(a),median=float(np.median(a)),minimum=float(min(a)),maximum=float(max(a)))
    result=dict(schema='v14_fast_domain_independent_v1',passed=all(c['passed'] for c in checks),check_count=len(checks),failures=[c for c in checks if not c['passed']],
        source_sha256=before,checker_sha256=sha(__file__),comparison='ALL EffortDomain21.to_dict public fields byte-equal to frozen PreparedValue5b, no excluded fingerprint or other fields',
        case_count=len(records),public_archive_packets=9,records=records,checks=checks,forbidden_calls=calls,actual_plant_constructions=constructors,goal_or_inverse_solve_calls=solves,
        timing=dict(baseline=timing('baseline_build_ms'),fast=timing('fast_build_ms'),scope='nine direct domain-only calls on public archived q/v; not controller/IO/physics or hard realtime'),
        boundary='fixed q/v native independent command map only; no future effort/pressure prediction, contact stiffness, world kinematics or global safety claim')
    out.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V14 两端点努力域独立核查','',f"结果：{len(checks)} 项检查，失败 {len(result['failures'])} 项；{len(records)} 个对照case，9个既存公开机器人包。",'',
        f'新源 SHA256 `{EXPECTED}`；原PreparedValue5b作为比较基准，全部公开字段（含model_signature）逐字节核对，未剔除任何字段。',
        '', '在固定q/v的已知原生执行器中，各motor只使用自己的参考、固定length/speed、自己的control/force clipping；原domain也禁止广义支持重叠与非box域。故对完整lower/upper向量各评估一次，可得到原逐axis锚上替换端点的同一21个区间。负gain使用排序，flat/±0同值保留首端点；该等价不适用于未来q变化或接触力预测。',
        '', 'AST独立比较确认：除anchor至context之间的端点评估策略外，原6f build_domain全部校验、界限交集、原p0、原clock/source/owner/checksum构造代码一致。实际API比较针对继承强guard的5b，包含负gain、flat、nativeforce平台、原nominal越proposal、signedzero、输入unknown/非法类型、机械域交集、初始activation不支持、重叠支持及group修改拒绝。',
        '',f"直接domain耗时：基准median {result['timing']['baseline']['median']:.6g}ms，新版median {result['timing']['fast']['median']:.6g}ms；范围与原包列于JSON。仅九个私有计算调用样本，不是20ms硬实时证书。",'',
        '14个step/forward/FK/contact入口全0调用；实际plant构造0、goal/inverse solve0、writer0。只编译公开nominal模型并在私有机器人数据上计算tendon/transmission。所有原源/档案前后SHA不变。新goal realizer的求解语义由另一份独审覆盖，本报告不替代该范围。']
    if result['failures']:lines+=['','## 未通过项','',*['- '+x['name']+': '+str(x['detail']) for x in result['failures']]]
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),cases=len(records),failures=result['failures'][:10],timing=result['timing'],out=str(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
