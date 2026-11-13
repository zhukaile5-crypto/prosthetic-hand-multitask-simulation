"""CompleteV14 pilot physics/metric audit; no solver/native/physics replay."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT))
from review.verify_v13_joint_pilot_independent_v1 import exact,json_equal,summary,stats
PLAN=ROOT/'review/v14_coherent_pilot_plan_v1.json'
OUT=ROOT/'review/v14_coherent_pilot_independent_v1.json'
MD=ROOT/'review/v14_coherent_pilot_independent_v1.md'
REGISTRATION_SHA='566df3891908fc610a4ec0cf5232d2ff4cfdb279ee0d24e1bf77136672410af4'
PREP=('normal','skin_normal','relative','position','raw_patches_json','legacy_common_sensor_json','environment_json','robot_actuator_target21','robot_control_time_s')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    if OUT.exists() or MD.exists():raise FileExistsError('Fresh independent evidence required')
    plan=read(PLAN);folder=ROOT/plan['output'];end=read(folder/'execution_complete.json')
    if not end['execution_complete']:raise RuntimeError('Only complete preassigned3 cohort is audited; wait for root terminal notice')
    checks=[];inputs={}
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def track(p):p=Path(p);inputs[str(p)]=sha(p);return p
    track(PLAN);manifest=read(track(folder/'manifest.json'));runplan=read(track(folder/'plan.json'));trials=read(track(folder/'trials.json'));ledger=read(track(folder/'assignment_ledger.json'))
    track(folder/'execution_complete.json');track(folder/'runtime_failures.json');env_record=read(track(folder/'environment.json'))
    registration_path=track(Path(runplan['registration']));registration=read(registration_path);identity=manifest['research_source_sha256'];snap=read(track(folder/'source_snapshot_manifest.json'))
    key=lambda r:(r['scenario'],r['method'],r['seed'])
    assigned={(s,m,n) for s in plan['assignment']['scenarios'] for m in plan['assignment']['methods'] for n in plan['assignment']['seeds']}
    check('predeclared3_exact',len(assigned)==len(manifest['jobs'])==len(trials)==3 and {key(x) for x in manifest['jobs']}==assigned=={key(x) for x in trials})
    check('ledger_all3_retained',len(ledger)==3 and {key(x['job']) for x in ledger}==assigned and all(x['status']=='evaluated' for x in ledger))
    check('no_runtime_failures',not end['runtime_failures'] and read(folder/'runtime_failures.json')==[])
    check('registration_exact',sha(registration_path)==REGISTRATION_SHA==runplan['registration_sha256'] and registration['assignment']==plan['assignment'] and registration['software_gate_passed'])
    for filename,field in (('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')):check('terminal_'+filename,sha(folder/filename)==end[field])
    check('105_sources',len(identity)==105 and identity==registration['source_sha256'] and set(identity)==set(snap['files']))
    check('snapshot_manifest_hash',sha(folder/'source_snapshot_manifest.json')==manifest['source_snapshot_manifest_sha256'])
    for name,h in identity.items():
        row=snap['files'][name];check('source_'+name,sha(track(row['original_path']))==h==sha(track(folder/row['snapshot_path']))==row['sha256'])
    env=registration['environment'];check('environment_record_exact',env==env_record['pinned_environment'])
    check('python_executable',Path(sys.executable).resolve()==Path(env['executable']).resolve() and sha(track(sys.executable))==env['executable_sha256'])
    check('binary42',len(env['native_binaries_sha256'])==42)
    for path,h in env['native_binaries_sha256'].items():check('native_binary_'+path,sha(track(path))==h)
    check('reported_sources_unchanged',end['sources_unchanged'])
    check('all_run_directories_assigned',{str(x.resolve()) for x in (folder/'runs').iterdir()}=={str(Path(x['destination']).resolve()) for x in manifest['jobs']})
    sys.path.insert(0,str(FOUNDATION))
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    wanted=list(dict.fromkeys(['time',*PREP,'substep_time','substep_duration','substep_skin_normal','substep_all_hand_object','warning_count','controller_active','control_ms','debug_json']))
    rows=[];oldrows=[];preparation=[];technical=[];all_old_cohorts=sorted({x['cohort'] for x in plan['old_comparators']})
    for cohort in all_old_cohorts:
        old_manifest=read(track(ROOT/'output'/cohort/'source_snapshot_manifest.json'))
        for name,row in old_manifest['files'].items():check('preserved_'+cohort+'_'+name,sha(track(row['original_path']))==row['sha256'])
    for job in manifest['jobs']:
        method=job['method'];dest=Path(job['destination']);meta=read(track(dest/'metadata.json'));result=read(track(dest/'result.json'));trace=track(dest/'trace.npz')
        check(method+'_trace_identity',sha(trace)==meta['trace_sha256']==result['trace_sha256'])
        check(method+'_source_identity',meta['research_source_sha256']==identity)
        check(method+'_identity_and_completed',(meta['scenario']['name'],meta['method'],meta['seed'])==key(job) and meta['split']=='robustness' and meta['completed'])
        with np.load(trace,allow_pickle=False) as z:a={k:z[k].copy() for k in wanted}
        t=a['time'];st=a['substep_time'];dt=a['substep_duration'];physical=float(meta['physics_timestep']);f=a['substep_skin_normal'];limits=np.asarray(meta['limits']);budget=float(meta['budget'])
        check(method+'_3201_samples16s',t.shape==(3201,) and exact(t,np.arange(3201)*.005) and t[-1]==16.)
        check(method+'_32000_intervals16s',st.shape==dt.shape==(32000,) and np.all(dt==physical) and np.allclose(st,np.arange(32000)*physical,atol=1e-10,rtol=0) and abs(st[-1]+dt[-1]-16)<1e-9)
        check(method+'_zero_warnings',np.max(a['warning_count'])==0)
        first=evaluate_trace(trace);second=evaluate_research(trace)
        check(method+'_two_evaluators_all_original_fields',all(k in result and json_equal(v,result[k]) for k,v in first.items()) and json_equal(second,result['research_metrics']))
        check(method+'_trial_is_exact_run_result',sum(json_equal(result,x) for x in trials)==1)
        direct_local=float(np.sum(np.maximum(f-limits,0.)*dt[:,None]));direct_total=float(np.maximum(f.sum(1)-budget,0.)@dt)
        safety=second['safety'];check(method+'_raw_excess_exact',direct_local==safety['local_excess_integral_Ns'] and direct_total==safety['total_excess_integral_Ns'])
        check(method+'_raw_guard_exact',safety['guard_pass']==bool((f<=limits+.02).all() and (f.sum(1)<=budget+.02).all() and (a['skin_normal'][-1]<=limits+.02).all() and a['skin_normal'][-1].sum()<=budget+.02))
        rows.append(dict(cohort='development_v14_coherent_pilot_v1',**summary(result,a)))
        same_method=[x for x in plan['old_comparators'] if x['method']==method]
        check(method+'_both_old_cohorts_exact',len(same_method)==2 and {x['cohort'] for x in same_method}==set(all_old_cohorts))
        for comp in same_method:
            label=comp['cohort'];oldtrace=track(comp['trace_path']);oldresultpath=track(comp['result_path']);oldmeta=read(track(oldtrace.with_name('metadata.json')));oldresult=read(oldresultpath)
            check(method+'_'+label+'_trace_result_pin',sha(oldtrace)==comp['trace_sha256'] and sha(oldresultpath)==comp['result_sha256'])
            for field in ('config','noise','gains','actual_ratio_task_gains','limits','budget','model_sha256','source_sha256','motion_mode','task_contract'):check(method+'_'+label+'_common_'+field,json_equal(meta[field],oldmeta[field]))
            with np.load(oldtrace,allow_pickle=False) as z:b={k:z[k].copy() for k in wanted}
            check(method+'_'+label+'_full_fixed_environment',exact(a['environment_json'],b['environment_json']))
            mask=t<7.5;prior=b['time']<7.5;check(method+'_'+label+'_1500prep_clocks',int(mask.sum())==int(prior.sum())==1500 and exact(t[mask],b['time'][prior]))
            columns={}
            for field in PREP:
                ok=exact(a[field][mask],b[field][prior]);check(method+'_'+label+'_prep_'+field,ok)
                columns[field]=dict(exact=ok,new_dtype=str(a[field].dtype),old_dtype=str(b[field].dtype),comparison='row UTF8 content bytes; whole-array trailing Unicode padding is not data' if a[field].dtype.kind in 'US' else 'dtype+shape+all bytes')
            preparation.append(dict(method=method,comparator=label,frames=1500,columns=columns));oldrows.append(dict(cohort=label,**summary(oldresult,b)))
        # Read logged technical status/cost only. Math/native properties were
        # independently checked before physics; this audit does not re-solve.
        debug=[json.loads(str(x)) for x in a['debug_json']]
        dec=[d for d,when in zip(debug,t) if d.get('decision_time_s') is not None and abs(d['decision_time_s']-when)<1e-10]
        decision_manifest=read(track(dest/'joint_decision_manifest.json'));decision_file=track(dest/decision_manifest['file'])
        check(method+'_decision_file_identity',sha(decision_file)==decision_manifest['sha256'] and len(dec)==decision_manifest['records'])
        counts=Counter();modes=Counter();wall=[];qpwall=[];nativewall=[];fallback=Counter()
        for d in dec:
            counts['qp:'+str(d.get('qp_status'))]+=1;native=d.get('native_summary');counts['native:'+str(None if native is None else native.get('status'))]+=1
            modes[str(d.get('mode'))]+=1;fallback[str(d.get('label'))]+=1
            timing=d.get('solver_timing_ms')
            if timing:wall.append(timing['whole_decision']);qpwall.append(timing['qp']);nativewall.append(timing['native_inverse'])
        technical.append(dict(method=method,decision_records=len(dec),by_mode=dict(modes),logged_status_counts=dict(counts),logged_labels=dict(fallback),whole_decision_ms=stats(wall),qp_ms=stats(qpwall),realizer_ms=stats(nativewall),
            execution_changed_sensor_ticks=sum(bool(d.get('execution_changed_plan')) for d in debug),stale_plan_replaced_ticks=sum(bool(d.get('stale_plan_replaced')) for d in debug),
            scope='logged execution counts/cost only; no repeated math3297/native702 or new solver replay; all posttakeover sensor ticks retained'))
    for path,h in inputs.items():check('read_only_'+path,sha(path)==h)
    output=dict(schema='v14_complete_first_pilot_independent_v1',passed=all(x['passed'] for x in checks),check_count=len(checks),failures=[x for x in checks if not x['passed']],checks=checks,
        no_new_physics=True,no_production_modification=True,scope='single seen nominal/capsule/hold617901 condition;3 force-intent methods, not3 independent samples or scenarios; same-objective solver+realizer change is a bundle, not solver-only causality',
        registration_sha256=REGISTRATION_SHA,sources=105,binaries=42,input_sha256=inputs,checker_sha256=sha(__file__),summary_helper_sha256=sha(ROOT/'review/verify_v13_joint_pilot_independent_v1.py'),
        release_scope='unchanged frozen V13 nominal release path+actor; holding endpoint changes can alter release response, not a new release-policy design',
        new=rows,old=oldrows,preparation=preparation,technical=technical,
        signal_scope='raw200Hz TV and50Hz TV distinct; raw integration unfiltered safety. Canonical latency remains200Hz; extra rawEND interval-dwell is descriptive, not gap/separation certificate.',
        noise_scope='same device model/config, seed policy/source and3201-clock; actual1500 prep packets exact. Source final RNG was not exported; no unsupported internal final-state equality claim.')
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V14首轮完整三条试跑独立核查','',f"{len(checks)}项，失败{len(output['failures'])}项。105源与42二进制、3任务账本、两套原评价全字段、原新两组准备各1500帧9列、物参/噪声/全环境及完整3201/32000时钟核对。无新物理、无源修改、未重复数学/原生求解。",'',
        '同一已见 nominal/capsule/hold617901 开发工况。三种方法不是三个独立场景或独立物理样本，无显著性/整体优势结论。V14是数值求解与目标兑现一同改变，不能单独归因于求解器。','',
        '|版本/方法|任务/全运动|全程力保护|最大指力 N|超限 Ns|静态均力 N|静态raw TV N/s|静态50Hz TV N/s|全持握50Hz TV N/s|200Hz释放 s|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort in [*all_old_cohorts,'development_v14_coherent_pilot_v1']:
        for r in [x for x in oldrows+rows if x['cohort']==cohort]:
            q=r['windows']['static_quality'];w=r['windows']['entire_holding_quality']
            name='native' if 'v12_native' in cohort else 'V13' if 'v13_' in cohort else 'V14'
            lines.append(f"|{name}/{r['method']}|{r['mission']}/{r['full_motion']}|{r['guard']}|{r['peak_N']:.9g}|{r['local_excess_Ns']:.9g}|{q['mean_N']:.9g}|{q['raw_200Hz']['finger_TV_rate_N_s']:.9g}|{q['bandlimited']['finger_TV_rate_N_s']:.9g}|{w['bandlimited']['finger_TV_rate_N_s']:.9g}|{r['release']['physical_clearance_latency_s']:.9g}|")
    lines+=['','原始与50Hz波动分别列示。任务成功不能抵消力保护失败；各阶段、指峰及rawEND峰时间、holding SD/CV、连续运动误差、raw低力确认另见JSON。未达到/失败或不合格窗口应保留NULL，不补零。','',
        '释放仍是V13原路径和actor，不是V14新生命周期方案。原持握终态不同可改变释放反应，不能把差值当单独释放设计的因果效应。','',
        '|方法|决策数|QP状态|native状态|总决策 median/p95/max ms|','|---|---:|---|---|---|']
    for d in technical:
        s=d['whole_decision_ms'];lines.append(f"|{d['method']}|{d['decision_records']}|{ {k:v for k,v in d['logged_status_counts'].items() if k.startswith('qp:')} }|{ {k:v for k,v in d['logged_status_counts'].items() if k.startswith('native:')} }|{s['median']:.6g}/{s['p95']:.6g}/{s['maximum']:.6g}|")
    lines+=['','该表是实际保存的20ms决策耗时而非所有5ms tick中位数。它不是硬实时通过证明；保存的数值不可用/暂态候选、fallback及执行变化均不隐去。来源/字段一致不等于接触安全或新算法优势。']
    if output['failures']:lines+=['','## 未通过项','',*['- '+x['name']+': '+str(x['detail']) for x in output['failures']]]
    MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=output['passed'],checks=len(checks),failed=output['failures'][:10],evidence=str(OUT)),ensure_ascii=False))
    return 0 if output['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
