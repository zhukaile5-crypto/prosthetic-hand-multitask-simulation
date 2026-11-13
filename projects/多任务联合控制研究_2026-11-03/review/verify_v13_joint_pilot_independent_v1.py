"""Read-only audit of the predeclared complete three-job V13Joint pilot.

Only new review evidence is written. No simulator, controller or plant is run.
"""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
PLAN=ROOT/'review/v13_joint_pilot_plan_v1.json'
OUT=ROOT/'review/v13_joint_pilot_independent_v1.json'
MD=ROOT/'review/v13_joint_pilot_independent_v1.md'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exact(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind in 'US' and b.dtype.kind in 'US':
        return all(str(x).encode('utf-8')==str(y).encode('utf-8') for x,y in zip(a.flat,b.flat))
    return a.dtype==b.dtype and a.tobytes()==b.tobytes()
def json_equal(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
def stats(a):
    a=np.asarray(a,float);return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(np.max(a))) if len(a) else None
def raw_clearance(st,dur,force,start,threshold=.2,dwell=.2):
    since=None
    for s,d,f in zip(st,dur,force):
        if s<start-1e-9:continue
        if f<=threshold:
            if since is None:since=float(s)
            if s+d-since>=dwell-1e-9:return dict(first_start_s=since,confirmation_end_s=float(s+d),latency_s=float(s+d-start))
        else:since=None
    return None
def summary(result,arrays):
    m=result['research_metrics'];t=arrays['time'];hold=(t>=7.5)&(t<=12.)
    whole=m['entire_holding_quality'];force=arrays['normal'][hold].sum(1)
    d=dict(method=m['method'],mission=m['mission_success'],full_motion=m['full_motion_mission_success'],guard=m['safety']['guard_pass'],
        peak_N=max(m['safety']['finger_peak_N']),finger_peak_N=m['safety']['finger_peak_N'],palm_peak_N=m['safety']['palm_peak_N'],
        local_excess_Ns=m['safety']['local_excess_integral_Ns'],total_excess_Ns=m['safety']['total_excess_integral_Ns'],stages=m['safety']['stages'],
        release=m['release'],stability_wait_s=m['initial_stability_wait_s'],progress=m['progress'],windows={},
        whole_total_force_SD_raw_N=float(np.std(force,ddof=1)),whole_total_force_CV_raw=float(np.std(force,ddof=1)/np.mean(force)),
        control_5ms_all_ms=stats(arrays['control_ms'][arrays['controller_active']]))
    for k in ('static_quality','dynamic_quality','entire_holding_quality'):
        x=m[k];d['windows'][k]=dict(eligible=x['eligible'],mean_N=x.get('mean_actual_finger_force_N'),bandlimited=x.get('bandlimited'),raw_200Hz=x.get('raw_200Hz'))
    st=arrays['substep_time'];du=arrays['substep_duration'];f=arrays['substep_skin_normal'];i,j=np.unravel_index(np.argmax(f),f.shape)
    d['largest_integration_force']=dict(N=float(f[i,j]),digit=int(j),start_s=float(st[i]),end_s=float(st[i]+du[i]),stage='holding' if st[i]<12 else 'release')
    d['raw_clearance_diagnostic']=raw_clearance(st,du,arrays['substep_all_hand_object'],12.)
    return d

def main():
    if OUT.exists() or MD.exists():raise FileExistsError('Independent attempts are immutable; choose a new audit version')
    plan=read(PLAN);folder=ROOT/plan['output'];end=read(folder/'execution_complete.json')
    if not end['execution_complete']:raise RuntimeError('Wait for the complete assigned cohort; partial performance is not audited')
    checks=[];tracked={}
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def track(path):path=Path(path);tracked[str(path)]=sha(path);return path
    track(PLAN)
    manifest=read(track(folder/'manifest.json'));runplan=read(track(folder/'plan.json'));ledger=read(track(folder/'assignment_ledger.json'));trials=read(track(folder/'trials.json'))
    registration_path=Path(runplan['registration']);registration=read(track(registration_path));snap=read(track(folder/'source_snapshot_manifest.json'))
    track(folder/'execution_complete.json');track(folder/'environment.json');track(folder/'runtime_failures.json')
    expected={(s,m,n) for s in plan['assignment']['scenarios'] for m in plan['assignment']['methods'] for n in plan['assignment']['seeds']}
    key=lambda j:(j['scenario'],j['method'],j['seed'])
    check('all_three_assigned_jobs_exact',len(manifest['jobs'])==len(expected)==3 and {key(j) for j in manifest['jobs']}==expected)
    check('all_three_trials_exact',len(trials)==3 and {key(j) for j in trials}==expected)
    check('all_assignments_evaluated',len(ledger)==3 and {key(x['job']) for x in ledger}==expected and all(x['status']=='evaluated' for x in ledger))
    check('no_hidden_runtime_failures',not end['runtime_failures'] and read(folder/'runtime_failures.json')==[])
    check('terminal_hashes',all(sha(folder/n)==end[k] for n,k in [('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')]))
    check('registration_identity',sha(registration_path)==runplan['registration_sha256'] and registration['assignment']==plan['assignment'] and registration['software_gate_passed'])
    identity=manifest['research_source_sha256'];check('source99',len(identity)==99 and identity==registration['source_sha256'])
    check('snapshot_manifest_identity',sha(folder/'source_snapshot_manifest.json')==manifest['source_snapshot_manifest_sha256'] and set(snap['files'])==set(identity))
    for name,h in identity.items():
        entry=snap['files'][name];original=Path(entry['original_path']);copy=folder/entry['snapshot_path']
        check('source_snapshot_'+name,sha(track(original))==h==sha(track(copy))==entry['sha256'])
    env=registration['environment'];env_record=read(folder/'environment.json')
    check('environment_record_matches',env==env_record['pinned_environment'])
    check('python_executable_identity',sha(env['executable'])==env['executable_sha256'] and Path(sys.executable).resolve()==Path(env['executable']).resolve())
    binaries=env['native_binaries_sha256'];check('binary42',len(binaries)==42)
    for path,h in binaries.items():check('binary_'+path,sha(track(path))==h)
    check('completed_sources_unchanged_flag',end['sources_unchanged'])
    check('no_unassigned_run_directories',{str(p.resolve()) for p in (folder/'runs').iterdir()}=={str(Path(j['destination']).resolve()) for j in manifest['jobs']})

    # Import only unchanged analysis functions. No simulator or executable policy.
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(FOUNDATION))
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    rows=[];old_rows=[];technical=[];prepared=[];shared_identity=None
    comparators={x['method']:x for x in plan['old_comparators']}
    wanted=['time','normal','skin_normal','position','relative','raw_patches_json','legacy_common_sensor_json','environment_json','robot_actuator_target21','robot_control_time_s',
        'substep_time','substep_duration','substep_skin_normal','substep_normal','substep_all_hand_object','substep_palm','warning_count','control_ms','controller_active','debug_json',
        'executed_reference21','reference_increment21','requested_reference21','mode_generation','transport_paused']
    for job in manifest['jobs']:
        method=job['method'];dest=Path(job['destination']);meta=read(track(dest/'metadata.json'));result=read(track(dest/'result.json'));old=comparators[method]
        check(method+'_old_trace_pin',sha(track(old['trace_path']))==old['trace_sha256']);check(method+'_old_result_pin',sha(track(old['result_path']))==old['result_sha256'])
        oldmeta=read(track(Path(old['trace_path']).with_name('metadata.json')));oldresult=read(old['result_path'])
        check(method+'_new_trace_identity',sha(track(dest/'trace.npz'))==meta['trace_sha256']==result['trace_sha256'])
        check(method+'_metadata_source_identity',meta['research_source_sha256']==identity)
        for name in ('config','noise','gains','actual_ratio_task_gains','limits','budget','model_sha256','source_sha256','motion_mode','task_contract'):
            check(method+'_common_'+name,json_equal(meta[name],oldmeta[name]))
        intersection=set(meta['research_source_sha256'])&set(oldmeta['research_source_sha256'])
        check(method+'_old80_preserved',len(oldmeta['research_source_sha256'])==80 and len(intersection)==80 and all(meta['research_source_sha256'][k]==oldmeta['research_source_sha256'][k] for k in intersection))
        check(method+'_assignment_metadata',(meta['scenario']['name'],meta['method'],meta['seed'])==key(job) and meta['split']=='robustness' and meta['completed'])
        with np.load(dest/'trace.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in wanted}
        old_wanted=[k for k in wanted if k not in ('executed_reference21','reference_increment21','requested_reference21','mode_generation')]
        with np.load(old['trace_path'],allow_pickle=False) as z:b={k:z[k].copy() for k in old_wanted}
        t=a['time'];st=a['substep_time'];du=a['substep_duration'];dt=float(meta['physics_timestep'])
        check(method+'_complete_16s_sensor',t.shape==(3201,) and exact(t,np.arange(3201,dtype=float)*.005) and t[-1]==16.)
        check(method+'_complete_all32000_substeps',st.shape==du.shape==(32000,) and np.isfinite(st).all() and np.isfinite(du).all() and np.allclose(st,np.arange(32000)*dt,rtol=0,atol=1e-10) and np.all(du==dt) and abs(st[-1]+du[-1]-16)<1e-9)
        check(method+'_zero_physical_warnings',np.max(a['warning_count'])==0)
        check(method+'_full_environment_equal',exact(a['environment_json'],b['environment_json']))
        mask=t<7.5;oldmask=b['time']<7.5
        check(method+'_prep_time_equal',exact(t[mask],b['time'][oldmask]))
        prep={}
        for col in plan['common_preparation_fields']:
            ok=exact(a[col][mask],b[col][oldmask]);check(method+'_prep_'+col,ok)
            prep[col]=dict(exact=ok,new_dtype=str(a[col].dtype),old_dtype=str(b[col].dtype),comparison='per-row UTF8 bytes (fixed Unicode array padding is not data)' if a[col].dtype.kind in 'US' else 'dtype+shape+bytes')
        prepared.append(dict(method=method,rows=int(mask.sum()),fields=prep))
        frozen=evaluate_trace(dest/'trace.npz');research=evaluate_research(dest/'trace.npz')
        check(method+'_original_evaluator_fields_exact',all(k in result and json_equal(v,result[k]) for k,v in frozen.items()))
        check(method+'_research_evaluator_fields_exact',json_equal(research,result['research_metrics']))
        check(method+'_trials_exact_run_result',sum(json_equal(result,x) for x in trials)==1)
        f=a['substep_skin_normal'];limits=np.asarray(meta['limits']);budget=float(meta['budget']);local=np.maximum(f-limits,0.)
        direct=dict(local_excess_integral_Ns=float(np.sum(local*du[:,None])),total_excess_integral_Ns=float(np.maximum(f.sum(1)-budget,0.)@du))
        for k,v in direct.items():check(method+'_direct_raw_'+k,v==result['research_metrics']['safety'][k])
        check(method+'_direct_raw_full_guard',result['research_metrics']['safety']['guard_pass']==bool((f<=limits+.02).all() and (f.sum(1)<=budget+.02).all() and (a['skin_normal'][-1]<=limits+.02).all() and a['skin_normal'][-1].sum()<=budget+.02))
        rows.append(summary(result,a));old_rows.append(summary(oldresult,b))

        evidence_manifest=read(track(dest/'joint_decision_manifest.json'));evidence_path=track(dest/evidence_manifest['file'])
        check(method+'_decision_file_hash',sha(evidence_path)==evidence_manifest['sha256'])
        counts=Counter();n=0;last=-np.inf;max_residual_diff=0.;timings=[];by_mode={};qpfail=[]
        with gzip.open(evidence_path,'rt',encoding='utf-8') as stream:
            for line in stream:
                e=json.loads(line);i=int(e['decision_id']);n+=1;when=float(e['time_s']);mode=e['mode'];by_mode[mode]=by_mode.get(mode,0)+1
                check(method+f'_decision_{i}_clock',i==n-1 and when>last and abs(e['nominal_target_time_s']-when-.02)<1e-12);last=when
                qp=e['qp'];native=e['native'];counts['domain:'+e['domain']['status']]+=1
                if qp is None:counts['qp:absent']+=1
                else:
                    counts['qp:'+qp['status']]+=1
                    if qp['status']!='converged':qpfail.append(dict(decision_id=i,time_s=when,mode=mode,status=qp['status'],reason=qp['reason']))
                    if qp['status']=='converged':check(method+f'_decision_{i}_qp_primal',all(v<=1e-10 for v in qp['diagnostics']['primal_violations'].values()))
                if native is None:counts['native:absent']+=1;continue
                counts['native:'+native['status']]+=1
                diag=native['diagnostics'];recheck=diag.get('prepared_value_final_full_recheck',{})
                if native['reference21'] is not None:
                    flags=('passed','native_bytes_equal','delta_bytes_equal','residual_bytes_equal','objective_bytes_equal','transmission_bytes_equal','no_zero_shortcut')
                    check(method+f'_decision_{i}_full_native_recheck',all(recheck.get(k) is True for k in flags))
                    A=np.asarray(e['domain']['transmission21x25']);p=np.asarray(native['native_effort21']);p0=np.asarray(native['p0_effort21']);delta=p-p0
                    z=np.asarray(qp['zP']) if qp is not None and qp['status']=='converged' else None
                    # When no QP is available the controller uses the public prior distributed across patches.
                    # Do not fabricate that unlogged fallback patch assignment here.
                    check(method+f'_decision_{i}_native_delta_exact',exact(delta,np.asarray(native['actual_delta_effort21'])))
                    if z is not None:
                        residual=A.T@delta-np.asarray(e['Jpx25']).T@(z-np.asarray(e['observed_patch_normal']))
                        actual=np.asarray(native['residual25']);diff=float(np.max(abs(residual-actual)));max_residual_diff=max(max_residual_diff,diff)
                        check(method+f'_decision_{i}_canonical_residual_exact',exact(residual,actual),diff)
                    check(method+f'_decision_{i}_same_fixed_p0',exact(p0,np.asarray(e['domain']['p0_effort21'])))
        check(method+'_all_decision_records',n==evidence_manifest['records'])
        debug=[json.loads(x) for x in a['debug_json']]
        decision_debug=[d for d,when in zip(debug,t) if d.get('decision_time_s') is not None and abs(float(d['decision_time_s'])-when)<1e-10]
        labels=Counter(d.get('label','missing') for d in decision_debug)
        check(method+'_decision_debug_count',len(decision_debug)==n)
        for d in decision_debug:
            if 'solver_timing_ms' in d:timings.append(d['solver_timing_ms']['whole_decision'])
        technical.append(dict(method=method,decisions=n,by_mode=by_mode,status_counts=dict(counts),labels=dict(labels),qp_unavailable=qpfail,
            actual_20ms_decision_wall_ms=stats(timings),max_canonical_residual_difference=max_residual_diff,
            changed_execution_sensor_ticks=sum(bool(d.get('execution_changed_plan')) for d in debug),
            stale_plan_replaced_sensor_ticks=sum(bool(d.get('stale_plan_replaced')) for d in debug),
            conditional_support_ticks=sum(bool(d.get('conditional_support_active')) for d in debug),
            pressure_prediction_available=False,hard_realtime_certified=False))

    for path,h in tracked.items():check('unchanged_input_'+path,sha(path)==h)
    out=dict(schema='v13_joint_complete_pilot_independent_audit_v1',passed=all(x['passed'] for x in checks),checks=len(checks),failures=[x for x in checks if not x['passed']],
        scope='one predeclared already-seen physical seed617901, nominal capsule hold; three policies, not three independent environments; no performance inference or global force-safety claim',
        no_new_physics=True,no_production_modifications=True,input_sha256=tracked,author_script_sha256=sha(__file__),
        sources=99,binaries=42,preparation=prepared,new=rows,old=old_rows,technical=technical,details=checks,
        noise_boundary='identical known device config, seed policy, frozen instrument source and full sensor clock; prep packets byte exact. Full post-takeover tactile values differ physically and are not claimed byte equal. RNG internals were not exported by this run.',
        clearance_boundary='canonical release latency is unchanged200Hz evaluation. Extra raw diagnosis uses recorded integration segments END=start+duration, not an infinite-time geometric clearance certificate; 200Hz mj_forward force samples are a distinct record.')
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V13Joint 首轮三条物理试跑独立核查','',f"独立检查 {len(checks)} 项，失败 {len(out['failures'])} 项。只读完整终态与预定旧三条对照；未运行物理、未改执行源。",'',
        '同一已见 nominal/capsule/hold 工况、物理种子617901；三种策略不是三个独立环境，不能据此给整体优势或显著性结论。','',
        '|版本/方法|任务/完整运动|全程力保护|最大指力 N|超限 Ns|静态均力 N|静态原始 TV N/s|静态50Hz TV N/s|全持握50Hz TV N/s|200Hz释放延迟 s|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,rr in [('旧native',old_rows),('新joint',rows)]:
        for r in rr:
            sq=r['windows']['static_quality'];w=r['windows']['entire_holding_quality'];lat=r['release']['physical_clearance_latency_s']
            lines.append(f"|{label}/{r['method']}|{r['mission']}/{r['full_motion']}|{r['guard']}|{r['peak_N']:.9g}|{r['local_excess_Ns']:.9g}|{sq['mean_N']:.9g}|{sq['raw_200Hz']['finger_TV_rate_N_s']:.9g}|{sq['bandlimited']['finger_TV_rate_N_s']:.9g}|{w['bandlimited']['finger_TV_rate_N_s']:.9g}|{lat:.9g}|")
    lines+=['','原始与50Hz列严格分开；任务成功、低均力或低TV不能抵消全程力保护失败。阶段、逐指峰、原始积分、运动误差、SD/CV及数值回退明细见JSON。完整99源/42二进制身份与快照逐项核对；原新3×9列准备内容逐字节比较，字符串按每行UTF8字节核对，不把整列Unicode尾部填充当数据。','',
        '|方法|20ms决策数|QP状态|native状态|决策耗时median/p95/max ms|','|---|---:|---|---|---|']
    for x in technical:
        q={k:v for k,v in x['status_counts'].items() if k.startswith('qp:')};n={k:v for k,v in x['status_counts'].items() if k.startswith('native:')};s=x['actual_20ms_decision_wall_ms']
        lines.append(f"|{x['method']}|{x['decisions']}|{q}|{n}|{s['median']:.6g}/{s['p95']:.6g}/{s['maximum']:.6g}|")
    lines+=['','耗时测量不是20ms硬实时通过。QP不可用与明确反馈/合法名义回退保留，不能把task_success解释为每次优化均收敛。所有可用native结果的最终full复核标志及固定p0、canonical实际增量/残差被再次核对；未重新求解全部优化器或重跑物理。','',
        '冻结评价的释放延迟保持200Hz口径。JSON另列raw integration END的持续低力确认；两类记录含额外forward查询语义差异，不把帧结果当连续几何清离证明。']
    if out['failures']:lines+=['','## 未通过项','',*['- '+x['name']+': '+str(x['detail']) for x in out['failures']]]
    MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=len(out['failures']),failures=out['failures'][:10],audit=str(OUT),report=str(MD)),ensure_ascii=False))
    return 0 if out['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
