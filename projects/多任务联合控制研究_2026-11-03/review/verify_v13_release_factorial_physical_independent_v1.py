"""Independent four-cell fixed-input physical readout. Never runs physics."""
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LOCK_SHA='acff94ad3edfabe3ac97a96b4d117c51be365d17b6069cda2ff547ae6622135f'
OUT=ROOT/'review/v13_release_factorial_physical_independent_v1.json'
MD=ROOT/'review/v13_release_factorial_physical_independent_v1.md'
BOOKKEEPING={'alpha','alpha_before','phase','command','environment_json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def exact(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return False
    if a.dtype.kind in 'US' and b.dtype.kind in 'US':return all(str(x).encode('utf-8')==str(y).encode('utf-8') for x,y in zip(a.flat,b.flat))
    return a.dtype==b.dtype and a.tobytes()==b.tobytes()
def dwell(t,flag,start=12.,end=15.5,period=.2):
    since=None
    for when,clear in zip(t,flag):
        if when<start-1e-9 or when>end+1e-9:continue
        if clear:
            if since is None:since=float(when)
            if when-since>=period-1e-9:return float(when)
        else:since=None
    return None
def main():
    if OUT.exists() or MD.exists():raise FileExistsError('Independent physical audits remain immutable')
    bridge_root=ROOT/'output/v13_release_factorial_bridge_v1';other_root=ROOT/'output/v13_release_factorial_other3_v1'
    terminal=[read(p/'execution_complete.json') for p in (bridge_root,other_root)]
    if not all(x['execution_complete'] for x in terminal):raise RuntimeError('Wait for every predeclared job; never use partial cells')
    checks=[];tracked={}
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def track(p):p=Path(p);tracked[str(p)]=sha(p);return p
    roots=(bridge_root,other_root);manifests=[];jobs=[];locks=[]
    for folder,done in zip(roots,terminal):
        track(folder/'execution_complete.json');manifest=read(track(folder/'manifest.json'));ledger=read(track(folder/'assignment_ledger.json'));lock=read(track(folder/'source_lock.json'));locks.append(lock);manifests.append(manifest)
        check(folder.name+'_source_lock',sha(folder/'source_lock.json')==LOCK_SHA==done['source_lock_sha256']==manifest['source_lock_sha256'])
        check(folder.name+'_true_terminal',done['passed'] and done['sources_unchanged'] and done['identical_noise_rng_final'] and not done['failures'])
        check(folder.name+'_all_jobs_ledger',len(ledger)==len(manifest['jobs']) and all(x['status']=='completed' for x in ledger))
        check(folder.name+'_no_unassigned_dirs',{p.name for p in folder.iterdir() if p.is_dir() and p.name!='sources'}=={Path(x['destination']).name for x in manifest['jobs']})
        check(folder.name+'_108_sources',len(lock['sources'])==108 and len(lock['parent99'])==99 and all(lock['sources'][k]==h for k,h in lock['parent99'].items()))
        for name,h in lock['sources'].items():
            original=ROOT/name if not name.startswith('foundation/') else ROOT.parent/'五指策略接入_2026-11-03'/name.removeprefix('foundation/')
            check(folder.name+'_source_'+name,sha(track(original))==h==sha(track(folder/'sources'/name)))
        for path,h in lock['environment']['native_package_binaries_sha256'].items():check(folder.name+'_binary_'+path,sha(track(path))==h)
        check(folder.name+'_42_binaries',len(lock['environment']['native_package_binaries_sha256'])==42)
        check(folder.name+'_python_executable',sha(track(lock['environment']['executable']))==lock['environment']['executable_sha256'] and Path(sys.executable).resolve()==Path(lock['environment']['executable']).resolve())
        jobs.extend(manifest['jobs'])
    check('same_identical_lock',locks[0]==locks[1]);check('four_exact_cells',len(jobs)==4 and {x['variant'] for x in jobs}=={'both','lift_only','fingers_only','neither'})
    check('bridge_explicit_gate',terminal[0]['both_bridge_all_recomputed_exact'])
    lock=locks[0];source_path=track(lock['input']['trace']);source_meta=read(track(source_path.with_name('metadata.json')))
    check('source_trace_and_meta_pinned',sha(source_path)==lock['input']['trace_sha256'] and sha(source_path.with_name('metadata.json'))==lock['input']['metadata_sha256'])
    check('source_parent_manifest_pinned',sha(track(ROOT/'output/development_v13_joint_pilot_v1/manifest.json'))==lock['input']['cohort_manifest_sha256'] and sha(track(ROOT/'output/development_v13_joint_pilot_v1/execution_complete.json'))==lock['input']['cohort_complete_sha256'])
    with np.load(source_path,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files if k!='metadata'}
    source_entry=np.asarray(json.loads(str(source['observation_json'][2400]))['robot_reference21'],float)
    sourcespec=json.loads(str(source['observation_json'][2400]))['sensor_characteristics']
    # Every fresh bridge field remains mandatory. Bookkeeping is separately
    # classified and never counted as independent policy/physical outputs.
    bridge_compare=read(track(bridge_root/'both/both_bridge_comparison.json'));required=bridge_compare['required_fields']
    check('no_source_alias_in_bridge_fields',not any(x.startswith('source_') for x in required))
    arrays={};results={};rows=[];input_checks=[];rng_final=[]
    for job in jobs:
        variant=job['variant'];folder=Path(job['destination']);result=read(track(folder/'result.json'));meta=read(track(folder/'metadata.json'));diag=read(track(folder/'diagnostics.json'))
        path=track(folder/'trace.npz');packets=track(folder/'cached_contact_packets.jsonl.gz');track(folder/'fixed_reference_program.npz');track(folder/'reset_integration_state.npz')
        check(variant+'_complete_result',result['complete'] and result['variant']==variant and result['rows']==3201 and result['physics_intervals']==32000 and sha(path)==result['trace_sha256'] and sha(packets)==result['cached_contact_packet_sha256'])
        check(variant+'_same_actual_config',meta['config']==source_meta['config'] and meta['model_sha256']==source_meta['model_sha256'] and meta['limits']==source_meta['limits'] and meta['budget']==source_meta['budget'])
        check(variant+'_not_new_policy_outcome',meta['new_feedback_decisions']==0 and meta['new_strategy_success_evaluated'] is False)
        check(variant+'_same_incoming_entry',exact(np.asarray(meta['source_entry_reference21']),source_entry) and meta['source_entry_time_s']==12.)
        with np.load(path,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files if not k.startswith('source_') and k!='metadata'};provenance={k:z[k].copy() for k in z.files if k.startswith('source_')}
        arrays[variant]=v;results[variant]=result;t=v['time'];st=v['substep_time'];du=v['substep_duration'];end=st+du;f=v['substep_skin_normal'];n=v['substep_normal'];limits=np.asarray(meta['limits']);budget=float(meta['budget']);release=st>=12.-1e-9
        check(variant+'_3201_clock',exact(t,np.arange(3201)*.005))
        check(variant+'_32000_clock',st.shape==du.shape==(32000,) and np.all(du==.0005) and np.allclose(st,np.arange(32000)*.0005,atol=1e-10,rtol=0) and abs(end[-1]-16)<1e-9)
        check(variant+'_raw_state_same_end_clock',exact(end,v['substep_state_end_time']) and v['substep_qpos'].shape==(32000,33) and v['substep_qvel'].shape==(32000,32) and v['substep_actual_ctrl'].shape==(32000,22))
        check(variant+'_zero_warnings',np.max(v['warning_count'])==np.max(v['substep_warning_count'])==0)
        program=source['executed_reference21'].copy();use_lift=variant in ('both','lift_only');use_fingers=variant in ('both','fingers_only')
        if not use_lift:program[2400:,18]=source_entry[18]
        if not use_fingers:program[2400:,:18]=source_entry[:18]
        check(variant+'_full_actual21_mask',exact(v['executed_reference21'],program) and exact(v['requested_reference21'],program))
        check(variant+'_actual_raw22_ZOH_all_intervals',exact(v['substep_actual_ctrl'],np.repeat(v['ctrl'][:-1],10,axis=0)))
        check(variant+'_full_environment_same_input',exact(v['environment_json'],source['environment_json']))
        check(variant+'_RNG_chain_single_sample_per_tick',exact(v['rng_before'][1:],v['rng_after'][:-1]) and str(v['rng_after'][-1])==result['noise_rng_final'])
        rng_final.append(result['noise_rng_final'])
        check(variant+'_source_provenance_never_real_output',all(k.removeprefix('source_') in source and exact(x,source[k.removeprefix('source_')]) for k,x in provenance.items()))
        if variant=='both':
            for col in required:check('both_direct_exact_'+col,col in v and col in source and exact(v[col],source[col]))
        # Recompute original stage diagnostics without importing its producer.
        stage={}
        for label,lo,hi in (('preparation',0.,7.5),('holding',7.5,12.),('release',12.,16.)):
            idx=np.flatnonzero((st>=lo-1e-9)&(st<hi-1e-9));i,j=np.unravel_index(np.argmax(f[idx]),f[idx].shape);k=int(idx[i])
            values=dict(duration_s=float(du[idx].sum()),peak_N=float(f[k,j]),digit=int(j),peak_start_s=float(st[k]),peak_end_s=float(end[k]),target_same_digit_N=float(n[k,j]),non_target_same_digit_N=float(max(0.,f[k,j]-n[k,j])),local_excess_Ns=float(np.sum(np.maximum(f[idx]-limits,0)*du[idx,None])),total_excess_Ns=float(np.maximum(f[idx].sum(1)-budget,0)@du[idx]),above_local_limit_duration_s=float(du[idx][(f[idx]>limits+.02).any(1)].sum()),palm_at_peak_N=float(v['substep_palm'][k]),support_at_peak_N=float(v['substep_support'][k]),max_penetration_m=float(v['substep_world_penetration'][idx].max()))
            check(variant+'_'+label+'_all_producer_metrics_direct',all(values[key]==diag[label][key] for key in values))
            stage[label]=values
        clearance200=dwell(t,v['all_hand_object']<=.2);clearanceraw=dwell(end,v['substep_all_hand_object']<=.2)
        check(variant+'_clearance_two_clocks_direct',clearance200==diag['low_force_clearance']['frame200Hz_first_by_deadline'] and clearanceraw==diag['low_force_clearance']['raw_integration_end_first_by_deadline'])
        support=v['substep_support'][release];rz=v['substep_qpos'][release,27]
        address_ok=exact(v['qpos'][:,25:28],v['position'])
        check(variant+'_object_free_xyz_schema_verified',address_ok)
        idx=np.flatnonzero(release);wp=v['substep_world_penetration'][release];maximum_i=int(idx[np.argmax(wp)]);firstsupport=np.flatnonzero(release&(v['substep_support']>.02))
        # Re-aggregate every stored release packet against the cached raw force
        # record. Target IDs are evaluation-only; never enter a command.
        packet_count=0;max_all_diff=0.;max_target_diff=0.;max_palm_diff=0.;all_exact=True;last_end=-np.inf;aux=[]
        with gzip.open(packets,'rt',encoding='utf-8') as stream:
            for line in stream:
                q=json.loads(line);when=float(q['end_s']);raw_i=int(round(when/.0005))-1
                if raw_i<0 or raw_i>=32000:raise ValueError('Contact packet clock outside original intervals')
                if when<=last_end:raise ValueError('Duplicate/rewound cached packet')
                last_end=when;packet_count+=1;allfinger=np.zeros(5);targetfinger=np.zeros(5);palm_target=0.;palm_other=0.;hand_other_target=0.;hand_other_non=0.
                for patch in q['patches']:
                    digit=int(patch['digit']);pressure=float(patch['normal']);target=bool(patch['is_target']);part=patch['part']
                    if digit>=0:
                        allfinger[digit]+=pressure
                        if target:targetfinger[digit]+=pressure
                    elif part=='palm':
                        if target:palm_target+=pressure
                        else:palm_other+=pressure
                    else:
                        if target:hand_other_target+=pressure
                        else:hand_other_non+=pressure
                all_exact=all_exact and when==end[raw_i] and exact(allfinger,f[raw_i]) and exact(targetfinger,n[raw_i]) and palm_target==v['substep_palm'][raw_i]
                max_all_diff=max(max_all_diff,float(max(abs(allfinger-f[raw_i]))));max_target_diff=max(max_target_diff,float(max(abs(targetfinger-n[raw_i]))));max_palm_diff=max(max_palm_diff,abs(palm_target-v['substep_palm'][raw_i]))
                if st[raw_i]>=12.-1e-9:aux.append((palm_target,palm_other,hand_other_target,hand_other_non))
        check(variant+'_all8001_cached_packet_forces_exact',packet_count==8001 and all_exact,dict(count=packet_count,all_diff=max_all_diff,target_diff=max_target_diff,palm_diff=max_palm_diff))
        aux=np.asarray(aux);check(variant+'_8000_release_aux_intervals',aux.shape==(8000,4))
        row=dict(variant=variant,use_original_fingers=use_fingers,use_original_lift=use_lift,stages=stage,frame_clearance_s=None if clearance200 is None else clearance200-12.,raw_END_clearance_s=None if clearanceraw is None else clearanceraw-12.,
            raw_recontact_after_first_dwell_N=None if clearanceraw is None else float(v['substep_all_hand_object'][end>clearanceraw+1e-9].max(initial=0.)),
            nonhand_object_support_peak_N=float(max(support)),nonhand_object_support_impulse_Ns=float(support@du[release]),first_nonhand_object_support_END_s=None if not len(firstsupport) else float(end[firstsupport[0]]),
            world_penetration_max_m=float(max(wp)),world_penetration_max_END_s=float(end[maximum_i]),object_z_200Hz_min_m=float(v['position'][t>=12.,2].min()),object_z_200Hz_final_m=float(v['position'][-1,2]),
            object_z_entry_m=float(v['position'][2400,2]),object_z_raw_min_m=float(rz.min()),object_z_raw_max_m=float(rz.max()),object_z_change_entry_to_final_m=float(v['position'][-1,2]-v['position'][2400,2]),
            evaluation_auxiliary_pressure=dict(axes=['palm_target','palm_nontarget','other_hand_target','other_hand_nontarget'],peaks_N=aux.max(0).tolist(),integrals_Ns=(aux*.0005).sum(0).tolist()),
            cached_packet_count=packet_count,warnings=0,algorithm_success_rate=None)
        rows.append(row)
    common=arrays['both']
    for variant,v in arrays.items():
        prefix=common['time']<=12.+1e-9;rawprefix=common['substep_state_end_time']<=12.+1e-9
        for col in ('qpos','qvel','position','rotation','normal','skin_normal','support','all_hand_object','legacy_common_sensor_json','observation_json','raw_patches_json','ctrl_before','robot_actuator_target21'):
            check(variant+'_common_entry_prefix_'+col,exact(v[col][prefix],common[col][prefix]))
        for col in ('substep_qpos','substep_qvel','substep_actual_ctrl','substep_skin_normal','substep_normal','substep_support','substep_warning_count'):
            check(variant+'_common_raw_prefix_'+col,exact(v[col][rawprefix],common[col][rawprefix]))
        check(variant+'_all_RNG_stream_states_equal',exact(v['rng_before'],common['rng_before']) and exact(v['rng_after'],common['rng_after']))
    check('all4_final_RNG_same',len(set(rng_final))==1 and rng_final[0]==terminal[0]['results'][0]['noise_rng_final'] and all(x['noise_rng_final']==rng_final[0] for x in terminal[1]['results']))
    for path,h in tracked.items():check('unchanged_'+path,sha(path)==h)
    ordered={x['variant']:x for x in rows};rows=[ordered[k] for k in ('neither','fingers_only','lift_only','both')]
    result=dict(schema='v13_release_factorial_physical_independent_v1',passed=all(c['passed'] for c in checks),check_count=len(checks),failures=[c for c in checks if not c['passed']],checks=checks,input_sha256=tracked,
        checker_sha256=sha(__file__),source_lock_sha256=LOCK_SHA,rows=rows,source_entry_reference21=source_entry.tolist(),device_spec=sourcespec,
        scope='one seen fixed-input source condition617901/fusion/hold/capsule;2×2 finger18 and lift program bundles, same wrist. Causal input-bundle counterfactual, not a new algorithm or success-rate comparison.',
        bookkeeping_fields=sorted(BOOKKEEPING),independent_bridge_fields=[x for x in required if x not in BOOKKEEPING],unrecomputed_source_policy_fields=list(bridge_compare['explicit_unrecomputed_source_fields']),
        force_scope='all-external finger raw safety; target/nontarget and auxiliary ownership evaluation only. Nonhand_object_support is recorded support proxy, not a pure material-damage metric.',
        clocks='raw START plusduration=END; force rows reflect recorded solver intervals and separate200Hz forward-query records. End-labelled dwell is finite-record low-force criterion, never a geometric certificate.',
        no_new_physics=True,no_source_modification=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V13释放固定输入2×2物理独立审查','',f"{len(checks)}项，失败{len(result['failures'])}项。四单元均完整16s/3201帧/32000积分段；同108源/42二进制锁、同物参/环境/传感时钟及新四分支全程RNG链。未启动物理或改源。",'',
        'both的全部真实重算物理/公共传感/端口/clock/原积分列与源直接核对；source_*不参与物理exact。alpha/phase/command/environment只是输入/仪器记账另列。所有单元在12s incoming时刻及此前q/v/forces/公共packet一致，严格共入口。','',
        '|原18指路径|原lift路径|单元|释放最大指力 N|超限剂量 Ns|超限时长 s|END力清空延迟 s|物体-非手支撑峰 N|支撑冲量 Ns|最大world穿透 m|物体最终高度变化 m|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        p=r['stages']['release'];clear='NULL' if r['raw_END_clearance_s'] is None else f"{r['raw_END_clearance_s']:.9g}"
        lines.append(f"|{r['use_original_fingers']}|{r['use_original_lift']}|{r['variant']}|{p['peak_N']:.9g}|{p['local_excess_Ns']:.9g}|{p['above_local_limit_duration_s']:.9g}|{clear}|{r['nonhand_object_support_peak_N']:.9g}|{r['nonhand_object_support_impulse_Ns']:.9g}|{r['world_penetration_max_m']:.9g}|{r['object_z_change_entry_to_final_m']:.9g}|")
    lines+=['','表内原始峰/剂量未经滤波。力清空NULL不补零。仅张指虽然可减少指部超限，仍必须同时呈现物体掉落/支撑冲击和穿透代价；不能将“指力未超阈值”写成完整物理安全或优越释放策略。没有增加任何材料阈值或算法成功率。','',
        '保存的contact packets与全部8001个相关raw缓存积分记录逐包重新聚合：所有指外力、target指力、target掌力及时间相同；所有权标签仅用于离线解释，不进入输入程序。全四单元warnings为0并不使穿透或撞击代价消失。','',
        '两个因子各为一个完整命令包，不识别单关节贡献、独立COM模型或真实材料刚度。结果只适用于该既存条件和固定输入时钟；尚不能据此宣称新闭环控制收益或现实硬件安全。']
    if result['failures']:lines+=['','## 未通过项','',*['- '+x['name']+': '+str(x['detail']) for x in result['failures']]]
    MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=result['failures'][:12],evidence=str(OUT)),ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
