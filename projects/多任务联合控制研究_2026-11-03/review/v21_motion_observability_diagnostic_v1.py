"""Read-only all16 V21 motion-observability diagnostic; no model or solve.

Scope fixed before execution: every saved release decision in all16 records;
per-record snapshots at 12,13,14,15s and its final saved decision. Algebraic
matrix ranks and recorded residuals are not object-twist observability ranks
or physical motion predictions. Offline object speed norms use the nearest
stored integration-end state, separately at decision and acquisition clocks.
"""
from pathlib import Path
from collections import Counter
import gzip, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
AUX=ROOT/'review/v21_complete16_auxiliary_v1.json'
AUX_SHA='b894592f38a8ca199088fd41bda9331ab819486b995b7db08d35e898be12d933'
FIXED_TIMES=(12.,13.,14.,15.)
SOURCES=('v20_contact_exit/adapter.py','v20_contact_exit/geometry.py',
         'v21_stable_handoff/task_solver.py','v21_stable_handoff/supervisor.py',
         'v21_stable_handoff/path_memory.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),minimum=None if not len(a) else float(a.min()),
        median=None if not len(a) else float(np.median(a)),maximum=None if not len(a) else float(a.max()))
def require(v,reason):
    if not v:raise ValueError(reason)

def collect():
    require(sha(AUX)==AUX_SHA,'final all16 auxiliary identity changed')
    aux=read(AUX);require(aux['complete'] is True and aux['issues']==[] and len(aux['records'])==16,'complete all16 required')
    rows=[];inputs={str(AUX):AUX_SHA};source={p:sha(ROOT/p) for p in SOURCES};overall=Counter();issues=[]
    for item in sorted(aux['records'],key=lambda r:tuple(r['context_key'])):
        run=Path(item['run']);log=run/'release_decisions.jsonl.gz';ledger=run/'delivery_intervals.npz'
        inputs[str(log)]=sha(log);inputs[str(ledger)]=sha(ledger)
        require(inputs[str(ledger)]==item['input_sha256']['delivery_intervals.npz'],'ledger identity mismatch')
        with np.load(ledger,allow_pickle=False) as z:
            ends=z['interval_end_s'];vel=z['object_free_qvel6_end'];valid=z['object_state_valid'];hand=z['target_hand_normal_N']
        require(ends.shape==(32000,) and vel.shape==(32000,6) and bool(valid.all()),'full valid ledger required')
        def physical(t):
            i=int(np.argmin(abs(ends-t)));require(abs(ends[i]-t)<1e-8,'no same-time stored integration-end state')
            return dict(time_s=float(ends[i]),linear_speed_m_s=float(np.linalg.norm(vel[i,:3])),
                angular_speed_norm_rad_s=float(np.linalg.norm(vel[i,3:])),target_hand_normal_N=float(hand[i]))
        count=Counter();phases=Counter();roles=Counter();omitted=Counter();failed=Counter();prefixes=Counter();ranks=Counter()
        arrays={k:[] for k in ('motion_age_s','observed_e_rms_m_s','observed_e_max_m_s','emitted_motion_proxy_rms_m_s','emitted_motion_normalized_loss')}
        snapshots={};last=None;last_motion=None;last_positive=None;last_motion_cert=None;first_p0=None;last_t=None
        with gzip.open(log,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);t=float(r['time_s']);require(t>=12. and t<=16. and (last_t is None or t>last_t),'release log clock/order')
                last_t=t;count['decisions']+=1;phases[r['phase']]+=1;roles[r['task_role']]+=1
                ct=r.get('contact_tasks');ct=ct if isinstance(ct,dict) else {};hd=(r.get('hierarchy') or {}).get('diagnostics') or {}
                h=r.get('hierarchy') or {};motion=(ct.get('diagnostics') or {}).get('motion') or {}
                P=len(ct['pressure_P_N']) if ct.get('available') is True else None
                if P is not None:count['known_P0' if P==0 else 'known_positive_pressure']+=1
                else:count['no_available_contact_task']+=1
                M=ct.get('motion_matrix_Mx21');e=ct.get('motion_error_M_m_s')
                available=M is not None and e is not None
                levels=h.get('levels') or [];cert=h.get('certified_prefix') or []
                names=[c['name'] for c in cert];mcert='public_external_projection_low_motion' in names
                mattempt=any(l.get('name')=='public_external_projection_low_motion' for l in levels)
                budget=(hd.get('motion_task') or {}).get('available') is True
                if h:prefixes[str(h.get('completed_prefix'))+'/'+str(h.get('fully_converged'))]+=1
                if h.get('failed_layer'):failed[str(h['failed_layer'].get('name'))+'/'+str(h['failed_layer'].get('status'))]+=1
                count['motion_budget_constructed']+=int(budget);count['motion_layer_attempted']+=int(mattempt);count['motion_layer_certified']+=int(mcert)
                count['positive_with_motion']+=int(P is not None and P>0 and available)
                count['P0_with_motion']+=int(P==0 and available);count['P0_without_motion']+=int(P==0 and not available)
                snap=dict(time_s=t,phase=r['phase'],task_role=r['task_role'],P=P,motion_available=available,
                    motion_diagnostics=motion,motion_budget_constructed=budget,motion_layer_attempted=mattempt,
                    motion_layer_certified=mcert,certified_level_names=names,completed_prefix=h.get('completed_prefix'),
                    failed_layer=h.get('failed_layer'),proposal_accepted=r['geometry_proposal_accepted'],
                    physical_at_decision=physical(t),physical_scope='offline ledger integration-end state; not controller input')
                if available:
                    M=np.asarray(M,float);e=np.asarray(e,float);require(M.shape==(len(e),21) and len(e)>0,'complete M/e shape')
                    rank=int(np.linalg.matrix_rank(M));ranks[str(rank)]+=1;count['motion_available']+=1
                    acq=float(motion['acquisition_time_s']);age=float(motion['age_s']);arrays['motion_age_s'].append(age)
                    require(abs(t-acq-age)<1e-9,'motion acquisition/age')
                    actual=np.asarray(r['execution_receipt']['actual_reference21'],float);current=np.asarray(r['current_reference21'],float)
                    # Recorded engineering intent evaluated at actual emitted du; no solve or forward.
                    proxy=M@(actual-current)/.005+e
                    snap.update(M_shape=list(M.shape),M_rank=rank,M_rank_role='reference-map algebraic rank, NOT object-twist observability rank',
                        e_M_m_s=e.tolist(),e_rms_m_s=float(np.sqrt(np.mean(e*e))),e_max_m_s=float(max(abs(e))),
                        emitted_motion_proxy_rms_m_s=float(np.sqrt(np.mean(proxy*proxy))),physical_at_acquisition=physical(acq))
                    for dest,field in [('observed_e_rms_m_s','e_rms_m_s'),('observed_e_max_m_s','e_max_m_s'),('emitted_motion_proxy_rms_m_s','emitted_motion_proxy_rms_m_s')]:arrays[dest].append(snap[field])
                    if budget:
                        mt=hd['motion_task'];u=np.asarray(mt['effective_row_units_M']);snap['row_units_M']=mt['row_units_M'];snap['effective_row_units_M']=u.tolist()
                        snap['witness_budget_normalized_abs']=mt['normalized_absolute_budget_M'];snap['emitted_motion_normalized_loss']=float(np.sum((.005*proxy/u)**2));arrays['emitted_motion_normalized_loss'].append(snap['emitted_motion_normalized_loss'])
                    last_motion=snap
                else:
                    count['motion_absent']+=1;reason=motion.get('reason') or motion.get('status') or ('no_contact_task_'+r['task_role']);omitted[str(reason)]+=1
                if P is not None and P>0:last_positive=snap
                if P==0 and first_p0 is None:first_p0=snap
                if mcert:last_motion_cert=snap
                for target in FIXED_TIMES:
                    if abs(t-target)<1e-9:snapshots[str(target)]=snap
                last=snap
        require(count['decisions'] in (800,801) and len(snapshots)==4,'all release clocks/fixed samples required')
        snapshots['last_saved_decision']=last
        raw_hand_idxs=np.flatnonzero((ends>=12.) & (hand>0));last_hand=None if not len(raw_hand_idxs) else float(ends[raw_hand_idxs[-1]])
        final=physical(16.);metrics=item['metrics'];require(abs(final['linear_speed_m_s']-metrics['end_linear_speed_m_s'])<1e-12 and abs(final['angular_speed_norm_rad_s']-metrics['end_angular_speed_rad_s'])<1e-12,'offline final speed differs from fixed-v3 readout')
        overall.update(count)
        rows.append(dict(context_key=item['context_key'],run=str(run),log_sha256=inputs[str(log)],ledger_sha256=inputs[str(ledger)],
            counts=dict(count),phase_counts=dict(phases),role_counts=dict(roles),motion_omission_reasons=dict(omitted),failed_layers=dict(failed),
            prefix_counts=dict(prefixes),M_reference_map_rank_counts=dict(ranks),motion_statistics={k:stats(v) for k,v in arrays.items()},
            fixed_snapshots=snapshots,last_available_motion=last_motion,last_motion_certified=last_motion_cert,
            last_positive_pressure=last_positive,first_knownP0=first_p0,last_raw_target_hand_contact_end_s=last_hand,
            physical_end=final,original_metrics=metrics))
        print('completed',len(rows),item['context_key'],flush=True)
    return plain(dict(schema='v21_motion_observability_diagnostic_v1',complete=True,assigned_count=16,all_assigned_checked=True,
        predetermined_scope=dict(all_release_decisions=True,fixed_snapshot_times_s=FIXED_TIMES,plus_last_saved_decision=True,
            no_winning_sample_selection=True,no_solver_resolve=True,no_model_construction=True),
        source_sha256=source,reader_sha256=sha(__file__),input_sha256=inputs,counts=dict(overall),records=rows,issues=issues,
        limitations=['M rank is actuator reference geometry rank, not a rigid-object 6D observability test.',
          'Device-plane projections aggregate same-digit pressure-weighted robot patches and unknown external actor mixture.',
          'e is acquired at delayed hand sample time; J/B are current robot geometry; emitted proxy is not a finite-motion prediction.',
          'No per-actor target identity or rigid-object twist observer in the online task.',
          'Target-hand contact and object speeds here are offline evaluation only, with cached-contact/end-state phase limitations.',
          'No counterfactual reordering or solver rerun: a hierarchy tradeoff is source-defined, not experimentally isolated causation.'],physics_steps=0))

def markdown(d):
    lines=['# V21运动可观测性与任务优先级诊断','',
      '完整16条所有释放决策；固定12、13、14、15秒及末个保存决策。没有重解求解器、构造模型或修改控制源。在线记录与离线对象速度分列，不能把离线真值作为下一策略可直接读取的状态。','',
      '|环境/运动/seed|有运动任务/全部决策|运动层认证|最后M时刻|末刻M|末速m/s|末角速rad/s|末段低运动|',
      '|---|---:|---:|---:|---|---:|---:|---|']
    for r in d['records']:
        c=r['counts'];m=r['last_available_motion'];f=r['physical_end'];last=r['fixed_snapshots']['last_saved_decision'];end='无' if m is None else f"{m['time_s']:.3f}"
        lines.append(f"|{'/'.join(map(str,r['context_key']))}|{c.get('motion_available',0)}/{c['decisions']}|{c.get('motion_layer_certified',0)}|{end}|{last['motion_available']}|{f['linear_speed_m_s']:.8f}|{f['angular_speed_norm_rad_s']:.8f}|{r['original_metrics']['final_low_motion_suffix_found']}|")
    lines+=['','源码确定事实：','',
      '- `v20_contact_exit/adapter.py:74–129`：每个有正压已接受几何的digit，只有两个设备平面投影。e=同采样机器人点投影速度+相对slip；同digit多patch按压力平均。未知actor混合、不是6D刚体twist。M使用当前机器人J及参考映射B，不能当物体动力响应或物体twist观测矩阵。',
      '- `v20_contact_exit/adapter.py:93–97,122–134,194–212`：没有同digit有效正压几何就不生成该digit运动行；无可用行明确None而不是测量0。',
      '- `v21_stable_handoff/task_solver.py:169–216`：L1只锁正向slack；从该L1当时anchor构建运动代理残差上界，随后最大scalar normal withdrawal先于低运动层，姿态最后。这个预算是不劣于L1 witness的工程代理，不是最小运动目标已优先实现，更不是物理耗散保证。',
      '- M/e消失后，没有接收面上对象角速/滚动的持续反馈任务。末段移动不能被读成“运动误差已经零”。','',
      '证据边界：JSON给全部时点的可用性/认证数量、固定点e与同acq/同decision离线速度、最后M/最后正压/原始target-hand接触时刻。即使某固定点e小，也仅表示已观测设备平面投影小；不推出整物体角速小。没有合法6D外物速度观察矩阵、actor对应或无接触后的持续状态观测，不能从现有日志计算对象twist的完整观测秩。','',
      '对V22的可证伪要求：把“带接触的清离可行性”与“可观测的接收面低运动”显式分开；运动证据缺失必须unknown，不能当任务完成。若改变优先级，应在同snapshot保存运动witness、normal/slack预算和层级attainment，比较是否真实改善离线末速/驻留，同时保留清离/力限/TV代价。仅靠清手后P0不再有e的当前接口，无法持续闭环控制已经脱手物体的转动；还需合法共享传感或明确被动运动假设，不能偷用对象真值。','',
      '本诊断是源结构与时序关联证据，不是“当前normal优先级导致全部残余运动”的隔离因果证明。全部16记录保留，未筛失败或赢样本。']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    out=ROOT/'review/v21_motion_observability_diagnostic_v1.json';md=out.with_suffix('.md')
    require(not out.exists() and not md.exists(),'fresh report paths required')
    result=collect()
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with md.open('x',encoding='utf-8') as f:f.write(markdown(result))
    print(json.dumps(dict(path=str(out),sha256=sha(out),md_sha256=sha(md),counts=result['counts']),ensure_ascii=False))
