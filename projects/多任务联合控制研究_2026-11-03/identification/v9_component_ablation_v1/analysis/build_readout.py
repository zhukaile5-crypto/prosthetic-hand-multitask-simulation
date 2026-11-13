"""Author readout only, separate from independent raw audit and frozen83."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
PACKAGE=Path(__file__).resolve().parents[1]
DATA=PACKAGE/'output/causal8_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    finish=read(DATA/'execution_complete.json');assert finish['execution_complete'] and finish['sources_unchanged']
    assert sha(DATA/'trials.json')==finish['trials_sha256']
    plan=read(DATA/'plan.json');rows=read(DATA/'trials.json');assert len(rows)==8
    records=[];index={};common={}
    for row in rows:
        scene=row['scenario'];variant=row['component_variant'];r=row['research_metrics']
        trace=DATA/'runs'/f'{scene}__{variant}__610212/trace.npz'
        with np.load(trace,allow_pickle=False) as z:
            t=z['time'];strings=z['debug_json'];pause=z['transport_paused'];target=z['actual_motion_target'];ref=z['executed_motion_reference']
        ix=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9));debug=[json.loads(str(strings[i])) for i in ix]
        extra=np.array([bool(d.get('component_ablation',{}).get('actual_additional_risk_hold',False)) for d in debug])
        evidence=np.array([bool(d.get('component_ablation',{}).get('counterfactual_risk_hold',False)) for d in debug])
        moving=np.any(abs(target[ix]-target[np.maximum(0,ix-1)])>1e-10,axis=1)
        st=r['safety']['stages'];holding=st['policy_holding_transport'];release=r['release']
        q={}
        for window in ('static_quality','dynamic_quality','entire_holding_quality'):
            saved=r[window]
            q[window]=dict(eligible=saved['eligible'],coverage=saved['coverage'],physical_support_fraction=saved['physical_support_fraction'],
                mean_force_N=saved['mean_actual_finger_force_N'],bandlimited=saved['bandlimited'],raw_200Hz=saved['raw_200Hz'])
        rec=dict(environment=scene,component_variant=variant,flags=row['component_flags'],
            legacy_task=r['mission_success'],full_motion_task=r['full_motion_mission_success'],failure=r['failure_reason'],failure_at=r['failure_at'],
            wrist_failure_at=r['wrist_tracking_guard']['failure_at'],whole_force_guard=r['safety']['guard_pass'],
            physical_release=release['physical_success'],task_qualified_release=release['task_qualified_success'],release_latency_s=release['physical_clearance_latency_s'],
            initial_stability_wait_s=r['initial_stability_wait_s'],whole_peak_by_finger_N=r['safety']['finger_peak_N'],
            whole_local_excess_Ns=r['safety']['local_excess_integral_Ns'],whole_total_excess_Ns=r['safety']['total_excess_integral_Ns'],
            holding_peak_by_finger_N=holding['finger_peak_N'],holding_local_excess_Ns=holding['local_excess_integral_Ns'],holding_total_excess_Ns=holding['total_excess_integral_Ns'],
            stages=st,qualities=q,tracking=r['progress']['dynamic_motion_tracking'],holding_tracking=r['progress']['holding_motion_tracking'],
            moving_requested_ticks=int(moving.sum()),additional_hold_ticks=int(extra.sum()),additional_hold_s=float(extra.sum()*.005),
            additional_hold_while_moving_s=float((extra&moving).sum()*.005),total_pause_s=float(pause[ix].sum()*.005),
            diagnostic_would_hold_on_current_stream_s=float(evidence.sum()*.005),
            max_original_target_reference_gap_while_extra_hold=None if not extra.any() else np.max(abs(target[ix][extra]-ref[np.maximum(0,ix-1)][extra]),axis=0).tolist(),
            risk_evidence_scope='counterfactual latch on THIS realized observation stream, not an alternate-world rollout',trace_sha256=sha(trace))
        records.append(rec);index[(scene,variant)]=rec
    for scene in plan['environments']:
        common[scene]={w:all(index[(scene,v)]['qualities'][w]['eligible'] for v in plan['variants']) for w in ('static_quality','dynamic_quality','entire_holding_quality')}
    def metrics(x):
        out={k:float(x[k]) for k in ('legacy_task','full_motion_task','whole_force_guard','physical_release','task_qualified_release','release_latency_s','initial_stability_wait_s','whole_local_excess_Ns','whole_total_excess_Ns','holding_local_excess_Ns','holding_total_excess_Ns','additional_hold_s','additional_hold_while_moving_s','total_pause_s')}
        out.update(whole_peak_N=max(x['whole_peak_by_finger_N']),holding_peak_N=max(x['holding_peak_by_finger_N']))
        for i,axis in enumerate(('lift_m','WRJ2_rad','WRJ1_rad')):
            out['tracking_RMSE_'+axis]=x['tracking']['rmse'][i];out['tracking_IAE_'+axis]=x['tracking']['absolute_error_integral'][i]
        for w,available in common[x['environment']].items():
            if not available:continue
            for sample in ('bandlimited','raw_200Hz'):
                for k in ('finger_TV_rate_N_s','finger_net_drift_rate_N_s','finger_reversal_rate_N_s'):
                    out[w+'/'+sample+'/'+k]=x['qualities'][w][sample][k]
        return out
    contrasts={}
    for scene in plan['environments']:
        values={v:metrics(index[(scene,v)]) for v in plan['variants']};contrasts[scene]={}
        for k in values['P0H0']:
            y={v:values[v][k] for v in values}
            contrasts[scene][k]=dict(values=y,projection_at_H0=y['P1H0']-y['P0H0'],projection_at_H1=y['P1H1']-y['P0H1'],
                additional_hold_at_P0=y['P0H1']-y['P0H0'],additional_hold_at_P1=y['P1H1']-y['P1H0'],
                interaction=y['P1H1']-y['P1H0']-y['P0H1']+y['P0H0'])
    dest=PACKAGE/'analysis/readout_v1';dest.mkdir(parents=True,exist_ok=False)
    result=dict(schema='v9_component_ablation_author_readout_v1',role='author descriptive readout, not independent audit',created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256={n:sha(DATA/n) for n in ('trials.json','plan.json','manifest.json','execution_complete.json','endpoint_execution_acceptance.json')},analysis_source_sha256=sha(__file__),
        producer_sources_unchanged=finish['sources_unchanged'],producer_endpoint_acceptance=read(DATA/'endpoint_execution_acceptance.json'),records=records,
        all_four_variant_common_quality_windows=common,within_environment_package_contrasts=contrasts,
        limits=['one seen physical/noise seed and two environments','projection includes novelmask/reserve/antiwindup; no single novelmask causal claim','common-force-quality cohort excludes collision for all4 variants, NULL not0','H0 retains original baseline pauses','conditional deterministic package effects, not population significance or a newly selected winning policy','native release unchanged; later release pilots not merged','independent fullraw/source verification is a separate artifact'])
    (dest/'readout.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# V9 2×2组件消融：作者读数','',
      '8/8真实16秒执行完成，83项来源未变；生产端00/11四个参考端点的40项全16秒物理数组均精确匹配。此文件是作者描述性汇总，独立raw校核另行提供。','',
      'P=投影包（含novel mask、最低储备、antiwindup）；H=额外风险运动暂停包。H0保留基线自身暂停。这不是四个新算法，也不能从P包结果单独归因novel mask。','',
      '| 环境 | P/H | 原任务/含腕任务 | 全程限力 | 持握超限 N·s | 全程最大指峰 N | 事件窗50Hz/raw TV N/s | 额外暂停 s | 事件窗RMSE mm / mrad / mrad |',
      '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for x in records:
        q=x['qualities']['dynamic_quality'];tv='NULL / NULL' if not common[x['environment']]['dynamic_quality'] else f"{q['bandlimited']['finger_TV_rate_N_s']:.4f} / {q['raw_200Hz']['finger_TV_rate_N_s']:.4f}"
        rmse=' / '.join(f'{v*1000:.3f}' for v in x['tracking']['rmse'])
        lines.append(f"| {x['environment']} | {x['component_variant']} | {int(x['legacy_task'])}/{int(x['full_motion_task'])} | {int(x['whole_force_guard'])} | {x['holding_local_excess_Ns']:.6f} | {max(x['whole_peak_by_finger_N']):.3f} | {tv} | {x['additional_hold_s']:.3f} | {rmse} |")
    lines += ['', '## 可证伪结论','',
      '在当前两种具体物理条件中，新增暂停H的任务代价可由开关实验复现：friction_drop两种P下H1均触发腕部失败（9.755s），H0均完整成功；collision两种P下H1均在10.305s出现升降跟踪失败并于10.59s触发腕部判据，H0均完整成功。所有8条都真实释放，不能把任务资格受损重记为物理释放失败。','',
      '碰撞初始TH峰四组合都是24.992N，完整力护栏都失败。H在P0下令持握超限积分由.173795增至.351294 N·s；在P1下由.173938增至.467853。P单独在H0几乎不改变该积分，在H1却再增加.116559 N·s，显示两包存在交互，不能视为两个独立可相加的小修补。','',
      '摩擦下降的P包使50Hz事件TV小幅降低（H0约3.9%、H1约3.4%），但raw TV升高；H约减半raw TV，却增加带限TV、峰力和跟踪误差并导致完整任务失败。不同频率指标与可靠性必须并列，不能用单个下降量宣布获胜。','',
      '四组合静态TV均为.145331 N/s，初稳均.200s；这是同一前事件输入的重复，不是独立重复。collision的事件/全持握品质窗均不满足连续支持条件，四组合共同窗为空，NULL不填零，任务/安全分母仍全部保留。','',
      'readout.json包含全部逐指峰、阶段/全程超限积分、真实与任务资格释放、固定50Hz/raw TV净漂移/往复、均力、三轴RMSE/IAE、原目标落后、额外与总暂停，以及每环境P/H简单效应和交互。所有对比只用四组合共同可用的固定品质窗。','',
      '这些是同一已见种子的两个环境下的组件包因果读数，没有总体置信/非劣/优势结论；不把P1H0或其他开关组合包装为新获胜算法。下一架构需要解释移动风险中的进度/力代价及交互，不以调门限掩盖反例。']
    (dest/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(dest),records=len(records),common=common,readout_sha256=sha(dest/'readout.json')),ensure_ascii=False))
if __name__=='__main__':main()
