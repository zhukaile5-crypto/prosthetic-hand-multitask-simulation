"""Describe a retained development matrix without inferential win claims."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
import numpy as np


def average(items):
    v=[x for x in items if x is not None]
    return None if not v else float(np.mean(v))


def quality(rows,key):
    valid=[r['research_metrics'][key] for r in rows if r['research_metrics'][key]['eligible']]
    return dict(available=len(valid),assigned=len(rows),
        TV_N_s=average([x['bandlimited']['finger_TV_rate_N_s'] for x in valid]),
        reversal_N_s=average([x['bandlimited']['finger_reversal_rate_N_s'] for x in valid]),
        drift_N_s=average([x['bandlimited']['finger_net_drift_rate_N_s'] for x in valid]),
        raw_TV_N_s=average([x['raw_200Hz']['finger_TV_rate_N_s'] for x in valid]),
        mean_force_N=average([x['mean_actual_finger_force_N'] for x in valid]))


def summarize(rows):
    out=[]
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];mm=[r['research_metrics'] for r in rr]
        out.append(dict(method=method,assigned=len(rr),legacy_missions=sum(m['mission_success'] for m in mm),
            full_motion_missions=sum(m['full_motion_mission_success'] for m in mm),
            raw_force_guard_pass=sum(bool(m['safety'] and m['safety']['guard_pass']) for m in mm),
            prepared=sum(m['preparation_passed'] for m in mm),
            static=quality(rr,'static_quality'),dynamic=quality(rr,'dynamic_quality'),holding=quality(rr,'entire_holding_quality'),
            holding_local_excess_Ns=sum(s['local_excess_integral_Ns'] for m in mm if m['safety'] for k,s in m['safety']['stages'].items() if k=='policy_holding_transport'),
            physical_releases=sum(m['release']['physical_success'] for m in mm),
            release_latency_s=average([m['release']['physical_clearance_latency_s'] for m in mm]),
            release_latency_observed=sum(m['release']['physical_clearance_latency_s'] is not None for m in mm),
            failure_reasons={reason:sum(m['failure_reason']==reason for m in mm) for reason in sorted({m['failure_reason'] for m in mm if m['failure_reason']})}))
    return out


def cell(x,digits=4):return '不可估计' if x is None else f'{x:.{digits}f}'


def paired_quality(rows,baseline,candidate,key):
    b={(r['scenario'],r['seed']):r['research_metrics'][key] for r in rows if r['method']==baseline}
    c={(r['scenario'],r['seed']):r['research_metrics'][key] for r in rows if r['method']==candidate}
    assigned=sorted(set(b)|set(c));good=[k for k in assigned if k in b and k in c and b[k]['eligible'] and c[k]['eligible']]
    fields=('finger_TV_rate_N_s','finger_net_drift_rate_N_s','finger_reversal_rate_N_s')
    contrasts={}
    for f in fields:
        x=[b[k]['bandlimited'][f] for k in good];y=[c[k]['bandlimited'][f] for k in good]
        contrasts[f]=dict(baseline_mean=average(x),candidate_mean=average(y),
            mean_paired_difference=average([yy-xx for xx,yy in zip(x,y)]))
    return dict(baseline=baseline,candidate=candidate,window=key,assigned_pairs=len(assigned),
        common_available_pairs=len(good),unavailable_pairs=[list(k) for k in assigned if k not in good],
        contrasts=contrasts,selection='Jointly complete physical force window only; all failures retained in assigned mission/safety denominators; not confirmation inference')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=(ROOT/a.input).resolve();out=(ROOT/a.out).resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or out.exists():raise ValueError('Research-only fresh output required')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    finish=json.loads((source/'execution_complete.json').read_text(encoding='utf-8'))
    rows=json.loads((source/'trials.json').read_text(encoding='utf-8'))
    expected=len(manifest['jobs'])
    if len(rows)!=expected or not finish['execution_complete'] or not finish['sources_unchanged']:raise RuntimeError('Matrix/source incomplete; do not publish a complete-matrix readout')
    method_rows=summarize(rows);scenes=[]
    for scene in sorted({r['scenario'] for r in rows}):scenes.append(dict(scene=scene,methods=summarize([r for r in rows if r['scenario']==scene])))
    tracking=[]
    for r in rows:
        if r['scenario']=='dynamic_transport':tracking.append(dict(method=r['method'],seed=r['seed'],**r['research_metrics']['progress']['dynamic_motion_tracking']))
    pairs=[paired_quality(rows,b,c,k) for b,c in [('ratio','joint'),('ratio_release','joint_release')]
        if any(r['method']==b for r in rows) and any(r['method']==c for r in rows)
        for k in ('static_quality','dynamic_quality','entire_holding_quality')]
    report=dict(phase='development',performance_superiority_proven=False,physical_randomization=manifest['physical_randomization'],
        execution=finish,source_matrix=str(source),matrix_sha256=hashlib.sha256((source/'trials.json').read_bytes()).hexdigest(),
        methods=method_rows,scenarios=scenes,transport_tracking=tracking,paired_quality=pairs,
        known_metric_limitations=['V2 entire-holding symmetric full-record FIR near 12s includes future release samples. Preserve V2 results; raw entire-holding has no such future filtering. Static and dynamic windows are outside this boundary region.'],
        interpretation='Paired descriptive development evidence only. Quality has explicit availability; raw safety and task outcomes retain all assigned trials.')
    out.mkdir(parents=True)
    (out/'readout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 任务驱动架构开发对照','',f'共{len(rows)}条，来源冻结检查通过。物理随机化：{manifest["physical_randomization"]}。这是开发结果，不能作独立确认或总体优势证明。','',
        '|方法|任务完成（含腕跟踪）|全程力门槛|静态TV N/s|动态TV N/s|全持握原始TV N/s|持握超限积分 N·s|释放延迟 s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in method_rows:lines.append(f'|{r["method"]}|{r["full_motion_missions"]}/{r["assigned"]}|{r["raw_force_guard_pass"]}/{r["assigned"]}|{cell(r["static"]["TV_N_s"])}|{cell(r["dynamic"]["TV_N_s"])}|{cell(r["holding"]["raw_TV_N_s"])}|{cell(r["holding_local_excess_Ns"])}|{cell(r["release_latency_s"])}|')
    lines+=['','释放延迟只在实际暴露且清离完成者中可估计，失败不填零。品质窗覆盖数及全部分场景数据见readout.json。原legacy任务字段与新腕部任务字段同时保留。','',
        'V2全持握滤波TV在12秒边界含未来释放样本，原值保留于JSON，但不能把不同释放路径导致的该数值差异归于持握策略。表中全持握使用原先已保存的200Hz原始TV；静态/动态仍使用预定50Hz品质口径。',
        '', '比较joint与ratio隔离相同旧释放下的持握效果；比较joint_release与ratio_release隔离相同新释放下的持握效果。不同释放路径之间的全程护栏收益不能全部归因于持握融合。']
    lines+=['','各方法均值可能对应不同有效窗。配对品质仅使用双方共同完整窗口，同时保留全部任务/安全分母。','',
        '|配对|窗口|共同窗口/分配|基线TV N/s|候选TV N/s|配对差 N/s|','|---|---|---:|---:|---:|---:|']
    for pair in pairs:
        v=pair['contrasts']['finger_TV_rate_N_s']
        lines.append(f'|{pair["candidate"]} vs {pair["baseline"]}|{pair["window"]}|{pair["common_available_pairs"]}/{pair["assigned_pairs"]}|{cell(v["baseline_mean"])}|{cell(v["candidate_mean"])}|{cell(v["mean_paired_difference"])}|')
    (out/'开发对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(out),methods=method_rows),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
