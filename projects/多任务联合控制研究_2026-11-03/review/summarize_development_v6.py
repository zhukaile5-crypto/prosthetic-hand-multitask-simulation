"""Source-backed, complete-matrix V3 development readout; no win inference."""
from pathlib import Path
import argparse,hashlib,json,sys
from collections import Counter
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
from summarize_development_v5 import summarize,paired_quality,cell,average


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(row):return (row['scenario'],row['method'],row['seed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=(ROOT/a.input).resolve();out=(ROOT/a.out).resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or out.exists():
        raise ValueError('Fresh research-only output required')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    finish=json.loads((source/'execution_complete.json').read_text(encoding='utf-8'))
    rows=json.loads((source/'trials.json').read_text(encoding='utf-8'))
    if (not finish['execution_complete'] or not finish['sources_unchanged'] or
        Counter(map(key,rows))!=Counter(map(key,manifest['jobs'])) or finish['trials_sha256']!=sha(source/'trials.json')):
        raise ValueError('Incomplete/unmatched matrix or source; retain errors, no complete readout')
    if any(r['research_metrics']['schema']!='joint-hand-research-evaluation-v3' or
        r['research_metrics']['filter_contract'].get('version')!=3 for r in rows):
        raise ValueError('V3-only readout; never silently mix historical evaluations')
    methods=summarize(rows)
    for r in methods:
        mm=[x['research_metrics'] for x in rows if x['method']==r['method']]
        r['initial_stability_wait_s']=average([m['initial_stability_wait_s'] for m in mm])
        r['event_recovery_statuses']=dict(Counter(e['status'] for m in mm for e in m['event_recovery']))
        r['whole_force_excess_Ns']=sum(m['safety']['local_excess_integral_Ns'] for m in mm if m['safety'])
    pairs=[paired_quality(rows,b,'joint',k) for b in ('ratio','passive')
        if any(r['method']==b for r in rows)
        for k in ('static_quality','dynamic_quality','entire_holding_quality')]
    for pair in pairs:
        v=pair['contrasts']['finger_TV_rate_N_s'];b=v['baseline_mean'];c=v['candidate_mean']
        v['relative_change_percent']=None if b is None or b==0 else 100.*(c/b-1.)
    tracking=[dict(scenario=r['scenario'],method=r['method'],seed=r['seed'],
        **r['research_metrics']['progress']['dynamic_motion_tracking']) for r in rows if r['scenario']=='dynamic_transport']
    by_scene=[dict(scenario=s,methods=summarize([r for r in rows if r['scenario']==s]))
        for s in sorted({r['scenario'] for r in rows})]
    identity={n:sha(source/'sources'/n) for n in manifest['research_source_sha256']}
    if identity!=manifest['research_source_sha256']:raise ValueError('Execution snapshot changed')
    report=dict(schema='v6-complete-development-readout-v1',phase='development',
        performance_superiority_proven=False,source_matrix=str(source),matrix_sha256=sha(source/'trials.json'),
        source_manifest_sha256=sha(source/'manifest.json'),execution=finish,
        physical_randomization=manifest['physical_randomization'],
        physical_seed_clusters=sorted({r['seed'] for r in rows}),
        methods=methods,scenarios=by_scene,paired_quality=pairs,transport_tracking=tracking,
        evaluator_contract='V3 lifecycle-isolated50Hz FIR; original full raw safety and all assigned failures retained',
        interpretation='Descriptive paired development only. Reused scenarios within a physical seed are clustered, not independent physical draws. Force quality cannot compensate mission/safety loss.',
        report_sources={Path(__file__).name:sha(__file__),'summarize_development_v5.py':sha(Path(__file__).with_name('summarize_development_v5.py'))})
    lines=['# V6整体架构：完整开发对照','',f'共{len(rows)}条，物理种子簇{report["physical_seed_clusters"]}。完整分配及冻结来源已核对。这是开发比较，尚不构成总体优势或独立确认。','',
        '|方法|完成/分配|全程原始力合格/分配|静态50Hz TV N/s|动态50Hz TV N/s|全持握50Hz TV N/s|持握超限 N·s|初始稳定等待 s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in methods:
        lines.append(f'|{r["method"]}|{r["full_motion_missions"]}/{r["assigned"]}|{r["raw_force_guard_pass"]}/{r["assigned"]}|{cell(r["static"]["TV_N_s"])}|{cell(r["dynamic"]["TV_N_s"])}|{cell(r["holding"]["TV_N_s"])}|{cell(r["holding_local_excess_Ns"])}|{cell(r["initial_stability_wait_s"])}|')
    lines+=['','上表各方法有效窗可能不同；下表仅比较双方共同完整窗口。所有失败仍保留在任务和安全分母。初始未稳定/未准入按预定1.4秒限制等待计入；事件恢复另列状态，不能与初始等待混用。','',
        '|候选/对照|窗口|共同窗口/分配|对照 TV N/s|候选 TV N/s|相对变化|','|---|---|---:|---:|---:|---:|']
    for pair in pairs:
        v=pair['contrasts']['finger_TV_rate_N_s']
        lines.append(f'|joint/{pair["baseline"]}|{pair["window"]}|{pair["common_available_pairs"]}/{pair["assigned_pairs"]}|{cell(v["baseline_mean"])}|{cell(v["candidate_mean"])}|{cell(v["relative_change_percent"],2)}%|')
    lines+=['','释放与握持使用相同路径才可直接归因。全程原始安全保留释放峰值；持握超限单列，不能据此免除释放失败。V3保持50Hz/50ms评价核并隔离生命周期边界，旧V1/V2实验不改写。','',
        '|动态运输|种子|升降RMSE mm|WRJ2 RMSE mrad|WRJ1 RMSE mrad|','|---|---:|---:|---:|---:|']
    for r in sorted(tracking,key=lambda x:(x['seed'],x['method'])):
        vals=[None]*3 if not r['available'] else [1000.*v for v in r['rmse']]
        lines.append(f'|{r["method"]}|{r["seed"]}|{cell(vals[0],3)}|{cell(vals[1],3)}|{cell(vals[2],3)}|')
    lines+=['','全部场景结果、原始TV、净漂移/往复分解、恢复状态、物理释放及其条件延迟、覆盖数和源摘要见readout.json。不得只选有利窗口、频段或种子解释优势。']
    out.mkdir(parents=True)
    (out/'readout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'开发对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),methods=methods,paired_quality=pairs),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
