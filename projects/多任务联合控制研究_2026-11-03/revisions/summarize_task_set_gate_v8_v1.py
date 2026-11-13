"""Complete paired development report; fixed V3 metrics, all comparators."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
from summarize_development_v5 import summarize,paired_quality,average,cell

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return r['scenario'],r['method'],r['seed']

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();src=(ROOT/a.input).resolve();dst=(ROOT/a.out).resolve()
    if not src.is_relative_to(ROOT) or not dst.is_relative_to(ROOT) or dst.exists():raise ValueError('New research-only output required')
    manifest=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    end=json.loads((src/'execution_complete.json').read_text(encoding='utf-8'))
    rows=json.loads((src/'trials.json').read_text(encoding='utf-8'))
    assert end['execution_complete'] and end['sources_unchanged']
    assert sha(src/'trials.json')==end['trials_sha256']
    assert Counter(map(key,rows))==Counter(map(key,manifest['jobs']))
    assert all(r['research_metrics']['schema']=='joint-hand-research-evaluation-v3' for r in rows)
    assert all(sha(src/'sources'/n)==v for n,v in manifest['research_source_sha256'].items())
    methods=summarize(rows)
    for m in methods:
        mm=[r['research_metrics'] for r in rows if r['method']==m['method']]
        m['initial_stability_wait_s']=average([x['initial_stability_wait_s'] for x in mm])
        m['event_recovery_statuses']=dict(Counter(e['status'] for x in mm for e in x['event_recovery']))
        m['whole_force_excess_Ns']=sum(x['safety']['local_excess_integral_Ns'] for x in mm if x['safety'])
    pairs=[paired_quality(rows,b,'joint',w) for b in sorted({r['method'] for r in rows}-{'joint'})
           for w in ('static_quality','dynamic_quality','entire_holding_quality')]
    for q in pairs:
        for c in q['contrasts'].values():
            b=c['baseline_mean'];c['relative_change_percent']=None if b in (None,0.) else 100*(c['candidate_mean']/b-1.)
    tracking=[dict(scenario=r['scenario'],method=r['method'],seed=r['seed'],
        **r['research_metrics']['progress']['dynamic_motion_tracking']) for r in rows if r['scenario']=='dynamic_transport']
    report=dict(schema='complete-task-set-development-readout-v1',phase='development',superiority_proven=False,
        source_matrix=str(src),matrix_sha256=sha(src/'trials.json'),source_manifest_sha256=sha(src/'manifest.json'),
        execution=end,physical_seed_clusters=sorted({r['seed'] for r in rows}),methods=methods,paired_quality=pairs,
        transport_tracking=tracking,scenarios=[dict(scenario=s,methods=summarize([r for r in rows if r['scenario']==s]))
            for s in sorted({r['scenario'] for r in rows})],
        limitations=['Seen development, not independent confirmation. Scenes within the same physical seed are clustered.',
        'Same-seed static pre-event windows may be exactly repeated; no independent replication claim.',
        'Mission/physical release, full raw force guard, quality coverage and tracking are separate. No scalar weighting can erase a guard or mission failure.',
        'Physical force forecasts are explicitly unavailable for V8 empirical task allocation. No safety certificate or global model admission.'],
        reporter_sources={Path(__file__).name:sha(__file__),'summarize_development_v5.py':sha(Path(__file__).with_name('summarize_development_v5.py'))})
    lines=['# 完整任务集合开发对照','',f'共{len(rows)}条，种子簇{report["physical_seed_clusters"]}；所有分配、原始失败及冻结来源均核对。属于已见开发检验，尚无总体优势证明。','',
        '|方法|运动任务完成/分配|全程原始力合格/分配|静态50Hz TV N/s|动态50Hz TV N/s|持握50Hz TV N/s|持握超限 N·s|初稳等待 s|释放延迟 s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in methods:
        lines.append(f'|{r["method"]}|{r["full_motion_missions"]}/{r["assigned"]}|{r["raw_force_guard_pass"]}/{r["assigned"]}|{cell(r["static"]["TV_N_s"])}|{cell(r["dynamic"]["TV_N_s"])}|{cell(r["holding"]["TV_N_s"])}|{cell(r["holding_local_excess_Ns"])}|{cell(r["initial_stability_wait_s"])}|{cell(r["release_latency_s"])}|')
    lines+=['','任务完成字段不包含材料合格；两列必须同时看。释放延迟仅统计实际清离者，其覆盖数在JSON保留，失败不填零。各方法可用窗不同；成对表只取共同完整窗，原失败仍保留在全部分配分母。','',
        '|joint/对照|窗口|共同窗/分配|对照 TV N/s|joint TV N/s|相对变化|','|---|---|---:|---:|---:|---:|']
    for q in pairs:
        c=q['contrasts']['finger_TV_rate_N_s']
        lines.append(f'|{q["baseline"]}|{q["window"]}|{q["common_available_pairs"]}/{q["assigned_pairs"]}|{cell(c["baseline_mean"])}|{cell(c["candidate_mean"])}|{cell(c["relative_change_percent"],2)}%|')
    lines+=['','相同预览强对照为ratio_preview/passive_preview；其原力策略不变。动态运动实际跟踪、品质均力、200Hz原始TV、净漂移/往复、事件恢复及分场景结果见JSON。静态四场景若完全重复，仅是一条轨迹的重复记录。','',
        '|动态运输方法|种子|升降RMSE mm|WRJ2 RMSE mrad|WRJ1 RMSE mrad|','|---|---:|---:|---:|---:|']
    for r in sorted(tracking,key=lambda x:(x['seed'],x['method'])):
        values=[None]*3 if not r['available'] else [1000*x for x in r['rmse']]
        lines.append(f'|{r["method"]}|{r["seed"]}|{cell(values[0])}|{cell(values[1])}|{cell(values[2])}|')
    dst.mkdir(parents=True);(dst/'readout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (dst/'开发对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(dst),methods=methods),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
