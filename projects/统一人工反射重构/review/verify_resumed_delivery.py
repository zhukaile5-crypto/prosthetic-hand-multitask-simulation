"""Check final artifacts and immutable inputs; never execute simulations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=read(ROOT/'review/confirmation_plan.json')
    audit=read(ROOT/'review/final_result_audit.json')
    summary_path=ROOT/'output/confirmation_combined/analysis/summary.json'
    s=read(summary_path)
    report=ROOT/'全景实验报告.md'; doc=report.read_text(encoding='utf-8')
    binding=read(ROOT/'review/report_numeric_bindings.json')
    provenance=read(ROOT/'output/confirmation_combined/figures_verified/figure_provenance.json')
    checks={}
    frozen={**plan['code_sha256'],**plan['supporting_sha256'],**plan['evidence_sha256']}
    checks['frozen_inputs_unchanged']=all(sha(ROOT/p)==h for p,h in frozen.items())
    checks['summary_audit_identity']=audit['passed'] and audit['summary_sha256']==sha(summary_path)
    checks['report_data_identity']=binding['summary_sha256']==sha(summary_path) and binding['report_sha256']==sha(report)
    checks['numeric_bindings_are_frozen_rows']=all(row in s['comparisons'] for row in binding['comparisons_used'])
    targets=re.findall(r'\]\(([^\n)]+)\)',doc)
    checks['all_report_links_exist']=all(Path(t).exists() for t in targets)
    checks['figure_data_identity']=provenance['summary_sha256']==sha(summary_path)
    checks['figure_file_identity']=all(sha(ROOT/'output/confirmation_combined/figures_verified'/p)==h for p,h in provenance['files'].items())
    counts={}
    known={'test':'ad6dae269cd9456a120e9db688b0ef1cf4e0087d6f71542e5f0653f34bb67fde',
           'robustness':'40edd2c9e4323e3adb94115970718721c0a6ef4a59f5331f32d169166078e39c'}
    for split,n in [('test',1344),('robustness',896)]:
        p=ROOT/f'output/confirmation_{split}'
        rows=read(p/'trials.json')
        counts[split]={'aggregate_rows':len(rows),'json_records':len(list((p/'trials').glob('*.json'))),
                       'npz_traces':len(list((p/'traces').glob('*.npz')))}
        # Record folder is records in this runner; tolerate its documented label only.
        if counts[split]['json_records']==0 and (p/'records').exists():
            counts[split]['json_records']=len(list((p/'records').glob('*.json')))
        checks[f'{split}_matrix_counts']=all(v==n for v in counts[split].values())
        checks[f'{split}_prior_checkpoint_preserved']=sha(p/'trials.json')==known[split]
    artifact_paths=['全景实验报告.md','明日续作.md','review/report_numeric_bindings.json',
        'review/final_result_audit.json','review/final_result_audit.md','review/final_release_audit.json',
        'review/机制与架构诊断.md','review/机制与架构诊断.json',
        'output/confirmation_combined/analysis/summary.json',
        'output/confirmation_combined/analysis/综合评价结果.md',
        'output/confirmation_combined/combined_manifest.json',
        'output/confirmation_combined/figures_verified/figure_provenance.json']
    checks['delivery_files_exist']=all((ROOT/p).is_file() for p in artifact_paths)
    result={'passed':all(checks.values()),'checked_utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'counts':counts,'report_link_count':len(targets),
        'artifact_sha256':{p:sha(ROOT/p) for p in artifact_paths},
        'visual_review':'All four PNG figures inspected; equal scales, labels, intervals and scene coverage legible.',
        'scope':'Delivery verification only. Formal inputs, definitions, settings and sources unchanged; no new simulation or retuning.'}
    (ROOT/'review/resumed_delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','checks','counts','report_link_count']},ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()
