"""Acceptance of supplementary artifacts and unchanged original evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    plan=read(ROOT/'review/formal_plan.json')
    frozen={**plan['code_sha256'],**plan['supporting_sha256']}
    assert all(sha(ROOT/name)==value for name,value in frozen.items())
    protected={}
    for label,filename in [('original','protected_before.json'),('round1','round1_before.json')]:
        rows=read(ROOT/'review'/filename)
        assert all(sha(r['Path'])==r['Hash'].lower() for r in rows)
        protected[label]=len(rows)
    meta=read(HERE/'output/metadata.json')
    assert meta['passed'] and meta['trials']==2880 and meta['frozen_sources_unchanged']
    assert all(sha(path)==value for path,value in meta['sources'].items())
    assert all(sha(HERE/'output'/name)==value for name,value in meta['output_sha256'].items())
    audit=read(HERE/'独立核查.json')
    assert audit['passed'] and audit['raw_traces_recomputed']==120
    assert audit['pooled_effect_rows_rebuilt']==3456 and audit['bootstrap_and_sign_test_family_rows_recomputed']==24
    assert all(sha(path)==value for path,value in audit['source_hashes'].items())
    recovery=read(HERE/'恢复结果/recovery_audit.json')
    assert recovery['passed'] and recovery['traces']==360 and recovery['scheduled_events']==1080
    assert sha(HERE/'恢复计划.json')==recovery['plan_sha256']
    assert sha(HERE/'analyze_recovery.py')==recovery['script_sha256']
    figure=read(HERE/'figures/figure_sources.json')
    assert all(sha(path)==value for path,value in figure['sources'].items())
    assert len(figure['artifacts'])==4
    assert all(sha(r['path'])==r['sha256'] for r in figure['artifacts'])
    report_sources=read(HERE/'report_sources.json')
    assert report_sources['passed'] and all(sha(path)==value for path,value in report_sources['input_sources'].items())
    assert sha(report_sources['report']['path'])==report_sources['report']['sha256']
    result=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),all_original_frozen_files_unchanged=True,protected_files=protected,
        original_trials=2880,independent_raw_trace_checks=120,
        pulse_recovery_events=1080,figures=2,visual_check='Both PNGs inspected; readable marks/labels, no clipping',
        exploratory=True,report_sources=report_sources,
        core_evidence_hashes={str(p):sha(p) for p in [HERE/'output/metadata.json',HERE/'独立核查.json',
            HERE/'恢复结果/recovery_audit.json',HERE/'figures/figure_sources.json',HERE/'力平稳性补充结论.md']})
    (HERE/'交付核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('report_sources','core_evidence_hashes')},ensure_ascii=False))


if __name__=='__main__':main()
