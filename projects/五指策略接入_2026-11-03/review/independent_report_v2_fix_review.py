"""Exact patch and actual canonical point audit for display-only report fix."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    old_path=ROOT/'analysis_report.py';new_path=ROOT/'analysis_report_v2.py'
    old=old_path.read_text(encoding='utf-8');new=new_path.read_text(encoding='utf-8')
    start="            comparisons.append(dict(group=group, title=title, endpoint=endpoint, label=label, unit=unit,"
    guard="            if 'endpoint' in stats and stats['endpoint'] != endpoint:\n                raise ValueError('Canonical comparison endpoint identity disagrees with the requested endpoint')\n            stats_without_endpoint = {name:value for name,value in stats.items() if name != 'endpoint'}\n"
    expected=old.replace(start,guard+start).replace("else 'descriptive secondary', **stats))","else 'descriptive secondary', **stats_without_endpoint))")
    checks=dict(old_frozen_sha=sha(old_path)=='8d6ab4ca8d1eba001924bc297a75f776ade0cdc57924b26a857f7cfa35c5e8a9',
        new_sha=sha(new_path)=='ffa34e2c96a0fd5f337e5a25252e4b9c05b9209dcaf4063ba52fd9c8a70fbe14',
        only_identity_guard_and_duplicate_key_removal=expected==new)
    canonical_path=ROOT/'output/confirmation_integration_v2/analysis/summary.json'
    canonical=json.loads(canonical_path.read_text(encoding='utf-8'))
    old_module=runpy.run_path(str(old_path));new_module=runpy.run_path(str(new_path))
    try:old_module['collect_metrics'](canonical['records'],canonical)
    except TypeError as exc:checks['original_failure_reproduced_offline']='endpoint' in str(exc)
    else:checks['original_failure_reproduced_offline']=False
    comparisons,denominators,stages=new_module['collect_metrics'](canonical['records'],canonical)
    selected={(r['group'],r['endpoint']):r for r in comparisons if r['group'] in ('main','boundary')}
    canonical_fields=('pairs','clusters','baseline_mean','candidate_mean','difference','ci95','p','p_holm')
    for official in canonical['comparisons']:
        if 'cohort' in official:continue
        actual=selected[(official['domain'],official['endpoint'])]
        checks[f"canonical/{official['domain']}/{official['endpoint']}/all_values_identity"]=all(actual.get(k)==official.get(k) for k in canonical_fields)
    for count in denominators:
        group=count['group'];rows=[r for r in canonical['records'] if r['method']==count['method']]
        if group.startswith('main'):rows=[r for r in rows if not r['boundary']]
        elif group.startswith('boundary'):rows=[r for r in rows if r['boundary']]
        if '/' in group:rows=[r for r in rows if r['shape']==group.split('/')[1]]
        checks[f"denominator/{group}/{count['method']}"]=(count['assigned']==len(rows) and
            count['task_success']==sum(r['task_success'] for r in rows) and count['static_eligible']==sum(r['static_eligible'] for r in rows) and
            count['release_planned']==sum(r['release_planned'] for r in rows))
    # Every numeric statistical helper, source loader and metric definition
    # function is byte-for-byte preserved by the exact source transformation.
    checks['all_original_synthetic_checks']=new_module['self_test']()['passed']
    original_bootstrap=new_module['bootstrap_pairs']
    def inconsistent(pairs,*args,**kwargs):return dict(original_bootstrap(pairs,*args,**kwargs),endpoint='wrong_identity')
    new_module['collect_metrics'].__globals__['bootstrap_pairs']=inconsistent
    try:new_module['collect_metrics'](canonical['records'],canonical)
    except ValueError as exc:checks['inconsistent_endpoint_rejected']='endpoint identity' in str(exc)
    else:checks['inconsistent_endpoint_rejected']=False
    certificate=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),old_source_sha256=sha(old_path),new_source_sha256=sha(new_path),
        canonical_sha256=sha(canonical_path),own_script_sha256=sha(__file__),
        scope='Offline display dictionary identity defect only; exact patch verified. No experiment, physical model, controller, metric definition, number, CI or assigned denominator changed. Canonical records used only for report construction and equality checks.')
    out=ROOT/('review/independent_report_v2_fix_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(certificate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=certificate['passed'],checks=len(checks),failed=certificate['failed'],path=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':main()
