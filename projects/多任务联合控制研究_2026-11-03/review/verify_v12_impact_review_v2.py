"""Read-only saved-record integrity / presentation revision verification. No physics."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    review = ROOT / 'review'
    old = read(review / 'v12_old48_impact_release_bottleneck_v1.json')
    new_path = review / 'v12_old48_impact_release_bottleneck_v2.json'
    new = read(new_path)
    checks = []
    def require(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)
    require('complete18records', len(new['records']) == len(old['records']) == 18)
    require('complete48stage_summary_unchanged', old['full48_stage_summary'] == new['full48_stage_summary'])
    for first, second in zip(old['records'], new['records']):
        identity = (second['method'], second['scenario'], second['seed'], second['motion_mode'])
        require(str(identity) + ' identity', all(first[k] == second[k] for k in ('role', 'method', 'scenario', 'seed', 'motion_mode', 'trace_path', 'trace_sha256')))
        for phase in ('preparation', 'holding', 'release'):
            # Only bracketing frame reference field names/interpretation changed.
            strip = lambda p: {k: v for k, v in p.items() if k != 'bracketing200Hz_states'}
            require(str(identity) + ' ' + phase + ' complete_peak_unchanged', strip(first['peaks'][phase]) == strip(second['peaks'][phase]))
        require(str(identity) + ' immutable_input', digest(second['trace_path']) == second['trace_sha256'] and second['input_unchanged'])
        meta = read(Path(second['trace_path']).with_name('metadata.json'))
        require(str(identity) + ' metadata_identity', meta['completed'] and meta['trace_sha256'] == second['trace_sha256'])
    for relative, expected in (
        ('v12/effort_tracker.py', 'eba6a670a6f4e013368f8ba70e268ac0d2ad02dd90cb01e9d10e0cdc6ee0658f'),
        ('v12_native/effort_tracker.py', '160eb7129237d695240d0639a6562dd7fb21c6bacbaea1c06414cf61ea098915'),
    ):
        require(relative + ' frozen', digest(ROOT / relative) == expected)
    records = read(ROOT / 'report/v12_full48_v1/metric_readout.json')['records']
    per_scene = {scene: {method: sum(r['safety_by_stage']['policy_holding_transport']['local_excess_integral_Ns'] for r in records if r['scenario'] == scene and r['method'] == method)
                         for method in ('ratio_effort', 'fixed_effort', 'fusion_effort')}
                 for scene in sorted({r['scenario'] for r in records})}
    report = review / 'V12轻握收益与碰撞释放峰值瓶颈_20261105.md'
    output = dict(schema='v12_impact_review_verification_v2', physics_steps=0, checks=checks,
                  passed=all(c['passed'] for c in checks), input_readout_sha256=new['input_full48_readout_sha256'],
                  evidence_sha256=digest(new_path), verifier_sha256=digest(__file__), report_sha256=digest(report),
                  holding_local_excess_Ns_by_scene=per_scene,
                  note='Original v1 and frozen executor sources preserved. Frame command labels clarified; raw peak numbers and episodes unchanged.')
    path = review / 'v12_old48_impact_release_validation_v2.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=output['passed'], checks=len(checks), out=str(path), sha256=digest(path), report_sha256=output['report_sha256']), ensure_ascii=False))

if __name__ == '__main__':
    main()
