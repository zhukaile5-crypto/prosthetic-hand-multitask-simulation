"""Final source/data/artifact acceptance; never changes experiment inputs."""
from pathlib import Path
import ast
import hashlib
import json
from physics import ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_hashes(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    names=('Common','EventBaseline','RatioBaseline','AdaptiveConeBaseline','Fusion')
    return {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names}


if __name__=='__main__':
    plan=read(ROOT/'review/formal_plan.json')
    for name,value in plan['code_sha256'].items():
        assert sha(ROOT/name)==value,name
    for name,value in plan['supporting_sha256'].items():
        assert sha(ROOT/name)==value,name
    checked={}
    for label,path in [('original','review/protected_before.json'),('round1','review/round1_before.json')]:
        rows=read(ROOT/path)
        current=[dict(Path=r['Path'],Hash=sha(r['Path'])) for r in rows]
        assert all(sha(r['Path'])==r['Hash'].lower() for r in rows),label
        checked[label]=dict(files=len(rows),all_hashes_unchanged=True)
        (ROOT/'review'/f'{label}_after.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    old=class_hashes(ROOT/'review/development_runtime_v1/controllers.py')
    new=class_hashes(ROOT/'controllers.py')
    assert len(old)==5 and old==new
    suites={}
    for name,expected in [('test_holdout_v2',1920),('robustness_holdout_v2',960)]:
        directory=ROOT/'output'/name
        manifest,audit=read(directory/'manifest.json'),read(directory/'audit.json')
        assert manifest['completed_trials']==manifest['expected_trials']==expected
        assert manifest['code_sha256']==manifest['final_code_sha256']==plan['code_sha256']
        assert audit['passed'] and audit['results']['independently_recomputed_trace_integrals']==expected
        assert audit['repeatability']['identical_physical_and_control_results']
        suites[name]=dict(trials=expected,manifest_sha256=sha(directory/'manifest.json'),
                          trials_sha256=sha(directory/'trials.json'),audit_sha256=sha(directory/'audit.json'))
    figures=read(ROOT/'output/final_figures/figure_sources.json')
    replay=read(ROOT/'output/final_replay/render_manifest.json')
    artifacts=[p for folder in ['output/final_figures','output/final_replay']
               for p in (ROOT/folder).glob('*') if p.suffix in ('.png','.svg','.gif')]
    assert sum(p.suffix=='.gif' for p in artifacts)==2
    assert sum(p.suffix=='.svg' for p in artifacts)>=4
    report=ROOT/'第二轮结果与改进路线.md'
    assert report.is_file() and report.stat().st_size>1000
    result=dict(passed=True,total_formal_trials=2880,protected=checked,
                baseline_class_ast_hashes_unchanged=old,formal_suites=suites,
                artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
                report_sha256=sha(report),figures_manifest_sha256=sha(ROOT/'output/final_figures/figure_sources.json'),
                replay_manifest_sha256=sha(ROOT/'output/final_replay/render_manifest.json'),
                caveat='Structural/numerical checks do not prove real contact calibration or object damage safety.')
    (ROOT/'review/final_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,trials=2880,artifacts=len(artifacts),protected=checked),ensure_ascii=False))
