"""Prove that later coordinator revisions did not change tuned baseline classes."""
from pathlib import Path
import json
from acceptance_round2 import ROOT, read, sha, class_hashes


if __name__=='__main__':
    before=ROOT/'review/development_runtime_v3/controllers.py'
    after=ROOT/'controllers.py'
    baseline=read(ROOT/'output/baseline_calibration_v2_clock/manifest.json')
    assert sha(before)==baseline['code_sha256']['controllers.py']
    old,new=class_hashes(before),class_hashes(after)
    assert len(old)==5 and old==new
    for name in ('physics.py','scenarios.py','run_experiments.py'):
        assert sha(ROOT/name)==baseline['code_sha256'][name],name
    result=dict(passed=True,classes_unchanged=True,before_sha256=sha(before),after_sha256=sha(after),
                before_class_sha256=old,after_class_sha256=new,
                baseline_calibration_manifest_sha256=sha(ROOT/'output/baseline_calibration_v2_clock/manifest.json'),
                shared_physics_scenarios_runner_unchanged=True)
    (ROOT/'review/baseline_kernel_equivalence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
