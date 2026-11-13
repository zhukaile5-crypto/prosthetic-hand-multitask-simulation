"""Materialize reviewable policy bodies from an immutable controller snapshot.

Scalar contact extraction and the model-dependent support-floor bound change.
All objectives, PI regulation, continuity terms, reserves and release priority remain.
The physical predictor/observer/executor are the shared migration adapter.
"""
from pathlib import Path
import ast,hashlib,json

ROOT=Path(__file__).resolve().parent
def main():
    source=ROOT/'vendor/frozen_controllers.py';text=source.read_text(encoding='utf-8')
    tree=ast.parse(text)
    def method(cls,name):
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        member=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
        lines=text.splitlines(keepends=True)
        return ''.join(lines[member.lineno-1:member.end_lineno])
    ratio=method('RatioController','_ratio_target')
    old='np.linalg.norm(self.state.tangential[supported], axis=1).sum()'
    assert ratio.count(old)==1
    ratio=ratio.replace(old,'self.state.shear_magnitude[supported].sum()')
    unified=method('UnifiedCoordinator','_decide')
    substitutions={
        'np.linalg.norm(self.state.tangential, axis=1)':'self.state.shear_magnitude',
        'np.linalg.norm(self.state.slip,axis=1)':'self.state.slip_magnitude'}
    for old,new in substitutions.items():
        assert unified.count(old)==1,(old,unified.count(old));unified=unified.replace(old,new)
    old_floor='''            attainable = (P@np.tile(caps, h)+c).reshape(h, 5)
            for i in selected:
                maximum_reference = self.state.normal[i]+self.state.force_gain[i]*max(
                    0., min(self.g.close_rate*self.g.decision_period,
                            self.state.closure_max[i]-self.state.alpha[i]))/max(.01, self.g.baseline_tracking_gain)
                reference_floor[:, i] = min(self.g.support_contact_floor, maximum_reference)
                if active[i]:
                    max_action_force = self.state.normal[i]+self.state.force_gain[i]*self.g.close_rate*self.g.decision_period*np.arange(1,h+1)
                    force_floor[:, i] = np.minimum(self.g.support_contact_floor,
                                                  np.minimum(attainable[:, i], max_action_force))
'''
    assert unified.count(old_floor)==1
    unified=unified.replace(old_floor,'            force_floor, reference_floor = self._support_floors(P,c,caps,selected,active)\n')
    output='''# Generated, reviewable migration of the existing two policy bodies.
import time
import numpy as np
from vendor.frozen_controllers import RatioController, UnifiedCoordinator, solve_qp, QPResult, project_force
from controller_adapter import HandControllerMixin

class MigratedRatio(HandControllerMixin, RatioController):
'''+ratio+'''

class MigratedUnified(HandControllerMixin, UnifiedCoordinator):
'''+unified+'\n'
    destination=ROOT/'policy_sources.py';destination.write_text(output,encoding='utf-8')
    record={'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'policy_body_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'substitutions':{'ratio':{'norm(projected 2D shear)':'sum of measured patch shear magnitudes'},
                         'unified':substitutions,
                         'model_bound':'positive-diagonal support-floor formula replaced with shared signed neutral-reference prediction'},
        'unchanged':'PI rule and all unified objective/constraint terms; no stable-corridor V3 policy',
        'shared_physical_migration':'controller_adapter.py, force_model.py, state_estimator.py, sensors.py',
        'prepared_without_performance_runs':True}
    (ROOT/'review/policy_migration_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'prepared':destination.name,'original_sha256':record['original_sha256']}))
if __name__=='__main__':main()
