"""V9 physical route retained as method=joint; replace controller at time zero."""
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v9.runtime import V9Session,source_files as v9_source_files
from .component_controller import ComponentController,VARIANTS

PACKAGE=Path(__file__).resolve().parent
STATIC=('__init__.py','component_controller.py','ablation_runtime.py','run_ablation.py','verify.py','plan.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_plan():return json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'))
def source_files():
    plan=read_plan();base=v9_source_files(['joint'])
    if set(base)!=set(plan['frozen_v9_source_sha256']):raise RuntimeError('Frozen77 V9 source set changed')
    for k,p in base.items():
        if sha(p)!=plan['frozen_v9_source_sha256'][k]:raise RuntimeError('Frozen V9 source changed: '+k)
    return {**base,**{'identification/v9_component_ablation_v1/'+n:PACKAGE/n for n in STATIC}}
def source_identity():return {k:sha(p) for k,p in sorted(source_files().items())}

class ComponentSession(V9Session):
    def __init__(self,component_variant,**kwargs):
        if component_variant not in VARIANTS:raise ValueError('Unknown component variant')
        if kwargs.get('method','joint')!='joint':raise ValueError('Physical route must remain joint')
        kwargs['method']='joint';kwargs['motion_mode']='transport'
        super().__init__(**kwargs)
        if self.time!=0. or self.tick!=0 or self.trace or self.plant.substep_records:
            raise RuntimeError('Controller replacement is allowed only before any physics tick')
        self.component_variant=component_variant
        self.controller=ComponentController(self.calibration,self.limits,self.budget,
            deepcopy(self.actual_ratio_task_gains),component_variant)
        self.gains=asdict(self.controller.g)
        self.run_id=self.run_id+'__'+component_variant
        self.research_sources=source_identity()

    def metadata(self):
        value=super().metadata()
        value.update(schema='v9_component_package_ablation_trial_v1',
            component_variant=self.component_variant,
            component_flags=dict(zip(('projection_package','additional_risk_motion_hold'),VARIANTS[self.component_variant])),
            component_registration_sha256=sha(PACKAGE/'plan.json'),
            ablation_scope='two component packages, not novel-mask-only causality; internal method remains joint')
        return value
