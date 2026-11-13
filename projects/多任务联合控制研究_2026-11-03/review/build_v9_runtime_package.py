"""Generate only new V9 runtime/runner/init/QA from frozen V8; no physics."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'v9';folder.mkdir(exist_ok=True)
texts={}
for filename in ['runtime.py','run_development.py','__init__.py']:
    text=(ROOT/'v8'/filename).read_text(encoding='utf-8-sig').replace('v8','v9').replace('V8','V9')
    if filename=='runtime.py':
        text=text.replace("    pending += [ROOT/'v7/belief.py',ROOT/'v7/__init__.py']\n",'',1)
        text=text.replace("RATIO_TASK_GAIN_KEYS=","MOTION_MODES=('scenario_default','hold','transport')\nRATIO_TASK_GAIN_KEYS=",1)
        text=text.replace("    def __init__(self,method='joint',**kwargs):\n","    def __init__(self,method='joint',**kwargs):\n        self.motion_mode=kwargs.pop('motion_mode','scenario_default')\n        if self.motion_mode not in MOTION_MODES:raise ValueError('Unknown public motion mode')\n",1)
        text=text.replace("self.observer_warmup_s=.5 if holding_method=='joint' else .2","self.observer_warmup_s=.2",1)
        old="""        self.user_trajectory=(public_transport_trajectory(self.config.lift_target)
            if self.mode=='scheduled' and self.scenario_name=='dynamic_transport'
            else constant_trajectory(self.config.lift_target))"""
        new="""        transport=(self.motion_mode=='transport' or self.motion_mode=='scenario_default'
            and self.mode=='scheduled' and self.scenario_name=='dynamic_transport')
        self.user_trajectory=(public_transport_trajectory(self.config.lift_target)
            if transport else constant_trajectory(self.config.lift_target))"""
        if old not in text:raise ValueError('Frozen trajectory constructor changed')
        text=text.replace(old,new,1)
        text=text.replace("value.update(schema='v9_task_research_trial_v1',","value.update(schema='v9_task_research_trial_v1',motion_mode=self.motion_mode,",1)
        text=text.replace('state update only before handoff; joint0.5s versus traditional0.2s; no preparation action change','state update only before handoff; all methods0.2s; joint uses the single baseline force posterior')
        text=text.replace("'foundation_unique_public_tactile_belief' if self.method=='joint'","'baseline_coupled_force_posterior_only' if self.method=='joint'",1)
        text=text.replace("        if not scheduled:\n","        if self.motion_mode in ('hold','transport'):\n            public_target=self.user_trajectory.position_at(t)\n            lift_ref=float(public_target[0]);wrist=list(public_target[1:])\n        if not scheduled:\n",1)
    if filename=='run_development.py':
        text=text.replace('V9Session,METHODS,source_files,source_identity','V9Session,METHODS,MOTION_MODES,source_files,source_identity',1)
        text=text.replace("    parser.add_argument('--out',required=True)\n","    parser.add_argument('--out',required=True)\n    parser.add_argument('--motion-mode',choices=MOTION_MODES,default='scenario_default')\n",1)
        text=text.replace("seed=job['seed'],split=job['split'])","seed=job['seed'],split=job['split'],motion_mode=job['motion_mode'])",1)
        text=text.replace("result['research_metrics']=evaluate_research(destination/'trace.npz')","result['research_metrics']=evaluate_research(destination/'trace.npz')\n        result['motion_mode']=job['motion_mode']",1)
        text=text.replace("split=split,destination=str(destination/'runs'","split=split,motion_mode=args.motion_mode,destination=str(destination/'runs'",1)
        text=text.replace("jobs=jobs,planned=len(jobs),physical_randomization=args.physical_randomization,workers=args.workers,","jobs=jobs,planned=len(jobs),physical_randomization=args.physical_randomization,workers=args.workers,motion_mode=args.motion_mode,",1)
        text=text.replace('observer_warmup_s=dict(joint=.5,traditional=.2)','observer_warmup_s=dict(joint=.2,traditional=.2)',1)
        text=text.replace("jobs=jobs,created_utc=datetime.now(timezone.utc).isoformat(),research_source_sha256=identity,","jobs=jobs,created_utc=datetime.now(timezone.utc).isoformat(),research_source_sha256=identity,motion_mode=args.motion_mode,",1)
        text=text.replace('seed=args.seeds[0],split=split)','seed=args.seeds[0],split=split,motion_mode=args.motion_mode)',1)
    ast.parse(text);texts[folder/filename]=text

qa=(ROOT/'review/verify_v8_runtime.py').read_text(encoding='utf-8-sig').replace('v8','v9').replace('V8','V9')
qa=qa.replace("'v9/controller.py','v9/feedback.py','v7/belief.py','v7/__init__.py',","'v9/controller.py','v9/feedback.py',",1)
qa=qa.replace("unique_new_joint_state=type(joint.controller.state).__module__=='v9.controller'","unique_new_joint_state=type(joint.controller.state.force_state) is type(old.controller.state) and joint.controller.nominal.state is joint.controller.state",1)
qa=qa.replace('declared_half_second_warmup=joint.observer_warmup_s==.5','declared_half_second_warmup=joint.observer_warmup_s==.2',1)
ast.parse(qa);texts[ROOT/'review/verify_v9_runtime.py']=qa
for path in texts:
    if path.exists():raise FileExistsError('Retain existing V9 file: '+str(path))
for path,text in texts.items():path.write_text(text,encoding='utf-8')
print('Created new V9 runtime/runner/init/QA; frozen V7/V8 untouched; no physics')
