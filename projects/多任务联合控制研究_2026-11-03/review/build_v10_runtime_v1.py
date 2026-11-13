"""Create isolated V10 execution wiring from frozen V9, never advance physics."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
outputs={}
for name in ('__init__.py','runtime.py','run_development.py'):
    value=(ROOT/'v9'/name).read_text(encoding='utf-8-sig').replace('v9','v10').replace('V9','V10')
    if name=='runtime.py':
        old="pending=[ROOT/'v10'/n for n in ('__init__.py','runtime.py','run_development.py','controller.py','feedback.py')]"
        new="""pending=[ROOT/'v10'/n for n in ('__init__.py','runtime.py','run_development.py','controller.py','directional_allocator.py','robot_contact_space.py','servo_bank.py')]
    pending.append(ROOT/'v9/__init__.py')
    from v10.servo_bank import Bank
    from v10.directional_allocator import ROBOT_AXES25
    bank=Bank(ROBOT_AXES25)
    if bank.load_errors or len(bank.entries)!=15:raise ValueError('All15 frozen empirical servo contexts required for this development version')
    for path,digest in bank.source_identity().items():
        p=Path(path).resolve()
        if not p.is_relative_to(ROOT.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Servo source scope/identity mismatch')
        pending.append(p)"""
        assert old in value;value=value.replace(old,new,1)
        old="self.controller=create_joint(self.calibration,self.limits,self.budget,gains=actual_ratio_gains)"
        new="self.controller=create_joint(self.calibration,self.limits,self.budget,gains=actual_ratio_gains,robot_model=self.plant.model)"
        assert old in value;value=value.replace(old,new,1)
        old="        obs.motion_reference_rate_limits=np.asarray(V10_REFERENCE_RATE_LIMITS,float).copy()"
        new=old+"\n        obs.lift_velocity_m_s=float(raw['lift_velocity']) # public robot encoder velocity, supplied to every method"
        assert old in value;value=value.replace(old,new,1)
        old="value.update(schema='v10_task_research_trial_v1',motion_mode=self.motion_mode,"
        new=old+"\n            public_robot_lift_velocity='same current robot encoder velocity supplied to every method; no external object state',"
        assert old in value;value=value.replace(old,new,1)
        value=value.replace("'baseline_coupled_force_posterior_only' if self.method=='joint'","'baseline_force_posterior_plus_robot_encoder_geometry_only' if self.method=='joint'",1)
    ast.parse(value);outputs[ROOT/'v10'/name]=value
for path in outputs:
    if path.exists():raise FileExistsError('Retain existing file '+str(path))
for path,value in outputs.items():path.write_text(value,encoding='utf-8')
print('Created isolated V10 runtime/runner/init; no dynamics or policy execution')
