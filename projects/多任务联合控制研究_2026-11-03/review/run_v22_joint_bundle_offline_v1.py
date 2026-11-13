"""Test the declared V22 joint-task prototype on fixed historical inputs only."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.task_bundle import assemble
from v21_same_problem_oracle_v1 import solve_problem,plain,sha

out=ROOT/'review/v22_joint_bundle_offline_v1.json'
assert not out.exists(),'Preserve earlier experiments.'
# One analytic conflicting-task example checks the intended joint optimum.
bundle=assemble(lower=[0.],upper=[1.],normal_cost_A=[[1.]],normal_target_b=[1.],
    projection_A=[[1.]],projection_target_b=[0.],original_base_G=[[1.],[-1.]],original_base_h=[1.,0.],
    original_fixed_E=[],original_fixed_f=[],witnessed_motion_G=[],witnessed_motion_h=[])
analytic=solve_problem(bundle.problem([0.]))
assert analytic['verification']['kkt_passed'] and abs(float(analytic['x'][0])-.5)<1e-8
src=ROOT/'review/v21_numerical_corpus_v1.json'
corpus=json.loads(src.read_text(encoding='utf-8'))
samples=[s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion']
assert len(samples)==16
rows=[]
for sample in samples:
    old=sample['failed_stage_problem'];first=sample['stage_problems'][0]
    motion=sample['original_record']['hierarchy']['diagnostics']['motion_task']
    assert first['layer_name']=='normal_inward_correction_slack' and motion['available']
    bundle=assemble(lower=first['lower'],upper=first['upper'],normal_cost_A=first['A'],normal_target_b=first['b'],
        projection_A=old['A'],projection_target_b=old['b'],original_base_G=first['G'],original_base_h=first['h'],
        original_fixed_E=first['E'],original_fixed_f=first['f'],
        witnessed_motion_G=motion['budget_inequality_matrix'],witnessed_motion_h=motion['budget_inequality_upper'])
    r=solve_problem(bundle.problem(old['initial_x']))
    row=dict(context_key=sample['context_key'],source=sample['source'],status=r['status'],result=r,
        input_normal_rows=bundle.correction_rows,input_motion_rows=bundle.motion_rows,
        projection_cost_scale=bundle.motion_cost_scale,physical_run=False,runtime_connected=False)
    if r.get('candidate_available'):
        a=np.asarray(first['A']);b=np.asarray(first['b']);m=np.asarray(old['A']);e=np.asarray(old['b'])
        x0=np.asarray(old['initial_x']);x=np.asarray(r['x'])
        row.update(normal_task_loss_before=float(np.sum((a@x0-b)**2)),normal_task_loss_after=float(np.sum((a@x-b)**2)),
            motion_task_loss_before=float(np.sum((m@x0-e)**2)),motion_task_loss_after=float(np.sum((m@x-e)**2)))
    rows.append(row)
    print(json.dumps({'context':row['context_key'],'status':r['status'],
        'motion_before':row.get('motion_task_loss_before'),'motion_after':row.get('motion_task_loss_after')},ensure_ascii=False),flush=True)
payload=dict(schema='v22_joint_bundle_offline_v1',assigned=16,assessed=len(rows),analytic_conflict_example=analytic,
    rows=rows,counts=dict(Counter(r['status'] for r in rows)),corpus_sha256=sha(src),runner_sha256=sha(__file__),
    prototype_source_sha256={'v22_joint_task_fusion/__init__.py':sha(ROOT/'v22_joint_task_fusion/__init__.py'),
        'v22_joint_task_fusion/task_bundle.py':sha(ROOT/'v22_joint_task_fusion/task_bundle.py')},
    oracle_sha256=sha(ROOT/'review/v21_same_problem_oracle_v1.py'),
    physics_steps=0,production_changes=0,core_prototype_added=True,runtime_connected=False,
    scope='Fixed prototype coefficients1 and1/motion_rows; fixed16 historical first-L3-failure inputs, not a parameter sweep or physical improvement.')
out.write_text(json.dumps(plain(payload),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'out':str(out),'counts':payload['counts'],'physics_steps':0},ensure_ascii=False))
