"""Only the axes revision and byte-diff binding; no repeated old8 or model."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'v16_release_geometry/geometry_tasks.py'
NEW='e7e0119824bfbd1bd865478a36758147f6e90b4cbde9cfdce2878673365e86e3'
OLD='1694ca8889eebd709e7f1e8925c359880b2255166cdb46acde836b41605a0f0c'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==NEW
from v16_release_geometry.geometry_tasks import make_length_proposal_map,RobotPatchGeometry,build_geometry_tasks
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
raw=SOURCE.read_bytes();newline=b'\r\n' if b'\r\n' in raw else b'\n';old=raw
patches=[(b'    robot_jacobian_m_per_q25: np.ndarray'+newline+b'    robot_axes25: tuple'+newline,b'    robot_jacobian_m_per_q25: np.ndarray'+newline),
    (b"        object.__setattr__(self, 'robot_axes25', _axes(self.robot_axes25, 25, 'patch robot_axes25'))"+newline,b''),
    (b'        if mapping is not None and p.robot_axes25 != mapping.robot_axes25:'+newline+b"            why.append('robot_jacobian_axis_order_does_not_match_proposal_map')"+newline,b'')]
counts=[]
for before,after in patches:counts.append(old.count(before));old=old.replace(before,after)
reconstructed=hashlib.sha256(old).hexdigest()
ck('only_declared_axes_revision_restores_exact_prior_source',counts==[1,1,1] and reconstructed==OLD,dict(reversed_edits=3,reconstructed_prior_sha256=reconstructed))
axes=tuple('axis'+str(i) for i in range(25));refs=tuple('ref'+str(i) for i in range(21))
m=make_length_proposal_map(transmission21x25=np.eye(21,25),reference_to_length21=np.ones(21),joint_scale25=np.ones(25),robot_axes25=axes,reference_axes21=refs,time_s=1.,declared_source='pure axes fixture only')
J=np.zeros((3,25));J[:,:3]=np.eye(3)
p=RobotPatchGeometry(robot_body=2,digit=0,acquisition_time_s=.99,robot_time_s=1.,observed_normal_N=1.,acquisition_position_world_m=np.zeros(3),acquisition_normal_world=np.array([1.,0,0]),robot_position_world_m=np.zeros(3),robot_jacobian_m_per_q25=J,robot_axes25=axes)
def build(ps):return build_geometry_tasks(patches=ps,mapping=m,current_time_s=1.,maximum_geometry_age_s=.1,step_lower21=-np.ones(21)*.01,step_upper21=np.ones(21)*.01,nominal_open_increment21=np.zeros(21),mode_generation=1,robot_geometry_authenticated=True)
good=build([p]);bad=build([p,replace(p,robot_axes25=(axes[1],axes[0],*axes[2:]))])
ck('matching_order_usable_single_mismatched_row_makes_whole_set_unknown',good.status=='EXPERIMENTAL_GEOMETRY_TASKS' and bad.status=='UNAVAILABLE_GEOMETRY_TASKS' and bad.normal_matrix_m_per_reference is None and bad.tangent_matrix_m_per_reference is None and bad.diagnostics['unavailable_patches'][0]['reasons']==['robot_jacobian_axis_order_does_not_match_proposal_map'])
reject=[]
for wrong in (axes[:-1],tuple(['same']*25),tuple([False,*axes[1:]])):
    try:replace(p,robot_axes25=wrong);reject.append(False)
    except ValueError:reject.append(True)
ck('missing_duplicate_or_nonstring_axes_rejected',all(reject))
readme=(ROOT/'v16_release_geometry/README.md').read_text(encoding='utf-8')
ck('new_field_not_a_model_certificate_and_readme_explicit',p.robot_axes25==m.robot_axes25 and 'typed' in readme and 'rank' in readme and '外层' in readme and good.diagnostics['actual_servo_response_validated'] is False and good.diagnostics['pressure_prediction_available'] is False and 'mujoco' not in sys.modules)
prior=ROOT/'review/v16_geometry_tasks_independent_v1.json';prior_report=json.loads(prior.read_text(encoding='utf-8'));assert prior_report['passed'] is True and prior_report['source_sha256']==OLD
report=dict(schema='v16_geometry_tasks_axes_independent_v2',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=NEW,source_unchanged=sha(SOURCE)==NEW,
    prior8=dict(path=str(prior),sha256=sha(prior),source_sha256=OLD,passed=True),checker_sha256=sha(__file__),scope='Axes-only amendment; reversing exact three edits reconstructs old source hash, so old8 numerical evidence is retained without repeated tests. No raw/model/physics.',model_created=False,raw_records_read=0,physics_steps=0)
out=ROOT/'review/v16_geometry_tasks_axes_independent_v2.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)
