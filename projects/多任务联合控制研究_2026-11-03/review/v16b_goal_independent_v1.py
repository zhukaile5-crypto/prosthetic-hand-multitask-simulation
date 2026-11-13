"""Pure V16b task-goal binding gate; no Session/model construction or physics."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v16b_release_goal.goal import release_clearance_goal
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry,release_nominal
from v13_joint.design import NONTHUMB_CLEAR_TIME_S,THUMB_TUCK_UNTIL_S,THUMB_UNFOLD_TIME_S,LIFT_DESCENT_TIME_S
from v16_release_geometry.supervisor import ReleaseSupervisor,RELEASE_DEADLINE_S
from v16_release_geometry.transfer_observer import TransferObserver
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={n:ROOT/'v16b_release_goal'/n for n in ('goal.py','runtime.py','run_development.py')}
expected={'goal.py':'0ad6d969092a6ffec63b9151d5b6b5c9b915059dfd71d20a6815873aa35990cb','runtime.py':'1f7fd5640d0a5f8a2c98781f4bfc51d9312abd471d91924dad7dd47aff2782ab','run_development.py':'100a9ca3ecfefd6661c582977f2c1094718c31ef9d00d0ff648400f5a9d66305'}
assert all(sha(files[n])==v for n,v in expected.items())
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
config=ROOT.parent/'五指策略接入_2026-11-03/foundation_config.py';tree=ast.parse(config.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OPEN' for t in n.targets));prep=np.concatenate(ast.literal_eval(node.value));prep_before=prep.tobytes()
lo=np.r_[np.full(18,-2.),0.,-2.,-2.];hi=np.r_[np.full(18,2.),.1,2.,2.];start=np.r_[prep*.7,.1,.01,-.02]
terminal=max(NONTHUMB_CLEAR_TIME_S,THUMB_TUCK_UNTIL_S+THUMB_UNFOLD_TIME_S,LIFT_DESCENT_TIME_S);records=[]
for origin in (0.,12.,123.125):
 entry=ReleaseEntry(origin,Reference21(start,origin),1)
 goal=release_clearance_goal(entry,preparation_open18=prep,mechanical_lower21=lo,mechanical_upper21=hi)
 legacy=release_nominal(entry,origin+100.,prep,lo,hi)
 records.append(dict(origin_s=origin,goal_thumb=goal.reference18[:5].tolist(),exact_legacy18=goal.reference18.tobytes()==legacy.values[:18].tobytes(),query_s=goal.extraction_query_s,saturated_after_s=goal.saturated_after_s))
ck('same_frozen_terminal18_for_three_origins',all(r['exact_legacy18'] for r in records),records)
expected_terminal=np.r_[[-1.,.1,0.,0.,0.],prep[5:]]
ck('preparation_thumb_and_release_endpoint_separated',np.allclose(goal.reference18,expected_terminal,rtol=0,atol=16*np.finfo(float).eps) and not np.array_equal(goal.reference18[:5],prep[:5]) and np.array_equal(goal.preparation_open18,prep) and prep.tobytes()==prep_before)
entry=ReleaseEntry(12.,Reference21(start,12.),1);tightlo=lo.copy();tighthi=hi.copy();tightlo[0]=-.8;tighthi[0]=-.5
bounded=release_clearance_goal(entry,preparation_open18=prep,mechanical_lower21=tightlo,mechanical_upper21=tighthi)
ck('goal_saturation_obeys_passed_real_domain_without_widening',bounded.reference18[0]==-.8 and np.all(bounded.reference18>=tightlo[:18]) and np.all(bounded.reference18<=tighthi[:18]) and np.array_equal(lo,np.r_[np.full(18,-2.),0.,-2.,-2.]))
try:release_clearance_goal(entry,preparation_open18=prep,mechanical_lower21=hi,mechanical_upper21=lo);reject=False
except ValueError:reject=True
ck('empty_permissions_rejected_not_retargeted',reject)
g=release_clearance_goal(entry,preparation_open18=prep,mechanical_lower21=lo,mechanical_upper21=hi)
observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=[.2,.2],receiver_up_axis3=[0,0,1])
graph=ReleaseSupervisor(entry,open18=g.reference18,observer=observer);command=graph.nominal(12.005,lo,hi)
ck('future_query_extracts_value_only_current_command_clock_retained',g.extraction_query_s>12.+terminal and command.time_s==12.005 and command.time_s!=g.extraction_query_s and g.to_public()['extraction_is_not_a_command_or_future_measurement'] is True)
ck('existing_duration_deadline_and_control_parameters_unchanged',terminal==2.2 and RELEASE_DEADLINE_S==3.5 and g.saturated_after_s==terminal and g.to_public()['original_release_deadline_changed'] is False and g.to_public()['execution_authority_widened'] is False)

runtime=ast.parse(files['runtime.py'].read_text(encoding='utf-8'));cls=next(n for n in runtime.body if isinstance(n,ast.ClassDef));methods=[n.name for n in cls.body if isinstance(n,ast.FunctionDef)];fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_lifecycle_step');source=ast.unparse(fn)
ck('runtime_only_prebuilds_goal_graph_then_calls_same_parent_lifecycle',methods==['__init__','_lifecycle_step','metadata'] and 'self.v16_supervisor is None' in source and 'release_clearance_goal(entry' in source and 'open18=self.release_clearance_goal.reference18' in source and source.index('self.plant.bind_supervisor')<source.index('super()._lifecycle_step(obs)') and 'extraction_query_s' not in source and 'commit_reference21' not in source and 'step_one_sensor_tick' not in source)
from v16b_release_goal.runtime import source_files,parent_files
child=source_files();parent=parent_files();newkeys=set(child)-set(parent)
ck('closure126_extends_same_parent123_by_exact_three_files',len(child)==126 and len(parent)==123 and newkeys=={'v16b_release_goal/'+n for n in files} and all(Path(child[k]).resolve()==Path(v).resolve() for k,v in parent.items()),dict(parent_sources=len(parent),new_sources=len(child),added=sorted(newkeys)))
report=dict(schema='v16b_goal_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=expected,source_unchanged=all(sha(files[n])==v for n,v in expected.items()),checker_sha256=sha(__file__),
    source_paths={k:str(v) for k,v in files.items()},scope='Pure frozen terminal extraction/bounds/current-clock and runtime AST/closure126 map only. No Session/model construction, actual pipeline, physics or algorithm-contribution/performance claim.',session_constructions=0,model_requests=0,physics_steps=0)
out=ROOT/'review/v16b_goal_independent_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)
