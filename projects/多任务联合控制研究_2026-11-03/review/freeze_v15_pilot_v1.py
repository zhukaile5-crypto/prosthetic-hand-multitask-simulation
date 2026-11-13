"""Freeze predeclared V15 pilot cells after substantive software gates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.dont_write_bytecode=True
from v15_support.runtime import source_identity
from v13_a1.run_bridge import environment

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 planpath=root/'review/v15_runtime_and_pilot_plan_v1.json';plan=read(planpath)
 required=['review/v15_runtime_root_zero_v2.json','review/v15_integrated_controller_independent_v1.json','review/v15_release_tasks_independent_v1.json','review/v15_runtime_bridge_independent_v1.json']
 gates={}
 for rel in required:
  p=root/rel;v=read(p)
  if not v.get('passed',v.get('all_passed',False)):raise RuntimeError('Required software gate not passed: '+rel)
  gates[rel]=dict(sha256=sha(p),scope=v.get('scope'))
 identity=source_identity();env=environment()
 # Freeze source files after all edits; old software tests do not certify untested physics.
 for stage in plan['sequential_stages']:
  p=root/('review/v15_'+stage['name']+'_registration_v1.json')
  if p.exists():raise FileExistsError(p)
  assignment={k:plan['common'][k] for k in ('methods','scenarios','seeds','motion_mode','physical_randomization')}
  assignment.update(task_policy=stage['task_policy'],receiver_policy=stage['receiver_policy'])
  value=dict(schema='v15_pilot_registration_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=identity,environment=env,assignment=assignment,software_gate_passed=True,software_gates=gates,plan=str(planpath.relative_to(root)),plan_sha256=sha(planpath),stage=stage,development_only=True,performance_claim_allowed=False)
  p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print(json.dumps(dict(registration=str(p.relative_to(root)),sha256=sha(p),source_count=len(identity),binaries=len(env['native_binaries_sha256'])),ensure_ascii=False),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())

