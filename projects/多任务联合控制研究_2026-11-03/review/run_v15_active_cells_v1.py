"""Root's sole sequential scheduler of the three predeclared active pilot cells."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 gate=ROOT/'review/v15_passive_blind_independent_attempt1.json';value=read(gate)
 if not value['passed']:raise RuntimeError('Passive blind independent result gate not passed')
 plan=read(ROOT/'review/v15_runtime_and_pilot_plan_v1.json');status=ROOT/'review/v15_active_sequence_progress_v1.json'
 if status.exists():raise FileExistsError(status)
 state=dict(schema='v15_root_sequential_active_pilot_v1',created_utc=datetime.now(timezone.utc).isoformat(),passive_gate=str(gate.relative_to(ROOT)),passive_gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),source_changes_allowed=False,records=[],current=None,complete=False)
 for stage in plan['sequential_stages'][1:]:
  if (ROOT/stage['output']).exists():raise FileExistsError(ROOT/stage['output'])
 write(status,state)
 for stage in plan['sequential_stages'][1:]:
  state['current']=stage['name'];state['current_started_utc']=datetime.now(timezone.utc).isoformat();write(status,state)
  registration='review/v15_'+stage['name']+'_registration_v1.json'
  cmd=[sys.executable,'-B','-m','v15_support.run_development','--out',stage['output'],'--registration',registration,'--task-policy',stage['task_policy'],'--receiver-policy',stage['receiver_policy'],'--motion-mode','hold','--methods','ratio_effort','fixed_effort','fusion_effort','--scenarios','nominal','--seeds','617901','--physical-randomization','--workers','3']
  print(json.dumps(dict(start_stage=stage['name'],output=stage['output']),ensure_ascii=False),flush=True)
  result=subprocess.run(cmd,cwd=ROOT)
  terminal_path=ROOT/stage['output']/'execution_complete.json'
  terminal=read(terminal_path) if terminal_path.exists() else None
  state['records'].append(dict(stage=stage['name'],output=stage['output'],exit_code=result.returncode,terminal=terminal))
  if result.returncode or terminal is None or not terminal['execution_complete']:
   state.update(current=None,complete=False,terminated_on_failure=True,ended_utc=datetime.now(timezone.utc).isoformat());write(status,state);return 1
  write(status,state)
 state.update(current=None,complete=True,terminated_on_failure=False,ended_utc=datetime.now(timezone.utc).isoformat());write(status,state)
 print(json.dumps(dict(sequence_complete=True,completed_cells=len(state['records']),physical_trials=sum(x['terminal']['trials'] for x in state['records']))),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
