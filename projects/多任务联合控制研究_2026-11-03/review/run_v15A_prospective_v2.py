"""Root-only prospective scheduler; source and evaluation definitions stay fixed."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 planpath=ROOT/'review/v15A_prospective_multiscene_plan_v2.json';plan=read(planpath);status=ROOT/'review/v15A_prospective_sequence_progress_v2.json'
 if status.exists():raise FileExistsError(status)
 for stage in plan['stages']:
  if (ROOT/stage['output']).exists():raise FileExistsError(ROOT/stage['output'])
 state=dict(schema='v15A_root_prospective_sequence_v2',created_utc=datetime.now(timezone.utc).isoformat(),plan=str(planpath.relative_to(ROOT)),plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),source_changes_allowed=False,assigned=64,records=[],current=None,complete=False)
 write(status,state)
 for stage in plan['stages']:
  state.update(current=stage['name'],current_started_utc=datetime.now(timezone.utc).isoformat());write(status,state)
  cmd=[sys.executable,'-B','-m','v15_support.run_development','--out',stage['output'],'--registration',stage['registration'],'--task-policy',stage['task_policy'],'--receiver-policy',stage['receiver_policy'],'--motion-mode',stage['motion_mode'],'--methods',*stage['methods'],'--scenarios',*stage['scenarios'],'--seeds',*[str(x) for x in stage['seeds']],'--physical-randomization','--workers','3']
  print(json.dumps(dict(start_stage=stage['name'],arm=stage['arm'],assigned=stage['planned'],output=stage['output'])),flush=True)
  result=subprocess.run(cmd,cwd=ROOT)
  terminal_path=ROOT/stage['output']/'execution_complete.json';terminal=read(terminal_path) if terminal_path.exists() else None
  state['records'].append(dict(stage=stage['name'],arm=stage['arm'],output=stage['output'],exit_code=result.returncode,terminal=terminal));write(status,state)
  if result.returncode or terminal is None or not terminal['execution_complete']:
   state.update(current=None,complete=False,terminated_on_runtime_failure=True,ended_utc=datetime.now(timezone.utc).isoformat());write(status,state);return 1
 state.update(current=None,complete=True,terminated_on_runtime_failure=False,ended_utc=datetime.now(timezone.utc).isoformat());write(status,state)
 print(json.dumps(dict(sequence_complete=True,completed_cells=len(state['records']),physical_trials=sum(x['terminal']['trials'] for x in state['records']))),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
