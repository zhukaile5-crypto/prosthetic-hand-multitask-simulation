"""Execute the frozen five-arm registration; preserve every assigned result."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def paired_prefix(full,holding):
    from review.run_v17_transfer_small_v1 import compare_original_fields
    from v17_contact_transfer.motion_domain_sensors import ADDED_FIELDS
    result=compare_original_fields(full,holding,True)
    same=True;count=0
    with np.load(full/'trace.npz',allow_pickle=False) as a,np.load(holding/'trace.npz',allow_pickle=False) as b:
        aa=a['legacy_common_sensor_json'][a['time']<12.-1e-9]
        bb=b['legacy_common_sensor_json'][b['time']<12.-1e-9]
        same=aa.shape==bb.shape
        if same:
            for x,y in zip(aa,bb):
                xx=json.loads(str(x));yy=json.loads(str(y));count+=1
                if any(key not in xx or key not in yy or xx[key]!=yy[key] for key in ADDED_FIELDS):same=False;break
    result.update(additional_domain_metadata_exact=same,metadata_packets_compared=count,
        passed=result['passed'] and same)
    return result


def main():
    from v17b_full_evidence.runtime import source_identity,METHODS
    from v13_a1.run_bridge import environment
    master=ROOT/'review/v17b_prospective80_plan_v1.json';plan=read(master)
    progress=ROOT/'review/v17b_prospective80_progress_v1.json'
    if progress.exists():raise FileExistsError('Never restart a retained attempt implicitly')
    if sha(Path(__file__))!=plan['scheduler_sha256']:raise RuntimeError('Registered scheduler changed')
    for stage in plan['stages']:
        if (ROOT/stage['output']).exists():raise FileExistsError(ROOT/stage['output'])
    def invariant():
        if source_identity(METHODS)!=plan['source_sha256'] or environment()!=plan['environment']:
            raise RuntimeError('Frozen source136/environment42 changed')
        for path,pin in ((plan['auxiliary_metrics']['contract'],plan['auxiliary_metrics']['contract_sha256']),
            (plan['auxiliary_metrics']['reader'],plan['auxiliary_metrics']['reader_sha256'])):
            if sha(ROOT/path)!=pin:raise RuntimeError('Registered metric definition/calculator changed')
        if any(sha(ROOT/path)!=pin for path,pin in plan['prefix_helper_sha256'].items()):
            raise RuntimeError('Registered original-prefix reader helper changed')
    invariant()
    state=dict(schema='v17b_prospective80_sequence_progress_v1',master_plan_sha256=sha(master),
        created_utc=datetime.now(timezone.utc).isoformat(),planned=80,records=[],current=None,
        physics_active=False,complete=False,source_changes_allowed=False)
    write(progress,state)
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    for stage in plan['stages']:
        invariant()
        state.update(current=stage['name'],physics_active=True,current_started_utc=datetime.now(timezone.utc).isoformat())
        write(progress,state)
        cmd=[sys.executable,'-B','-m','v17b_full_evidence.run_development','--out',stage['output'],
            '--registration',stage['registration'],'--methods',*stage['methods'],'--scenarios',*stage['scenarios'],
            '--seeds',*[str(s) for s in stage['seeds']],'--motion-mode',stage['motion_mode'],
            '--task-policy',stage['task_policy'],'--receiver-policy',stage['receiver_policy'],
            '--physical-randomization','--workers',str(stage['workers'])]
        print(json.dumps(dict(starting=stage['name'],planned=stage['planned'])),flush=True)
        child=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy())
        terminal_path=ROOT/stage['output']/'execution_complete.json'
        terminal=read(terminal_path) if terminal_path.exists() else None
        record=dict(stage=stage['name'],exit_code=child.returncode,terminal=terminal,prefix_comparisons=[])
        state['records'].append(record)
        if child.returncode or terminal is None or not terminal.get('execution_complete'):
            state.update(physics_active=False,current=None,stopped_on_runtime_failure=True,
                ended_utc=datetime.now(timezone.utc).isoformat());write(progress,state);return 1
        invariant()
        if stage['arm']=='full_V17':
            holding=next(s for s in plan['stages'] if s['arm']=='holding_A' and s['motion_mode']==stage['motion_mode'])
            for seed in stage['seeds']:
                for scene in stage['scenarios']:
                    rel=Path('runs')/(scene+'__ratio_effort__'+str(seed))
                    record['prefix_comparisons'].append(paired_prefix(ROOT/stage['output']/rel,ROOT/holding['output']/rel))
            if not all(x['passed'] for x in record['prefix_comparisons']):
                state.update(physics_active=False,current=None,stopped_on_prefix_contract_failure=True,
                    ended_utc=datetime.now(timezone.utc).isoformat());write(progress,state);return 1
        write(progress,state)
    state.update(current=None,physics_active=False,complete=True,stopped_on_runtime_failure=False,
        ended_utc=datetime.now(timezone.utc).isoformat(),completed=sum(x['terminal']['trials'] for x in state['records']))
    write(progress,state)
    print(json.dumps(dict(complete=True,completed=state['completed'],planned=80)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
