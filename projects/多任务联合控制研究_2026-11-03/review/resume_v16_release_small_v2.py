"""Correct a Unicode container comparison; execute only unexecuted assignments.

V1, its terminal failed gate, original registration and both completed runs are
preserved. No policy, evaluator, physics, metric or source123 file is modified.
"""
from pathlib import Path
import importlib.util, json, os, subprocess, sys
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('v16_initial_scheduler',ROOT/'review/run_v16_release_small_v1.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def compare_payload(new,old,before_release):
    checks={}
    with np.load(new/'trace.npz',allow_pickle=False) as a,np.load(old/'trace.npz',allow_pickle=False) as b:
        ma=a['time']<12.-1e-9 if before_release else np.ones(len(a['time']),bool)
        mb=b['time']<12.-1e-9 if before_release else np.ones(len(b['time']),bool)
        for name in base.PHYSICAL_COLUMNS:
            if name not in a or name not in b:
                checks[name]=dict(equal=False,reason='missing_required_column');continue
            aa=a[name][ma];bb=b[name][mb]
            same_shape=aa.shape==bb.shape
            storage=bool(same_shape and aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes())
            text=aa.dtype.kind=='U' and bb.dtype.kind=='U'
            equal=bool(same_shape and all(x.encode('utf-8')==y.encode('utf-8') for x,y in zip(aa.flat,bb.flat))) if text else storage
            checks[name]=dict(equal=equal,storage_equal=storage,new_dtype=str(aa.dtype),old_dtype=str(bb.dtype),
                rule='exact_UTF8_string_payload_not_global_NPZ_padding' if text else 'same_shape_dtype_and_bytes')
    return dict(passed=all(x['equal'] for x in checks.values()),checks=checks,
        scope='strictly before12' if before_release else 'complete16s',
        new=str(new),old=str(old),new_trace_sha256=base.sha(new/'trace.npz'),old_trace_sha256=base.sha(old/'trace.npz'))


def main():
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    plan=base.read(ROOT/'review/v16_release_small_plan_v1.json')
    original=base.read(ROOT/'review/v16_release_small_progress_v1.json')
    if original.get('stopped_after_stage')!='coordinated_hold' or len(original['stages'])!=2:
        raise RuntimeError('Only the known terminal reader-gate stop can be resumed')
    if base.source_identity(['ratio_effort'])!=plan['source_sha256'] or base.environment()!=plan['environment']:
        raise RuntimeError('Source123/environment changed; no execution')
    progress=ROOT/'review/v16_release_small_progress_v2.json'
    if progress.exists():raise FileExistsError(progress)
    stages=[]
    for entry in original['stages']:
        item=dict(entry);item['comparisons']=[]
        stage=next(s for s in plan['stages'] if s['id']==entry['id'])
        for scene in stage['scene']:
            new=ROOT/stage['output']/'runs'/f'{scene}__ratio_effort__619732'
            item['comparisons'].append(compare_payload(new,base.previous_A(scene,stage['motion'],619732),stage['policy']!='blind'))
        if not entry['execution']['execution_complete'] or not all(x['passed'] for x in item['comparisons']):
            raise RuntimeError('A genuine completed-run divergence remains; do not continue')
        stages.append(item)
    incident=dict(schema='v16_unicode_storage_gate_correction_v2',created_utc=datetime.now(timezone.utc).isoformat(),
        original_scheduler_sha256=base.sha(ROOT/'review/run_v16_release_small_v1.py'),
        original_terminal_progress_sha256=base.sha(ROOT/'review/v16_release_small_progress_v1.json'),
        corrected_reader_sha256=base.sha(Path(__file__)),source123_unchanged=True,
        storage_is_not_payload_reason='Unicode width derives from longest string across the full16s array; a new post12 receiver payload changes padding of equal pre12 strings.',
        old_failed_gate_retained=True,completed_runs_not_rerun=True,completed_rechecks=stages,
        no_metric_or_label_changed=True)
    base.write(ROOT/'review/v16_release_reader_storage_correction_v2.json',incident)
    state=dict(master_plan_sha256=base.sha(ROOT/'review/v16_release_small_plan_v1.json'),complete=False,
        physics_started=True,physics_active=True,resumed_unexecuted_only=True,stages=stages,
        reader_correction_sha256=base.sha(ROOT/'review/v16_release_reader_storage_correction_v2.json'))
    base.write(progress,state)
    stage=plan['stages'][2]
    if (ROOT/stage['output']).exists():raise FileExistsError('Unexecuted stage output must be absent')
    cmd=[sys.executable,'-B','-m','v16_release_geometry.run_development','--out',stage['output'],
        '--registration',stage['registration'],'--methods','ratio_effort','--scenarios',*stage['scene'],
        '--seeds','619732','--motion-mode',stage['motion'],'--task-policy','integrated',
        '--receiver-policy',stage['policy'],'--physical-randomization','--workers','2']
    print(json.dumps(dict(corrected_rechecks_passed=True,starting_unexecuted=stage['id'],planned=2)),flush=True)
    done=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy())
    finish=base.read(ROOT/stage['output']/'execution_complete.json')
    row=dict(id=stage['id'],returncode=done.returncode,execution=finish,comparisons=[])
    for scene in stage['scene']:
        new=ROOT/stage['output']/'runs'/f'{scene}__ratio_effort__619732'
        if (new/'trace.npz').exists():
            row['comparisons'].append(compare_payload(new,base.previous_A(scene,stage['motion'],619732),True))
    state['stages'].append(row)
    state.update(complete=bool(not done.returncode and finish['execution_complete'] and len(row['comparisons'])==2 and all(x['passed'] for x in row['comparisons'])),
        physics_active=False,ended_utc=datetime.now(timezone.utc).isoformat())
    base.write(progress,state)
    print(json.dumps(dict(complete=state['complete'],assigned=4,completed=sum(x['execution']['trials'] for x in state['stages']))),flush=True)
    return 0 if state['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
