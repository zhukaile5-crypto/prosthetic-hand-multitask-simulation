"""Isolated dispatcher failure witnesses. No package/simulator import."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,traceback,time
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='18168f389e414367210e6a3cb4fa6eb2542ebd5f56a0953b00a09a5e65d042a6'
BASE=ROOT/'review/v13_release_factorial_dispatch_old18168_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=ROOT/'review/revisions/v13_release_factorial_run_18168_exact.py'
    payload=source.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=EXPECTED:raise ValueError('Use an explicitly preserved18168 source; do not silently audit a revision')
    BASE.mkdir(exist_ok=False);saved=BASE/'run_18168.py';saved.write_bytes(payload)
    tree=ast.parse(payload.decode('utf-8-sig'));part=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('execute','write')],type_ignores=[])
    cases=[]
    for name in ('snapshot_failure','pool_creation_failure','failed_bridge_still_dispatches'):
        root=BASE/name;root.mkdir();target=root/'output/attempt'
        class Pool:
            def __init__(self,*args,**kwargs):raise OSError('INDEPENDENT_POOL_FAILURE')
        def snapshot(*args):
            if name=='snapshot_failure':raise OSError('INDEPENDENT_SNAPSHOT_FAILURE')
        false_bridge=dict(execution_complete=True,sources_unchanged=True,both_bridge_all_recomputed_exact=True,source_lock_sha256='LOCK',passed=False,identical_noise_rng_final=False)
        ns=dict(Path=Path,ROOT=root,load=lambda:{},require=lambda r:{'input':{}},identity=lambda:{'fixture':'unchanged'},
            sha=lambda p:'LOCK',LOCK=Path('lock'),snapshot=snapshot,ProcessPoolExecutor=Pool,read=lambda p:false_bridge,time=time,json=json,traceback=traceback)
        exec(compile(part,str(saved),'exec'),ns)
        try:ns['execute'](SimpleNamespace(phase='factorial' if name=='failed_bridge_still_dispatches' else 'bridge',bridge=str(root/'bridge'),out=str(target),workers=1));exception=None
        except Exception as e:exception=dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        cases.append(dict(name=name,exception=exception,files=sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()) if target.exists() else [],
            assignment_ledger=(target/'assignment_ledger.json').exists(),completion=(target/'execution_complete.json').exists(),
            failed_bridge_reached_pool=name=='failed_bridge_still_dispatches' and exception is not None and exception['error']=='INDEPENDENT_POOL_FAILURE'))
    result=dict(schema='v13_release_dispatch_old18168_independent_failures_v1',source_sha256=EXPECTED,saved_source=str(saved),
        independently_reproduced=len(cases)==3 and all(not c['assignment_ledger'] and not c['completion'] for c in cases) and cases[-1]['failed_bridge_reached_pool'],
        blockers=['snapshot/pool dispatch exceptions leave no assigned/outcome ledger','factorial gate does not explicitly require bridge.passed or RNG flag'],cases=cases,
        no_simulator_import=True,no_workers_spawned=True,physics_steps=0,checker_sha256=sha(__file__))
    out=BASE/'audit.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(reproduced=result['independently_reproduced'],cases=len(cases),evidence=str(out)),ensure_ascii=False))
if __name__=='__main__':main()
