"""Finite author cases + all65 original saved problems, no model/physics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import is_dataclass
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from v22_joint_task_fusion import finite_solver as fs
from v22_joint_task_fusion.numerics import numerical_identity


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if is_dataclass(x): return plain(vars(x))
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return plain(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    return x
def rejected(fn):
    try: fn()
    except (ValueError, TypeError): return True
    return False
def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main(out):
    if out.exists(): raise FileExistsError('Fresh author evidence required')
    prod = ('v22_joint_task_fusion/finite_solver.py', 'v22_joint_task_fusion/numerics.py')
    before = {p: sha(ROOT/p) for p in prod}; checks = []; cases = []; times = []
    def ck(name, ok, detail=None): checks.append(dict(name=name, passed=bool(ok), detail=detail))
    def solve(name, lower, upper, levels, initial, **kw):
        start = time.perf_counter_ns(); r = fs.solve_hierarchy(lower=lower, upper=upper, levels=levels, initial=initial, **kw); ms = (time.perf_counter_ns()-start)/1e6
        times.append(ms); cases.append(dict(name=name, result=r, elapsed_ms=ms)); return r
    r = solve('unconstrained interior', [-1., -1.], [1., 1.], [fs.Level('goal', np.eye(2), [.2, -.3])], [0., 0.])
    ck('unconstrained nonzero LS analytic solution', r.fully_converged and np.allclose(r.x, [.2, -.3], atol=1e-8, rtol=0))
    r = solve('fixed coordinate', [.3, -1.], [.3, 1.], [fs.Level('goal', np.eye(2), [.7, .4])], [.3, 0.])
    ck('fixed coordinate not clipped or dropped', r.fully_converged and r.x[0] == .3 and abs(r.x[1]-.4) < 1e-8)
    r = solve('one-sided anchor already optimal', [0.], [1.], [fs.Level('goal', [[1.]], [-1.])], [0.])
    ck('degenerate original anchor can certify without fake legacy loop', r.fully_converged and r.x[0] == 0 and r.certified_prefix[0]['candidate_source'] == 'original_feasible_layer_anchor')
    r = solve('coupled true inequality', [-1., -1.], [1., 1.], [fs.Level('goal', np.eye(2), [1., 1.])], [0., 0.], inequality_matrix=[[1., 1.]], inequality_upper=[.1])
    ck('coupled original hard row preserved', r.fully_converged and np.allclose(r.x, [.05, .05], atol=1e-8, rtol=0))
    r = solve('hierarchy with attained constraints', [-1., -1.], [1., 1.], [fs.Level('first', [[1., 0.]], [.2]), fs.Level('second', [[0., 1.]], [.4]), fs.Level('third', np.eye(2), [.9, .9])], [0., 0.])
    ck('previous attained is frozen, not reweighted', r.fully_converged and r.completed_prefix == 3 and np.allclose(r.x, [.2, .4], atol=1e-8, rtol=0))
    r = solve('redundant exact equality', [-1., -1.], [1., 1.], [fs.Level('goal', [[1., 0.]], [.1])], [0., 0.], equality_matrix=[[1., 1.], [2., 2.]], equality_target=[0., 0.])
    ck('dependent E candidate compression retains all original rows', r.fully_converged and np.allclose(r.x, [.1, -.1], atol=1e-8, rtol=0) and len(r.certified_prefix[0]['hard_domain']['fixed_equality_target']) == 2)
    r = solve('known empty objective', [-1., -1.], [1., 1.], [fs.Level('empty known task', np.empty((0, 2)), np.empty(0))], [.1, .2])
    ck('P0 empty task is known flat not unknown zero response', r.fully_converged and np.array_equal(r.x, [.1, .2]))
    r = solve('nonzero feasible domain anchor', [.2], [.5], [fs.Level('goal', [[1.]], [0.])], [.2])
    ck('no hidden clip or current-origin retarget', r.fully_converged and r.x[0] == .2)
    r = solve('large finite original hard coefficient', [-1.], [1.], [fs.Level('goal', [[1.]], [.5])], [0.], inequality_matrix=[[1e200]], inequality_upper=[0.])
    ck('large finite hard row cannot normalize to zero', r.fully_converged and r.x[0] == 0 and r.certified_prefix[0]['hard_domain']['normalized_inequality_matrix'][0, 0] == 1)
    with patch.object(fs, '_slsqp_candidate', lambda *a, **kw: (np.array([.1]), dict(author_injection=True))):
        r = solve('once face LS required', [-1.], [1.], [fs.Level('goal', [[1.]], [.4])], [0.])
    ck('only one fixed-threshold face refinement and original certificate', r.fully_converged and abs(r.x[0]-.4) < 1e-8 and r.certified_prefix[0]['candidate_source'] == 'one_tight_face')
    def no_candidate(*a, **kw): raise RuntimeError('author injected generator failure')
    with patch.object(fs, '_slsqp_candidate', no_candidate):
        r = solve('later failure retained prefix', [-1.], [1.], [fs.Level('flat first', [[0.]], [0.]), fs.Level('second', [[1.]], [.5])], [0.])
    ck('failed lower trial cannot replace certified prefix', r.available and not r.fully_converged and r.completed_prefix == 1 and r.x[0] == 0 and r.failed_layer['index'] == 1)
    with patch.object(fs, '_dual_witness', lambda *a, **kw: (None, None, dict(author_injected_missing_dual=True))):
        r = solve('candidate without dual never certified', [-1.], [1.], [fs.Level('goal', [[1.]], [.25])], [0.])
    ck('library candidate success alone cannot certify task', not r.available and r.x[0] == 0 and r.completed_prefix == 0)
    ck('infeasible original initial rejected, not projected', rejected(lambda: fs.solve_hierarchy(lower=[.2], upper=[.5], initial=[0.], levels=[fs.Level('goal', [[1.]], [.4])])))
    ck('numeric strings and constant inconsistent hard rows rejected', rejected(lambda: fs.solve_hierarchy(lower=['0'], upper=[1.], initial=[0.], levels=[])) and rejected(lambda: fs.solve_hierarchy(lower=[-1.], upper=[1.], initial=[0.], levels=[], inequality_matrix=[[0.]], inequality_upper=[-1.])))
    ck('readonly result and certificate arrays', not cases[0]['result'].x.flags.writeable and not cases[0]['result'].certified_prefix[0]['matrix'].flags.writeable)
    # Original65 are not rerun physics: exact saved single-stage matrices only.
    corpus_path = ROOT/'review/v21_numerical_corpus_v1.json'; corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    verifier = load('independent_original_matrix_kkt_author_use', ROOT/'review/v21_original_matrix_kkt_verifier_v1.py')
    rows = []
    for sample in corpus['samples']:
        p = sample['failed_stage_problem']; n = p['variable_count']; lower = np.asarray(p['lower']); upper = np.asarray(p['upper']); fixed = np.flatnonzero(lower == upper); free = np.flatnonzero(lower != upper)
        E = np.asarray(p['E']).reshape(-1, n); f = np.asarray(p['f']); k = len(fixed)
        if not np.array_equal(E[:k], np.eye(n)[fixed]) or not np.array_equal(f[:k], lower[fixed]): raise ValueError('Original fixed E is not exact implicit family')
        Gin = np.asarray(p['input_inequality_matrix']).reshape(-1, n); hin = np.asarray(p['input_inequality_upper'])
        if not np.array_equal(np.vstack((Gin, np.eye(n)[free], -np.eye(n)[free])), np.asarray(p['G'])) or not np.array_equal(np.r_[hin, upper[free], -lower[free]], p['h']): raise ValueError('Original G family not exact')
        begin = time.perf_counter_ns()
        result = fs.solve_hierarchy(lower=lower, upper=upper, initial=p['initial_x'], levels=[fs.Level(p['layer_name'], p['A'], p['b'])],
            inequality_matrix=Gin, inequality_upper=hin, equality_matrix=E[k:], equality_target=f[k:], max_iterations=p['max_iterations'], atol=p['atol'], rtol=p['rtol'])
        elapsed = (time.perf_counter_ns()-begin)/1e6
        independent = None
        if result.available:
            c = result.certified_prefix[-1]; kk = c['KKT']; domain = c['hard_domain']
            exact_domain = np.array_equal(domain['original_inequality_matrix_with_box'], p['G']) and np.array_equal(domain['original_inequality_upper_with_box'], p['h']) and np.array_equal(domain['fixed_equality_matrix'], E) and np.array_equal(domain['fixed_equality_target'], f)
            independent = verifier.verify_stage(p, result.x, duals=dict(inequality_unit=kk['full_normalized_inequality_dual'], equality_raw=kk['equality_raw_dual'], provenance='new_backend_current_original_problem_dual_not_historical'))
            verified = exact_domain and independent['kkt_passed']
        else: verified = False
        rows.append(dict(sample_id=sample['sample_id'], context_key=sample['context_key'], original_selection_kind=sample['selection_kind'], layer_name=p['layer_name'], available=result.available,
            status=result.status, original_matrix_independent_KKT=verified, elapsed_ms=elapsed, result=result,
            independent_verification=independent))
    ck('all65 same original problem candidates independently KKT verified', len(rows) == 65 and all(r['original_matrix_independent_KKT'] for r in rows), [r['sample_id'] for r in rows if not r['original_matrix_independent_KKT']])
    after = {p: sha(ROOT/p) for p in prod}; ck('production source subset stable', before == after)
    identity = numerical_identity()
    corpus_times = np.asarray([r['elapsed_ms'] for r in rows])
    report = dict(schema='v22_finite_solver_author_v1', passed=all(c['passed'] for c in checks), production_source_sha256=after,
        checker_sha256=sha(__file__), checks=checks, issues=[c['name'] for c in checks if not c['passed']], finite_cases=cases,
        corpus_sha256=sha(corpus_path), corpus_assigned=65, corpus_assessed=len(rows), corpus_available=sum(r['available'] for r in rows),
        corpus_original_KKT=sum(r['original_matrix_independent_KKT'] for r in rows), corpus_rows=rows,
        independent_verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'), numerical_environment=identity,
        timing=dict(scope='single host actual solver call walltime; not5ms guarantee; includes dual LP, SLSQP/face if needed',
            corpus_median_ms=float(np.median(corpus_times)), corpus_p95_ms=float(np.percentile(corpus_times, 95)), corpus_max_ms=float(np.max(corpus_times))),
        physics_steps=0, model_constructions=0, old170_source_changed=False,
        limitations=['Author gate, not independent numerical audit.', 'No contact-pressure, actual-motion or real-time certificate.', 'All65 saved failures/controls included with no outcome filtering.'])
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=len(checks), issues=report['issues'], original_KKT=report['corpus_original_KKT'], out=str(out), sha256=sha(out), timing=report['timing']), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a = p.parse_args(); raise SystemExit(main(a.out))
