"""Fixed all32 L4 review experiment; no production edits, models or physics.

Exactly one original-polytope gradient LP, exact [0,1] LS segment minimizer,
and one original-tx tight-face refinement. No threshold/goal selection.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from v22_joint_task_fusion import finite_solver as fs
from scipy.optimize import linprog

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return plain(x.item())
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return None
    return x
def load(p):
    s = importlib.util.spec_from_file_location('independent_v21_original_verifier', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main(out):
    if out.exists(): raise FileExistsError(out)
    production = ROOT/'v22_joint_task_fusion/finite_solver.py'; before = sha(production)
    corpus_path = ROOT/'review/v21_numerical_corpus_v1.json'
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    verifier = load(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py')
    selected = [s for s in corpus['samples'] if s['failed_stage_problem']['layer_name'] == 'canonical_or_explicit_path_reference']
    assert len(selected) == 32
    rows = []
    for sample in selected:
        start = time.perf_counter_ns(); p = sample['failed_stage_problem']
        pp = verifier.canonical_problem(p, p['initial_x']); A,b,G,h,E,f,x0 = [pp[k] for k in ('A','b','G_unit','h_unit','E','f','x')]
        frozen = pp['frozen']; atol,rtol = pp['atol'],pp['rtol']; mi=p['max_iterations']
        def assess(x, name):
            k = fs._certify(A,b,x,G,h,E,f,frozen,atol,rtol,mi)
            d = None if k['full_normalized_inequality_dual'] is None else dict(inequality_unit=k['full_normalized_inequality_dual'], equality_raw=k['equality_raw_dual'], provenance='current_original_KKT_review_dual')
            iv = verifier.verify_stage(p,x,duals=d,comparison_x=x0)
            return dict(candidate=name, x=x, KKT=k, independent=iv, independently_certified=iv['kkt_passed'], exact_coordinate_box=bool(np.all(x >= np.asarray(p['lower'])) and np.all(x <= np.asarray(p['upper']))))
        attempts = [assess(x0, 'original_anchor')]
        res0=fs._dot(A,x0)-b; gradient=2*fs._dot(A.T,res0)
        # Every original G/E row is retained, only positive row normalization.
        Eu,fu,_=fs._normalize(E,f)
        gradscale=max(1.,fs._inf(gradient))
        lp=linprog(gradient/gradscale,A_ub=G if len(G) else None,b_ub=h if len(G) else None,
            A_eq=Eu if len(E) else None,b_eq=fu if len(E) else None,bounds=[(None,None)]*len(x0),
            method='highs',options=dict(fs.DUAL_LP_OPTIONS,maxiter=int(mi)))
        note=dict(success=bool(lp.success),status=int(lp.status),message=str(lp.message),gradient_scale=gradscale,
            original_G_rows=len(G),original_E_rows=len(E),no_original_rows_dropped=True)
        line=None; face=None
        if lp.x is not None and np.isfinite(lp.x).all():
            xLP=np.asarray(lp.x); attempts.append(assess(xLP,'gradient_LP_endpoint'))
            if attempts[-1]['independent']['primal']['passed']:
                d=xLP-x0; Ad=fs._dot(A,d)
                denominator=math.fsum(float(v)*float(v) for v in Ad)
                numerator=-math.fsum(float(v)*float(w) for v,w in zip(Ad,res0))
                alpha=0. if denominator==0 else min(1.,max(0.,numerator/denominator))
                line=x0+alpha*d
                note.update(segment_alpha=alpha,segment_numerator=numerator,segment_denominator=denominator,
                    alpha_role='exact_minimizer_on_original_convex_feasible_segment_not_coordinate_clip')
                attempts.append(assess(line,'one_exact_LS_segment'))
                try:
                    face,face_note=fs._refine_once(A,b,G,h,E,f,line,atol,rtol)
                    rr=assess(face,'one_original_tx_tight_face');rr['generation']=face_note;attempts.append(rr)
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc: note['face_error']=repr(exc)
        accepted=[r for r in attempts if r['independently_certified']]
        rows.append(dict(sample_id=sample['sample_id'],context_key=sample['context_key'],selection_kind=sample['selection_kind'],
            layer_name=p['layer_name'],LP=note,attempts=attempts,accepted=bool(accepted),
            exact_box_certified=any(r['independently_certified'] and r['exact_coordinate_box'] for r in attempts),
            elapsed_ms=(time.perf_counter_ns()-start)/1e6))
    counts={name:sum(any(r['candidate']==name and r['independently_certified'] for r in row['attempts']) for row in rows)
        for name in ('original_anchor','gradient_LP_endpoint','one_exact_LS_segment','one_original_tx_tight_face')}
    report=dict(schema='v22_L4_fixed_LP_direction_review_v1',experiment_complete=True,production_unchanged=sha(production)==before,
        production_source_sha256={'v22_joint_task_fusion/finite_solver.py':before},corpus_sha256=sha(corpus_path),checker_sha256=sha(__file__),
        assigned=32,original_failure_cases=sum(s['selection_kind']=='first_failed_layer' for s in selected),
        fullsuccess_final_layer_controls=sum(s['selection_kind']=='first_full_success_control' for s in selected),
        accepted=sum(r['accepted'] for r in rows),exact_box_certified=sum(r['exact_box_certified'] for r in rows),
        candidate_acceptance_counts=counts,rows=rows,physics_steps=0,model_constructions=0,
        no_original_goals_domains_tolerances_changed=True,not_new_production_gate=True)
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('assigned','accepted','exact_box_certified','candidate_acceptance_counts','production_unchanged')},ensure_ascii=False))
    print('evidence='+str(out));print('sha256='+sha(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();main(args.out)
