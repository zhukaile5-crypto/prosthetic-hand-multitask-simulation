"""Four strict-equality spots, exact rational scalar dot reference; no physics."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT.parent/'五指策略接入_2026-11-03'));sys.dont_write_bytecode=True
from evaluate_runs import evaluate_trace
from evaluation import evaluate_research
CASES=[('v15A_prospective_baseline_hold_v2','collision__fixed_effort__619731'),('v15A_prospective_baseline_hold_v2','collision__fusion_effort__619732'),('v15A_prospective_candidate_A_transport_v2','collision__ratio_effort__619732'),('v15A_prospective_baseline_transport_v2','collision__fixed_effort__619732')]
def dif(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):return sum([dif(v,b.get(k),p+'.'+k) for k,v in a.items()],[])
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):return sum([dif(x,y,p+'.'+str(i)) for i,(x,y) in enumerate(zip(a,b))],[])
    if a==b:return []
    return [dict(path=p,recomputed=a,stored=b)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    prior=json.loads((ROOT/'review/v15A_metric_recompute_mismatch_spots_v1.json').read_text(encoding='utf-8'));rows=[];all_same=True
    for j,(batch,name) in enumerate(CASES):
        d=ROOT/'output'/batch/'runs'/name;path=d/'trace.npz';stored=json.loads((d/'result.json').read_text(encoding='utf-8'));old=evaluate_trace(path);new=evaluate_research(path);od=dif(old,stored);nd=dif(new,stored['research_metrics']);all_same &=not od and not nd
        with np.load(path,allow_pickle=False) as z:force=z['substep_skin_normal'];dt=z['substep_duration'];t=z['substep_time']
        meta=json.loads((d/'metadata.json').read_text(encoding='utf-8'));total=np.maximum(force.sum(axis=1)-float(meta['budget']),0.);witness=[]
        old_diffs=prior['rows'][j]['direct_old']
        for hit in old_diffs:
            field=hit['path'].split('.')[-1];mask=t>=7.5-1e-9 if field.startswith('post_handoff') else np.ones(len(t),bool)
            a=total[mask];b=dt[mask];exact=sum((Fraction(float(x))*Fraction(float(y)) for x,y in zip(a,b) if x),Fraction(0));reference=float(exact)
            n=len(a);u=2.**-53;gamma=(2*n*u)/(1-2*n*u);bound=gamma*float(exact)
            original=float(hit['stored']);recomputed=float(hit['recomputed']);err0=float(abs(Fraction(original)-exact));err1=float(abs(Fraction(recomputed)-exact));ulp=abs(int(np.float64(original).view(np.uint64))-int(np.float64(recomputed).view(np.uint64)))
            witness.append(dict(field=field,vector_length=n,nonzero_terms=int(np.count_nonzero(a)),nearest_binary64_exact_product_sum=reference,
                original_error_against_exact_inputs=err0,unrestricted_audit_error_against_exact_inputs=err1,original_vs_unrestricted_ULPs=ulp,
                conservative_dot_roundoff_bound_Ns=bound,both_within_bound=err0<=bound and err1<=bound,
                derivation='Exact Fraction sum of products of saved finite nonnegative binary64 total-force excess and duration. Bound gamma_(2n)*sum_abs_products, u=2^-53; no physical threshold or score tolerance change.'))
        rows.append(dict(batch=batch,run=name,trace_sha256=sha(path),old_differences=od,research_differences=nd,rational_scalar_dot_witness=witness))
    report=dict(schema='v15A_metric_thread_reproducibility_v1',all_original_metrics_exact_under_this_environment=bool(all_same),rows=rows,
        thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},checker_sha256=sha(__file__),physics_steps=0,
        caveat='Reference bounds only the scalar dot arithmetic on identical saved double inputs, not physical model/discretization uncertainty. Original exact-equality failure remains preserved.')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(all_original_metrics_exact=bool(all_same),sha256=sha(out),out=str(out)),ensure_ascii=False));raise SystemExit(0 if all_same else 1)
