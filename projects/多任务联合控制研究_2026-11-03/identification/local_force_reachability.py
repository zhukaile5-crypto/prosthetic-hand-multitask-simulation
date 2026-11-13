"""Algebraic one-interval force reachability at saved states, NOT simulation.

Uses the pre-existing public signed calibration and actual one-sided coordinate
bounds. It is a model diagnostic, not a physical safe-set certificate.
"""
from __future__ import annotations

import argparse,json,itertools
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from identify_seen_dynamics import ROOT,DEFAULT_SOURCE,load,latent_coordinates,causal_slope,serial,digest


def bounded_vertices(A,b):
    """Enumerate the vertices of this five-dimensional bounded polytope."""
    values=[]
    for rows in itertools.combinations(range(len(b)),A.shape[1]):
        C=A[list(rows)];rhs=b[list(rows)]
        try:x=np.linalg.solve(C,rhs)
        except np.linalg.LinAlgError:continue
        if np.max(A@x-b)<=1e-8:values.append(x)
    return np.asarray(values)


def box_least_squares(J,target,lo,hi):
    best=None;objective=float('inf')
    for status in itertools.product((-1,0,1),repeat=J.shape[1]):
        status=np.asarray(status);free=status==0;x=np.where(status<0,lo,hi)
        x[free]=0.;x[free]=np.linalg.lstsq(J[:,free],target-J@x,rcond=None)[0]
        if np.any(x<lo-1e-9) or np.any(x>hi+1e-9):continue
        cost=float(np.sum((J@x-target)**2))
        if cost<objective:best=x.copy();objective=cost
    if best is None:raise RuntimeError('No box-constrained least-squares candidate')
    return best


def inspect(path,stamp):
    a=load(path);t=a['time'];i=int(np.argmin(abs(t-stamp)));meta=a['meta'];cal=meta['calibration']
    with np.load(path,allow_pickle=False) as saved:
        packet=saved['observation_json'];frame=json.loads(str(packet[i]))
    alpha=a['alpha_before'][i];lower=np.array(frame['closure_min']);upper=np.array(frame['closure_max'])
    period=.02;dt=.005;tau=float(cal['force_tau']);g=meta['gains'];B=np.array(cal['force_matrix']);active=a['contact_model_active'][i]
    Ba=active[:,None]*B;z=latent_coordinates(a);bias=a['estimated_normal']-(z-np.asarray(cal['reference_alpha']))@B.T
    rate=causal_slope(t,bias,.12)[i];normal=a['estimated_normal'][i]+.11*rate
    decay=np.exp(-period/tau);aa=np.exp(-dt/tau);ramp=1.-sum(aa**j for j in range(1,5))/4
    J=ramp*Ba;carry=Ba@((1.-decay)*(alpha-z[i]));constant=normal+period*rate+carry
    lo=np.maximum(lower-alpha,-float(g.get('open_rate',.6))*period)
    hi=np.minimum(upper-alpha,float(g.get('close_rate',.45))*period)
    # Roundoff-only signs; no expansion of the saved physical/model action box.
    lo=np.minimum(lo,0.);hi=np.maximum(hi,0.)
    error=np.array(cal['response_error_bound']);error=np.where(active,error,0.)
    limits=np.asarray(meta['limits']);budget=float(meta['budget'])
    A=np.vstack([J,np.ones((1,5))@J]);b=np.r_[limits-error-constant,budget-error.sum()-constant.sum()]
    poly_A=np.vstack([A,np.eye(5),-np.eye(5)]);poly_b=np.r_[b,hi,-lo]
    vertices=bounded_vertices(poly_A,poly_b);intervals=[]
    for digit in range(5):
        values=vertices@J[digit]+constant[digit] if len(vertices) else []
        intervals.append(dict(digit=digit,minimum_predicted_pressure_N=None if not len(values) else float(np.min(values)),
             maximum_predicted_pressure_N=None if not len(values) else float(np.max(values))))
    requested=-(period*rate+carry)
    fit=box_least_squares(J,requested,lo,hi)
    fit_safe=bool(np.max(A@fit-b)<=1e-8)
    exact=bool(np.max(abs(J@fit-requested))<=1e-8 and fit_safe)
    return dict(scene=meta['scenario']['name'],method=meta['method'],seed=meta['seed'],time_s=float(t[i]),
         calibration_tau_s=tau,active_pressure=active,one_sided_action_lower=lo,one_sided_action_upper=hi,
         fast_ramp_gain=ramp,fast_J=J,fast_J_singular_values=np.linalg.svd(J,compute_uv=False),
         fast_J_rank=int(np.linalg.matrix_rank(J)),measured_coordinate=alpha,
         inferred_slow_rate_Nps=rate,model_current_pressure_N=normal,
         zero_action_pressure_change_N=period*rate+carry,
         per_digit_model_interval=intervals,enumerated_feasible_vertices=len(vertices),
         drift_cancel_exact_feasible=exact,
         bounded_least_squares_action=fit,bounded_least_squares_pressure_error_N=J@fit-requested,
         least_squares_action_satisfies_pressure_constraints=fit_safe,
         diagnostic_not_a_safety_certificate=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--out',type=Path,default=ROOT/'model_data/local_force_reachability_v1.json');args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Write only inside new research root')
    if out.exists():raise FileExistsError(out)
    rows=[]
    for scene,shape in [('nominal','capsule'),('contact_robustness','capsule'),('sphere','sphere')]:
        path=args.source_root/'output/confirmation_integration_v2/runs'/f'{scene}__{shape}__unified__530101'/'trace.npz'
        rows.append(inspect(path,8.2))
    result=dict(schema='seen_local_force_reachability_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(__file__),seen_old_confirmation=True,new_performance_claim=False,rows=rows,
        limitations=['No MuJoCo step or controller execution occurs.',
            'Full matrix rank does not imply bidirectional authority at a one-sided domain corner.',
            'Intervals are per-output extrema and generally cannot all be realized simultaneously.',
            'The old calibration error bound does not certify changed contact family, new motion gain or unobserved nonlinear transitions.',
            'Least-squares cancellation is an algebraic candidate, not a tested controller action or prediction guarantee.'])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(out),sha256=digest(out)),ensure_ascii=False))


if __name__=='__main__':main()
