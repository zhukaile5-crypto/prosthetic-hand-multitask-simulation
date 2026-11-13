"""Frozen-geometry normal-force reserve feasibility on archived V21 public observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

ATTR=ROOT/'review/v21_stage_attribution_v1.json'
PROTOCOL=ROOT/'protocol.json'
LENGTH_M=.050
SCALE_N=.030
REQUEST_N=.20
TICKS=range(1850,1860)
CHECK_TOL=1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def solve(linprog, objective, aub, bub, bounds, aeq=None, beq=None):
    result=linprog(objective,A_ub=aub,b_ub=bub,A_eq=aeq,b_eq=beq,bounds=bounds,
                   method='highs',options={'primal_feasibility_tolerance':1e-9,
                                           'dual_feasibility_tolerance':1e-9})
    if result.status==2:
        return None
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def maximize_reserve(linprog, p, w, lower, upper, budget, mode):
    """Keep all six rows, allow instrument-scale variation, or keep only net force."""
    wb=np.column_stack((w,np.zeros(6)))
    zero=np.zeros(5)
    aub=np.vstack((np.column_stack((np.eye(5),np.ones(5))),np.r_[np.ones(5),0.]))
    bub=np.r_[upper,budget]
    aeq=beq=None
    if mode=='exact_wrench':
        aeq,beq=wb,w@p
    elif mode=='instrument_scale':
        aub=np.vstack((aub,wb,-wb))
        bub=np.r_[bub,w@p+SCALE_N,-w@p+SCALE_N]
    elif mode=='exact_force_only':
        aeq,beq=wb[:3],w[:3]@p
    else:
        raise ValueError(mode)
    x=solve(linprog,np.r_[zero,-1.],aub,bub,list(zip(lower,upper))+[(0.,None)],aeq,beq)
    if x is None:
        return dict(status='infeasible')
    q,reserve=x[:5],x[5]
    delta=w@(q-p)
    assert np.all(aub@x<=bub+CHECK_TOL)
    assert np.all(q>=lower-CHECK_TOL) and np.all(q<=upper+CHECK_TOL)
    assert reserve>=-CHECK_TOL and np.min(upper-q)>=reserve-CHECK_TOL
    if aeq is not None:
        assert np.allclose(aeq@x,beq,atol=CHECK_TOL,rtol=0)
    return dict(status='optimal',min_reserve_N=float(reserve),candidate_pressures_N=q.tolist(),
                delta_force_N=delta[:3].tolist(),delta_moment_Nm=(delta[3:]*LENGTH_M).tolist(),
                model_residual_equivalent_N=float(np.max(np.abs(delta))))


def minimum_wrench_change(linprog,p,w,lower,upper,budget,requested):
    caps=upper-requested
    if np.any(caps<lower):
        return dict(status='infeasible_under_pressure_floor')
    # x is five candidate pressures and the nonnegative worst component of W delta.
    aub=np.vstack((np.column_stack((w,-np.ones(6))),np.column_stack((-w,-np.ones(6))),
                   np.r_[np.ones(5),0.]))
    bub=np.r_[w@p,-w@p,budget]
    x=solve(linprog,np.r_[np.zeros(5),1.],aub,bub,list(zip(lower,caps))+[(0.,None)])
    if x is None:
        return dict(status='infeasible')
    q,slack=x[:5],x[5]
    delta=w@(q-p)
    assert np.all(aub@x<=bub+CHECK_TOL)
    assert np.all(q>=lower-CHECK_TOL) and np.all(q<=caps+CHECK_TOL)
    assert abs(np.max(np.abs(delta))-slack)<=CHECK_TOL
    return dict(status='optimal',minimum_model_residual_equivalent_N=float(slack),
                candidate_pressures_N=q.tolist(),delta_force_N=delta[:3].tolist(),
                delta_moment_Nm=(delta[3:]*LENGTH_M).tolist())


def checked_cases(linprog):
    lower=np.zeros(5)
    upper=np.array([1.,2.,2.,2.,2.])
    p=np.array([.9,.4,.4,.4,.4])
    w=np.vstack((np.eye(5),np.zeros(5)))
    rigid=maximize_reserve(linprog,p,w,lower,upper,9.,'exact_wrench')
    assert abs(rigid['min_reserve_N']-.1)<CHECK_TOL
    tolerant=maximize_reserve(linprog,p,w,lower,upper,9.,'instrument_scale')
    assert abs(tolerant['min_reserve_N']-.13)<CHECK_TOL
    required=minimum_wrench_change(linprog,p,w,lower,upper,9.,REQUEST_N)
    assert abs(required['minimum_model_residual_equivalent_N']-.1)<CHECK_TOL
    w[:,1]=w[:,0]
    swapped=maximize_reserve(linprog,p,w,lower,upper,9.,'exact_wrench')
    assert swapped['min_reserve_N']>.1 and swapped['model_residual_equivalent_N']<CHECK_TOL


def read_trial(record, contact_floor, linprog):
    src=record['metrics']['full_V21']
    trace=ROOT/src['trace']
    assert sha(trace)==src['trace_sha256']
    rows=[]
    with np.load(trace,allow_pickle=False) as a:
        metadata=json.loads(str(a['metadata'].item()))
        upper=np.asarray(metadata['limits'],float)
        budget=float(metadata['budget'])
        observations=a['observation_json']
        for i in TICKS:
            obs=json.loads(str(observations[i]))
            t=i*.005
            assert abs(float(a['time'][i])-t)<1e-9
            assert abs(obs['time']-t)<1e-9 and abs(obs['sample_time']-(t-.010))<1e-9
            row=dict(read_s=t,acquisition_s=obs['sample_time'])
            if not all(len(obs[key])==5 and all(obs[key]) for key in ('valid','distributed_valid','contact_present')):
                rows.append(dict(**row,status='five_valid_contacts_unavailable'))
                continue
            w_raw=np.array(obs['normal_wrench_map'],float)
            pressure=np.array(obs['normal'],float)
            if w_raw.shape!=(6,5) or not np.isfinite(w_raw).all() or pressure.shape!=(5,) or not np.isfinite(pressure).all():
                rows.append(dict(**row,status='pressure_or_wrench_unavailable'))
                continue
            if np.any(pressure<=0) or np.any(pressure>upper) or pressure.sum()>budget:
                rows.append(dict(**row,status='reference_pressure_not_inside_nominal_caps'))
                continue
            if any(abs(obs['sensor_characteristics'][key]-value)>1e-12
                   for key,value in (('force_sigma_N',.01),('auxiliary_moment_sigma_Nm',.0005))):
                rows.append(dict(**row,status='instrument_scale_mismatch'))
                continue
            w=w_raw*np.r_[np.ones(3),np.full(3,1./LENGTH_M)][:,None]
            lower=np.minimum(pressure,contact_floor)
            sv=np.linalg.svd(w,compute_uv=False)
            solutions={mode:maximize_reserve(linprog,pressure,w,lower,upper,budget,mode)
                       for mode in ('exact_wrench','instrument_scale','exact_force_only')}
            assert all(x['status']=='optimal' for x in solutions.values())
            minimum=minimum_wrench_change(linprog,pressure,w,lower,upper,budget,REQUEST_N)
            rows.append(dict(**row,status='assessed',observed_pressure_N=pressure.tolist(),
                             current_min_reserve_N=float(np.min(upper-pressure)),wrench_rank=int(np.linalg.matrix_rank(w)),
                             singular_values=sv.tolist(),maximum_reserve=solutions,
                             minimum_wrench_change_at_0p20N=minimum))
    return dict(context=record['context'],trace=src['trace'],trace_sha256=src['trace_sha256'],
                limits_N=upper.tolist(),budget_N=budget,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out
    from v22_joint_task_fusion.numerics import ensure_numeric_environment
    ensure_numeric_environment()
    from scipy.optimize import linprog

    checked_cases(linprog)
    source=read(ATTR)
    assert source['source_sha256']['protocol.json']==sha(PROTOCOL) and source['contexts']==16
    floor=read(PROTOCOL)['thresholds']['finger_contact_min_N']
    assert floor==.2
    cases=[read_trial(record,floor,linprog) for record in source['rows']]
    data=dict(schema='v21_frozen_normal_reserve_feasibility_v1',physics_steps=0,
              contexts=len(cases),physical_seed_clusters=2,
              contract=dict(length_m=LENGTH_M,component_engineering_scale_N=SCALE_N,requested_reserve_N=REQUEST_N,
                            fixed_read_times_s=[t*.005 for t in TICKS],contact_floor_N=floor,
                            model='fixed public palm-origin normal-wrench map only; no actor association or command response'),
              software_checks_passed=True,
              sources={x.name:sha(x) for x in (ATTR,PROTOCOL,Path(__file__))},contexts_data=cases)
    payload=json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)
    with out.open('x',encoding='utf-8') as f:
        f.write(payload+'\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':
    main()
