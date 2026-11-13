"""Read-only V2 audit; no simulation trials, tuning, or formal seeds."""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib,inspect
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
import controllers as c
import final_preflight_extra as prior
from physics import Plant,SensorPipeline
from scenarios import SCENARIOS,parameters
from run_experiments import hashes

def all_qps():
    rng=np.random.default_rng(927381);results=[];real=c.solve_qp
    shape_names=['nominal','box_heavy','sphere']
    smoke=ROOT/'output/common_model_smoke_v2'
    files={r['scenario']:r['trace_file'] for r in json.loads((smoke/'trials.json').read_text(encoding='utf-8')) if r['method']=='fixed'}
    for method in ['ratio','unified','unified_no_continuity','unified_no_geometry','unified_fixed_stage']:
        for case in range(5):
            name=shape_names[case%3];s=next(x for x in SCENARIOS if x.name==name);p=parameters(s,410181)
            plant=Plant(s,p);sf=SensorPipeline(s,p,410181).sample(plant.truth(),0.)
            z=np.load(smoke/files[name])
            index=1660
            normal=rng.uniform(.2,.7,5)*rng.uniform(1.3,5.,5)
            limits=np.maximum(normal+.35,rng.uniform(1.3,5.,5));budget=float(normal.sum()+rng.uniform(.6,2.))
            sf=replace(sf,normal=normal,tangential=z['observed_tangential'][index].copy(),
                slip_local=np.zeros((5,2)),contact_pos=z['observed_contact_pos'][index].copy(),
                contact_normal=z['observed_contact_normal'][index].copy(),tangent_basis=z['observed_tangent_basis'][index].copy(),
                joint=z['observed_joint'][index].copy(),joint_velocity=np.zeros(5),alpha=z['alpha'][index].copy(),
                valid=np.ones(5,bool),age=.01,phase='HOLDING',load_estimate=.8,load_valid=True,load_uncertainty=.15,load_confidence=.85)
            controller=c.make_controller(method,limits,budget)
            for tick in range(80):controller.state.update(replace(sf,time=tick*.005),.005)
            captured=[]
            def capture(Q,q,A,b,passes,tolerance,**kwargs):
                outcome=real(Q,q,A,b,passes,tolerance,**kwargs)
                captured.append((Q.copy(),q.copy(),A.copy(),b.copy(),kwargs,outcome))
                return outcome
            c.solve_qp=capture
            try:controller._decide(replace(sf,time=1.),'HOLDING')
            finally:c.solve_qp=real
            for Q,q,A,b,kwargs,outcome in captured:
                reference=real(Q,q,A,b,5000,1e-9,**kwargs)
                fun=lambda x:float(.5*x@Q@x+q@x)
                active=np.abs(A@outcome.x-b)<1e-7
                if np.any(active):
                    multipliers=np.linalg.lstsq(A[active].T,-(Q@outcome.x+q),rcond=1e-10)[0]
                    stationarity=float(np.max(abs(Q@outcome.x+q+A[active].T@multipliers)))
                else:stationarity=float(np.max(abs(Q@outcome.x+q)))
                results.append(dict(method=method,case=case,dimension=len(q),status=outcome.status,
                    converged=outcome.converged,residual=max(0.,float(np.max(A@outcome.x-b))),
                    reference_status=reference.status,relative_objective_gap=(fun(outcome.x)-fun(reference.x))/max(1.,abs(fun(reference.x))),
                    independent_stationarity_residual=stationarity))
    return dict(count=len(results),outer_count=sum(r['dimension']==18 for r in results),
        inner_count=sum(r['dimension']==15 for r in results),
        passed=all(r['converged'] and r['residual']<=1e-8 and abs(r['relative_objective_gap'])<1e-8 for r in results),cases=results)

def ratio_clock():
    r=c.make_controller('ratio',[5.]*5,22.);r.state.normal[:]=4.;r.state.delayed_normal[:]=2.
    r.state.tangential[:]=[.2,0.];r.state.contact[:]=True;r.state.slip_speed=0.;r.state.time=1.
    r._ratio_target('HOLDING')
    return dict(measured_ratio=r.debug['ratio'],expected_aligned_ratio=.1,
        passed=abs(r.debug['ratio']-.1)<1e-12)

def smoke_readback():
    source=inspect.getsource(prior.smoke_evidence).replace('def smoke_evidence():','def smoke_v2():').replace('common_model_smoke_v1','common_model_smoke_v2')
    namespace=dict(prior.__dict__);exec(source,namespace);r=namespace['smoke_v2']()
    manifest=json.loads((ROOT/'output/common_model_smoke_v2/manifest.json').read_text(encoding='utf-8'))
    changed=[key for key,value in hashes().items() if manifest['code_sha256'][key]!=value]
    r['source_changes_since_smoke']=changed
    r['accepted_pre_correction_controller_sha256']=manifest['code_sha256']['controllers.py']
    r['physics_and_evaluation_current']=changed==['controllers.py'] or changed==[]
    r['scope']='21 immutable development traces pre one-line ratio time-alignment correction; no current formal efficacy claim'
    return r

def tuning_inputs():
    plan=json.loads((ROOT/'controller_candidates.json').read_text(encoding='utf-8'));manifests=[]
    for j in range(3):manifests.append(json.loads((ROOT/f'output/candidates_v2_final_{j}/manifest.json').read_text(encoding='utf-8')))
    return dict(candidates_per_method={k:len(v) for k,v in plan['candidates'].items()},
        expected=360,same_matrix=all(m['code_sha256']==hashes() and m['seeds']==plan['development_seeds'] and
            m['scenarios']==plan['development_scenarios'] and m['methods']==['fixed','ratio','independent','unified'] and
            m['expected_trials']==120 for m in manifests),selection_rule=plan['selection_rule'],
        scope='input matrix only; no candidate result selection or formal seed inspection')

if __name__=='__main__':
    before=hashes();r=dict(qps=all_qps(),ratio_clock=ratio_clock(),sensor_boundary=prior.sensor_boundary(),
        formal_missing_record=prior.formal_missing_record(),smoke_evidence=smoke_readback(),tuning_budget=tuning_inputs())
    r['runtime_sha256']=before;r['source_unchanged']=before==hashes();r['formal_seeds_used']=False
    r['passed']=(r['source_unchanged'] and r['qps']['passed'] and r['ratio_clock']['passed'] and r['sensor_boundary']['passed'] and
        r['formal_missing_record']['passed'] and r['smoke_evidence']['all_checks_pass'] and
        r['smoke_evidence']['physics_and_evaluation_current'] and r['tuning_budget']['same_matrix'])
    (ROOT/'review/final_preflight_v2_checks.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],qp_count=r['qps']['count'],outer=r['qps']['outer_count'],inner=r['qps']['inner_count'],
        source_unchanged=r['source_unchanged'],ratio_clock=r['ratio_clock'],smoke_traces=r['smoke_evidence']['count'],
        max_objective_gap=max(abs(z['relative_objective_gap']) for z in r['qps']['cases']),
        same_tuning_matrix=r['tuning_budget']['same_matrix'],runtime_sha256=r['runtime_sha256']),ensure_ascii=False))
