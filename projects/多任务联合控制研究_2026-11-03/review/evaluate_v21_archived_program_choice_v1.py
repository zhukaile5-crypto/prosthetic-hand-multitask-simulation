"""Historical exact-program selection audit; all simulation results are read-only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'calibration/phase3_path_validation_v3'
AUDIT=ROOT/'review/independent_phase3_v3_saved_20261104T105201916786Z.json'
TARGET=ROOT/'review/v21_normal_reserve_locality_v1.json'
HORIZONS=(.02,.04,.06,.10)
MATCH_N=.03
TOL=1e-8


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def effect(kernel_t,kernel,delta_path,time_s,interval):
    starts=np.arange(len(delta_path)-1)*interval
    use=starts<time_s-1e-10
    if not use.any():
        return np.zeros(kernel.shape[1])
    delta=np.diff(delta_path,axis=0)[use]
    ages=time_s-starts[use]
    flat=kernel.reshape(len(kernel_t),-1)
    response=np.stack([np.interp(ages,kernel_t,flat[:,i],left=0.,right=flat[-1,i])
                       for i in range(flat.shape[1])],axis=-1).reshape(len(ages),kernel.shape[1],kernel.shape[2])
    return np.einsum('koi,ki->o',response,delta)


def load_arm(folder,expected_hash,interval):
    m=read(folder/'metadata.json')
    if not m['available']:
        return None
    trace=folder/'trace.npz'
    assert m['complete'] and m['trace_sha256']==expected_hash==sha(trace)
    with np.load(trace,allow_pickle=False) as z:
        raw_t=z['time']
        pressure=z['pressure']
        control=z['input_reference']
        physical=z['physical_input_reference']
    assert raw_t.shape==(101,) and pressure.shape==(101,5) and control.shape==(101,8)
    assert physical.shape==(1000,8) and np.isfinite(pressure).all()
    assert np.array_equal(control,np.array(m['executed_path']))
    assert np.allclose(physical,np.repeat(control[1:],10,axis=0),rtol=0,atol=1e-12)
    assert np.allclose(raw_t-raw_t[0],np.arange(101)*.005,rtol=0,atol=TOL)
    path=np.array(m['requested_path'])
    stride=round(interval/.005)
    assert np.allclose(control[::stride],path,rtol=0,atol=1e-12)
    assert np.all(path>=np.array(m['lower'])-1e-12) and np.all(path<=np.array(m['upper'])+1e-12)
    return dict(path=path,pressure=pressure,control=control,trace_sha256=expected_hash,time=raw_t)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out
    review=read(AUDIT)
    plan=read(BASE/'plan.json')
    assert review['passed'] and review['plan_sha256']==sha(BASE/'plan.json')
    assert review['summary_sha256']==sha(BASE/'summary.json')
    model_path=Path(plan['input_model']['path'])
    assert sha(model_path)==plan['input_model']['sha256']
    model=read(model_path)
    k=next(q for q in model['shapes'] if q['shape']=='capsule')['coordinates']['current_palm']
    kernel_t=np.array(k['response_kernel_times_s'])
    kernel=np.array(k['response_to_20ms_interval_increment'])[:,:5,:]
    assert kernel.shape==(len(kernel_t),5,8) and np.isfinite(kernel).all()
    interval=plan['interval_s']
    assert interval==.02 and tuple(plan['forecast_horizons_s'])==HORIZONS
    registry={(r['checkpoint'],r['branch']):r['trace_sha256'] for r in review['records'] if r['shape']=='capsule'}
    assert len(registry)==129
    target=read(TARGET)
    wanted={}
    for r in target['rows']:
        if abs(r['read_s']-9.295)<TOL:
            d=r['results']['scaled_force_and_moment']
            assert d['status']=='optimal'
            direction=np.array(d['force_adjustment_N'])
            seed=r['context'][2]
            assert seed not in wanted or np.array_equal(wanted[seed],direction)
            wanted[seed]=direction
    assert len(wanted)==2
    decisions, measured, skipped, differential_rows=[],[],[],[]
    labels=['zero','zero_replay']+[f'micro_{axis}_{sign}' for axis in range(8) for sign in ('plus','minus')]
    labels.extend(f'mixed_{i:02d}' for i in range(plan['mixed_sequences']))
    assert len(labels)==plan['planned_branches_per_checkpoint']
    for n in range(5):
        ck=f'checkpoint_{n:02d}'
        folder=BASE/'capsule'/ck
        assert read(folder/'restore_replay_check.json')['passed']
        arms={}
        for name in labels:
            if (ck,name) not in registry:
                m=read(folder/name/'metadata.json')
                assert not m['available']
                skipped.append(dict(checkpoint=ck,arm=name,reason=m.get('failure')))
                continue
            arms[name]=load_arm(folder/name,registry[ck,name],interval)
        zero=arms.pop('zero')
        repeat=arms.pop('zero_replay')
        assert all(np.array_equal(zero[key],repeat[key]) for key in ('pressure','path','control','time'))
        available={'zero':zero,**arms}
        forecast, actual={},{}
        for name,arm in available.items():
            assert np.array_equal(arm['time'],zero['time']) and np.array_equal(arm['path'][0],zero['path'][0])
            assert np.array_equal(arm['pressure'][0],zero['pressure'][0])
            dp=arm['path']-zero['path']
            pred=np.array([effect(kernel_t,kernel,dp,h,interval) for h in HORIZONS])
            truth=np.array([arm['pressure'][round(h/.005)]-zero['pressure'][round(h/.005)] for h in HORIZONS])
            forecast[name],actual[name]=pred,truth
            if name!='zero':
                grid=np.arange(26)*interval
                measured_error=(arm['pressure'][::4]-zero['pressure'][::4]) - np.array(
                    [effect(kernel_t,kernel,dp,h,interval) for h in grid])
                summary=read(folder/name/'validation.json')['differential_response_error']
                rmse=np.sqrt(np.mean(measured_error**2,axis=0))
                assert np.allclose(rmse,summary['rmse_per_output'][:5],atol=1e-9,rtol=0.)
                differential_rows.append(dict(checkpoint=ck,arm=name,
                    four_horizon_abs_delta_error_N=np.abs(pred-truth).tolist(),
                    all_knot_rmse_N=rmse.tolist()))
                measured.append(dict(checkpoint=ck,arm=name,trace_sha256=arm['trace_sha256'],
                                     actual_delta_N=truth.tolist(),predicted_delta_N=pred.tolist()))
        for pool in ('mixed_only','all_measured'):
            names=[name for name in available if pool=='all_measured' or name.startswith('mixed_') or name=='zero']
            for seed,desired in sorted(wanted.items()):
                for i,h in enumerate(HORIZONS):
                    def error(d):return float(np.max(np.abs(d-desired)))
                    chosen=min(names,key=lambda name:error(forecast[name][i]))
                    oracle=min(names,key=lambda name:error(actual[name][i]))
                    decisions.append(dict(checkpoint=ck,pool=pool,target_seed=seed,horizon_s=h,
                                          zero_error_N=error(actual['zero'][i]),chosen_arm=chosen,
                                          chosen_predicted_error_N=error(forecast[chosen][i]),
                                          chosen_actual_error_N=error(actual[chosen][i]),
                                          chosen_actual_change_N=actual[chosen][i].tolist(),
                                          oracle_arm=oracle,oracle_actual_error_N=error(actual[oracle][i])))
    assert len(measured)==119 and len(skipped)==1 and len(decisions)==80
    artifact=dict(schema='v21_fragile_exact_archived_program_selection_v1',physics_steps=0,
                  support='Phase3 standard capsule, single preparation and seed, five checkpoints, fixed archived 8D programmes',
                  label='ideal public pressure difference to offline same-state zero',
                  selector='minimum L-infinity target error on frozen kernel action response, no future zero used',
                  target_application='numerical V21 force change comparison only, no transplant or physical outcome claim',
                  target_match_scale_N=MATCH_N,targets={str(k):v.tolist() for k,v in wanted.items()},
                  horizon_s=HORIZONS,measured=measured,decisions=decisions,
                  differential_error=differential_rows,unavailable=skipped,
                  source_sha256={path.name:sha(path) for path in (BASE/'plan.json',AUDIT,model_path,TARGET,Path(__file__))})
    data=json.dumps(artifact,indent=2,ensure_ascii=False,allow_nan=False)
    with out.open('x',encoding='utf-8') as f:
        f.write(data+'\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':
    main()
