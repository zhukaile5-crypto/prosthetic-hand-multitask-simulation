"""Small independent read-only protection gate; no prepare/dispatch/results."""
from pathlib import Path
from unittest.mock import patch
import argparse, ast, hashlib, importlib.util, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FILES=('review/prepare_v22_small_v1.py','review/read_v22_small_readout_v1.py',
       'review/verify_v22_v21_full_trace_bridges_v1.py','review/v18_passive_trace_comparison_v3.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(out):
    if out.exists():raise FileExistsError(out)
    before={f:sha(ROOT/f) for f in FILES};checks=[]
    def ck(n,x):checks.append(dict(name=n,passed=bool(x)))
    r=module('independent_v22_readout_guard',ROOT/FILES[1])
    b=module('independent_v22_frozen_compare',ROOT/FILES[3])
    stages=[dict(name=f'{v}_{i}',variant=v,context_key=list(c),output='__independent_nonexistent_output__',
        assignment=dict(release_variant=v,scenarios=[c[0]],motion_mode=c[1],seeds=[c[2]]))
        for v in r.VARIANTS for i,c in enumerate(r.CONTEXTS)]
    plan=dict(stages=stages);good=dict(execution_complete=True,sources_unchanged=True,
        numerical_unchanged=True,runtime_failures=[],planned=1,trials=1)
    markers={s['name']:dict(good) for s in stages}
    ck('exact nine terminal roster accepted',r.barrier(plan,markers)['ready'])
    for field,value in [('execution_complete',False),('sources_unchanged',False),('numerical_unchanged',False),('runtime_failures',[dict(error='retained')])]:
        changed={k:dict(v) for k,v in markers.items()};changed[stages[-1]['name']][field]=value
        ck('unfinished/invalid '+field+' blocked',not r.barrier(plan,changed)['ready'])
    called=[]
    with patch.object(r.Path,'exists',return_value=False),patch.object(r,'module',side_effect=lambda *a,**k:called.append('calculator_import')),patch.object(r.np,'load',side_effect=lambda *a,**k:called.append('npz_read')):
        try:r.collect(plan);blocked=False
        except ValueError as e:blocked='All nine terminal' in str(e)
    ck('missing one/all results blocked before calculator or any NPZ read',blocked and not called)
    dup=dict(plan,stages=stages[:-1]+[stages[0]])
    try:r.barrier(dup,markers);blocked=False
    except ValueError:blocked=True
    ck('duplicate/missing assigned context rejected',blocked)
    unknown=r.difference({'physical_success':False,'latency':None},{'physical_success':True,'latency':.8})
    ck('failed success retained and NULL latency not zero',unknown['physical_success']['candidate'] is False and unknown['latency']['candidate'] is None and unknown['latency']['delta'] is None)
    base={'time':np.array([0.,.005]),'normal':np.ones((2,5)),'ctrl':np.zeros((2,21)),
          'metadata':np.array('old provenance'),'control_ms':np.array([1.,2.]),
          'debug_json':np.array([json.dumps({'solver_timing_ms':{'qp':1.,'native_inverse':2.,'whole_decision':3.},'actual_state':7})]*2)}
    other={k:v.copy() for k,v in base.items()};other['metadata']=np.array('new provenance');other['control_ms'][:]=7.
    other['debug_json']=np.array([json.dumps({'solver_timing_ms':{'qp':9.,'native_inverse':8.,'whole_decision':7.},'actual_state':7})]*2)
    ck('only declared walltime/provenance difference accepted',b.compare_arrays(base,other)['passed'])
    other['normal'][0,0]+=1e-12
    ck('physical force difference even tiny rejected',not b.compare_arrays(base,other)['passed']);other['normal']=base['normal'].copy()
    other['debug_json']=np.array([json.dumps({'solver_timing_ms':{'qp':9.,'native_inverse':8.,'whole_decision':7.},'actual_state':8})]*2)
    ck('non-timing debug difference rejected',not b.compare_arrays(base,other)['passed'])
    other={k:v.copy() for k,v in base.items()};other['control_ms'][0]=-1.
    ck('invalid walltime not covered by whitelist',not b.compare_arrays(base,other)['passed'])
    after={f:sha(ROOT/f) for f in FILES};ck('reviewed files unchanged during gate',before==after)
    data=dict(schema='v22_registration_readout_bridge_independent_v1',passed=all(c['passed'] for c in checks),
        checks=checks,reviewed_source_sha256=after,checker_sha256=sha(__file__),
        physics_steps=0,prepare_calls=0,dispatch_calls=0,production_imports=0,new_result_reads=0,
        scope='Finite barrier/NULL/whitelist counterexamples plus source reading; not execution-source/physical audit.')
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=data['passed'],checks=len(checks),sha256=sha(out),out=str(out))))
    return 0 if data['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();raise SystemExit(main(a.out))
