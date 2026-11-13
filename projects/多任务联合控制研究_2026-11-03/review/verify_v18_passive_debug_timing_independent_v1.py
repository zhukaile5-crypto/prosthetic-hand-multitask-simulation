"""Strict leaf-level timing-only check of the single completed passive bridge."""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import argparse,hashlib,json,struct
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TIMING={'/solver_timing_ms/qp','/solver_timing_ms/native_inverse','/solver_timing_ms/whole_decision'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def ptr(path,key):return path+'/'+str(key).replace('~','~0').replace('/','~1')
def differences(a,b,path=''):
    if type(a) is not type(b):return [dict(path=path,kind='type')]
    if isinstance(a,dict):
        if set(a)!=set(b):return [dict(path=path,kind='keyset')]
        return [d for k in sorted(a) for d in differences(a[k],b[k],ptr(path,k))]
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,kind='length')]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,ptr(path,i))]
    equal=struct.pack('d',a)==struct.pack('d',b) if isinstance(a,float) else a==b
    return [] if equal else [dict(path=path,kind='value',new=a,original=b)]
def projected(a,path=''):
    if path in TIMING:
        if type(a) not in (int,float) or not np.isfinite(a) or a<0:raise ValueError('invalid timing '+path)
        return 'REGISTERED_WALLCLOCK_ONLY'
    if isinstance(a,dict):return {k:projected(v,ptr(path,k)) for k,v in a.items()}
    if isinstance(a,list):return [projected(v,ptr(path,i)) for i,v in enumerate(a)]
    return a
def accepts(a,b):
    try:
        if any(x['kind']!='value' or x['path'] not in TIMING for x in differences(a,b)):return False
        return not differences(projected(a),projected(b))
    except (ValueError,TypeError):return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    plan=read(ROOT/'review/v18_small_plan_v1.json');stage=next(s for s in plan['stages'] if s['name']=='passive_ratio_transport')
    base=ROOT/stage['output'];marker=read(base/'execution_complete.json')
    assert marker['execution_complete'] and marker['sources_unchanged'] and not marker['runtime_failures']
    new=base/'runs/nominal__ratio_effort__619741/trace.npz';old=ROOT/stage['original_same_condition']/'trace.npz'
    with np.load(new,allow_pickle=False) as z:nd=z['debug_json'];nt=z['time']
    with np.load(old,allow_pickle=False) as z:od=z['debug_json'];ot=z['time']
    assert np.array_equal(nt,ot) and len(nt)==3201
    diff=defaultdict(list);unknown=[];nh=hashlib.sha256();oh=hashlib.sha256();all_accepted=True
    for i,(x,y) in enumerate(zip(nd,od)):
        a=json.loads(str(x));b=json.loads(str(y));ds=differences(a,b)
        for item in ds:
            if item['kind']!='value' or item['path'] not in TIMING:unknown.append(dict(time_s=float(nt[i]),**item))
            else:diff[item['path']].append(dict(time_s=float(nt[i]),new=item['new'],original=item['original']))
        all_accepted=all_accepted and accepts(a,b)
        nh.update((json.dumps(projected(a),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
        oh.update((json.dumps(projected(b),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    j=int(np.argmin(abs(nt-7.5)));a=json.loads(str(nd[j]));b=deepcopy(a)
    assert type(a['decision_time_s']) is float
    b['decision_time_s']=float(np.nextafter(a['decision_time_s'],np.inf))
    oneulp=not accepts(a,b)
    b=deepcopy(a);b['solver_timing_ms']['unregistered_other_ms']=1.
    newkey=not accepts(a,b)
    b=deepcopy(a);b['solver_timing_ms']['qp']=-1.
    negative=not accepts(a,b)
    b=deepcopy(a);b['solver_timing_ms']['qp']+=.1
    positive=accepts(a,b)
    summary={p:dict(count=len(v),first_time_s=v[0]['time_s'],last_time_s=v[-1]['time_s'],example=v[0]) for p,v in sorted(diff.items())}
    result=dict(schema='v18_passive_debug_timing_independent_v1',passed=all_accepted and not unknown and nh.hexdigest()==oh.hexdigest() and all([oneulp,newkey,negative,positive]),packets=3201,all_structural_and_non_timing_leaves_exact_floatbytes=all_accepted,unapproved_differences=unknown,exact_approved_paths=sorted(TIMING),path_counts=summary,projected_new_sha256=nh.hexdigest(),projected_original_sha256=oh.hexdigest(),counterexamples=dict(non_timing_one_ULP_rejected=oneulp,new_debug_key_even_ms_suffix_rejected=newkey,negative_registered_timing_rejected=negative,finite_nonnegative_registered_timing_only_accepted=positive),input_sha256={str(new.relative_to(ROOT)):sha(new),str(old.relative_to(ROOT)):sha(old)},source_proof={'v14_coord/controller.py:207':'three named leaves from qp_ns/inverse_ns/perf_counter_ns divided by1e6; diagnostic only'},checker_sha256=sha(__file__),physics_steps=0,models_created=0,source_or_original_data_modified=False,other_three_candidate_outcomes_read=False,scope='Debug only: five external wallclock-column structural/full-array validation is root v2 separate evidence; original literal failed bridge remains preserved; no policy/physics benefit claim.')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=result['passed'],packets=3201,paths=summary,counterexamples=result['counterexamples'],path=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
