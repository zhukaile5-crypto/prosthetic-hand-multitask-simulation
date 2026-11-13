"""Prospective arithmetic oracle only; does not enable vector clips anywhere."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    ranges=[(-2.,2.),(-.01,.01),(-0.,0.),(0.,0.),(-0.,-0.),(-2.,-0.),(0.,2.),(-2.,np.nextafter(0.,-np.inf)),(np.nextafter(0.,np.inf),2.)]
    points=[-np.finfo(float).max,-2.,-.01,-1e-12,-0.,0.,1e-12,.01,2.,np.finfo(float).max]
    checks=[];comparisons=0
    for lo,hi in ranges:
        values=points+[lo,hi,np.nextafter(lo,-np.inf),np.nextafter(lo,np.inf),np.nextafter(hi,-np.inf),np.nextafter(hi,np.inf)]
        for value in values:
            x=np.full(21,value);lower=np.full(21,lo);upper=np.full(21,hi)
            scalar=np.array([np.clip(x[i],lower[i],upper[i]) for i in range(21)])
            vector=np.clip(x,lower,upper);ok=scalar.tobytes()==vector.tobytes();comparisons+=21
            checks.append(dict(lower=float(lo).hex(),upper=float(hi).hex(),input=float(value).hex(),passed=ok))
    # Same mixed vector/ranges repeated fixed1000 times, with output consumed.
    x=np.resize(np.array(points),21);lower=np.full(21,-.01);upper=np.full(21,.01);timing={}
    for mode in ('scalar','vector'):
        start=time.perf_counter_ns()
        for _ in range(1000):
            if mode=='scalar':y=np.array([np.clip(x[i],lower[i],upper[i]) for i in range(21)])
            else:y=np.clip(x,lower,upper)
        timing[mode]=(time.perf_counter_ns()-start)/1e6
    report=dict(schema='native_scalar_vector_clip_prospective_oracle_v1',passed=all(c['passed'] for c in checks),
        numpy_version=np.__version__,checks=checks,component_value_comparisons=comparisons,timing_1000_vectors_ms=timing,
        scope='Finite float64 scalar versus elementwise vector np.clip on this interpreter, signedzeros and adjacent endpoints included. No production source changed. Not all floatingpoint inputs or policy throughput proof.',physics_steps=0)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(passed=report['passed'],comparisons=comparisons,timing_1000_vectors_ms=timing,path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
