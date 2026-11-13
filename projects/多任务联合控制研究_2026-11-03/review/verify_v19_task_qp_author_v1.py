"""Finite independent-formula author math fixtures; no model/native/physics."""
from pathlib import Path
from itertools import product
from dataclasses import replace
import sys,hashlib,importlib.util,json,copy,argparse,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SOURCE=ROOT/'v19_contact_reference/task_qp.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def base(P=0):
    return dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),gain21=np.ones(21),offset21=np.zeros(21),
        Jpx25=np.zeros((P,25)),y_p=np.ones(P),tau_extra25=np.zeros(25),delta_lower21=-np.ones(21),delta_upper21=np.ones(21),
        open_step21=np.zeros(21),motion_step21=np.zeros(21),unload_fraction=.5,patch_digit_p=np.arange(P,dtype=int)%5,caps5=np.full(5,100.),total_budget=1000.,
        scales=dict(residual25=np.ones(25),delta21=np.ones(21),reference21=np.ones(21),intentP=np.ones(P),progress=1.))
def raw(p,u,z,gamma,actual=None):
    d=p['offset21']+p['gain21']*u if actual is None else actual
    r=p['A21x25'].T@d-(p['Jpx25'].T@(z-p['y_p'])+p['tau_extra25'])
    sr,sd,su,sf,sg=(p['scales'][s] for s in ('residual25','delta21','reference21','intentP','progress'))
    a=sr*r;b=sd*d;c=su*(u-gamma*p['open_step21']-p['motion_step21']);e=sf*(z+gamma*p['unload_fraction']*p['y_p']-p['y_p']);f=sg*(gamma-1.)
    return float(a@a+b@b+c@c+e@e+f*f),r,d
def oracle(a,g,c,j,y,tau,o,m,rho,lo,hi,cap,gl,gh):
    M=np.array([[a*g,-j,0.],[g,0.,0.],[1.,0.,-o],[0.,1.,rho*y],[0.,0.,1.]])
    k=np.array([a*c+j*y-tau,c,-m,-y,-1.]);H=M.T@M;linear=M.T@k
    lower=np.array([lo,0.,gl]);upper=np.array([hi,cap,gh]);candidates=[]
    for state in product((-1,0,1),repeat=3):
        fixed=[i for i,v in enumerate(state) if v];free=[i for i,v in enumerate(state) if not v]
        x=np.zeros(3)
        for i in fixed:x[i]=lower[i] if state[i]<0 else upper[i]
        if free:x[free]=np.linalg.solve(H[np.ix_(free,free)],-linear[free]-H[np.ix_(free,fixed)]@x[fixed])
        if np.all(x>=lower) and np.all(x<=upper):candidates.append(x)
    best=min(candidates,key=lambda x:float(np.square(M@x+k).sum()))
    return best,float(np.square(M@best+k).sum()),len(candidates)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='review/v19_task_qp_author_attempt1.json');args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    spec=importlib.util.spec_from_file_location('v19_task_author',SOURCE);q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
    snap=ROOT/'review/revisions'/(out.stem+'_source.py');snap.parent.mkdir(exist_ok=True)
    if snap.exists():raise FileExistsError(snap)
    snap.write_bytes(SOURCE.read_bytes());pin=sha(SOURCE);checks=[];details={};started=time.perf_counter()
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def run(name,fn):
        try:fn()
        except Exception as e:ck(name+'/exception',False,dict(error=repr(e),traceback=traceback.format_exc()))
    def solve(p,name):
        r=q.solve_task_qp(**p);ck(name+'/converged',r.available,dict(status=r.status,reason=r.reason))
        if r.du21 is None:return r
        ev=q.evaluate_task(p,r.du21,r.zP,r.gamma);v,res,d=raw(p,r.du21,r.zP,r.gamma)
        ck(name+'/same raw objective/residual/d',np.float64(v).tobytes()==np.float64(r.objective).tobytes() and res.tobytes()==r.residual25.tobytes() and d.tobytes()==r.d21.tobytes())
        ck(name+'/fresh feasible directKKT',ev['available'] and ev['feasible'] and ev['direct_KKT_met'])
        ck(name+'/sameproblem merit',q.check_same_problem_candidate(p,r)['accepted'])
        details[name]=dict(du=r.du21,z=r.zP,gamma=r.gamma,d=r.d21,objective=r.objective,iterations=r.iterations)
        return r
    def closed():
        p=base();p['gain21'][:]=0;p['open_step21'][0]=.2
        r=solve(p,'zero_gain_keeps_geometry');ck('plateau reference moves without fake effort inverse',abs(r.du21[0]-.2)<1e-10 and abs(r.gamma-1)<1e-10 and np.max(abs(r.d21))==0)
        p['gamma_lower']=p['gamma_upper']=0.;p['motion_step21'][0]=.1
        r=solve(p,'SEARCH_fixedgamma0_motion_alive');ck('search no OPEN but motion retained',r.gamma==0 and abs(r.du21[0]-.1)<1e-10)
        for sign in (1.,-1.):
            p=base();p['gamma_lower']=p['gamma_upper']=0.;p['A21x25'][0,0]=1;p['gain21'][0]=2.*sign;p['offset21'][0]=.3;p['tau_extra25'][0]=.1
            r=solve(p,'offset_gain_'+str(sign));ck('offset enters mechanical AND delta '+str(sign),abs(r.du21[0]+sign/9)<1e-10 and abs(r.d21[0]-(.3-2/9))<1e-10)
        p=base();p['A21x25'][0,0]=1;p['open_step21'][0]=1.
        r=solve(p,'progress_conflict');ck('gamma transparent coupled compromise',abs(r.gamma-.6)<1e-10 and abs(r.du21[0]-.2)<1e-10)
        p=base(1);p['gain21'][:]=0;p['open_step21'][0]=.3;p['unload_fraction']=1.
        r=solve(p,'normal_progress_joint');ck('normal unload and reference share gamma',abs(r.gamma-1)<1e-10 and abs(r.zP[0])<1e-10 and abs(r.du21[0]-.3)<1e-10)
    run('closed',closed)
    def faces():
        cases=[(1.,1.,.1,1.,.8,.2,.3,.05,.7,-.3,.4,.35,0.,1.),
               (-1.,-2.,.3,.7,.4,-.2,.5,-.1,.5,-.15,.3,.8,0.,1.),
               (1.,0.,.6,1.,1.,.2,1.,.1,1.,-.2,.4,.3,0.,1.),
               (.3,2.,-.2,-.5,.6,1.1,-.4,.05,.8,.05,.2,.4,.3,.3),
               (0.,0.,0.,0.,.2,0.,.2,.1,0.,-.3,.4,0.,0.,1.)]
        for i,(a,g,c,j,y,tau,o,m,rho,lo,hi,cap,gl,gh) in enumerate(cases):
            p=base(1);p['A21x25'][0,0]=a;p['gain21'][0]=g;p['offset21'][0]=c;p['Jpx25'][0,0]=j;p['y_p'][0]=y;p['tau_extra25'][0]=tau
            p['open_step21'][0]=o;p['motion_step21'][0]=m;p['unload_fraction']=rho;p['delta_lower21'][0]=lo;p['delta_upper21'][0]=hi;p['caps5'][0]=p['total_budget']=cap;p['gamma_lower']=gl;p['gamma_upper']=gh
            expected,value,count=oracle(a,g,c,j,y,tau,o,m,rho,lo,hi,cap,gl,gh);r=solve(p,'independent3D_faces_'+str(i))
            ck('independent3D oracle '+str(i),np.max(abs(np.r_[r.du21[0],r.zP[0],r.gamma]-expected))<1e-8 and abs(value-r.objective)<1e-10,dict(candidates=count,expected=expected.tolist(),expected_objective=value))
    run('faces',faces)
    def gradients():
        rng=np.random.default_rng(19081);p=base(3);p['A21x25']=rng.normal(size=(21,25))*.1;p['gain21']=rng.normal(size=21);p['gain21'][2]=0
        p['offset21']=rng.normal(size=21)*.1;p['Jpx25']=rng.normal(size=(3,25))*.2;p['tau_extra25']=rng.normal(size=25)*.2
        p['open_step21']=rng.normal(size=21)*.1;p['motion_step21']=rng.normal(size=21)*.05;p['unload_fraction']=.6
        p['scales']['reference21'][:]=3.;u=rng.normal(size=21)*.1;z=np.array([.3,.4,.5]);gamma=.4;ev=q.evaluate_task(p,u,z,gamma);x=np.r_[u,z,gamma];fd=[]
        for j in range(len(x)):
            a=x.copy();b=x.copy();a[j]+=2e-7;b[j]-=2e-7;fd.append((raw(p,a[:21],a[21:-1],a[-1])[0]-raw(p,b[:21],b[21:-1],b[-1])[0])/4e-7)
        err=float(np.max(abs(np.r_[ev['gradient_du21'],ev['gradient_zP'],ev['gradient_gamma']]-fd)))
        ck('independent central differences all25 derivatives',err<1e-7,err)
        actual=p['offset21']+p['gain21']*u;actual[0]+=.01
        native=q.evaluate_task(p,u,z,gamma,actual_delta21=actual);value,res,_=raw(p,u,z,gamma,actual)
        ck('actual native substitution same full objective and reference terms',native['available'] and np.float64(value).tobytes()==np.float64(native['objective']).tobytes() and res.tobytes()==native['residual25'].tobytes() and native['native_delta_substituted'] and abs(native['affine_realization_error21'][0]-.01)<1e-15)
    run('gradients',gradients)
    def anchors_and_rejects():
        p=base(1);p['gamma_lower']=p['gamma_upper']=.4;p['delta_lower21'][:]=.05;p['delta_upper21'][:]=.2;p['motion_step21'][:]=.3;p['caps5'][0]=p['total_budget']=.2
        r=solve(p,'offdomain_zero_anchor');anchor=r.diagnostics['merit_anchor']
        ck('anchor gamma and du exact actual domain not false infeasible',anchor['gamma']==.4 and not anchor['zero_gamma_in_domain'] and not anchor['zero_reference_increment_in_domain'] and np.min(anchor['du21'])==.05 and anchor['zP']==[.2])
        ck('provisional sameproblem not fake optimum',q.check_same_problem_candidate(p,replace(r,status='nonconvergence'))['accepted'] and not q.check_same_problem_candidate(p,replace(r,status='nonconvergence'))['optimality_certified'])
        ck('bad objective rejects',not q.check_same_problem_candidate(p,replace(r,objective=r.objective+.1))['accepted'])
        ck('bad gamma rejects',not q.check_same_problem_candidate(p,replace(r,gamma=1.2))['accepted'])
        ck('nonzero offset gain2 proves du not effort alias',not np.array_equal(details['offset_gain_1.0']['du'],details['offset_gain_1.0']['d']))
        for key,value in [('gain21',None),('offset21',None),('tau_extra25',None),('unload_fraction',None),('Jpx25',None)]:
            p=base();p[key]=value;res=q.solve_task_qp(**p);ck('unknown '+key,res.status=='unknown' and res.du21 is None)
        p=base();p['unload_fraction']=1.1;ck('unload fraction not silently clamped',q.solve_task_qp(**p).status=='invalid_input')
        p=base();p['gamma_lower']=.7;p['gamma_upper']=.3;ck('empty gamma domain rejects',q.solve_task_qp(**p).status=='invalid_input')
        p=base();ck('bool gamma candidate rejected',not q.evaluate_task(p,np.zeros(21),np.empty(0),True)['available'])
        p=base(128);p['gamma_lower']=p['gamma_upper']=0;res=solve(p,'full128_no64variablecap');ck('full P128 accepted domain',res.available and len(res.zP)==128)
        p=base(129);ck('P129 unknown not zero omission',q.solve_task_qp(**p).status=='unknown')
    run('anchors_rejects',anchors_and_rejects)
    ck('no model/physics import',not any(k.startswith('mujoco') or k=='hand_plant' for k in sys.modules))
    old=read(ROOT/'review/v18_small_plan_v1.json')['source_sha256']
    ck('old144 exact retained',all(sha((ROOT.parent/'五指策略接入_2026-11-03'/k[11:]) if k.startswith('foundation/') else ROOT/k)==v for k,v in old.items()))
    ck('candidate stable during QA',sha(SOURCE)==pin)
    report=dict(schema='v19_task_qp_author_v1',role='mathematical author gate, not independent/native/physical validation',passed=all(c['passed'] for c in checks),checks=checks,details=details,
        source_sha256=pin,source_snapshot=str(snap.relative_to(ROOT)),checker_sha256=sha(__file__),elapsed_s=time.perf_counter()-started,physics_steps=0,model_constructions=0,
        limitations=['No native branch/model membership is certified here; caller owns offset/gain and reference-cell validity.','Actual native substitution tests are arithmetic only, not originalF067 reevaluation.','No geometry/contact/pressure prediction or hardware realtime/placement certificate.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],source_sha256=pin,report=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
if __name__=='__main__':raise SystemExit(main())
