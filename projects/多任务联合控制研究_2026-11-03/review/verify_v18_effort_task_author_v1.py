"""Independent-formula mathematical author QA; no model/native/physics calls."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import argparse,copy,hashlib,importlib.util,json,sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SOURCE=ROOT/'v18_effort_transfer/effort_task.py'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def problem(P=0):
    return dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),Jpx25=np.zeros((P,25)),y_p=np.zeros(P),z_intent_p=np.ones(P),
        effort_lower21=-2*np.ones(21),effort_upper21=2*np.ones(21),patch_digit_p=np.arange(P,dtype=int)%5,
        caps5=np.full(5,10.),total_budget=50.,support_coeff_p=np.zeros(P),load_proxy_N=0.,tau_extra25=np.zeros(25),
        scales=dict(residual25=np.ones(25),intentP=np.ones(P),delta21=np.ones(21),load=1.))
def raw(p,d,z):
    r=p['A21x25'].T@d-(p['Jpx25'].T@(z-p['y_p'])+p['tau_extra25'])
    sr,sf,sd,sl=(p['scales'][s] for s in ('residual25','intentP','delta21','load'))
    short=max(0.,p['load_proxy_N']-float(p['support_coeff_p']@z))
    value=float((sr*r)@(sr*r)+(sf*(z-p['z_intent_p']))@(sf*(z-p['z_intent_p']))+(sd*d)@(sd*d)+(sl*short)**2)
    e=p.get('extra_task_rows')
    if e is not None:
        v=e['matrix']@np.r_[d,z]-e['target'];value+=float(v@v)
    return value,r


def scalar_oracle(a,j,y,target,tau,dlo,dhi,zmax,c,load,ed=0.,ez=0.,b=0.):
    """Exact finite active faces of 2D box plus squared-hinge pieces."""
    intervals=[(0.,zmax,False)]
    if c>0 and load>0:
        kink=load/c
        intervals=[]
        if min(zmax,kink)>=0:intervals.append((0.,min(zmax,kink),True))
        if max(0.,kink)<=zmax:intervals.append((max(0.,kink),zmax,False))
    candidates=[]
    def f(d,z):return (a*d-(j*(z-y)+tau))**2+(z-target)**2+d*d+max(0.,load-c*z)**2+(ed*d+ez*z-b)**2
    for zl,zh,active in intervals:
        v=np.array([a,-j]);w=np.array([ed,ez])
        H=np.outer(v,v)+np.eye(2)+np.outer(w,w)
        h=(j*y-tau)*v+np.array([0.,-target])-b*w
        if active:H[1,1]+=c*c;h[1]-=load*c
        x=-np.linalg.solve(H,h)
        if dlo<=x[0]<=dhi and zl<=x[1]<=zh:candidates.append(x)
        for d in (dlo,dhi):candidates.append(np.array([d,np.clip(-(h[1]+H[1,0]*d)/H[1,1],zl,zh)]))
        for z in (zl,zh):candidates.append(np.array([np.clip(-(h[0]+H[0,1]*z)/H[0,0],dlo,dhi),z]))
    best=min(candidates,key=lambda x:f(*x))
    return best,float(f(*best)),len(candidates)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='review/v18_effort_task_author_attempt1.json');args=ap.parse_args()
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    spec=importlib.util.spec_from_file_location('v18_effort_math_author',SOURCE);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    snapshot=ROOT/'review/revisions'/(out.stem+'_source.py');snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():raise FileExistsError(snapshot)
    snapshot.write_bytes(SOURCE.read_bytes());before=sha(SOURCE)
    plan=ROOT/'review/v17b_prospective80_plan_v1.json';old=json.loads(plan.read_text(encoding='utf-8'))['source_sha256']
    checks=[];details={};start=time.perf_counter()
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def run(name,fn):
        try:fn()
        except Exception as exc:ck(name+'/exception',False,dict(error=repr(exc),traceback=traceback.format_exc()))
    def solution(p,name,expected_d=None,expected_z=None):
        r=m.solve_effort_task(**p);ev=m.evaluate_effort_task(p,r.d21,r.zP) if r.d21 is not None else None
        ck(name+'/converged',r.available,dict(status=r.status,reason=r.reason))
        if ev is None:return r
        value,res=raw(p,r.d21,r.zP)
        ck(name+'/raw objective residual exact',np.float64(value).tobytes()==np.float64(r.objective).tobytes() and res.tobytes()==r.residual25.tobytes())
        ck(name+'/fresh original-unit directKKT feasible',ev['available'] and ev['feasible'] and ev['direct_KKT_met'])
        ck(name+'/sameproblem merit',m.check_same_problem_candidate(p,r)['accepted'])
        if expected_d is not None:ck(name+'/analytic delta',np.max(abs(r.d21-expected_d))<1e-9)
        if expected_z is not None:ck(name+'/analytic normal',np.max(abs(r.zP-expected_z),initial=0.)<1e-9)
        details[name]=dict(d=r.d21,z=r.zP,objective=r.objective,residual=r.residual25,iterations=r.iterations)
        return r
    def basic():
        p=problem();p['A21x25'][0,0]=1;p['tau_extra25'][0]=.6
        want=np.zeros(21);want[0]=.3;r=solution(p,'nonzero_rhs_noP',want,np.empty(0))
        q=copy.deepcopy(p);q['tau_extra25']*=-1
        s=solution(q,'signed_rhs_flip',-want,np.empty(0));ck('signed target not forced nonnegative',s.d21[0]<0 and r.d21[0]>0)
        p=problem(1);p['A21x25'][0,0]=p['Jpx25'][0,0]=1;p['y_p'][0]=.3;p['z_intent_p'][0]=.7;p['tau_extra25'][0]=.2
        want=np.zeros(21);want[0]=.2;r=solution(p,'coupled_nonzero_rhs',want,np.array([.5]))
        q=copy.deepcopy(p);q['caps5'][0]=q['total_budget']=.4;want[0]=.15;solution(q,'normal_cap_budget_exact',want,np.array([.4]))
        q=copy.deepcopy(p);q['tau_extra25'][24]=.9;r=solution(q,'unreachable_generalized_component')
        ck('unreachable remains real residual',r.residual25[24]==-.9)
        p=problem();p['A21x25'][0,0]=1;p['tau_extra25'][0]=.2;E=np.zeros((1,21));E[0,0]=2
        p['extra_task_rows']=dict(matrix=E,target=np.array([.8]));want=np.zeros(21);want[0]=.3;solution(p,'softshape_nonzero_target',want,np.empty(0))
    run('basic',basic)
    def plateau():
        p=problem(2);p['A21x25'][0,0]=p['A21x25'][1,1]=1;p['Jpx25'][0,0]=p['Jpx25'][1,1]=1;p['y_p'][:]=1
        p['p0_21'][0]=1;p['effort_lower21'][0]=p['effort_upper21'][0]=1;p['tau_extra25'][:2]=[.2,.4]
        want=np.zeros(21);want[1]=.4/3
        r=solution(p,'known_plateau_keeps_other_axis_and_both_normals',want,np.array([.9,1-.4/3]))
        ck('singleton no inverse and unattainable pressure effort visible',r.d21[0]==0 and abs(r.residual25[0]+.1)<1e-12 and len(r.zP)==2)
    run('plateau',plateau)
    def oracles():
        rows=[(1.,1.,.3,.7,.2,-2.,2.,2.,0.,0.,0.,0.,0.),
              (-1.,.7,.4,.9,-.2,-.15,.3,.35,.5,.5,0.,0.,0.),
              (.4,-.5,.8,.2,1.2,-.3,.2,.6,.9,.4,.3,.5,.4),
              (1.,0.,0.,0.,.7,-.1,.2,0.,0.,0.,0.,0.,0.),
              (0.,1.,1.,1.,.5,0.,0.,1.5,.3,.8,.2,-.1,.3)]
        for i,(a,j,y,target,tau,dl,dh,cap,c,load,ed,ez,b) in enumerate(rows):
            p=problem(1);p['A21x25'][0,0]=a;p['Jpx25'][0,0]=j;p['y_p'][0]=y;p['z_intent_p'][0]=target;p['tau_extra25'][0]=tau
            p['effort_lower21'][0]=dl;p['effort_upper21'][0]=dh;p['caps5'][0]=p['total_budget']=cap;p['support_coeff_p'][0]=c;p['load_proxy_N']=load
            E=np.zeros((1,22));E[0,0]=ed;E[0,21]=ez;p['extra_task_rows']=dict(matrix=E,target=np.array([b]))
            expected,obj,count=scalar_oracle(a,j,y,target,tau,dl,dh,cap,c,load,ed,ez,b)
            r=solution(p,'independent2D_activeface_'+str(i))
            ck('independent2D_'+str(i)+'/solution and objective',np.max(abs(np.array([r.d21[0],r.zP[0]])-expected))<1e-9 and abs(r.objective-obj)<1e-10,dict(enumerated=count,expected=expected.tolist(),objective=obj))
    run('oracles',oracles)
    def gradient():
        rng=np.random.default_rng(71823);p=problem(3);p['A21x25']=rng.normal(size=(21,25))*.1;p['Jpx25']=rng.normal(size=(3,25))*.2
        p['y_p']=np.array([.2,.3,.5]);p['tau_extra25']=rng.normal(size=25)*.3;p['load_proxy_N']=2.;p['support_coeff_p']=np.array([.3,.4,.2])
        p['extra_task_rows']=dict(matrix=rng.normal(size=(4,24))*.1,target=np.array([.4,.3,.2,.1]));d=rng.normal(size=21)*.1;z=np.array([.1,.2,.3])
        ev=m.evaluate_effort_task(p,d,z);x=np.r_[d,z];numeric=[]
        for i in range(24):
            a=x.copy();b=x.copy();a[i]+=1e-6;b[i]-=1e-6;numeric.append((raw(p,a[:21],a[21:])[0]-raw(p,b[:21],b[21:])[0])/2e-6)
        err=float(np.max(abs(np.r_[ev['gradient_d21'],ev['gradient_zP']]-numeric)))
        ck('independent finite-difference all24 gradient',err<1e-8,err)
    run('gradient',gradient)
    def numerical():
        p=problem(1);p['A21x25'][0,0]=1;p['Jpx25'][0,1]=1;p['scales']['residual25'][0]=2e8
        r=solution(p,'original_hugeL_counterexample');ck('hugeL true z optimum',abs(r.zP[0]-.5)<1e-9)
        p=problem(1);p['z_intent_p'][0]=5e-13;p['caps5'][0]=p['total_budget']=1e-12;p['support_coeff_p'][0]=1;p['load_proxy_N']=1
        r=solution(p,'tiny_normal_box_counterexample');ck('tiny strict interior not falseKKT',r.zP[0]==1e-12)
        p=problem();p['A21x25'][0,0]=1;p['p0_21'][0]=2.**54;p['effort_lower21'][0]=p['p0_21'][0]-8;p['effort_upper21'][0]=p['p0_21'][0]+8;p['tau_extra25'][0]=1
        r=solution(p,'original_delta_survives_rounded_absolute_target');ck('raw delta not retargeted to rounded p21',abs(r.d21[0]-.5)<1e-9 and r.p21[0]==p['p0_21'][0])
        details['huge_p0_problem']=p;details['huge_p0_result']=r
    run('numerical',numerical)
    def unknown():
        for key,value in [('tau_extra25',None),('tau_extra25',np.full(25,np.nan)),('Jpx25',None),('load_proxy_N',None)]:
            p=problem();p[key]=value;r=m.solve_effort_task(**p);ck('unknown_'+key+'_'+str(value is None),r.status=='unknown' and r.d21 is None)
        p=problem(1);p['y_p'][0]=-.1;ck('signed data cannot masquerade as normal',m.solve_effort_task(**p).status=='invalid_input')
        p=problem();p['tau_extra25']=['0']*25;ck('string rhs rejected',m.solve_effort_task(**p).status=='invalid_input')
        p=problem();p['effort_lower21'][0]=1;p['effort_upper21'][0]=0;ck('empty effort domain rejected',not m.solve_effort_task(**p).available)
    run('unknown',unknown)
    def merit_native():
        p=problem(1);p['A21x25'][0,0]=p['Jpx25'][0,0]=1;p['y_p'][0]=.3;p['z_intent_p'][0]=.7;p['tau_extra25'][0]=.2
        r=m.solve_effort_task(**p);ev=m.evaluate_effort_task(p,r.d21,r.zP)
        provisional=replace(r,status='nonconvergence');rr=m.check_same_problem_candidate(p,provisional)
        ck('provisional feasible same-goal candidate no falsecert',rr['accepted'] and not rr['optimality_certified'])
        bad=replace(r,objective=r.objective+1e-3);ck('wrong objective refused',not m.check_same_problem_candidate(p,bad)['accepted'])
        bad=replace(r,d21=np.full(21,10.));ck('primal violation refused',not m.check_same_problem_candidate(p,bad)['accepted'])
        wrongp=copy.deepcopy(p);wrongp['tau_extra25'][0]=0;ck('old zero-rhs score refused',not m.check_same_problem_candidate(wrongp,r)['accepted'])
        def native_stub(p,wanted,d,z):
            native=p['p0_21']+d;actual=native-p['p0_21'];val,res=raw(p,actual,z);sd=p['scales']['delta21'];sr=p['scales']['residual25']
            return SimpleNamespace(available=True,p0_effort21=p['p0_21'].copy(),requested_delta_effort21=wanted.copy(),native_effort21=native,actual_delta_effort21=actual,
                residual25=res,objective=float((sr*res)@(sr*res)+(sd*actual)@(sd*actual)),diagnostics=dict(prepared_value_final_full_recheck=dict(passed=True)))
        stub=native_stub(p,r.d21,r.d21,r.zP);score=m.score_native_realization(p,original_d21=r.d21,zP=r.zP,native_result=stub)
        ck('final native total uses SAME normal and extra terms',score['available'] and np.float64(score['objective']).tobytes()==np.float64(ev['objective']).tobytes() and score['objective']!=stub.objective)
        ck('fixed correction bytes canonical',m.native_correction25(p,r.zP).tobytes()==ev['correction25'].tobytes())
        stub.residual25=stub.residual25+np.eye(25)[0]*.2;ck('stale zero-rhs native residual refused',not m.score_native_realization(p,original_d21=r.d21,zP=r.zP,native_result=stub)['available'])
        p=details['huge_p0_problem'];r=details['huge_p0_result'];stub=native_stub(p,r.d21,np.zeros(21),r.zP);score=m.score_native_realization(p,original_d21=r.d21,zP=r.zP,native_result=stub)
        ck('rounded absolute native mismatch retained not falsezero',score['available'] and abs(score['realization_error21'][0]+.5)<1e-9 and score['objective']>r.objective)
        stub.p0_effort21=stub.p0_effort21.copy();stub.p0_effort21[1]=1;ck('foreign p0 native report refused',not m.score_native_realization(p,original_d21=r.d21,zP=r.zP,native_result=stub)['available'])
    run('merit_native',merit_native)
    ck('readonly outputs',all(not np.asarray(details[k]['d']).flags.writeable for k in ('nonzero_rhs_noP','coupled_nonzero_rhs')))
    ck('no model or physics import',not any(s.startswith('mujoco') or s=='hand_plant' for s in sys.modules))
    ck('frozen136 unchanged',all(sha((ROOT.parent/'五指策略接入_2026-11-03'/k[11:]) if k.startswith('foundation/') else ROOT/k)==v for k,v in old.items()))
    ck('owned candidate source unchanged during tests',sha(SOURCE)==before)
    details.pop('huge_p0_result',None);details.pop('huge_p0_problem',None)
    report=dict(schema='v18_effort_task_author_v1',role='author_math_QA_not_independent_review_or_physical_gate',passed=all(c['passed'] for c in checks),checks=checks,details=details,
        source_sha256=before,source_snapshot=str(snapshot.relative_to(ROOT)),checker_sha256=sha(__file__),elapsed_s=time.perf_counter()-start,
        inherited_solver_sha256=sha(ROOT/'v15_support/task_qp.py'),direct_KKT_helper_sha256=sha(ROOT/'v13_joint/intent_qp.py'),physics_steps=0,model_constructions=0,
        limitations=['Native scoring tests use declared arithmetic stubs, not an F067 model or native admission.','No runtime, actor, pressure forecast, task graph, sensor/RNG, control authority or physical claim is implemented.','One fixed-budget convex current-state task; not future pressure safety, global floating optimum, or guaranteed real-time solver.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],source_sha256=before,report=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
