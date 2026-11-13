"""Pure pressure-measure task objective oracles. No solver/model/physics."""
from copy import deepcopy
from pathlib import Path
import argparse, hashlib, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v15_support.fusion_tasks import build_fusion_tasks

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixture():
    return dict(patch_digit_p=np.array([0,1,2,0,1,1]),normal_p=np.array([2.,3.,4.,1.,2.,6.]),
        reference_point_p3=[[-.02,0.,.01],[.03,.01,-.01],[.01,.04,.02],None,None,None],
        reference_normal_p3=[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],None,None,None],
        selected_p=np.array([True,True,True,False,False,False]),
        previous_conditional_force5=np.array([1.,2.,3.,0.,0.]),caps5=np.array([5.,4.,3.,5.,5.]),
        force_cost=.4,reference_continuity=3.,balance_weight=8.,support_contact_floor=.2,characteristic_length=.05)

def objective(spec,z):
    x=np.r_[np.zeros(21),z];e=spec.extra_matrix@x-spec.extra_target
    return float(np.square(spec.intentP*(z-spec.base_prior_p)).sum()+e@e)

def gradient(spec,z):
    x=np.r_[np.zeros(21),z]
    return 2.*spec.intentP**2*(z-spec.base_prior_p)+2.*spec.extra_matrix[:,21:].T@(spec.extra_matrix@x-spec.extra_target)

def direct(q,z):
    """Scalar group decomposition independent of constructed sf/rows/metadata."""
    d=q['patch_digit_p'];p=q['normal_p'];sel=q['selected_p'];caps=q['caps5'];floor=q['support_contact_floor']
    center=sum((p[j]*np.array(q['reference_point_p3'][j]) for j in np.flatnonzero(sel)),start=np.zeros(3))/p[sel].sum()
    cost=0.;g=np.zeros(len(z));groups=[]
    for digit in range(5):
        for conditional in (True,False):
            ids=np.flatnonzero((d==digit)&(sel==conditional))
            if not len(ids):continue
            shares=p[ids]/sum(p[i] for i in ids);total=sum(z[i] for i in ids);target=floor if conditional else 0.
            factor=q['force_cost']/max(1.,caps[digit])**2
            econ=factor*(total-target)**2
            distribution=factor*sum((z[i]-total*s)**2/s for i,s in zip(ids,shares))
            cost+=econ+distribution
            for i,s in zip(ids,shares):g[i]+=2.*factor*(z[i]-target*s)/s
            groups.append(dict(digit=digit,conditional=conditional,total=total,target=target,economy=econ,distribution=distribution))
    totals=np.array([sum(z[i] for i in range(len(z)) if sel[i] and d[i]==j) for j in range(5)])
    prev=q['previous_conditional_force5']
    if prev is None:prev=np.array([sum(p[i] for i in range(len(z)) if sel[i] and d[i]==j) for j in range(5)])
    norm=q['reference_continuity']/np.maximum(1.,caps)**2
    cost+=float(sum(norm*(totals-prev)**2))
    for i in np.flatnonzero(sel):g[i]+=2.*norm[d[i]]*(totals[d[i]]-prev[d[i]])
    wrench=np.zeros(5);rows={}
    for i in np.flatnonzero(sel):
        n=np.array(q['reference_normal_p3'][i]);v=np.r_[n[:2],np.cross(np.array(q['reference_point_p3'][i])-center,n)/q['characteristic_length']]
        rows[i]=v;wrench+=v*z[i]
    factor=q['balance_weight']/max(1.,float(caps.mean()))**2
    cost+=float(factor*(wrench@wrench))
    for i,v in rows.items():g[i]+=2.*factor*(v@wrench)
    return cost,g,groups,center

def split(q,index,fractions):
    """z_new=T z_old, same geometry, observed pressure divided alike."""
    P=len(q['normal_p']);parents=list(range(index))+[index]*len(fractions)+list(range(index+1,P))
    scale=np.ones(len(parents));scale[index:index+len(fractions)]=fractions
    T=np.zeros((len(parents),P));T[np.arange(len(parents)),parents]=scale
    b=deepcopy(q)
    for key in ('patch_digit_p','selected_p'):b[key]=q[key][parents].copy()
    b['normal_p']=T@q['normal_p']
    for key in ('reference_point_p3','reference_normal_p3'):b[key]=[deepcopy(q[key][i]) for i in parents]
    return b,T,np.array(parents)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    source=ROOT/'v15_support/fusion_tasks.py';before=sha(source);checks=[];failures=[];records=[]
    old=ROOT/'review/revisions/v15_fusion_tasks_f390_before_representation_revision.py'
    original_evidence=ROOT/'review/v15_fusion_tasks_author_v2.json';evidence_before=sha(original_evidence)
    def check(name,value,detail=None):
        checks.append(dict(name=name,passed=bool(value),detail=detail))
        if not value:failures.append(name)
    def close(name,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        # Predeclared arithmetic envelope: at most 256 eps times the summed
        # magnitudes. No impossible byte-equality demand after split reductions.
        error=float(np.max(abs(a-b),initial=0.));bound=256*np.finfo(float).eps*max(1.,float(np.sum(abs(a))),float(np.sum(abs(b))))
        check(name,error<=bound,dict(max_error=error,arithmetic_bound=bound))
    try:
        q=fixture();s=build_fusion_tasks(**q);check('base_available',s.available,s.reason)
        P=len(q['normal_p']);rng=np.random.default_rng(151102)
        for trial in range(8):
            z=rng.uniform(0.,3.,P);cost,g,groups,center=direct(q,z)
            close('direct_cost_'+str(trial),objective(s,z),cost);close('direct_gradient_'+str(trial),gradient(s,z),g)
            for i in range(P):
                h=1e-5;plus=z.copy();minus=z.copy();plus[i]+=h;minus[i]-=h
                fd=(objective(s,plus)-objective(s,minus))/(2*h)
                check('finite_difference_'+str(trial)+'_'+str(i),abs(fd-g[i])<=2e-8*max(1.,abs(fd),abs(g[i])),dict(fd=fd,gradient=float(g[i])))
            records.append(dict(trial=trial,z=z.tolist(),cost=cost,group_decomposition=groups))
        close('pressure_weighted_anchor',s.metadata['center_public_pressure_weighted_anchor'],direct(q,np.zeros(P))[3])
        check('anchor_not_com_or_arithmetic_mean','not center of mass' in s.metadata['center_role'] and not np.allclose(s.metadata['center_public_pressure_weighted_anchor'],np.mean(np.array(q['reference_point_p3'][:3]),axis=0)))
        for index in (0,4):
            b,T,parents=split(q,index,[.125,.375,.5]);ss=build_fusion_tasks(**b);check('split_available_'+str(index),ss.available,ss.reason)
            z=np.array([2.,1.5,2.5,.4,.7,1.3]);zs=T@z
            close('split_prior_'+str(index),ss.base_prior_p,T@s.base_prior_p)
            close('split_economy_hessian_pullback_'+str(index),T.T@np.diag(ss.intentP**2)@T,np.diag(s.intentP**2))
            close('split_total_cost_'+str(index),objective(ss,zs),objective(s,z))
            close('split_gradient_pullback_'+str(index),T.T@gradient(ss,zs),gradient(s,z))
            close('split_anchor_'+str(index),ss.metadata['center_public_pressure_weighted_anchor'],s.metadata['center_public_pressure_weighted_anchor'])
            close('split_digit_force_'+str(index),np.array(ss.metadata['conditional_digit_sum_map'])@zs,np.array(s.metadata['conditional_digit_sum_map'])@z)
            close('split_balance_'+str(index),np.array(ss.metadata['unweighted_balance_map'])@zs,np.array(s.metadata['unweighted_balance_map'])@z)
            close('split_linear_rows_'+str(index),ss.extra_matrix[:,21:]@T,s.extra_matrix[:,21:])
            close('split_linear_targets_'+str(index),ss.extra_target,s.extra_target)
            J=rng.normal(size=(P,25));close('split_JT_action_'+str(index),J[parents].T@zs,J.T@z)
            # Reverse proportional merge has the same cost. Deviating only
            # inside a duplicated point retains physics but pays distribution.
            if index==0:
                vv=np.zeros(len(zs));vv[index]=.05;vv[index+1]=-.05
                pert=zs+vv
                expected=float(sum(ss.intentP**2*vv**2))
                close('off_share_distribution_penalty',objective(ss,pert)-objective(ss,zs),expected)
                check('merge_keeps_distribution_scope_explicit',objective(ss,pert)>objective(s,z))
        order=np.array([5,2,0,4,1,3]);b=deepcopy(q)
        for key in ('patch_digit_p','normal_p','selected_p'):b[key]=q[key][order]
        for key in ('reference_point_p3','reference_normal_p3'):b[key]=[q[key][i] for i in order]
        perm=build_fusion_tasks(**b);z=np.array([2.,1.5,2.5,.4,.7,1.3])
        close('permutation_cost',objective(perm,z[order]),objective(s,z));close('permutation_gradient',gradient(perm,z[order]),gradient(s,z)[order])
        for factor in (2.**-20,32.,2.**20):
            b=deepcopy(q);b['normal_p']*=factor;ss=build_fusion_tasks(**b)
            check('uniform_scale_available_'+str(factor),ss.available,ss.reason)
            for name,a,c in [('prior',ss.base_prior_p,s.base_prior_p),('multiplier',ss.intentP,s.intentP),('extra_matrix',ss.extra_matrix,s.extra_matrix),('extra_target',ss.extra_target,s.extra_target)]:close('uniform_scale_'+str(factor)+'_'+name,a,c)
        b=deepcopy(q);b['normal_p'][~b['selected_p']]*=7.;ss=build_fusion_tasks(**b)
        close('selected_foreign_separate_shares',ss.intentP,s.intentP);close('foreign_scale_does_not_move_anchor',ss.metadata['center_public_pressure_weighted_anchor'],s.metadata['center_public_pressure_weighted_anchor'])
        check('foreign_unknown_geometry_inactive',all(s.metadata['geometry_not_used_p'][3:]) and np.array_equal(np.array(s.metadata['unweighted_balance_map'])[:,3:],np.zeros((5,3))))
        for name,key,value,status in [
            ('selected_point_unknown','reference_point_p3',[None]+q['reference_point_p3'][1:],'UNAVAILABLE_UNKNOWN'),
            ('selected_normal_unknown','reference_normal_p3',[None]+q['reference_normal_p3'][1:],'UNAVAILABLE_UNKNOWN'),
            ('zero_foreign_pressure','normal_p',np.array([2.,3.,4.,0.,2.,6.]),'UNAVAILABLE_DEGENERATE_GROUP'),
            ('selected_zero_group','normal_p',np.array([0.,3.,4.,1.,2.,6.]),'UNAVAILABLE_DEGENERATE_GROUP'),
            ('all_zero_group','normal_p',np.zeros(P),'UNAVAILABLE_DEGENERATE_GROUP'),
            ('unknown_pressure','normal_p',np.array([None,3.,4.,1.,2.,6.],object),'UNAVAILABLE_UNKNOWN')]:
            b=deepcopy(q);b[key]=value;ss=build_fusion_tasks(**b)
            check(name,ss.status==status and ss.intentP is None and ss.extra_matrix is None,dict(status=ss.status,reason=ss.reason))
        b=deepcopy(q);b['caps5'][:]=.1;b['support_contact_floor']=.8;ss=build_fusion_tasks(**b)
        close('floor_unclipped_by_caps',np.array(ss.metadata['conditional_digit_sum_map'])@ss.base_prior_p,np.array([.8,.8,.8,0.,0.]))
        check('floor_reports_overcap_not_retargeted',all(ss.metadata['floor_preference_above_cap5'][:3]) and not ss.metadata['floor_was_cap_or_budget_retargeted'])
        b,T,_=split(q,0,[.5,.5]);b['normal_p'][0]=np.nextafter(0.,1.);b['normal_p'][1]=1e308;ss=build_fusion_tasks(**b)
        check('underflow_share_unavailable_no_epsilon',not ss.available and ss.reason=='observed_pressure_share_not_representably_positive',dict(status=ss.status,reason=ss.reason))
        b,T,_=split(q,0,[.5,.5]);b['normal_p'][0]=np.nextafter(0.,1.);b['force_cost']=1e308;ss=build_fusion_tasks(**b)
        check('overflow_multiplier_unavailable_no_cap',not ss.available and ss.reason=='intent_multiplier_not_representably_positive',dict(status=ss.status,reason=ss.reason))
        b['normal_p'][0]=1e-200;b['force_cost']=q['force_cost'];ss=build_fusion_tasks(**b)
        check('tiny_representable_share_not_arbitrarily_clamped',ss.available and ss.intentP[0]>1e98,dict(available=ss.available,multiplier=None if ss.intentP is None else float(ss.intentP[0])))
        b=deepcopy(q);b['previous_conditional_force5']=None;ss=build_fusion_tasks(**b)
        close('first_frame_previous_is_observed_group',ss.metadata['previous_conditional_force5'],[2.,3.,4.,0.,0.])
        b['balance_weight']=b['reference_continuity']=0.;ss=build_fusion_tasks(**b);check('zero_weights_skip_rows',ss.extra_matrix.shape==(0,21+P))
        check('original_f390_snapshot_preserved',sha(old)=='f3902e626db3e22a8095896f8bf11de27c80d2ed05880753556d0b70345123eb')
        check('original_57_evidence_preserved',sha(original_evidence)==evidence_before)
        check('source_stable',sha(source)==before)
        check('no_model_solver_physics_import',not any(x in sys.modules for x in ('mujoco','v15_support.task_qp','v14_coord.task_qp')))
    except Exception:
        failures.append('exception');records.append(dict(exception=traceback.format_exc()))
    result=dict(schema='v15_representation_invariant_fusion_tasks_author_v1',passed=not failures,checks=checks,failures=failures,records=records,
        source_sha256=before,checker_sha256=sha(__file__),original_source_sha256=sha(old),original_author57_evidence_sha256=sha(original_evidence),
        arithmetic_rule='256*binary64_eps*max(1,sum_abs(lhs),sum_abs(rhs)) for algebraic representation transformations; finite-difference 2e-8 relative to max(1,gradient). No physical/model tolerances.',
        scope='Pure task objective and representation tests only. Proportional same-geometry subdivision/merge; arbitrary redistributed pressure intent retains a real distribution cost. No solver, robot, pressure prediction, runtime or physics.',physics_steps=0)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=failures,source_sha256=before,out=str(out)),ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
