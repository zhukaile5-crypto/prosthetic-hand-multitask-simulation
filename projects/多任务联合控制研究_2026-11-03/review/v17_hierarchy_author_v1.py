"""Finite analytical 2/3-axis fixtures embedded in21; no model/physics import."""
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import sys,importlib.util,hashlib,json,datetime,warnings,time
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'v17_contact_transfer/hierarchy.py'
EXPECTED='a3c9f090a5cf4368d7d1ab36344b0aac4cfd76aac3c0367a298a38fdf2932457'
CHECKS=[]
CASE_TIMINGS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(name,value,detail=None):CHECKS.append(dict(name=name,passed=bool(value),detail=detail))
def embed(a,k):
    a=np.asarray(a,float).reshape((-1,k));out=np.zeros((len(a),21));out[:,:k]=a;return out
def tasks(C,p,canonical,*,lower=None,upper=None,tangent=None,tangent_target=None,time_s=2.):
    k=len(canonical);lo=np.zeros(21);hi=np.zeros(21);lo[:k]=-1 if lower is None else lower;hi[:k]=1 if upper is None else upper
    nom=np.zeros(21);nom[:k]=canonical;C=embed(C,k)
    T=embed([] if tangent is None else tangent,k);target=np.zeros(len(T)) if tangent_target is None else np.asarray(tangent_target,float)
    return SimpleNamespace(normal_matrix_m_per_reference=C,normal_offset_m=np.zeros(len(C)),step_lower21=lo,step_upper21=hi,open_target_increment21=nom,tangent_matrix_m_per_reference=T,tangent_target_m=target,diagnostics=dict(current_time_s=time_s,patches=[dict(observed_normal_N=float(x)) for x in p]))
def motion(a,b,k,stamp=2.):return SimpleNamespace(matrix_Mx21=embed(a,k),target_M=np.asarray(b,float),time_s=stamp,metadata=dict(role='explicit_public_motion_increment_fixture'))
def vertex_oracle(C,p,lo,hi):
    """Independent 2D LP maximum over every pairwise polygon intersection."""
    C=np.asarray(C,float);weight=np.asarray(p,float)/sum(p);out=-weight@C
    G=np.vstack((C,np.eye(2),-np.eye(2)));h=np.r_[np.zeros(len(C)),hi,-np.asarray(lo)];points=[np.zeros(2)]
    for i,j in combinations(range(len(G)),2):
        B=G[[i,j]]
        if abs(np.linalg.det(B))<1e-14:continue
        x=np.linalg.solve(B,h[[i,j]])
        if np.max(G@x-h)<1e-12:points.append(x)
    return max(points,key=lambda x:float(out@x)),out

def extra_checks(new):
    # Reuse the already separate 2D exhaustive active-face/KKT oracle, not its
    # old candidate or main. These are author checks; root arranges independent.
    helper=ROOT/'review/v16_hierarchy_independent_v2.py'
    spec=importlib.util.spec_from_file_location('_v17_author_2d_reference',helper);ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    cases=[
        ('triangle',[(-.5,.75)],[1.25],np.eye(2),[1,1],np.array([[-1.,0.],[1.,-1.]]),np.zeros(2)),
        ('saturated_sum',[(1.,1.)],[1.8],np.eye(2),[-1,1],None,None),
        ('duplicate_faces',[(1.,1.)],[2.],np.array([[1.,-1.]]),[.3],np.array([[1.,1.],[2.,2.]]),np.array([1.,2.])),
    ]
    for name,A,b,B,c,G,h in cases:
        began=time.perf_counter_ns();r=ref.run_case(new,'2D_enumeration_'+name,[-1,-1],[1,1],[(A,b),(B,c)],G,h)
        CASE_TIMINGS.append(dict(name=name,whole_ms=(time.perf_counter_ns()-began)/1e6,face_ms=r.diagnostics['degenerate_face_elapsed_ms'],face_events=r.diagnostics['degenerate_face_events'],rational_events=r.diagnostics['exact_rational_events']))
    began=time.perf_counter_ns();r=ref.run_case(new,'2D_strong_cross_priority',[-1,-1],[1,1],[([[1,0]],[.25]),([[0,1]],[.6]),(np.eye(2),[-100,-100])])
    CASE_TIMINGS.append(dict(name='strong_cross_priority',whole_ms=(time.perf_counter_ns()-began)/1e6,face_ms=r.diagnostics['degenerate_face_elapsed_ms']))
    ck('strong_conflict_does_not_trade_attained_higher_rows',np.allclose(r.x,[.25,.6],atol=2e-12,rtol=0))
    for c in ref.CHECKS:CHECKS.append(dict(name=c['name'],passed=c['passed'],detail=c['details']))
    ck('original_KKT_tolerances_not_relaxed',all(z['numerical_atol']==1e-9 and z['numerical_rtol']==1e-8 for z in r.levels))
    # Truly tiny original hard row amplitude remains a real normalized blocker.
    tiny=new.solve_hierarchy(lower=[0,-1],upper=[0,1],levels=(new.Level('y',np.array([[0.,1.]]),np.array([1.])),),inequality_matrix=[[0.,1e-200]],inequality_upper=[0.])
    ck('tiny_independent_hard_row_not_dropped',tiny.available and np.array_equal(tiny.x,[0.,0.]) and tiny.diagnostics['original_hard_constraint_violation']==0.)
    # A near-span but EXACTLY independent row is not certified dependent. With
    # inherited numerical-rank cutoff it may be unavailable, never crossed.
    near=new.solve_hierarchy(lower=[0,-1],upper=[0,1],levels=(new.Level('y',np.array([[0.,1.]]),np.array([1.])),),inequality_matrix=[[1.,1e-16]],inequality_upper=[0.])
    notes=[q for l in near.levels for q in l['degenerate_face_repairs']]
    ck('tiny_exact_independence_not_swallowed_by_span_threshold',not near.available and near.x[1]==0 and all(not q['certified_dependent_rows'] for q in notes),dict(status=near.status,x=near.x.tolist(),notes=notes))
    # A non-coordinate dependent face exercises the bounded rational quotient.
    raw_equal=np.array([[1.,1.,0.]]);raw_g=np.array([[0.,1.,1.],[1.,2.,1.]])
    lengths=np.linalg.norm(raw_g,axis=1);g=raw_g/lengths[:,None];h=np.array([0.,1.])/lengths
    x=np.array([1.,0.,0.]);p=np.array([.125,-.125,.125]);p[0]+=1e-16
    z=new._null(np.vstack((raw_equal,g[[0]])),3);cache={}
    began=time.perf_counter_ns();fixed,zero,note=new._repair_degenerate_direction(p=p,z=z,g=g,h=h,x=x,gp=g@p,active=[0],raw_equal=raw_equal,raw_g=raw_g,cache=cache)
    CASE_TIMINGS.append(dict(name='rational_quotient_proof',whole_ms=(time.perf_counter_ns()-began)/1e6,proof=note))
    ck('noncoordinate_exact_face_proof_and_repair',zero==[1] and note['corrected'] and note['rational_used'] and note['rational_ops']>0 and np.max(abs(raw_equal@fixed))<1e-15 and np.max(abs(raw_g@fixed))<1e-15,note)
    again,zero2,note2=new._repair_degenerate_direction(p=p,z=z,g=g,h=h,x=x,gp=g@p,active=[0],raw_equal=raw_equal,raw_g=raw_g,cache=cache)
    ck('exact_face_cache_reused',note2['cached'] and zero2==zero and np.array_equal(again,fixed))
    with patch.object(new,'EXACT_OPERATION_BUDGET',0):
        unmodified,zero3,budget=new._repair_degenerate_direction(p=p,z=z,g=g,h=h,x=x,gp=g@p,active=[0],raw_equal=raw_equal,raw_g=raw_g,cache={})
    ck('proof_budget_failure_not_permission_to_drop_row',not zero3 and np.array_equal(unmodified,p) and not budget['corrected'] and 'budget_exceeded' in budget['reason'],budget)
    ck('empty_level_list_still_checks_original_hard_domain',new.solve_hierarchy(lower=[0.],upper=[0.],levels=()).diagnostics['original_hard_constraint_violation']==0.)
def main():
    before=sha(SOURCE)
    if before!=EXPECTED:raise RuntimeError('Bind a stable reviewed source before execution')
    spec=importlib.util.spec_from_file_location('_v17_new_hierarchy_author',SOURCE);new=importlib.util.module_from_spec(spec);sys.modules[spec.name]=new;spec.loader.exec_module(new)
    wrapper=ROOT/'review/revisions/v17_task_solver_b375_before_degenerate_face.py'
    assert sha(wrapper)=='b37536496537a5c6ee6688cc40663d133317fbc3bdc3c376ab56072c2af05e24'
    spec=importlib.util.spec_from_file_location('_v17_b375_wrapper_author',wrapper);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    # Proposed wiring only, outside production; root owns the actual import.
    m.Level=new.Level;m.solve_hierarchy=new.solve_hierarchy
    ck('frozen_generic5564_unchanged',sha(ROOT/'v16_release_geometry/hierarchy.py')=='5564c7165279e46b11642719461737418bf31be23a66acf3c9f672982b4401bf')
    # Primary prefers x=+1 although canonical OPEN requests x=-1. Motion then
    # chooses z=.25; tangent/posture cannot trade either higher-level result.
    a=tasks([[-1,-1,0]],[8],[-1,1,-.7],tangent=[[0,0,1]])
    began=time.perf_counter_ns();r=m.solve_contact_unloading(a,reference_step_scale21=np.ones(21),motion_task=motion([[1,0,0],[0,0,1]],[-1,.25],3))
    CASE_TIMINGS.append(dict(name='original_failed_3D',whole_ms=(time.perf_counter_ns()-began)/1e6,face_ms=r.diagnostics['degenerate_face_elapsed_ms'],face_events=r.diagnostics['degenerate_face_events'],rational_events=r.diagnostics['exact_rational_events'],repairs=[q for level in r.levels for q in level['degenerate_face_repairs']]))
    ck('original_failure_fixed_by_sparse_face_without_fraction',r.available and r.diagnostics['degenerate_face_events']>=1 and r.diagnostics['exact_rational_events']==0)
    ck('normal_retreat_can_oppose_canonical_OPEN',r.available and np.allclose(r.x[:3],[1,1,.25],atol=2e-12,rtol=0),r.x[:3].tolist())
    ck('lower_motion_tangent_pose_do_not_trade_primary',abs(float(-a.normal_matrix_m_per_reference[0]@r.x)-2)<2e-12 and [x['name'] for x in r.levels]==['pressure_weighted_normal_withdrawal','observed_external_contact_motion_correction','low_unnecessary_tangent_increment','canonical_clearance_posture'])
    ck('motion_precedes_tangent_and_softpose',abs(r.x[2]-.25)<2e-12 and a.open_target_increment21[2]==-.7)
    ck('virtual_work_and_safety_roles_honest',abs(r.diagnostics['normal_virtual_unloading_work_J']-16)<2e-12 and not r.diagnostics['physical_safety_certified'] and not r.diagnostics['object_motion_predicted'] and not r.diagnostics['force_prediction_available'] and not r.diagnostics['placement_certified'])
    # Nontrivial pressure objective: feasible triangle vertices (0,0),(0,1),(1,1).
    # Weighted withdrawal max is (0,1), not the canonical feasible pose (1,1).
    C=[[-1,0],[1,-1]];pressure=[1.,3.];oracle,outward=vertex_oracle(C,pressure,[-1,-1],[1,1])
    b=tasks(C,pressure,[1,1],tangent=[[0,1]])
    rb=m.solve_contact_unloading(b,reference_step_scale21=np.ones(21))
    ck('independent2D_vertex_maximum',rb.available and np.allclose(rb.x[:2],oracle,atol=2e-12,rtol=0) and np.allclose(oracle,[0,1],atol=1e-14,rtol=0),dict(actual=rb.x[:2].tolist(),oracle=oracle.tolist()))
    ck('box_upper_not_falsely_assumed_attainable',abs(rb.diagnostics['normal_progress_box_upper_m']-1.25)<1e-14 and abs(rb.diagnostics['normal_progress_achieved_m']-.75)<2e-12 and abs(float(outward@rb.x[:2])-.75)<2e-12)
    # Same-point splitting conserves pressure and duplicate hard rows; other
    # task blocks are held fixed. This tests NORMAL primary representation.
    split=tasks([C[0],C[0],C[1]],[.25,.75,3.],[1,1],tangent=[[0,1]])
    perm=tasks([C[1],C[0],C[0]],[3.,.75,.25],[1,1],tangent=[[0,1]])
    rs=m.solve_contact_unloading(split,reference_step_scale21=np.ones(21));rp=m.solve_contact_unloading(perm,reference_step_scale21=np.ones(21))
    effective=lambda t,result:-np.asarray(result.diagnostics['observed_pressure_weights'])@t.normal_matrix_m_per_reference
    ck('same_point_pressure_split_preserves_normal_objective_and_solution',rs.available and np.allclose(effective(b,rb),effective(split,rs),atol=1e-15,rtol=0) and np.allclose(rb.x,rs.x,atol=2e-12,rtol=0) and abs(rb.diagnostics['normal_virtual_unloading_work_J']-rs.diagnostics['normal_virtual_unloading_work_J'])<2e-12)
    ck('patch_row_permutation_preserves_normal_objective_and_solution',rp.available and np.allclose(effective(b,rb),effective(perm,rp),atol=1e-15,rtol=0) and np.allclose(rb.x,rp.x,atol=2e-12,rtol=0))
    # Correct reference units: normalized y=.3 means actual increment .15.
    scale=np.ones(21);scale[:2]=[.2,.5]
    u=tasks([[-1,0]],[4],[-.2,-.5],lower=[-.2,-.5],upper=[.2,.5],tangent=[[0,1]])
    ru=m.solve_contact_unloading(u,reference_step_scale21=scale,motion_task=motion([[0,1]],[.15],2))
    ck('motion_target_and_progress_preserve_declared_reference_units',ru.available and np.allclose((ru.x*scale)[:2],[.2,.15],atol=2e-12,rtol=0) and abs(ru.diagnostics['normal_progress_achieved_m']-.2)<2e-12)
    zero=tasks([[-1,0]],[1],[1,1],lower=[0,0],upper=[0,0],tangent=[[0,1]])
    rz=m.solve_contact_unloading(zero,reference_step_scale21=np.ones(21))
    ck('genuine_zero_only_domain_is_solved_not_called_physical_infeasible',rz.available and np.array_equal(rz.x,np.zeros(21)) and rz.diagnostics['normal_progress_box_upper_m']==0 and rz.diagnostics['normal_progress_achieved_m']==0 and not rz.diagnostics['physical_safety_certified'])
    ck('unknown_motion_omits_layer_not_measured_zero',not rb.diagnostics['observed_motion_task']['available'] and all(x['name']!='observed_external_contact_motion_correction' for x in rb.levels) and 'unavailable_not_zero_motion' in rb.diagnostics['observed_motion_task']['reason'])
    knownzero=m.solve_contact_unloading(b,reference_step_scale21=np.ones(21),motion_task=motion([[1,0]],[0.],2))
    ck('explicit_observed_zero_motion_is_distinct_from_unknown',knownzero.available and knownzero.diagnostics['observed_motion_task']['available'] and any(x['name']=='observed_external_contact_motion_correction' for x in knownzero.levels))
    def rejected(t,mt,exception=ValueError):
        try:m.solve_contact_unloading(t,reference_step_scale21=np.ones(21),motion_task=mt)
        except exception:return True
        return False
    ck('motion_clock_mismatch_and_unknown_geometry_clock_rejected',rejected(b,motion([[1,0]],[0],2,2.005)) and rejected(tasks(C,pressure,[1,1],time_s=None),motion([[1,0]],[0],2)))
    huge=tasks(C,[1e308,1e308],[1,1])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');rh=m.solve_contact_unloading(huge,reference_step_scale21=np.ones(21))
    ck('extreme_finite_pressures_normalize_without_false_finite_work',rh.available and np.array_equal(rh.diagnostics['observed_pressure_weights'],[.5,.5]) and np.isfinite(rh.x).all() and rh.diagnostics['normal_virtual_unloading_work_J'] is None,dict(weights=rh.diagnostics['observed_pressure_weights'],work=rh.diagnostics['normal_virtual_unloading_work_J'],warnings=[str(x.message) for x in ws]))
    ck('underflowed_positive_pressure_row_rejected_not_dropped',rejected(tasks(C,[1e-308,1e308],[1,1]),None,FloatingPointError))
    unsupported=tasks(C,pressure,[1,1]);unsupported.normal_matrix_m_per_reference=None
    ck('unknown_geometry_returns_none_not_zero_prediction',m.solve_contact_unloading(unsupported,reference_step_scale21=np.ones(21)) is None)
    extra_checks(new)
    ck('source_stable_and_no_model_framework_import',before==sha(SOURCE) and 'mujoco' not in sys.modules)
    report=dict(schema='v17-hierarchy-author-v1',role='author_numeric_revision_gate_not_independent',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(SOURCE),source_sha256=before,source_after_sha256=sha(SOURCE),generic_solver_sha256=sha(ROOT/'v16_release_geometry/hierarchy.py'),test_source_sha256=sha(Path(__file__)),wrapper_b375_sha256=sha(wrapper),proposed_wrapper_import_binding_only=True,case_timings=CASE_TIMINGS,proof_budgets=dict(sparse_visits=new.SPARSE_VISIT_BUDGET,rational_operations=new.EXACT_OPERATION_BUDGET,integer_bits=new.EXACT_INTEGER_BIT_BUDGET),passed=all(x['passed'] for x in CHECKS),counts=dict(total=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=sum(not x['passed'] for x in CHECKS)),checks=CHECKS,scope=dict(embedded_axes=[2,3],total_reference_axes=21,independent_2D_vertex_enumeration=True,model_constructions=0,physics_steps=0,FK_calls=0,solver_replays_from_physical_data=0,scipy=False),limitations=['Normal-primary split invariance is tested with conserved pressure and unchanged lower task blocks; no claim about arbitrary upstream tangent discretization.','Declared old-plane virtual displacement/work is not future servo response, actual contact work, pressure decrease or placement proof.','Finite analytical cases do not certify all numerical domains or runtime integration.','Returned x remains normalized and caller must apply reference_step_scale21 exactly once.','Timings are single pure calls on this machine, not realtime guarantees; reference oracle cost is included where named.'])
    out=ROOT/'review/v17_hierarchy_author_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],counts=report['counts'],source_sha256=before,path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
