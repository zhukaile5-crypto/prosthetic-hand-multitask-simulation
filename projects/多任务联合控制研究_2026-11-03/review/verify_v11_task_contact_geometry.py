"""Analytic/oracle SDF tests and fixed saved-public-flow coverage, no physics."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v11.object_belief import DeclaredShapeTemplate,RelativeObjectBelief
from v11.task_contact_geometry import query_hypothesis_gap,query_signed_gap
EXPECTED_BELIEF='b35553f4fa1496b09318b55ce443f0acf494cf6eeb1846829668af86f6381712'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def hypothesis(axis=(0.,0.,1.),origin=(0.,0.,0.),interval=(-.02,.03)):
    return dict(axis_reference=list(axis),center_line_origin_reference=list(origin),axial_center_interval_m=list(interval))

def capsule_oracle(point,axis,origin,lo,hi,L,r):
    # Independent explicit3D closest-segment oracle, includes interior minimizer
    # and a dense interior sweep, not the author's closed-form bounds.
    s=float(axis@(point-origin));coordinates=np.unique(np.r_[lo,hi,np.clip(s,lo,hi),np.linspace(lo,hi,2001)])
    centers=origin+coordinates[:,None]*axis
    offsets=point-centers;along=offsets@axis
    nearest=centers+np.clip(along,-L,L)[:,None]*axis
    distance=np.linalg.norm(point-nearest,axis=1)-r
    return float(distance.min()),float(distance.max())

def synthetic():
    checks=[]
    def check(name,flag):
        if not flag:raise AssertionError(name)
        checks.append(name)
    template=DeclaredShapeTemplate('capsule',.028,.047,'synthetic explicit dimensions')
    h=hypothesis();p=np.array([.04,0.,0.]);q=query_hypothesis_gap(template,h,p,observation_time_s=1.,sample_time_s=1.)
    check('cylinder exact signedgap despite unknown center',abs(q.gap_lower_m-.012)<1e-12 and q.gap_lower_m==q.gap_upper_m and q.all_centers_cylinder)
    check('cylinder unique outward normal and contact inward sign',np.allclose(q.outward_normal_reference,[1,0,0]) and np.allclose(q.inward_contact_normal_reference,[-1,0,0]))
    cap=query_hypothesis_gap(template,h,[.04,0,.1],observation_time_s=1.,sample_time_s=1.)
    check('endcap interval not zero-filled',cap.gap_upper_m>cap.gap_lower_m and cap.outward_normal_reference is None and not cap.all_centers_cylinder)
    check('cap extrema endpoints',cap.center_coordinate_for_min_gap_m==.03 and cap.center_coordinate_for_max_gap_m==-.02)
    wide=DeclaredShapeTemplate('capsule',.028,.01,'synthetic wide center interval')
    wi=query_hypothesis_gap(wide,hypothesis(interval=(-.1,.1)),[.04,0,0],observation_time_s=1.,sample_time_s=1.)
    check('interior center minimum not endpoint-only oracle',abs(wi.gap_lower_m-.012)<1e-12 and wi.center_coordinate_for_min_gap_m==0. and wi.gap_upper_m>wi.gap_lower_m)
    ax=query_hypothesis_gap(template,h,[0.,0.,0.],observation_time_s=1.,sample_time_s=1.)
    check('axis-line normal undefined but signedgap exists',ax.available and ax.outward_normal_reference is None and ax.gap_lower_m<0)
    known=query_hypothesis_gap(template,hypothesis(interval=(0.,0.)),[.04,0,.1],observation_time_s=1.,sample_time_s=1.)
    check('known-center cap has exact gap and normal',known.gap_lower_m==known.gap_upper_m and known.normal_available)
    sphere=DeclaredShapeTemplate('sphere',.028,source='synthetic explicit radius')
    sh=dict(conditional_center_reference=[.1,.2,.3]);ss=query_hypothesis_gap(sphere,sh,[.14,.2,.3],observation_time_s=1.,sample_time_s=1.)
    check('sphere standard signed distance',abs(ss.gap_lower_m-.012)<1e-12 and ss.gap_lower_m==ss.gap_upper_m)
    sc=query_hypothesis_gap(sphere,sh,[.1,.2,.3],observation_time_s=1.,sample_time_s=1.)
    check('sphere center normal undefined',sc.gap_lower_m==-.028 and sc.outward_normal_reference is None)
    late=query_hypothesis_gap(template,h,p,observation_time_s=1.,sample_time_s=1.01)
    check('different acquisition time unavailable no zero-motion propagation',not late.available and late.gap_lower_m is None)
    flip=query_hypothesis_gap(template,hypothesis(axis=(0.,0.,-1.),interval=(-.03,.02)),p,observation_time_s=1.,sample_time_s=1.)
    check('axis-sign gauge leaves signedgap normal invariant',flip.gap_lower_m==q.gap_lower_m and flip.gap_upper_m==q.gap_upper_m and np.allclose(flip.outward_normal_reference,q.outward_normal_reference))
    shifted=query_hypothesis_gap(template,hypothesis(origin=(0.,0.,2.),interval=(-2.02,-1.97)),[.04,0,.1],observation_time_s=1.,sample_time_s=1.)
    check('line-origin gauge interval transformation',abs(shifted.gap_lower_m-cap.gap_lower_m)<1e-12 and abs(shifted.gap_upper_m-cap.gap_upper_m)<1e-12)
    bad=query_hypothesis_gap(template,hypothesis(axis=(0.,0.,0.)),p,observation_time_s=1.,sample_time_s=1.)
    check('undefined axis not fabricated',not bad.available)
    inverted=query_hypothesis_gap(template,hypothesis(interval=(.1,-.1)),p,observation_time_s=1.,sample_time_s=1.)
    check('inconsistent center interval unavailable not sorted',not inverted.available)
    unknown=query_hypothesis_gap(template,dict(axis_reference=None),p,observation_time_s=1.,sample_time_s=1.)
    check('unknown geometry not zero-filled',not unknown.available and unknown.gap_upper_m is None)
    snap=dict(grouping_unambiguous=False,observation_time_s=1.,hypotheses=[h],selected_index=0,template=template)
    check('ambiguous grouping not silently selected',not query_signed_gap(snap,p,sample_time_s=1.).available)
    snap['grouping_unambiguous']=True;snap['current_time_s']=2.
    check('historical snapshot queried at acquisition only',query_signed_gap(snap,p,sample_time_s=1.).available and not query_signed_gap(snap,p,sample_time_s=2.).available)
    rng=np.random.default_rng(510205);max_error=0.;normal_error=0.
    for case in range(150):
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis);origin=rng.normal(size=3)*.1;lo=float(rng.uniform(-.2,.2));hi=lo+float(rng.uniform(0,.3));L=float(rng.uniform(.005,.1));r=float(rng.uniform(.005,.05));point=rng.normal(size=3)*.2
        t=DeclaredShapeTemplate('capsule',r,L,'synthetic oracle dimensions')
        result=query_hypothesis_gap(t,hypothesis(axis,origin,(lo,hi)),point,observation_time_s=3.,sample_time_s=3.)
        lower,upper=capsule_oracle(point,axis,origin,lo,hi,L,r)
        max_error=max(max_error,abs(result.gap_lower_m-lower),abs(result.gap_upper_m-upper))
        angle=.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        trans=np.array([.1,-.2,.3]);rot=query_hypothesis_gap(t,hypothesis(R@axis,R@origin+trans,(lo,hi)),R@point+trans,observation_time_s=3.,sample_time_s=3.)
        normal_error=max(normal_error,abs(rot.gap_lower_m-result.gap_lower_m),abs(rot.gap_upper_m-result.gap_upper_m))
    check('150capsules analytic extrema equal endpoint-and-interior3D oracle',max_error<1e-12)
    check('150rigid coordinate transforms preserve gap interval',normal_error<1e-12)
    check('no forcegradient stiffness safety or permission claim',not q.contact_force_gradient_available and not q.pressure_prediction_available and not q.physical_gap_certified and not q.mechanical_closing_permission_modified)
    check('unknown JSON null not NaN',isinstance(json.dumps(unknown.to_dict(),allow_nan=False),str))
    check('no simulator import',not any(k=='mujoco' or k.startswith('mujoco.') for k in sys.modules))
    return dict(passed=True,checks=checks,oracle_cases=150,max_abs_bound_error_m=max_error,max_rigid_transform_error_m=normal_error)

def qmatrix(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def stats(values):
    values=np.asarray(values,float)
    return None if not len(values) else dict(n=len(values),mean=float(values.mean()),p95=float(np.quantile(values,.95)),max=float(values.max()))

def saved_streams():
    basis=ROOT/'review/v11_object_belief_author_real_v1.json';old=read(basis)
    template=DeclaredShapeTemplate('capsule',.028,.047,old['real']['template_source']);streams=[]
    for prior in old['real']['streams']:
        path=Path(prior['path']);before=sha(path)
        if before!=prior['trace_sha256']:raise ValueError('Retained real-stream identity changed')
        with np.load(path,allow_pickle=False) as z:
            times=z['time'];public=z['observation_json'];truth_position=z['position'];truth_quaternion=z['qpos'][:,28:32]
        belief=RelativeObjectBelief(template);rows=[];matching_status=True
        for previous in prior['records']:
            i=int(np.argmin(abs(times-previous['frame_time_s'])));frame=json.loads(str(public[i]));snap=belief.update(frame)
            if snap.status!=previous['status']:raise AssertionError('Frozen belief reevaluation status changed')
            inliers=set(snap.hypotheses[snap.selected_index].inlier_indices) if snap.selected_index is not None else set()
            for patch_index,patch in enumerate(frame.get('skin_patches',[])):
                result=query_signed_gap(snap,patch['reference_position'],sample_time_s=patch['sample_time'])
                row=dict(frame_time_s=float(times[i]),sample_time_s=float(patch['sample_time']),patch_index=patch_index,
                    digit=patch['digit'],is_geometric_group_inlier=patch_index in inliers,
                    geometry_status=snap.status,query=result.to_dict(),evaluation_only=None)
                if result.available:
                    k=int(np.argmin(abs(times-patch['sample_time'])))
                    if abs(times[k]-patch['sample_time'])>1e-9:raise AssertionError('Acquisition-aligned truth required')
                    center=truth_position[k];axis=qmatrix(truth_quaternion[k])@np.array([0.,-1.,0.]);p=np.asarray(patch['reference_position'])
                    nearest=center+axis*np.clip(axis@(p-center),-.047,.047);signed=float(np.linalg.norm(p-nearest)-.028)
                    miss=max(result.gap_lower_m-signed,signed-result.gap_upper_m,0.)
                    angle=None
                    if result.normal_available and patch_index in inliers:
                        inward=np.asarray(patch['reference_normal']);inward/=np.linalg.norm(inward)
                        angle=float(np.rad2deg(np.arccos(np.clip(inward@result.inward_contact_normal_reference,-1.,1.))))
                    row['evaluation_only']=dict(truth_time_s=float(times[k]),declared_template_true_geometry_signed_gap_m=signed,
                        distance_outside_conditional_gap_interval_m=float(miss),
                        strict_contains_true_geometry_gap=bool(result.gap_lower_m-1e-12<=signed<=result.gap_upper_m+1e-12),
                        inward_normal_angle_to_public_contact_normal_deg=angle,
                        interpretation='Only axial-center ambiguity is enclosed. Estimated-axis/line and template/measurement errors are not part of this conditional interval; strict containment is a diagnostic, not a calibrated coverage test.')
                rows.append(row)
        if sha(path)!=before:raise AssertionError('Input modified')
        summary={}
        for name,select in [('all_public_patch_rows',rows),('selected_geometric_inlier_rows',[r for r in rows if r['is_geometric_group_inlier']])]:
            available=[r for r in select if r['query']['status']=='AVAILABLE_CONDITIONAL_GEOMETRIC_INTERVAL']
            cylinder=[r for r in available if r['query']['all_centers_cylinder']]
            summary[name]=dict(denominator=len(select),gap_interval_available=len(available),
                all_centers_cylinder_exact_conditional_gap_rows=len(cylinder),unique_geometric_normal_rows=sum(r['query']['outward_normal_reference'] is not None for r in available),
                gap_interval_width_m=stats([r['query']['gap_upper_m']-r['query']['gap_lower_m'] for r in available]),
                conditional_interval_miss_to_truth_m=stats([r['evaluation_only']['distance_outside_conditional_gap_interval_m'] for r in available]),
                strict_truth_containment_count=sum(r['evaluation_only']['strict_contains_true_geometry_gap'] for r in available),
                contact_normal_angle_error_deg=stats([r['evaluation_only']['inward_normal_angle_to_public_contact_normal_deg'] for r in available if r['evaluation_only']['inward_normal_angle_to_public_contact_normal_deg'] is not None]))
        streams.append(dict(mode=prior['mode'],scene=prior['scene'],method=prior['method'],trace_path=str(path),trace_sha256=before,
            frame_denominator=prior['denominator'],input_unchanged=True,summary=summary,records=rows))
    return dict(prior_belief_evidence_path=str(basis),prior_belief_evidence_sha256=sha(basis),streams=streams,
        scope='same four predeclared20Hz capsule streams; no sphere real validation, force response or global admission')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--include-real',action='store_true');args=parser.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh evidence required')
    if sha(ROOT/'v11/object_belief.py')!=EXPECTED_BELIEF:raise RuntimeError('Frozen b355 belief changed')
    result=dict(schema='v11_task_relevant_partial_geometry_author_v1',new_physics=False,synthetic=synthetic(),
        real=saved_streams() if args.include_real else None,
        source_sha256=dict(module=sha(ROOT/'v11/task_contact_geometry.py'),frozen_belief=sha(ROOT/'v11/object_belief.py'),verifier=sha(__file__)))
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(out=str(out),sha256=sha(out),synthetic=result['synthetic'],
        real=None if result['real'] is None else [{k:v for k,v in s.items() if k!='records'} for s in result['real']['streams']]),ensure_ascii=False))

if __name__=='__main__':main()
