"""Fixed synthetic and saved-public-stream tests, no simulator import/physics."""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v11.object_belief import DeclaredShapeTemplate,RelativeObjectBelief

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def patch(p,n,digit,index=0,sample=1.):
    return dict(digit=int(digit),robot_body=int(digit+1),link_point=[.001*index,0.,.02],link_normal=list(n),
        reference_position=list(p),reference_normal=list(n),normal=1.,sample_time=sample,age=.01)

def frame(patches,now=1.01,sigma=.0001):return dict(time=now,sample_time=1.,skin_patches=patches,sensor_characteristics=dict(position_sigma_m=sigma))

def sphere(center,r=.028,digits=(0,1,2),sample=1.):
    directions=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0],[0,-1.,0]])
    return [patch(center+r*directions[i],-directions[i],d,i,sample) for i,d in enumerate(digits)]

def capsule(center,with_caps=False,sample=1.):
    r=.028;axis=np.array([0.,1.,0]);rows=[]
    for i,s in enumerate([-.03,-.015,0.,.015,.03]):
        angle=i*1.3;out=np.array([np.cos(angle),0.,np.sin(angle)])
        rows.append(patch(center+axis*s+r*out,-out,i,i,sample))
    if with_caps:
        for side,d in [(-1,0),(1,4)]:
            out=np.array([1.,side,0.])/np.sqrt(2);rows.append(patch(center+axis*.047*side+r*out,-out,d,len(rows),sample))
    return rows

def synthetic():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    c=np.array([.05,.01,.45]);st=DeclaredShapeTemplate('sphere',.028,source='synthetic exact explicit radius')
    ct=DeclaredShapeTemplate('capsule',.028,.047,source='synthetic exact explicit nominal radius/halfsegment')
    obj=RelativeObjectBelief(st);s=obj.update(frame(sphere(c)))
    check('sphere exact inward-normal center',np.allclose(s.center_reference,c,atol=1e-12))
    check('sphere only center no invented orientation',s.axis_reference is None and not s.diagnostics['sphere_orientation_observable'])
    check('sphere covariance shared3fingers',np.allclose(s.hypotheses[0].conditional_position_noise_covariance3,np.eye(3)*1e-8/3))
    repeated=sphere(c);repeated += [deepcopy(repeated[0]) for _ in range(100)]
    dup=RelativeObjectBelief(st).update(frame(repeated))
    check('100same-finger patches no extra precision',np.array_equal(dup.hypotheses[0].conditional_position_noise_covariance3,s.hypotheses[0].conditional_position_noise_covariance3))
    check('same-finger duplication no biased center',np.allclose(dup.center_reference,c,atol=1e-12))
    r=obj.update(frame(sphere(c),now=1.02));check('duplicate sample no information accumulation',r.diagnostics['unique_packet_updates']==1 and r.diagnostics['duplicate_packet_no_new_information'])
    check('acquisition time not current propagation',r.observation_time_s==1. and r.current_time_s==1.02 and not r.current_state_prediction_available)
    old=sphere(c,sample=.99);o=obj.update(frame(old,now=1.025));check('old packet rejected',o.status=='OUT_OF_ORDER_ACQUISITION')
    r=obj.update(frame(sphere(c),now=1.03));check('old packet cannot poison last valid packet',np.allclose(r.center_reference,c) and r.observation_time_s==1.)
    stale=obj.update(frame(sphere(c),now=1.1));check('stale not current geometry',stale.status=='STALE_ACQUISITION' and stale.center_reference is None)
    missing=RelativeObjectBelief(st).update(dict(time=1.));check('missing observations unavailable',missing.status=='OBSERVATION_UNAVAILABLE' and missing.center_reference is None)
    no_spec=frame(sphere(c));del no_spec['sensor_characteristics'];u=RelativeObjectBelief(st).update(no_spec)
    check('unknown covariance never zero-filled',u.hypotheses[0].conditional_position_noise_covariance3 is None)
    extra=frame(sphere(c));extra['position']=[100,200,300];extra['object_pose']=[0]*7
    check('frame truth fields ignored',np.array_equal(RelativeObjectBelief(st).update(extra).center_reference,s.center_reference))
    forbidden=sphere(c);forbidden[0]['is_target']=True
    forbidden_result=RelativeObjectBelief(st).update(frame(forbidden))
    check('patch identity field rejected not consumed',len(forbidden_result.rejected_patches)==1)
    invalid=sphere(c);invalid[0]['reference_normal']=[0,0,0]
    check('undefined normals rejected',len(RelativeObjectBelief(st).update(frame(invalid)).rejected_patches)==1)
    mixedtime=sphere(c);mixedtime[0]['sample_time']=1.005
    mt=RelativeObjectBelief(st).update(frame(mixedtime))
    check('different times not merged into rigid fit',len(mt.rejected_patches)==2 and not mt.grouping_unambiguous)
    outlier=sphere(c,digits=(0,1,2,3))+sphere(c+np.array([.1,0,0]),digits=(4,))
    out=RelativeObjectBelief(st).update(frame(outlier))
    check('sphere outlier does not dominate pressure-free grouping',np.allclose(out.center_reference,c) and out.unassigned_patch_indices==(4,))
    outlier[-1]['normal']=1000000.
    check('huge outlier pressure cannot create geometric votes',np.allclose(RelativeObjectBelief(st).update(frame(outlier)).center_reference,c))
    multiple=sphere(c,digits=(0,1,2))+sphere(c+np.array([.1,0,0]),digits=(0,3,4))
    mm=RelativeObjectBelief(st).update(frame(multiple))
    check('two equal bodies retain ambiguous hypotheses',mm.status=='AMBIGUOUS_GROUPING' and len(mm.hypotheses)>=2 and mm.center_reference is None)
    one=RelativeObjectBelief(st).update(frame(sphere(c,digits=(0,))))
    check('one finger geometry not object ownership',not one.grouping_unambiguous and one.center_reference is None)
    cc=RelativeObjectBelief(ct).update(frame(capsule(c)))
    check('capsule exact axis projector',np.allclose(np.outer(cc.axis_reference,cc.axis_reference),np.diag([0,1,0]),atol=1e-12))
    check('cylinder-only axial center not invented',cc.center_reference is None and cc.axial_center_interval_m[0]<c[1]<cc.axial_center_interval_m[1])
    check('capsule roll gauge unobservable',not cc.hypotheses[0].rotation_about_axis_observable)
    ce=RelativeObjectBelief(ct).update(frame(capsule(c,True)))
    check('cap evidence identifies center conditionally',np.allclose(ce.center_reference,c,atol=1e-12) and ce.hypotheses[0].axial_center_identified)
    endpoint=c+np.array([0,.047,0]);endrows=sphere(endpoint)
    ed=RelativeObjectBelief(ct).update(frame(endrows))
    check('single coincident cap locus axis degeneracy explicit',ed.axis_reference is None and ed.center_reference is None)
    co=capsule(c,True)+[patch(np.array([.5,.5,.5]),np.array([1.,0,0]),0,9)]
    cop=RelativeObjectBelief(ct).update(frame(co))
    check('capsule off-line body remains unassigned',cop.axis_reference is not None and len(co)-1 in cop.unassigned_patch_indices)
    check('prototype never force or mechanical permission',not cop.force_prediction_available and not cop.mechanical_closing_permission_modified and not cop.global_model_admitted)
    check('JSON no NaN unknown represented null',isinstance(json.dumps(ed.to_dict(),allow_nan=False),str))
    check('no MuJoCo import or physics',not any(x=='mujoco' or x.startswith('mujoco.') for x in sys.modules))
    return dict(passed=True,checks=checks)

def qmatrix(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def stats(x):
    x=np.asarray(x,float)
    return None if not len(x) else dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))

def real_streams():
    # Fixed beforehand: no selection by reconstruction error or pressure result.
    jobs=[('hold','nominal','ratio_preview'),('hold','collision','ratio_preview'),
          ('hold','collision','joint'),('transport','collision','joint')]
    template_source='frozen foundation/hand_plant.py _model nominal capsule size1: radius.028m halfsegment.047m; XML local axis[0,-1,0] from quat[1,1,0,0]'
    template=DeclaredShapeTemplate('capsule',.028,.047,template_source)
    streams=[]
    for mode,scene,method in jobs:
        cohort=ROOT/f'output/development_v10_{mode}_v1'
        if not read(cohort/'execution_complete.json')['execution_complete']:raise ValueError('Full cohort required')
        base=cohort/'runs'/f'{scene}__{method}__610212';path=base/'trace.npz';meta=read(base/'metadata.json');before=sha(path)
        if before!=meta['trace_sha256'] or not meta['completed']:raise ValueError('Complete/hash-bound trace required')
        with np.load(path,allow_pickle=False) as z:
            t=z['time'];public=z['observation_json'];positions=z['position'];true_orientation=z['qpos'][:,28:32];raw_patches=z['raw_patches_json']
        ids=np.flatnonzero((t>=7.3-1e-9)&(t<12.-1e-9))[::10] # Fixed20Hz audit subset, not wholestream coverage claim.
        belief=RelativeObjectBelief(template);rows=[]
        for i in ids:
            f=json.loads(str(public[i]));result=belief.update(f);selected=None if result.selected_index is None else result.hypotheses[result.selected_index]
            row=dict(frame_time_s=float(t[i]),observation_time_s=result.observation_time_s,status=result.status,
                grouping_unambiguous=result.grouping_unambiguous,patch_count=len(f.get('skin_patches',[])),
                independent_fingers=[] if selected is None else list(selected.independent_fingers),
                inlier_indices=[] if selected is None else list(selected.inlier_indices),
                unassigned_indices=list(result.unassigned_patch_indices),center_reference=None if result.center_reference is None else result.center_reference.tolist(),
                axis_reference=None if result.axis_reference is None else result.axis_reference.tolist(),
                axial_center_interval_m=result.axial_center_interval_m,evaluation_only=None)
            if result.observation_time_s is not None:
                k=int(np.argmin(abs(t-result.observation_time_s)))
                if abs(t[k]-result.observation_time_s)>1e-9:raise AssertionError('Acquisition truth unavailable; no interpolation/now substitution')
                # Everything below is evaluation-only. It never returns to update.
                center_truth=positions[k];axis_truth=qmatrix(true_orientation[k])@np.array([0.,-1.,0.])
                ev=dict(truth_time_s=float(t[k]),current_time_not_used_for_truth=True,
                    center_truth_reference=center_truth.tolist(),axis_truth_reference=axis_truth.tolist(),
                    center_error_m=None,center_line_distance_m=None,axis_unsigned_angle_deg=None,axial_truth_inside_interval=None)
                if result.center_reference is not None:ev['center_error_m']=float(np.linalg.norm(result.center_reference-center_truth))
                if result.axis_reference is not None:
                    axis=result.axis_reference;origin=result.center_line_origin_reference
                    ev['center_line_distance_m']=float(np.linalg.norm(np.cross(center_truth-origin,axis)))
                    ev['axis_unsigned_angle_deg']=float(np.rad2deg(np.arccos(np.clip(abs(axis@axis_truth),0.,1.))))
                    value=float(axis@center_truth);interval=result.axial_center_interval_m
                    ev['axial_truth_inside_interval']=bool(interval[0]-1e-9<=value<=interval[1]+1e-9)
                labels=[]
                if selected is not None:
                    truth_contacts=json.loads(str(raw_patches[k]))
                    for patch_id in selected.inlier_indices:
                        p=f['skin_patches'][patch_id];candidates=[x for x in truth_contacts if x['digit']==p['digit'] and 'world_position' in x]
                        if not candidates:labels.append(None);continue
                        distances=np.array([np.linalg.norm(np.asarray(x['world_position'])-p['reference_position']) for x in candidates]);order=np.argsort(distances)
                        if distances[order[0]]>.003:labels.append(None);continue
                        if len(order)>1 and abs(distances[order[0]]-distances[order[1]])<1e-7 and candidates[order[0]]['is_target']!=candidates[order[1]]['is_target']:labels.append(None)
                        else:labels.append(bool(candidates[order[0]]['is_target']))
                ev['inlier_target_labels_from_acquisition_truth_nearest_match']=labels;row['evaluation_only']=ev
            rows.append(row)
        after=sha(path)
        if before!=after:raise AssertionError('Input changed')
        statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
        available=[r['evaluation_only'] for r in rows if r['grouping_unambiguous'] and r['evaluation_only'] is not None]
        axis_errors=[r['axis_unsigned_angle_deg'] for r in available if r['axis_unsigned_angle_deg'] is not None]
        center_errors=[r['center_error_m'] for r in available if r['center_error_m'] is not None]
        line_errors=[r['center_line_distance_m'] for r in available if r['center_line_distance_m'] is not None]
        coverage=[r['axial_truth_inside_interval'] for r in available if r['axial_truth_inside_interval'] is not None]
        labels=[v for r in available for v in r['inlier_target_labels_from_acquisition_truth_nearest_match'] if v is not None]
        streams.append(dict(mode=mode,scene=scene,method=method,path=str(path),trace_sha256=before,input_unchanged=True,
            fixed_evaluation_sampling='every10th200Hz frame over7.3<=t<12 (20Hz); not full200Hz validation',
            denominator=len(rows),statuses=statuses,unambiguous_frames=len(available),center_error_m=stats(center_errors),
            axis_angle_deg=stats(axis_errors),center_line_distance_m=stats(line_errors),
            axial_interval_truth_coverage=dict(covered=sum(coverage),n=len(coverage)),
            geometric_inlier_target_purity_evaluation_only=dict(target=sum(labels),matched=len(labels),unknown_or_ambiguous_not_zero_filled=True),records=rows))
    return dict(template_source=template_source,sphere_real_data_available=False,calibration_or_policy_admission=False,streams=streams)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--include-real',action='store_true');args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh evidence required')
    result=dict(schema='v11_relative_object_geometry_author_v1',synthetic=synthetic(),new_physics=False,
        source_sha256={'module':sha(ROOT/'v11/object_belief.py'),'verifier':sha(__file__)},
        priors=dict(fixed_geometric_tolerance_m=.003,minimum_axis_span_m=.006,cap_normal_projection_threshold=.15,
            purpose='engineering prior set before truth evaluation; not a confidence or force-safety bound'),
        real=None if not args.include_real else real_streams())
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(out=str(out),sha256=sha(out),synthetic_checks=len(result['synthetic']['checks']),
        real=None if result['real'] is None else [{k:v for k,v in s.items() if k!='records'} for s in result['real']['streams']]),ensure_ascii=False))

if __name__=='__main__':main()
