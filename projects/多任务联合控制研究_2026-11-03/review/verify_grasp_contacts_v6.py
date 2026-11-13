"""Pure and saved-public-stream checks; never constructs a physical plant."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grasp_contacts import PersistentGraspContactObserver


def patch(digit=0,body=25,point=(0.,-.009,.026),normal=(0.,-1.,0.),force=2.,sample=0.):
    return dict(digit=digit,robot_body=body,link_point=list(point),link_normal=list(normal),
        normal=force,sample_time=sample,age=0.,reference_position=[100.,200.,300.],
        reference_normal=[0.,0.,1.])


def frame(t,patches=None,normal=None,valid=None,sample=None):
    sample=t if sample is None else sample
    result=SimpleNamespace(time=t,sample_time=sample,normal=np.asarray([2.]*5 if normal is None else normal),
        valid=np.asarray([True]*5 if valid is None else valid))
    if patches is not None:
        result.skin_patches=deepcopy(patches)
        for p in result.skin_patches:p['sample_time']=sample
    return result


def warmed():
    observer=PersistentGraspContactObserver()
    for k in range(31):
        t=k*.005;observer.update(frame(t,[patch(i,25+i,sample=t) for i in range(5)]),allow_enrollment=True)
    observer.freeze_handoff()
    return observer


def synthetic_checks():
    checks={}
    o=warmed();h=o.update(frame(.16,[patch(i,25+i) for i in range(5)]))
    checks['common_warmup_sustained_regions_are_frozen_support']=bool(h.frozen and h.known_support.all() and np.all(h.anchor_count==1))
    h=o.update(frame(.165,[patch(),patch(body=23,point=(.02,0.,0.),force=3.)],normal=[5.,0.,0.,0.,0.]))
    checks['new_robot_link_pressure_is_not_credited_to_grasp']=bool(h.persistent_normal[0]==2. and h.new_normal[0]==3. and h.status[0]=='MIXED_NEW')
    checks['pressure_is_conserved_for_whole_skin_protection']=bool(np.allclose(h.persistent_normal+h.new_normal,h.all_normal))
    h=o.update(frame(.17,[patch(point=(.012,-.009,.026))],normal=[2.,0.,0.,0.,0.]))
    checks['far_new_patch_on_same_robot_link_is_unassigned']=bool(not h.known_support[0] and h.new_normal[0]==2.)
    h=o.update(frame(.175,[patch(normal=(0.,1.,0.))],normal=[2.,0.,0.,0.,0.]))
    checks['opposite_surface_normal_is_unassigned']=bool(not h.known_support[0])
    p=patch();p['reference_position']=[-1e6,1e6,-9e7];p['reference_normal']=[1.,0.,0.]
    h=o.update(frame(.18,[p],normal=[2.,0.,0.,0.,0.]))
    checks['robot_local_association_is_invariant_to_world_motion']=bool(h.known_support[0])
    h=o.update(frame(.185,[],normal=[0.]*5))
    checks['valid_absence_is_distinct_from_unknown_sensor']=bool(np.all(h.status=='ABSENT') and not h.support_present.any())
    h=o.update(frame(.19,[],normal=[2.]*5,valid=[False]*5))
    checks['dropout_has_no_support_evidence']=bool(np.all(h.status=='UNOBSERVED') and not h.known_support.any() and np.isnan(h.all_normal).all())
    h=o.update(frame(.25,[patch()],normal=[2.,0.,0.,0.,0.],sample=.19))
    checks['stale_delayed_geometry_has_no_support_credit']=bool(h.status[0]=='STALE' and not h.known_support.any())
    h=o.update(frame(.255,[patch()],normal=[2.,0.,0.,0.,0.]))
    checks['original_region_can_recover_after_dropout_without_new_enrollment']=bool(h.known_support[0])
    for k in range(52,100):
        h=o.update(frame(k*.005,[patch(body=23)],normal=[2.,0.,0.,0.,0.]),allow_enrollment=True)
    checks['posthandoff_new_contact_cannot_be_learned_as_original_grasp']=bool(not h.known_support[0] and h.anchor_count[0]==1)
    no=PersistentGraspContactObserver()
    for k in range(30):no.update(frame(k*.001,[patch()],sample=0.),allow_enrollment=True)
    no.freeze_handoff()
    checks['duplicate_acquisition_samples_do_not_invent_enrollment_dwell']=bool(len(no.anchors)==0)
    gap=PersistentGraspContactObserver()
    for k in range(6):gap.update(frame(k*.03,[patch()]),allow_enrollment=True)
    gap.freeze_handoff()
    checks['unobserved_sample_gaps_do_not_invent_continuous_contact']=bool(len(gap.anchors)==0)
    h=o.update(frame(.5,normal=[2.]*5))
    checks['missing_skin_geometry_is_explicitly_unknown']=bool(np.all(h.status=='GEOMETRY_UNAVAILABLE') and not h.known_support.any() and np.all(h.new_normal==2.))
    h=o.update(frame(.505,[patch(force=20.)],normal=[2.,0.,0.,0.,0.]))
    checks['patch_attribution_never_exceeds_measured_finger_pressure']=bool(h.persistent_normal[0]==2. and h.new_normal[0]==0.)
    h=o.update(frame(.51,[patch()],normal=[2.,0.,0.,0.,0.],sample=.6))
    checks['future_acquisition_is_unavailable_not_future_evidence']=bool(not h.fresh.any() and not h.known_support.any())
    h=o.update(frame(.515,[patch()],normal=[2.,0.,0.,0.,0.],sample=.50))
    checks['acquisition_clock_regression_is_not_current_support']=bool(not h.fresh.any())
    a=warmed();b=warmed()
    fa=frame(.16,[patch()],normal=[2.,0.,0.,0.,0.]);fb=deepcopy(fa)
    fb.object_pose=[123.,-555.,999.];fb.scenario='poison';fb.is_target=False
    fb.skin_patches[0].update(external_geom=-987,is_target=False,object_pose=[1.,2.,3.])
    checks['external_identity_and_object_truth_poison_has_no_effect']=bool(a.update(fa).to_dict()==b.update(fb).to_dict())
    try:o.update(frame(.1,[]));backwards=False
    except ValueError:backwards=True
    checks['backwards_observation_time_rejected']=backwards
    return checks


def saved_stream_checks():
    checks={};readouts={};sources={}
    for scene in ['nominal','collision','sphere','sensor_noise','sensor_dropout','dynamic_transport']:
        path=ROOT/'output/development_v5_physical_all16/runs'/f'{scene}__joint__610212'/'trace.npz'
        with np.load(path,allow_pickle=False) as z:
            times=z['time'];observations=z['observation_json']
            observer=PersistentGraspContactObserver();rows=[]
            for k,t in enumerate(times):
                if t<7.3-1e-9 or t>11.7+1e-9:continue
                if t>=7.5-1e-9 and not observer.frozen:observer.freeze_handoff()
                data=json.loads(str(observations[k]))
                h=observer.update(SimpleNamespace(**data),allow_enrollment=t<7.5-1e-9)
                if t>=7.5-1e-9:
                    rows.append(dict(time=float(t),persistent_normal=h.persistent_normal.tolist(),new_normal=h.new_normal.tolist(),
                        known_support=h.known_support.tolist(),status=h.status.tolist()))
                    mask=h.valid
                    checks[f'{scene}_pressure_conservation']=checks.get(f'{scene}_pressure_conservation',True) and bool(np.allclose((h.persistent_normal+h.new_normal)[mask],h.all_normal[mask]))
                    checks[f'{scene}_no_invalid_support']=checks.get(f'{scene}_no_invalid_support',True) and bool(not np.any(h.known_support&~h.fresh))
            readouts[scene]=dict(rows=rows,anchor_count=observer.snapshot.anchor_count.tolist())
            if scene=='nominal':
                checks['saved_nominal_support_remains_associated']=all(sum(r['known_support'])>=3 and r['known_support'][0] for r in rows)
            if scene=='collision':
                before=min(rows,key=lambda r:abs(r['time']-9.43));impact=min(rows,key=lambda r:abs(r['time']-9.6))
                checks['saved_collision_new_thumb_pressure_is_unassigned']=bool(before['new_normal'][0]<.15 and impact['new_normal'][0]>3.)
                checks['saved_collision_missing_target_regions_are_not_filled_by_fixture']=bool(sum(impact['known_support'])<3)
                readouts[scene]['selected']=dict(before=before,after=impact)
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return checks,readouts,sources


def main():
    checks=synthetic_checks();saved,readouts,sources=saved_stream_checks();checks.update(saved)
    result=dict(schema='grasp-contact-v6-interface-checks-v1',passed=all(checks.values()),checks=checks,
        source_sha256=hashlib.sha256((ROOT/'grasp_contacts.py').read_bytes()).hexdigest(),input_sources=sources,
        scope='Pure synthetic and replay of saved public observations; no new physical simulation or object identity certification',
        saved_readouts=readouts)
    out=ROOT/'review/grasp_contacts_v6_validation.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v],output=str(out)),ensure_ascii=False))
    if not result['passed']:raise AssertionError('Grasp contact checks failed; evidence retained')


if __name__=='__main__':main()
