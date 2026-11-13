"""Finite analytical author gate. No model, physics, FK, runtime or real data read."""
from pathlib import Path
import sys,json,hashlib,datetime,dataclasses
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v16_release_geometry.transfer_observer import (TransferObserver,PreReleaseLoadPrior,HandSupportSample)
from v15_support.receiver_sensor import ReceiverMeasurement

CHECKS=[]
def check(name,condition,detail=None):
    CHECKS.append(dict(name=name,passed=bool(condition),detail=detail))
def raises(call):
    try:call()
    except (ValueError,TypeError,AttributeError):return True
    return False
SIGMA=np.array([.01]*3+[.0005]*3)
def receiver(t,R,x=.01,y=.02,*,valid=True,force=None):
    if not valid:return ReceiverMeasurement(t,None,None,False,None,None,SIGMA,3*SIGMA,None,None,'acquisition_unavailable')
    f=np.array([0.,0.,-R]) if force is None else np.asarray(force,float)
    w=np.r_[f,[y*f[2],-x*f[2],.004]]
    return ReceiverMeasurement(t,t-.01,.01,True,w,max(0.,-f[2]),SIGMA,3*SIGMA,None,'boundary_after_forward',None)
def prior():return PreReleaseLoadPrior(1.,.1,11.995)
def observer(p='default',up=(0.,0.,1.)):
    return TransferObserver(request_time_s=12.,receiver_half_size_xy_m=[.2,.15],receiver_up_axis3=up,prior=prior() if p=='default' else p)
def hand(t,H=.5,u=.02):return HandSupportSample(t,t-.01,True,H,u)
def series(o,kind='transfer',geometry=False,first=0,last=40):
    result=[]
    for i in range(first,last+1):
        t=12.01+i*.005
        H,R=(.8-.6*i/40,.2+.6*i/40) if kind=='transfer' else ((-.5,1.5) if kind=='downward' else (0.,1.))
        x=.1*(i*.005) if kind=='moving' else .01
        kwargs={} if geometry is None else dict(hand_receiver_confounded=geometry,geometry_acquisition_time_s=t-.01)
        result.append(o.update(time_s=t,hand=hand(t,H),receiver=receiver(t,R,x=x),**kwargs))
    return result

def cop_regression(origin,velocity=0.,offset=0.,sigma=None):
    """Constant R, analytic nonzero wrench point; no object or motion model."""
    o=TransferObserver(request_time_s=origin,receiver_half_size_xy_m=[.2,.15],receiver_up_axis3=[0.,0.,1.])
    sigma=np.zeros(6) if sigma is None else np.asarray(sigma,float)
    for i in range(41):
        t=origin+.01+i*.005;x=.01+offset+velocity*i*.005;y=.005
        r=ReceiverMeasurement(t,t-.01,.01,True,np.array([0.,0.,-1.,-y,x,0.]),1.,sigma,3*sigma,None,'boundary_after_forward',None)
        e=o.update(time_s=t,hand=HandSupportSample(t,t-.01,True,0.,0.),receiver=r)
    return e.continuous_support_evidence['cop_trend']

def main():
    o=observer();e=series(o);p=e[-1].continuous_support_evidence
    check('41_fresh_samples_span_200ms',p['window_complete'] and p['window_samples']==41)
    check('40_samples_are_not_complete',not e[-2].continuous_support_evidence['window_complete'])
    check('analytical_signed_load_means',np.allclose([p['hand_support_N'],p['receiver_support_N']],[.5,.5],rtol=0,atol=2e-15))
    check('coupled_fraction_interval_closed_form',np.allclose(p['coupled_transfer_fraction'],[.24/1.1,.34/.9],rtol=0,atol=2e-15),p['coupled_transfer_fraction'])
    check('declared_engineering_width_not_sqrt_n',np.allclose([p['hand_engineering_width_N'],p['receiver_engineering_width_N']],[.02,.03],rtol=0,atol=1e-16))
    check('static_cop_and_all_six_wrench_channels',np.allclose(p['cop']['xy'],[.01,.02],rtol=0,atol=1e-16) and p['receiver_mean_wrench6'][5]!=0 and not p['cop_trend']['motion_resolved_against_engineering_proxy'])
    check('experimental_permission_has_no_placement_or_holding_authority',e[-1].stage_permission=='EXPERIMENTAL_COORDINATION_EVIDENCE' and not e[-1].to_public()['placement_certified'] and not e[-1].to_public()['holding_support_may_be_requested'] and not e[-1].to_public()['release_request_may_be_delayed'])
    obj=e[-1].to_public();obj['continuous_support_evidence']['hand_support_N']=99
    check('public_payload_copy_cannot_mutate_snapshot',e[-1].continuous_support_evidence['hand_support_N']!=99)
    last_t=12.01+40*.005
    check('same_tick_identical_is_idempotent',o.update(time_s=last_t,hand=hand(last_t,.8-.6*40/40),receiver=receiver(last_t,.2+.6*40/40),hand_receiver_confounded=False,geometry_acquisition_time_s=last_t-.01) is e[-1])
    check('same_tick_changed_rejected',raises(lambda:o.update(time_s=12.21,hand=hand(12.21,.3),receiver=receiver(12.21,.8),hand_receiver_confounded=False,geometry_acquisition_time_s=12.2)))
    check('clock_rewind_rejected',raises(lambda:o.update(time_s=12.205,hand=hand(12.205),receiver=receiver(12.205,1.))))
    unknown=o.update(time_s=12.215,hand=HandSupportSample(12.215,None,False,None,None,'missing'),receiver=receiver(12.215,0,valid=False))
    check('unknown_is_null_and_resets_window',unknown.status=='UNKNOWN' and unknown.continuous_support_evidence['hand_support_N'] is None and len(o.window)==0)
    for name,geom in [('unknown',None),('confounded',True)]:
        a=series(observer(),geometry=geom)[-1]
        check(name+'_geometry_cannot_authorize',a.stage_permission=='OBSERVATION_ONLY')
    a=observer();series(a,geometry=None,last=39);one=series(a,geometry=False,first=40)[-1]
    check('one_clear_geometry_tick_cannot_authorize_whole_window',one.stage_permission=='OBSERVATION_ONLY' and not one.continuous_support_evidence['geometry_clear_throughout_window'])
    no=series(observer(None))[-1]
    check('no_load_prior_is_not_zero_weight',no.continuous_support_evidence['coupled_transfer_fraction'] is None and no.stage_permission=='OBSERVATION_ONLY')
    down=series(observer(),kind='downward')[-1]
    check('negative_hand_support_preserved',down.continuous_support_evidence['hand_support_N']==-.5 and down.status=='DOWNWARD_HAND_LOADING_PROXY' and down.stage_permission=='OBSERVATION_ONLY')
    moving=series(observer(),kind='moving')[-1];mp=moving.continuous_support_evidence
    check('moving_cop_slope_is_past_window_proxy',np.allclose(mp['cop_trend']['velocity_m_s'],[.1,0.],atol=1e-14,rtol=0) and mp['cop_trend']['motion_resolved_against_engineering_proxy'] and moving.stage_permission=='OBSERVATION_ONLY')
    low=observer().update(time_s=12.01,hand=hand(12.01),receiver=receiver(12.01,.01))
    check('valid_low_load_has_unknown_cop_not_unknown_force',low.continuous_support_evidence['cop'] is None and low.continuous_support_evidence['receiver_support_N']==.01)
    zero=observer().update(time_s=12.01,hand=hand(12.01,0.),receiver=receiver(12.01,0.))
    check('valid_zero_receiver_distinct_from_missing',zero.status!='UNKNOWN' and zero.continuous_support_evidence['receiver_support_N']==0.)
    pre=observer().update(time_s=12.,hand=hand(12.,.8),receiver=receiver(12.,.2))
    eq=observer().update(time_s=12.01,hand=hand(12.01,.8),receiver=receiver(12.01,.2))
    check('postrequest_acquisition_eligibility',pre.status=='UNKNOWN' and 'not_yet_postrequest' in pre.unknown_reasons and eq.status=='ACCUMULATING_WINDOW')
    check('future_prior_and_geometry_rejected',raises(lambda:observer(PreReleaseLoadPrior(1.,.1,12.005))) and raises(lambda:observer().update(time_s=12.01,hand=hand(12.01),receiver=receiver(12.01,.5),geometry_acquisition_time_s=12.02)))
    times=11.8+np.arange(40)*.005
    pp=PreReleaseLoadPrior.from_history(12.,times,np.ones(40),np.full(40,.1),np.ones(40,bool))
    bad=np.ones(40,bool);bad[2]=False
    check('preprior_fixed_40sample_grid_and_missing',pp.value_N==1. and np.isclose(pp.uncertainty_N,.1) and PreReleaseLoadPrior.from_history(12.,times,np.ones(40),np.full(40,.1),bad) is None and PreReleaseLoadPrior.from_history(12.,times+.005,np.ones(40),np.full(40,.1),np.ones(40,bool)) is None)
    tilted=observer(up=[1.,0.,0.]).update(time_s=12.01,hand=hand(12.01),receiver=receiver(12.01,1.2,force=[-.5,0.,-1.2]))
    check('receiver_vertical_support_not_normal_compression',tilted.continuous_support_evidence['receiver_support_N']==.5 and tilted.continuous_support_evidence['receiver_compression_N']==1.2)
    M=np.zeros((6,5));M[0,:2]=[1.,-1.];F=np.zeros((5,6));F[0,2]=.9;B=np.tile([[0.,1.,0.],[0.,0.,1.]],(5,1,1));aux=np.array([0.,0.,.1,0.,0.,0.]);n=np.array([4.,4.,0.,0.,0.])
    kw=dict(read_time_s=12.01,acquisition_time_s=12.,up_at_acquisition3=[0.,0.,1.],axis_time_s=12.,normal5=n,normal_wrench_map6x5=M,non_normal5x6=F,tangent_basis5x2x3=B,auxiliary_wrench6=aux,valid5=[True]*5,distributed_valid5=[True]*5,auxiliary_valid=True,auxiliary_acquisition_s=12.,force_sigma_N=.01,tangential_sigma_N=.005,auxiliary_uncertainty_N=.1)
    hs=HandSupportSample.from_components(**kw)
    check('signed_wrench_projection_not_clamp_sum',hs.valid and hs.signed_support_N==1. and np.isclose(hs.uncertainty_N,.175) and n.sum()==8.)
    Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    Mr=M.copy();Mr[:3]=Q@M[:3];Fr=F.copy();Fr[:,:3]=F[:,:3]@Q.T;ar=aux.copy();ar[:3]=Q@aux[:3]
    rot=HandSupportSample.from_components(**{**kw,'up_at_acquisition3':Q@np.array([0.,0.,1.]),'normal_wrench_map6x5':Mr,'non_normal5x6':Fr,'tangent_basis5x2x3':B@Q.T,'auxiliary_wrench6':ar})
    split=F.copy();split[0,2]=.45;split[1,2]=.45
    reshared=HandSupportSample.from_components(**{**kw,'non_normal5x6':split})
    check('frame_rotation_and_force_redistribution_invariant',rot.signed_support_N==hs.signed_support_N and rot.uncertainty_N==hs.uncertainty_N and reshared.signed_support_N==hs.signed_support_N)
    stale=HandSupportSample.from_components(**{**kw,'axis_time_s':12.005});neg=HandSupportSample.from_components(**{**kw,'normal5':[-1.,4.,0.,0.,0.]})
    check('acquisition_axis_mismatch_and_negative_pressure_unknown',not stale.valid and stale.signed_support_N is None and not neg.valid)
    check('typed_no_truth_extra_and_strict_scalars',raises(lambda:HandSupportSample(12.01,12.,True,.5,.1,target_id=2)) and raises(lambda:PreReleaseLoadPrior(True,.1,11.995)) and raises(lambda:observer().update(time_s=12.01,hand=hand(12.01),receiver=receiver(12.01,.5),hand_receiver_confounded='False')))
    check('geometry_arrays_immutable',raises(lambda:o.half.setflags(write=True)) and raises(lambda:o.receiver_up.setflags(write=True)))
    ambiguous1=series(observer(),geometry=None)[-1].to_public();ambiguous2=series(observer(),geometry=None)[-1].to_public()
    check('aggregate_ambiguity_no_hidden_identity_certificate',ambiguous1==ambiguous2 and not ambiguous1['placement_certified'] and ambiguous1['stage_permission']=='OBSERVATION_ONLY')
    constants=[];lines=[]
    for origin in (1.,12.):
        c=cop_regression(origin);l=cop_regression(origin,.1);constants.append(c);lines.append(l)
        check('zero_sigma_constant_nonzero_cop_origin_'+str(origin),c['velocity_m_s']==[0.,0.] and c['engineering_width_m_s']==[0.,0.] and not c['motion_resolved_against_engineering_proxy'])
        check('zero_sigma_true_linear_trend_origin_'+str(origin),np.allclose(l['velocity_m_s'],[.1,0.],atol=1e-15,rtol=0) and l['motion_resolved_against_engineering_proxy'])
    check('registered_clock_origin_does_not_change_cop_regression',constants[0]==constants[1] and lines[0]==lines[1])
    shifted=cop_regression(12.,.1,.125)
    check('point_translation_preserves_linear_regression_to_float_roundoff',np.allclose(shifted['velocity_m_s'],lines[1]['velocity_m_s'],atol=2e-15,rtol=0))
    noisy=cop_regression(12.,sigma=SIGMA)
    # Independent affine L1 interval propagation: constant nonzero CoP gives
    # the same per-sample ratio interval; no sqrt(n) or pair independence.
    width=[]
    for mean_m,sign,center in ((.01,-1.,.01),(-.005,1.,.005)):
        corners=[sign*m/f for m in (mean_m-.0015,mean_m+.0015) for f in (-1.-.03,-1.+.03)]
        width.append(max(center-min(corners),max(corners)-center))
    j=np.arange(41,dtype=float)-20.;w=j/(.005*float(j@j))
    expected=(abs(w[1:]).sum()+abs(w[1:].sum()))*np.asarray(width)
    check('shared_anchor_uncertainty_uses_combined_anchor_coefficient',np.allclose(noisy['engineering_width_m_s'],expected,atol=2e-15,rtol=0),dict(actual=noisy['engineering_width_m_s'],closed_form=expected.tolist()))
    source=ROOT/'v16_release_geometry/transfer_observer.py'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(schema='v16-transfer-observer-author-pure-v2',role='component_author_not_independent_physical_audit',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),source_sha256=sha(source),test_source_sha256=sha(Path(__file__)),receiver_source_sha256=sha(ROOT/'v15_support/receiver_sensor.py'),six_case_diagnostic_sha256=sha(ROOT/'review/v16_release_geometry_six_cases_v1.json'),supersedes=dict(source_395e_archived=str(ROOT/'review/revisions/v16_transfer_observer_395e_before_centered_cop.py'),source_sha256=sha(ROOT/'review/revisions/v16_transfer_observer_395e_before_centered_cop.py'),author_report_v1_sha256=sha(ROOT/'review/v16_transfer_observer_author_v1.json'),reason='independent constant zero-sigma CoP false motion; no numeric threshold added'),scope=dict(analytical_fixtures=True,model_constructions=0,physics_steps=0,FK_calls=0,real_data_reanalysis=False,runtime_integration=False),checks=CHECKS,passed=all(c['passed'] for c in CHECKS),counts=dict(total=len(CHECKS),passed=sum(c['passed'] for c in CHECKS),failed=sum(not c['passed'] for c in CHECKS)))
    out=ROOT/'review/v16_transfer_observer_author_v2.json'
    if out.exists():raise FileExistsError('Preserve prior evidence; choose new report filename')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(report=str(out),sha256=sha(out),source_sha256=report['source_sha256'],counts=report['counts'],passed=report['passed']),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
