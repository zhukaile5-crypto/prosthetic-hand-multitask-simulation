"""Independent synthetic contract audit: no hand model or dynamics imported.

Only artificial matrices, pressures and contact maps are used. These checks
establish execution/prediction/interface consistency, not hand performance.
"""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,hashlib,json,sys,textwrap
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from force_model import ForceResponse
from controller_adapter import create_controller,HandControllerMixin
from vendor import frozen_controllers as original

SOURCES=('controller_adapter.py','prepare_policy_sources.py','policy_sources.py','force_model.py','state_estimator.py','contracts.py','vendor/frozen_controllers.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
CHECKS={};DETAILS={}
def check(name,value,detail=None):
    CHECKS[name]=bool(value)
    if detail is not None:DETAILS[name]=detail

def frame(B,t=0.,**changes):
    theta=np.arange(5)*2*np.pi/5
    direction=np.c_[-np.cos(theta),-np.sin(theta),np.zeros(5)]
    basis=np.stack([np.c_[[0.,0.,1.],np.cross(n,[0.,0.,1.])].T for n in direction])
    zero=np.zeros((5,6))
    data=dict(time=t,normal=np.full(5,2.),tangential=np.zeros((5,2)),
        contact_pos=np.zeros((5,3)),contact_normal=direction,tangent_basis=basis,
        slip_local=np.zeros((5,2)),joint=np.full(5,.5),joint_velocity=np.zeros(5),
        valid=np.ones(5,bool),age=0.,alpha=np.full(5,.5),phase='HOLDING',lift=.1,lift_ref=.1,
        support_axis=np.array([0.,0.,1.]),raw_joint=np.zeros(24),raw_joint_velocity=np.zeros(24),
        force_gain=np.maximum(.05,np.linalg.norm(B,axis=0)),closure_permission=np.ones(5,bool),
        closure_max=np.full(5,1.2),closure_min=np.zeros(5),nominal_force_weights=np.ones(5),
        motor_effort=1.2,handload_prior=0.,load_estimate=1.2,load_uncertainty=.2,load_valid=True,load_confidence=.9,
        force_tau=.04,force_matrix=B.copy(),model_reference_alpha=np.full(5,.5),response_error_bound=np.full(5,.05),
        aggregation_valid=np.ones(5,bool),aggregation_error=np.zeros(5),sample_time=t,dt=.005,
        auxiliary_wrench=np.zeros(6),auxiliary_pressure=0.,auxiliary_valid=True,auxiliary_age=0.,auxiliary_uncertainty=.05,
        normal_wrench_map=np.vstack((direction.T,np.zeros((3,5)))),
        finger_non_normal_wrench=np.zeros(6),finger_non_normal_by_finger=zero,
        normal_support_coeff=np.zeros(5),tangent_support_coeff=np.ones(5),
        shear_magnitude=np.zeros(5),slip_magnitude=np.zeros(5),contact_present=np.ones(5,bool),
        distributed_valid=np.ones(5,bool),distributed_error=np.zeros(5))
    data.update(changes)
    return SimpleNamespace(**data)

def calibration(B):return dict(force_matrix=B,force_tau=.04)

def explicit_ramp(B,n,alpha,latent,increments,active,tau=.04):
    """Direct four 5-ms increments; no predictor timing helper is used."""
    a=np.exp(-.005/tau);q=alpha.copy();z=latent.copy();z0=z.copy();normal=[]
    for increment in increments:
        for _ in range(4):
            q+=increment/4.
            z=a*z+(1.-a)*q
        normal.append(n+active*(B@(z-z0)))
    return np.asarray(normal)

def response_checks():
    rng=np.random.default_rng(938117)
    B=np.array([[-2.,.7,0.,0.,0.],[.4,2.5,.3,0.,0.],[0.,-.5,2.,.2,0.],
                [0.,0.,.2,1.8,.4],[.1,0.,0.,-.3,2.2]])
    for case in range(8):
        active=np.ones(5,bool) if case==0 else rng.random(5)>.25
        n=1.+rng.random(5);alpha=.3+.2*rng.random(5);latent=alpha+rng.normal(0.,.03,5)
        model=ForceResponse(B,.04)
        increments=rng.normal(0.,.008,(3,5))
        P,c=model.action_prediction(n,3,.02,active,alpha=alpha,latent=latent)
        expected=explicit_ramp(B,n,alpha,latent,increments,active)
        check(f'action_ramp_{case}',np.allclose((P@increments.ravel()+c).reshape(3,5),expected,atol=1e-12,rtol=0))
        references=1.+rng.random((3,5));search=np.where(active,0.,.003)
        Q,d,U,uc=model.reference_prediction(n,3,.02,.2,active,search,alpha=alpha,latent=latent)
        selected=np.flatnonzero(active);inverse=np.zeros((5,5))
        if len(selected):
            block=B[np.ix_(selected,selected)]
            inverse[np.ix_(selected,selected)]=np.linalg.solve(block.T@block+.02**2*np.eye(len(selected)),block.T)
        q=alpha.copy();z=latent.copy();z0=z.copy();previous=n.copy();forces=[];actions=[];a=np.exp(-.005/.04)
        for reference in references:
            increment=.2*inverse@(reference-previous)+search
            actions.append(increment.copy())
            for _ in range(4):q+=increment/4.;z=a*z+(1.-a)*q
            previous=n+active*(B@(z-z0));forces.append(previous.copy())
        check(f'reference_ramp_{case}',np.allclose(Q@references.ravel()+d,np.ravel(forces),atol=1e-12,rtol=0))
        check(f'reference_actual_action_{case}',np.allclose(U@references.ravel()+uc,np.ravel(actions),atol=1e-12,rtol=0))
        changed=references.copy();changed[:,~active]+=999.
        check(f'unobserved_reference_no_action_effect_{case}',np.allclose(U@changed.ravel()+uc,U@references.ravel()+uc,atol=1e-12,rtol=0))
    check('signed_negative_response_not_abs',B[0,0]<0 and model.inverse()[0,0]<0)
    return B

def geometry_and_policy_checks(B):
    controllers={name:create_controller(name,calibration(B),[5.]*5,20.) for name in ('ratio','unified')}
    check('only_two_factory_methods',all(
        (c.name==name and c._optimize.__func__ is HandControllerMixin._optimize)
        for name,c in controllers.items()))
    rejected=[]
    for name in ('fixed','legacy_demo','unified_no_geometry','unknown'):
        try:create_controller(name,calibration(B),[5.]*5,20.)
        except ValueError:rejected.append(name)
    check('unsupported_factory_methods_rejected',len(rejected)==4)
    try:create_controller('ratio',{},[5.]*5,20.)
    except (KeyError,ValueError):check('no_five_pad_calibration_fallback',True)
    else:check('no_five_pad_calibration_fallback',False)
    W=np.array([[.2,-.3,.4,.1,-.2],[.1,.2,-.1,.3,-.3],[-.8,.2,.0,.15,.05],
                [.012,.006,-.008,.009,.005],[-.004,.011,.002,-.006,.003],[.002,-.003,.007,.001,-.004]])
    f=frame(B,contact_normal=np.zeros((5,3)),normal_wrench_map=W,
            normal_support_coeff=W[2].copy(),tangent_support_coeff=np.array([.3,.9,.7,.6,.8]))
    c=controllers['unified'];c.state.update(f,.005);c.latest_frame=f
    active,wrench,shear,capacity,safe_mu=c._geometry()
    expected=W.copy();expected[3:]/=c.g.characteristic_length
    check('distributed_contact_survives_mean_normal_cancellation',bool(np.all(active)))
    check('distributed_six_dimensional_map_preserved',np.allclose(wrench,expected,atol=1e-12,rtol=0))
    expected_capacity=W[2]+safe_mu*f.tangent_support_coeff
    check('signed_net_support_capacity',np.allclose(capacity,expected_capacity,atol=1e-12,rtol=0) and capacity[0]<0)
    check('normal_map_not_unit_normalized',np.allclose(wrench[:3],W[:3],atol=1e-12,rtol=0))
    # Modeled force-output rows and valid feedback channels are different under
    # dropout; the reference-to-action predictor must match the shared tracker.
    missing=create_controller('unified',calibration(B),[5.]*5,20.)
    missing.state.update(frame(B),.005)
    dropout=frame(B,t=.005)
    dropout.valid[1]=False;dropout.normal[1]=np.nan
    dropout.tangential[1]=np.nan;dropout.slip_local[1]=np.nan
    dropout.shear_magnitude[1]=np.nan;dropout.slip_magnitude[1]=np.nan
    missing.state.update(dropout,.005)
    _,_,U,uc=missing._trajectory_model()
    reference=np.tile(missing.state.normal,3)+np.linspace(.1,.8,15)
    expected_u=missing.g.baseline_tracking_gain*missing.state.force_model.inverse(
        active=missing.state.contact_model_active & missing.state.valid)@(reference[:5]-missing.state.normal)+missing._search()
    check('partial_dropout_predictor_matches_actual_feedback_mask',np.allclose(U[:5]@reference+uc[:5],expected_u,atol=1e-12,rtol=0))
    c.state.shear_magnitude[:]=.3;c.state.tangential[:]=0.;c.state.delayed_normal[:]=2.
    r=controllers['ratio'];r.state=c.state;r._ratio_target('HOLDING')
    check('ratio_uses_distributed_shear_scalar',abs(r.debug['ratio']-.15)<1e-12)
    f2=frame(B,t=.005,slip_magnitude=np.full(5,.05),slip_local=np.zeros((5,2)))
    r.state.update(f2,.005)
    check('slip_scalar_not_projected_cancellation',r.state.slip_speed>.004)
    # A signed carry-over can lower the neutral prediction despite positive N.
    c.state.normal=np.full(5,.35);c.state.alpha=np.full(5,.8);c.state.latent_alpha=np.full(5,.3)
    c.state.contact_model_active[:]=True;c.state.response_error_bound[:]=.05
    P,k,_,_=c._trajectory_model();neutral=(P@np.tile(c.state.normal,3)+k).reshape(3,5)
    force_floor,reference_floor=c._support_floors(P,k,np.full(5,4.88),[0,1,2],np.ones(5,bool))
    expected_floor=np.zeros((3,5));expected_floor[:,:3]=np.minimum(c.g.support_contact_floor,
        np.maximum(0.,np.minimum(neutral[:,:3],c.state.normal[:3])-.05))
    check('support_floor_uses_signed_neutral_prediction',np.allclose(force_floor,expected_floor,atol=1e-12,rtol=0))
    old=force_floor.copy();c.state.force_gain[:]=999.
    again,_=c._support_floors(P,k,np.full(5,4.88),[0,1,2],np.ones(5,bool))
    check('support_floor_not_positive_gain_capacity_claim',np.array_equal(old,again))
    # The factory/step release contract is checked on artificial sensor data.
    for name in ('ratio','unified'):
        trial=create_controller(name,calibration(B),[5.]*5,20.)
        release=trial.step(frame(B),command='release')
        check(f'{name}_release_requests_only_opening',np.all(release.delta_closure<=0))
        check(f'{name}_release_forecast_unavailable',np.isnan(release.predicted_normal).all() and release.debug['prediction_available'] is False)
        check(f'{name}_no_monotonic_infeasibility_certificate',release.debug['model_infeasibility_certified'] is False)
    # An excessive synthetic pressure with B00<0 requires positive u0 to unload.
    c=create_controller('ratio',calibration(B),[1.]*5,4.)
    c.state.update(frame(B,normal=np.full(5,10.)),.005)
    P,k=c._predictor();A,b=c._action_constraints(P,k)
    plan,residual,iterations=c._optimize(np.eye(15),np.zeros(15),P,k,A,b)
    check('bounded_unload_uses_negative_gain_sign',plan[0]>0.,{'first_action':plan[:5].tolist(),'status':c.debug['qp_status']})
    check('unresolved_force_not_false_physical_infeasibility',c.debug['model_force_infeasible'] is False and c.debug['model_infeasibility_certified'] is False and c.pause_transport)
    check('bounded_unload_respects_actual_action_domain',np.max(plan)<=c.g.close_rate*c.g.decision_period+1e-6 and np.min(plan)>=-c.g.open_rate*c.g.decision_period-1e-6)

def source_checks():
    frozen=ast.parse((ROOT/'vendor/frozen_controllers.py').read_text(encoding='utf-8'))
    migrated=ast.parse((ROOT/'policy_sources.py').read_text(encoding='utf-8'))
    def node(tree,cls,name):
        parent=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        return next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name==name)
    def canonical(fn):return ast.dump(fn,include_attributes=False)
    # Declarative source audit: permitted sensor feature substitutions only.
    class Features(ast.NodeTransformer):
        def visit_Call(self,n):
            text=ast.unparse(n)
            if text in ('np.linalg.norm(self.state.tangential[supported], axis=1).sum()',
                        'self.state.shear_magnitude[supported].sum()'):
                return ast.copy_location(ast.Name(id='AUDIT_SHEAR_SUM',ctx=ast.Load()),n)
            if text in ('np.linalg.norm(self.state.tangential, axis=1)',):
                return ast.copy_location(ast.Name(id='AUDIT_SHEAR_MAGNITUDES',ctx=ast.Load()),n)
            if text in ('np.linalg.norm(self.state.slip, axis=1)',):
                return ast.copy_location(ast.Name(id='AUDIT_SLIP_MAGNITUDES',ctx=ast.Load()),n)
            return self.generic_visit(n)
        def visit_Attribute(self,n):
            text=ast.unparse(n)
            if text=='self.state.shear_magnitude':return ast.copy_location(ast.Name(id='AUDIT_SHEAR_MAGNITUDES',ctx=ast.Load()),n)
            if text=='self.state.slip_magnitude':return ast.copy_location(ast.Name(id='AUDIT_SLIP_MAGNITUDES',ctx=ast.Load()),n)
            return self.generic_visit(n)
    import copy
    old=Features().visit(copy.deepcopy(node(frozen,'RatioController','_ratio_target')))
    new=Features().visit(copy.deepcopy(node(migrated,'MigratedRatio','_ratio_target')))
    check('ratio_body_only_declared_feature_adaptation',canonical(old)==canonical(new))
    old=node(frozen,'UnifiedCoordinator','_decide');new=node(migrated,'MigratedUnified','_decide')
    old=copy.deepcopy(old);new=copy.deepcopy(new)
    # Replace the old positive-diagonal attainable-floor block with a marker.
    class Floors(ast.NodeTransformer):
        def visit_If(self,n):
            if ast.unparse(n.test)=='not acquisition':
                first=ast.unparse(n.body[0])
                marker=ast.Expr(value=ast.Name(id='AUDIT_SHARED_SUPPORT_FLOOR',ctx=ast.Load()))
                if first.startswith('attainable = '):
                    assert isinstance(n.body[1],ast.For) and ast.unparse(n.body[1].iter)=='selected'
                    n.body=[marker]+n.body[2:]
                    return n
                if first.startswith('force_floor, reference_floor = self._support_floors('):
                    n.body=[marker]+n.body[1:]
                    return n
            return self.generic_visit(n)
    old=Features().visit(Floors().visit(old));new=Features().visit(Floors().visit(new))
    equal=canonical(old)==canonical(new)
    if not equal:
        import difflib
        DETAILS['unified_normalized_ast_diff']=list(difflib.unified_diff(ast.unparse(old).splitlines(),ast.unparse(new).splitlines(),fromfile='vendor_normalized',tofile='migrated_normalized'))
    check('unified_body_only_declared_features_and_model_floor_adaptation',equal)
    check('vendor_snapshot_identity',sha(ROOT/'vendor/frozen_controllers.py')=='1277c9656259132c2529964284e58927bf4f6385f09eeabb3f85f3e9581d9805')

def main():
    before={p:sha(ROOT/p) for p in SOURCES}
    source_checks();B=response_checks();geometry_and_policy_checks(B)
    check('no_hand_physics_module_imported','hand_plant' not in sys.modules and 'mujoco' not in sys.modules)
    check('reviewed_sources_unchanged_during_audit',before=={p:sha(ROOT/p) for p in SOURCES})
    report={'schema':'independent-synthetic-adapter-audit-v1','checked_utc':datetime.now(timezone.utc).isoformat(),
        'passed':all(CHECKS.values()),'checks':CHECKS,'details':DETAILS,'source_sha256':before,
        'scope':'Pure artificial matrix/contact/interface/QP tests; no actual hand, physical calibration data, dynamics or policy-performance experiment.',
        'limits':['A synthetic pass does not establish force-model accuracy or physical grasp performance.',
                  'A separate physical and admission gate is still mandatory before real strategy runs.']}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/f'review/independent_adapter_{stamp}.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(CHECKS),'failed':[k for k,v in CHECKS.items() if not v],
                      'evidence':str(out)},ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
