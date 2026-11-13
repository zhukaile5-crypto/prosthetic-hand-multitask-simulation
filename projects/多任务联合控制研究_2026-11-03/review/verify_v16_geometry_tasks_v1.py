"""Analytic and representation oracles; no model/simulator import or physics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v16_release_geometry.geometry_tasks import (make_length_proposal_map,
    RobotPatchGeometry, build_geometry_tasks, task_loss)


def main(out):
    checks = []
    def check(name, value, details=None):
        checks.append(dict(name=name, passed=bool(value), details=details))
    def rejects(name, fn):
        try:
            fn()
            check(name, False)
        except (ValueError, TypeError):
            check(name, True)
    A = np.eye(21,25)
    def mapping(**kw):
        args = dict(transmission21x25=A, reference_to_length21=np.ones(21),
            joint_scale25=np.ones(25), robot_axes25=tuple('q'+str(i) for i in range(25)),
            reference_axes21=tuple('r'+str(i) for i in range(21)), time_s=1.,
            declared_source='analytic fixture only; identity length/reference assumption NOT robot evidence')
        args.update(kw)
        return make_length_proposal_map(**args)
    H = mapping()
    check('identity length proposal exact, four null dimensions explicit',
          np.array_equal(H.H25x21,np.eye(25,21)) and H.rank==21 and H.nullity==4)
    check('proposal never asserts dynamics/pressure accuracy',
          not H.actual_servo_response_validated and not H.pressure_prediction_available)
    J = np.zeros((3,25)); J[:3,:3]=np.eye(3)
    def patch(**kw):
        args = dict(robot_body=1,digit=0,acquisition_time_s=.99,robot_time_s=1.,
            observed_normal_N=1.,acquisition_position_world_m=np.zeros(3),
            acquisition_normal_world=np.array([1.,0,0]),robot_position_world_m=np.array([.001,0,0]),
            robot_jacobian_m_per_q25=J,robot_axes25=tuple('q'+str(i) for i in range(25)),
            robot_point_velocity_world_m_s=np.array([.2,.3,0]))
        args.update(kw)
        return RobotPatchGeometry(**args)
    def tasks(ps=None, **kw):
        args = dict(patches=[patch()] if ps is None else ps,mapping=H,current_time_s=1.,
            maximum_geometry_age_s=.12,step_lower21=-np.ones(21)*.1,
            step_upper21=np.ones(21)*.1,nominal_open_increment21=np.zeros(21),
            mode_generation=3,robot_geometry_authenticated=True)
        args.update(kw)
        return build_geometry_tasks(**args)
    t=tasks(); u=np.zeros(21);u[:3]=[.01,.02,.03]
    check('normal positive is inward, old-plane offset diagnostic only',
        np.array_equal(t.normal_matrix_m_per_reference[0],np.eye(21)[0]) and
        t.normal_offset_m[0]==0 and t.diagnostics['patches'][0]['robot_old_plane_inward_offset_m']==.001)
    loss=task_loss(t,u)
    check('closed-form separate normal and tangent losses',
          np.isclose(loss['normal_noninward_loss_m2'],.01**2) and
          np.isclose(loss['tangent_increment_loss_m2'],.02**2+.03**2))
    check('current normal/tangent velocities observed separately, never finite response',
        t.diagnostics['patches'][0]['current_robot_normal_velocity_m_s']==.2 and
        t.diagnostics['patches'][0]['current_robot_tangent_speed_m_s']==.3 and
        t.diagnostics['current_robot_velocity_not_substituted_as_a_finite_servo_model'])
    # Independent Gram/gradient oracle checks physical patch subdivision.
    base=tasks([patch(observed_normal_N=3.),patch(digit=1,observed_normal_N=2.,acquisition_normal_world=np.array([0,1.,0]))])
    split=tasks([patch(observed_normal_N=1.),patch(observed_normal_N=2.),
                 patch(digit=1,observed_normal_N=2.,acquisition_normal_world=np.array([0,1.,0]))])
    def cost_gradient(t,x):
        G=t.normal_matrix_m_per_reference; w=t.normal_weight
        r=G@x;active=np.maximum(r,0);T=t.tangent_matrix_m_per_reference
        return np.sum((w*active)**2)+np.sum((T@x)**2), 2*G.T@(w*w*active)+2*T.T@(T@x)
    c0,g0=cost_gradient(base,u);c1,g1=cost_gradient(split,u)
    bound=64*np.finfo(float).eps*max(1.,abs(c0),np.max(abs(g0)))
    check('split/merge loss and full gradient invariant with explicit arithmetic bound',
          abs(c0-c1)<=bound and np.max(abs(g0-g1))<=bound,dict(bound=bound,max_gradient_difference=float(np.max(abs(g0-g1)))))
    perm=tasks(list(reversed([patch(observed_normal_N=3.),patch(digit=1,observed_normal_N=2.,acquisition_normal_world=np.array([0,1.,0]))])))
    cp,gp=cost_gradient(perm,u)
    check('patch permutation objective/gradient invariant',abs(cp-c0)<=bound and np.max(abs(gp-g0))<=bound)
    scaled=tasks([patch(observed_normal_N=30.),patch(digit=1,observed_normal_N=20.,acquisition_normal_world=np.array([0,1.,0]))])
    cs,gs=cost_gradient(scaled,u)
    check('uniform measured pressure scaling does not retune geometry',abs(cs-c0)<=bound and np.max(abs(gs-g0))<=bound)
    opposite=tasks([patch(),patch(acquisition_normal_world=np.array([-1.,0,0]))])
    check('opposite normals kept as two conflicting rows, no mean cancellation',
          opposite.normal_matrix_m_per_reference.shape==(2,21) and
          task_loss(opposite,u)['normal_noninward_loss_m2']>0)
    negJ=J.copy();negJ[0,0]=-1
    upper=np.zeros(21); lower=-np.ones(21)*.1; op=np.zeros(21);op[0]=-.1
    conflict=tasks([patch(robot_jacobian_m_per_q25=negJ)],step_lower21=lower,step_upper21=upper,nominal_open_increment21=op)
    check('one-way OPEN conflict honest; no permission widening/action solving',
        conflict.diagnostics['strict_outward_unavailable_patch_indices']==[0] and
        conflict.diagnostics['unresolved_OPEN_noninward_patch_indices']==[0] and
        np.array_equal(conflict.step_upper21,upper) and np.array_equal(conflict.open_target_increment21,op))
    fixed=tasks(step_lower21=np.zeros(21),step_upper21=np.zeros(21))
    check('all fixed bounds no fake outward direction',fixed.diagnostics['strict_outward_unavailable_patch_indices']==[0])
    for name,kw in [('missing map',dict(mapping=None)),('unknown robot membership',dict(robot_geometry_authenticated=False)),
                    ('wrong map clock',dict(mapping=mapping(time_s=.9))),('empty observed set',dict(ps=[])),
                    ('wrong robot Jacobian axis order',dict(ps=[patch(robot_axes25=tuple(reversed(H.robot_axes25)))])),
                    ('partial stale geometry',dict(ps=[patch(),patch(acquisition_time_s=.1)]))]:
        result=tasks(**kw)
        check(name+' returns unknown matrices, never zero gains/clearance',result.normal_matrix_m_per_reference is None and not result.diagnostics['delivery_certified'])
    known=np.ones(21,bool);known[3]=False
    check('unknown reference-to-length column refuses whole map',mapping(known_reference21=known) is None)
    zeros=mapping(reference_to_length21=np.zeros(21))
    check('known zero mapping distinct from unknown, labelled experimental',zeros is not None and np.count_nonzero(zeros.H25x21)==0)
    extreme=tasks([patch(observed_normal_N=1e-300),patch(observed_normal_N=1e300)])
    check('positive pressure share underflow unavailable without epsilon',extreme.normal_matrix_m_per_reference is None)
    large=tasks([patch(observed_normal_N=1e308),patch(observed_normal_N=1e308)])
    check('large finite positive pressures normalized without overflow',large.status=='EXPERIMENTAL_GEOMETRY_TASKS' and np.allclose(large.normal_weight,np.sqrt(.5)))
    rejects('numeric string rejected',lambda:patch(observed_normal_N='1'))
    rejects('numpy bool rejected',lambda:patch(observed_normal_N=np.bool_(True)))
    rejects('zero pressure is outside positive-patch task domain',lambda:patch(observed_normal_N=0))
    rejects('truth field not in schema',lambda:patch(object_mass=.12))
    rejects('future acquisition rejected',lambda:tasks([patch(acquisition_time_s=1.01)]))
    rejects('empty real bounds rejected, not quietly clipped',lambda:tasks(step_lower21=np.ones(21),step_upper21=np.zeros(21)))
    rejects('non-unit normal not silently renormalized',lambda:patch(acquisition_normal_world=np.array([2.,0,0])))
    rejects('nonpositive joint metric rejected',lambda:mapping(joint_scale25=np.zeros(25)))
    rejects('nonfinite Jacobian rejected',lambda:patch(robot_jacobian_m_per_q25=np.full((3,25),np.nan)))
    source=ROOT/'v16_release_geometry/geometry_tasks.py'
    result=dict(schema='v16_geometry_tasks_analytic_author_v1',passed=all(x['passed'] for x in checks),
        checks=checks,check_count=len(checks),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope=dict(physics_steps=0,models_constructed=0,simulator_imported=False,optimization_not_run=True,
                   claims_only='task representation/units/permissions and unknown semantics; no servo or pressure accuracy'))
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),path=str(out),source=result['source_sha256']),ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
