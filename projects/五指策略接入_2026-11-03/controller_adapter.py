"""Common articulated-hand execution adapter for the two existing policies.

Upper policy objectives are preserved. Signed/coupled response, distributed
contact geometry, real action domains and observer timing are shared changes.
No positive-diagonal infeasibility certificate is applicable to this model.
"""
import numpy as np
from vendor import frozen_controllers as frozen
from state_estimator import CoupledTactileState


class HandControllerMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.state=CoupledTactileState(self.g)
        self.latest_frame=None

    def step(self,frame,phase=None,command='hold'):
        self.latest_frame=frame
        action=super().step(frame,phase,command)
        if command=='release' or str(phase or frame.phase).upper() in ('RELEASE','RELEASING'):
            # The common mechanical thumb-clearance path is outside the local
            # held-grasp response model. Do not show a fabricated force forecast.
            self.predicted=np.full(5,np.nan)
            action.predicted_normal=self.predicted.copy()
            action.debug.update(prediction_available=False,
                                prediction_reason='common mechanical release outside local held-grasp model')
        action.debug.update(hand_model='signed_coupled_calibrated_response',
                            model_infeasibility_certified=False,
                            force_gain_role='normalization_only',
                            passive_palm_support_subtracted=False)
        return action

    def _search(self):
        s=self.state
        delta=np.clip(s.model_reference_alpha-s.alpha,
                      -self.g.open_rate*self.g.decision_period,
                      self.g.close_rate*self.g.decision_period)
        # A contact-loss search uses only a public calibrated reference posture.
        return np.where(~s.contact_model_active & s.valid & s.closure_permission,delta,0.)

    def _predictor(self):
        return self.state.force_model.action_prediction(
            self.state.normal,self.g.horizon,self.g.decision_period,
            self.state.contact_model_active,alpha=self.state.alpha,
            latent=self.state.latent_alpha,execution_dt=.005)

    def _trajectory_model(self):
        return self.state.force_model.reference_prediction(
            self.state.normal,self.g.horizon,self.g.decision_period,
            self.g.baseline_tracking_gain,self.state.contact_model_active,self._search(),
            alpha=self.state.alpha,latent=self.state.latent_alpha,execution_dt=.005,
            feedback_active=self.state.contact_model_active & self.state.valid)

    def _optimize(self,Q,q,P,constant,A,b,support_floor=None):
        size=len(q);zero=np.zeros(size)
        candidates=[zero,np.asarray(self.previous_plan)]
        feasible=[x for x in candidates if x.shape==(size,) and np.max(A@x-b)<=1e-9]
        seed=min(feasible,key=lambda x:.5*x@Q@x+q@x) if feasible else None
        result=frozen.solve_qp(Q,q,A,b,self.g.qp_passes,self.g.qp_tolerance,
                               feasible_initial=seed)
        self.debug.update(qp_status=result.status,qp_optimal=bool(result.converged),
            qp_primal_residual=float(result.residual),qp_iterations=int(result.iterations),
            model_force_infeasible=False,model_infeasibility_certified=False,
            signed_coupled_response=True)
        if not result.converged:self.numerical_failures+=1
        if result.residual<=self.g.qp_tolerance:
            plan=result.x
        elif seed is not None:
            plan=seed;self.debug['qp_status']='feasible_seed_after_numerical_nonconvergence'
        else:
            # Finite mechanics may not remove an already excessive real force in
            # one interval. Optimize unloading inside the REAL action domain,
            # report that force constraints remain unresolved and pause motion.
            h=self.g.horizon;period=self.g.decision_period
            I=np.eye(size);C=np.kron(np.tril(np.ones((h,h))),np.eye(5))
            rows=[I,-I,C,-C]
            rhs=[np.full(size,self.g.close_rate*period),np.full(size,self.g.open_rate*period),
                 np.tile(self.state.closure_max-self.state.alpha,h),
                 np.tile(self.state.alpha-self.state.closure_min,h)]
            blocked=~self.state.valid|~self.state.closure_permission
            if np.any(blocked):rows.append(I[np.tile(blocked,h)]);rhs.append(np.zeros(np.count_nonzero(blocked)*h))
            DA,Db=np.vstack(rows),np.concatenate(rhs)
            target=np.tile(np.maximum(0.,self.limits-self.g.force_margin),h)
            target=np.minimum(np.asarray(constant),target)
            unload=frozen.solve_qp(2.*(P.T@P+1e-3*I),2.*P.T@(constant-target),DA,Db,
                                  self.g.qp_passes,self.g.qp_tolerance,feasible_initial=zero)
            plan=unload.x if unload.residual<=self.g.qp_tolerance else zero
            self.pause_transport=True
            self.debug.update(qp_status='force_constraints_unresolved_bounded_unloading',
                              model_feasibility_unknown=True,unload_status=unload.status)
        self.previous_plan=np.asarray(plan).copy()
        return self.previous_plan,float(result.residual),int(result.iterations)

    def _baseline_action(self,target):
        caps=np.maximum(0.,self.limits-self.g.force_margin)
        target=frozen.project_force(target,caps,max(0.,self.budget-2.*self.g.force_margin))
        self.desired=target.copy()
        s=self.state;P,c=self._predictor()
        active=s.contact_model_active & s.valid
        L=self.g.baseline_tracking_gain*s.force_model.inverse(active=active)
        desired=L@(target-s.normal)+self._search()
        horizon=np.zeros(5*self.g.horizon);horizon[:5]=desired
        A,b=self._action_constraints(P,c)
        candidate,residual,iterations=self._optimize(np.eye(len(horizon)),-horizon,P,c,A,b)
        action=self._clip_first(candidate[:5])
        first=np.zeros(len(horizon));first[:5]=action
        self.predicted=np.maximum(0.,c[:5]+P[:5]@first)
        self.debug.update(force_target=target.tolist(),baseline_tracking_gain=self.g.baseline_tracking_gain,
                          qp_primal_residual=residual,qp_iterations=iterations,
                          closure_search=self._search().tolist())
        return action

    def _geometry(self):
        s=self.state
        active=s.contact & s.distributed_valid
        center=np.mean(s.pos[active],axis=0) if np.any(active) else np.zeros(3)
        W=s.normal_wrench_map.copy()
        W[3:]=(W[3:]-np.cross(np.broadcast_to(center,(5,3)),W[:3].T).T)/self.g.characteristic_length
        W[:,~active]=0.
        shear=s.finger_non_normal_by_finger[active].sum(axis=0)
        shear[3:]=(shear[3:]-np.cross(center,shear[:3]))/self.g.characteristic_length
        safe_mu=s.mu*self.g.friction_margin*np.clip(s.confidence,.35,1.)
        # NET support retains downward normal contributions; never normalize a
        # distributed map into an invented single contact normal.
        capacity=s.normal_support_coeff+safe_mu*s.tangent_support_coeff
        capacity=np.where(active,capacity,0.)
        if self.name=='unified_no_geometry':capacity=np.where(active,safe_mu,0.)
        return active,W,shear,capacity,safe_mu

    def _support_floors(self,P,c,caps,selected,active):
        """Preserve support without a fictitious positive-diagonal bound.

        A held-force reference is propagated through the actual signed tracker;
        the floor never exceeds that neutral prediction or measured support.
        This is a model bound, not a proof of physical capability.
        """
        h=self.g.horizon;s=self.state
        neutral=(P@np.tile(s.normal,h)+c).reshape(h,5)
        force=np.zeros((h,5));reference=np.zeros((h,5))
        for i in selected:
            reference[:,i]=min(self.g.support_contact_floor,max(0.,min(caps[i],s.normal[i])))
            if active[i]:
                floor=np.maximum(0.,np.minimum(neutral[:,i],s.normal[i])-s.response_error_bound[i])
                force[:,i]=np.minimum(self.g.support_contact_floor,floor)
        return force,reference


def create_controller(method,calibration,limits,budget,gains=None):
    from policy_sources import MigratedRatio,MigratedUnified
    if method not in ('ratio','unified'):raise ValueError('Only the two admitted current policies are exposed')
    values=dict(gains or {})
    B=np.asarray(calibration['force_matrix'],float)
    values.update(force_gain=tuple(np.maximum(.05,np.linalg.norm(B,axis=0))),
                  force_tau=float(calibration['force_tau']))
    cls=MigratedRatio if method=='ratio' else MigratedUnified
    return cls(method,np.asarray(limits,float),float(budget),values)
