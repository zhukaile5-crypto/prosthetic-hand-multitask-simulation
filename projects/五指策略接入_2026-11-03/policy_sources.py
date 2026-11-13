# Generated, reviewable migration of the existing two policy bodies.
import time
import numpy as np
from vendor.frozen_controllers import RatioController, UnifiedCoordinator, solve_qp, QPResult, project_force
from controller_adapter import HandControllerMixin

class MigratedRatio(HandControllerMixin, RatioController):
    def _ratio_target(self, phase):
        supported = self.state.contact
        slipping = self.state.slip_speed > .004 and np.count_nonzero(supported) >= 2
        if np.count_nonzero(supported) >= 2:
            shear = self.state.shear_magnitude[supported].sum()
            ratio = float(shear/max(.25, self.state.delayed_normal[supported].sum()))
            self.samples.append(ratio)
        ratio = float(np.mean(self.samples)) if self.samples else 0.
        if phase in ('CLOSING', 'ACQUIRE', 'CONTACT_ESTABLISHMENT', 'LIFT', 'LIFTING'):
            self.label = 'shared_acquisition_reference'
            return np.asarray(self.g.acquisition_force)*self.state.nominal_weights
        if self.target_ratio is None:
            self.label = 'ratio_initialization'
            if slipping or np.count_nonzero(supported) < 3:
                if slipping:
                    self.boost += self.g.slip_boost_rate*self.g.decision_period
                self.stable_since = None
            else:
                self.boost *= np.exp(-self.g.decision_period/1.5)
                if self.stable_since is None:
                    self.stable_since = self.state.time
                if self.state.time-self.stable_since >= 1.:
                    self.target_ratio = max(.025, ratio)
                    self.scalar = self.g.base+self.boost
            target = self.g.base+self.boost
        else:
            self.label = 'ratio_feedback'
            error = ratio-self.target_ratio
            integral = np.clip(self.integral+error*self.g.decision_period, -.35, .75)
            raw = self.scalar+self.g.ratio_kp*error+self.g.ratio_ki*integral
            if self.g.ratio_floor <= raw <= 5. or raw > 5. and error < 0. or raw < self.g.ratio_floor and error > 0.:
                self.integral = integral
            target = np.clip(raw, self.g.ratio_floor, 5.)
        self.debug.update(target_ratio=self.target_ratio, ratio=ratio)
        return target*self.state.nominal_weights


class MigratedUnified(HandControllerMixin, UnifiedCoordinator):
    def _decide(self, frame, phase):
        started = time.perf_counter()
        h = self.g.horizon
        nr, dimension = 5*h, 6*h
        P, c, U, uc = self._trajectory_model()
        PN = np.column_stack((P, np.zeros((nr, h))))
        UA = np.column_stack((U, np.zeros((nr, h))))
        R = np.column_stack((np.eye(nr), np.zeros((nr, h))))
        slack_rows = np.column_stack((np.zeros((h, nr)), np.eye(h)))
        active, wrench, shear_wrench, capacity, safe_mu = self._geometry()
        selected = self._support_selection(active, capacity)
        acquisition = phase in ('CLOSING', 'ACQUIRE', 'CONTACT_ESTABLISHMENT')
        lifting = phase in ('LIFT', 'LIFTING')
        rebuilding = (not acquisition and (len(selected) < 3 or
            not all(active[i] and self.state.normal[i] >= self.g.support_contact_floor-.025 for i in selected)))
        profile_reference = ((np.asarray(self.g.acquisition_force) if acquisition or lifting or rebuilding
                              else np.full(5, self.g.base))*self.state.nominal_weights)
        Q = np.eye(dimension)*1e-6
        q = np.zeros(dimension)

        def objective(matrix, offset, weight):
            nonlocal Q, q
            if weight <= 0.:
                return
            Q += 2.*weight*(matrix.T@matrix)
            q += 2.*weight*(matrix.T@offset)

        if self.name == 'unified_fixed_stage':
            contact_weight, continuity_scale = 1., 1.
        elif acquisition or lifting or rebuilding:
            contact_weight, continuity_scale = 4., .45
        elif self.state.slip_speed > .004:
            contact_weight, continuity_scale = .0, .45
        else:
            contact_weight, continuity_scale = .0, 1.
        objective(R, np.zeros(nr), self.g.force_cost)
        objective(R, -np.tile(profile_reference, h), contact_weight)
        if self.name != 'unified_no_geometry':
            actual_normal_wrench = wrench@self.state.normal
            GW = np.kron(np.eye(h), wrench)
            weights = np.r_[np.full(3, self.g.balance_weight), np.full(3, self.g.moment_weight)]
            weights = np.tile(np.sqrt(weights), h)
            objective(weights[:, None]*(GW@PN), weights*(GW@c-np.tile(actual_normal_wrench, h)), 1.)
        D = np.kron(np.eye(h)-np.eye(h, k=-1), np.eye(5))
        if self.name != 'unified_no_continuity':
            previous_r = np.zeros(nr)
            previous_r[:5] = self.force_reference
            objective(D@R, -previous_r, self.g.reference_continuity*continuity_scale)
            previous_n = np.zeros(nr)
            previous_n[:5] = self.state.normal
            objective(D@PN, D@c-previous_n, self.g.force_continuity*continuity_scale)
            previous_u = np.zeros(nr)
            previous_u[:5] = self.previous_action
            action_scale = np.tile(self.state.force_gain, h)
            objective(action_scale[:, None]*(D@UA), action_scale*(D@uc-previous_u),
                      self.g.action_continuity*continuity_scale)
        objective(slack_rows, np.zeros(h), self.g.support_slack_cost)
        identity = np.eye(nr)
        cumulative = np.kron(np.tril(np.ones((h, h))), np.eye(5))
        sums = np.kron(np.eye(h), np.ones((1, 5)))
        caps = np.maximum(0., self.limits-self.g.force_margin)
        budget = max(0., self.budget-2.*self.g.force_margin)
        rows = [R, -R, sums@R, UA, -UA, cumulative@UA, -cumulative@UA,
                PN, -PN, sums@PN, -slack_rows]
        bounds = [np.tile(caps, h), np.zeros(nr), np.full(h, budget),
                  np.full(nr, self.g.close_rate*self.g.decision_period)-uc,
                  np.full(nr, self.g.open_rate*self.g.decision_period)+uc,
                  np.tile(self.state.closure_max-self.state.alpha, h)-cumulative@uc,
                  np.tile(self.state.alpha-self.state.closure_min, h)+cumulative@uc,
                  np.tile(caps, h)-c, c, np.full(h, budget)-sums@c, np.zeros(h)]
        blocked = ~self.state.valid | ~self.state.closure_permission
        if np.any(blocked):
            rows.append(UA[np.tile(blocked, h)])
            bounds.append(-uc[np.tile(blocked, h)])
        force_floor = np.zeros((h, 5))
        reference_floor = np.zeros((h, 5))
        if not acquisition:
            force_floor, reference_floor = self._support_floors(P,c,caps,selected,active)
            rows += [-R, -PN]
            bounds += [-reference_floor.ravel(), c-force_floor.ravel()]
        load_demand = (self.state.load*(1.+self.g.support_reserve_fraction)
                       +self.g.load_uncertainty_weight*self.state.load_uncertainty+self.g.support_buffer
                       +self.g.slip_load_gain*min(.04, self.state.slip_speed))
        tangent_demand = ((1.+self.g.support_reserve_fraction)*self.state.shear_magnitude
                          +self.g.slip_load_gain*np.minimum(.015, self.state.slip_magnitude)/5.)
        # Measured shear demand and potential capacity are inequalities with
        # explicit reserve slack. They are not fictitious realized force vectors.
        if np.count_nonzero(active) >= 3 and active[0]:
            for j in range(h):
                for i in np.flatnonzero(active):
                    row = -safe_mu[i]*PN[5*j+i].copy()
                    row[nr+j] = -1.
                    rows.append(row[None, :])
                    bounds.append(np.array([safe_mu[i]*c[5*j+i]-tangent_demand[i]]))
                row = -(capacity@PN[5*j:5*j+5])
                row[nr+j] = -1.
                rows.append(row[None, :])
                bounds.append(np.array([capacity@c[5*j:5*j+5]-load_demand]))
        A, b = np.vstack(rows), np.concatenate(bounds)
        feasible = []
        for reference in (np.tile(self.state.normal,h), self.reference_plan,
                          np.tile(self.force_reference,h), np.tile(project_force(profile_reference,caps,budget),h)):
            trial = np.r_[reference, np.zeros(h)]
            # Slack only relaxes the declared support reserve, never the
            # pressure, action-domain, cumulative-coordinate or force-floor rows.
            for j in range(h):
                includes = A[:,nr+j] < -.5
                if np.any(includes):
                    trial[nr+j] = max(0., float(np.max(A[includes]@trial-b[includes])))+1e-8
            if np.max(A@trial-b) <= 1e-9:
                feasible.append(trial)
        seed = min(feasible,key=lambda x:float(.5*x@Q@x+q@x)) if feasible else None
        result = solve_qp(Q,q,A,b,self.g.qp_passes,self.g.qp_tolerance,feasible_initial=seed)
        if not result.converged:
            self.numerical_failures += 1
        if result.residual > 1e-8:
            if seed is not None:
                result = QPResult(seed,max(0.,float(np.max(A@seed-b))),result.iterations,
                                  'numerical_failure_feasible_fallback',False,{'optimizer_status':result.status})
            else:
                reference = project_force(profile_reference,caps,budget)
                self.label = 'contact_reestablishment' if rebuilding else 'trajectory_feasibility_unknown'
                action = self._baseline_action(reference)
                self.pause_transport = True
                self.debug.update(trajectory_qp_status=result.status,trajectory_qp_residual=result.residual,
                                  support_reserve_slack=None,contact_reestablishment=bool(rebuilding),
                                  selected_support_contacts=list(selected),local_force_reference=profile_reference.tolist())
                return action
        reference = result.x[:5]
        self.reference_plan = result.x[:nr].copy()
        self.force_reference = reference.copy()
        reserve_slack = result.x[nr:].copy()
        planned_prediction = c[:5]+P[:5]@result.x[:nr]
        action = self._baseline_action(reference)
        # Common execution reports the same requested reference semantics as
        # every baseline, with a separately timestamped actual-force forecast.
        self.desired = reference.copy()
        self.pause_transport |= rebuilding or float(np.max(reserve_slack)) > .20
        self.label = ('contact_reestablishment' if rebuilding else 'support_reserve_limited'
                      if np.max(reserve_slack) > .20 else 'contact_establishment'
                      if acquisition else 'disturbance_recovery' if self.state.slip_speed > .004 else 'stable_hold')
        self.debug.update(trajectory_qp_status=result.status,trajectory_qp_optimal=result.converged,
                          trajectory_qp_residual=result.residual,trajectory_qp_iterations=result.iterations,
                          qp_seconds=time.perf_counter()-started,
                          local_force_reference=profile_reference.tolist(),force_reference=reference.tolist(),
                          planned_actual_force=planned_prediction.tolist(),support_demand=float(load_demand),
                          support_reserve_fraction=self.g.support_reserve_fraction,
                          support_reserve_slack=reserve_slack.tolist(),
                          support_slack_is_capability_certificate=False,
                          predicted_support=float(capacity@self.predicted),geometry_contacts=active.tolist(),
                          selected_support_contacts=list(selected),predicted_support_floor=force_floor.tolist(),
                          support_contact_floor=self.g.support_contact_floor,contact_reestablishment=bool(rebuilding),
                          estimated_actual_normal_wrench=(wrench@self.state.normal).tolist(),
                          measured_shear_wrench=shear_wrench.tolist(),
                          characteristic_length=self.g.characteristic_length,
                          stage_weights=[contact_weight,continuity_scale],
                          force_prediction=self.predicted.tolist(),
                          local_model='delay_aligned_force_offset_observer_and_shared_force_trajectory_executor')
        return action

