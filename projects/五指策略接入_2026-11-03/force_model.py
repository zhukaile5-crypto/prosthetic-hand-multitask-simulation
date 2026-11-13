"""Shared signed/coupled local response model for the articulated hand.

This is an execution/observation model, not a grasp policy. Calibration must
come from the isolated hand foundation; no five-pad gain is a default.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ForceResponse:
    matrix: np.ndarray
    tau: float
    residual_bound: float = 0.
    source: str = 'explicit calibration'

    def __post_init__(self):
        matrix=np.asarray(self.matrix,dtype=float).copy()
        if matrix.shape != (5,5) or not np.isfinite(matrix).all():
            raise ValueError('A finite signed 5 by 5 response matrix is required')
        if not np.isfinite(self.tau) or self.tau<=0:
            raise ValueError('A positive calibrated response time is required')
        if self.residual_bound<0 or not np.isfinite(self.residual_bound):
            raise ValueError('Invalid calibration residual bound')
        matrix.flags.writeable=False
        object.__setattr__(self,'matrix',matrix)

    @property
    def scales(self):
        """Positive units for costs only; these are not signed force gains."""
        return np.maximum(.05,np.linalg.norm(self.matrix,axis=0))

    def inverse(self,damping=.02,active=None):
        """Bounded least-squares force-error to closure mapping, preserving sign."""
        active=np.ones(5,dtype=bool) if active is None else np.asarray(active,bool)
        indices=np.flatnonzero(active);result=np.zeros((5,5))
        if len(indices):
            B=self.matrix[np.ix_(indices,indices)]
            result[np.ix_(indices,indices)]=np.linalg.solve(
                B.T@B+float(damping)**2*np.eye(len(indices)),B.T)
        return result

    def _timing(self,period,execution_dt):
        steps=round(period/execution_dt)
        if steps<1 or abs(steps*execution_dt-period)>1e-10:
            raise ValueError('Decision period must contain whole execution ticks')
        a=np.exp(-execution_dt/self.tau)
        decay=a**steps
        # A decision increment is divided into `steps` equal actual commands.
        ramp=1.-sum(a**j for j in range(1,steps+1))/steps
        return decay,ramp

    def action_prediction(self,normal,horizon,period,active=None,*,alpha=None,latent=None,execution_dt=.005):
        """n_j = constant_j + P_j @ stacked interval closure increments."""
        active=np.ones(5) if active is None else np.asarray(active,float)
        B=active[:,None]*self.matrix
        decay,ramp=self._timing(period,execution_dt)
        alpha=np.zeros(5) if alpha is None else np.asarray(alpha,float)
        latent=alpha.copy() if latent is None else np.asarray(latent,float)
        P=np.zeros((5*horizon,5*horizon));c=np.zeros(5*horizon)
        qP=np.zeros((5,5*horizon));zP=np.zeros_like(qP)
        qC=alpha.copy();zC=latent.copy();normal=np.asarray(normal,float)
        for j in range(horizon):
            select=np.zeros_like(qP);select[:,5*j:5*j+5]=np.eye(5)
            zP=decay*zP+(1.-decay)*qP+ramp*select
            zC=decay*zC+(1.-decay)*qC
            qP=qP+select
            P[5*j:5*j+5]=B@zP
            c[5*j:5*j+5]=normal+B@(zC-latent)
        return P,c

    def reference_prediction(self,normal,horizon,period,beta,active=None,search=None,*,alpha=None,latent=None,execution_dt=.005,feedback_active=None):
        """Propagate the SAME shared force tracker used by the first action.

        u_j = L (r_j-n_{j-1}) + search. Each u is then split across the
        actual execution ticks, with a calibrated first-order latent coordinate.
        The affine search is a public contact-acquisition action, not slip
        compensation. No sign/diagonal monotonicity is assumed.
        """
        active=np.ones(5,dtype=bool) if active is None else np.asarray(active,bool)
        feedback_active=active if feedback_active is None else np.asarray(feedback_active,bool)&active
        search=np.zeros(5) if search is None else np.asarray(search,float)
        B=active[:,None]*self.matrix
        # Both output and error-input channels of unobserved fingers are absent.
        L=float(beta)*self.inverse(active=feedback_active)
        decay,ramp=self._timing(period,execution_dt)
        alpha=np.zeros(5) if alpha is None else np.asarray(alpha,float)
        latent=alpha.copy() if latent is None else np.asarray(latent,float)
        P=np.zeros((5*horizon,5*horizon));c=np.zeros(5*horizon)
        U=np.zeros_like(P);uc=np.zeros_like(c)
        normal=np.asarray(normal,float)
        qP=np.zeros((5,5*horizon));zP=np.zeros_like(qP)
        qC=alpha.copy();zC=latent.copy()
        previous_P=np.zeros_like(qP);previous_c=normal.copy()
        for j in range(horizon):
            select=np.zeros((5,5*horizon));select[:,5*j:5*j+5]=np.eye(5)
            uP=L@(select-previous_P);uC=-L@previous_c+search
            zP=decay*zP+(1.-decay)*qP+ramp*uP
            zC=decay*zC+(1.-decay)*qC+ramp*uC
            qP=qP+uP;qC=qC+uC
            next_P=B@zP
            next_c=normal+B@(zC-latent)
            P[5*j:5*j+5]=next_P;c[5*j:5*j+5]=next_c
            U[5*j:5*j+5]=uP;uc[5*j:5*j+5]=uC
            previous_P,previous_c=next_P,next_c
        return P,c,U,uc

    def propagate_delayed(self,measured,alpha_now,alpha_at_measurement):
        return np.maximum(0.,np.asarray(measured)+self.matrix@(
            np.asarray(alpha_now)-np.asarray(alpha_at_measurement)))


def verify_algebra():
    """Independent recurrence oracle, including negative/cross-coupled response."""
    B=np.array([[-2.,1.,0.,0.,0.],[.3,3.,.2,0.,0.],
                [0.,-.4,2.5,.3,0.],[0.,0.,.2,2.,.2],[.2,0.,0.,-.3,1.8]])
    model=ForceResponse(B,.04)
    n=np.array([2.,1.,1.5,.7,.9]);reference=np.arange(15).reshape(3,5)/20+1.
    P,c,U,uc=model.reference_prediction(n,3,.02,.2)
    actual=[];actions=[];previous=n.copy();L=.2*model.inverse()
    q=np.zeros(5);z=np.zeros(5);decay=np.exp(-.005/.04)
    for r in reference:
        u=L@(r-previous);actions.append(u)
        for _ in range(4):q+=u/4;z=decay*z+(1.-decay)*q
        previous=n+B@z;actual.append(previous.copy())
    checks={'signed_negative_gain_preserved':bool(model.matrix[0,0]<0),
        'coupled_reference_prediction':bool(np.allclose(P@reference.ravel()+c,np.ravel(actual),atol=1e-12)),
        'same_action_model':bool(np.allclose(U@reference.ravel()+uc,np.ravel(actions),atol=1e-12)),
        'zero_increment_zero_change':bool(np.allclose(model.propagate_delayed(n,n,n),n))}
    if not all(checks.values()):raise AssertionError(checks)
    return checks


if __name__=='__main__':
    import json
    print(json.dumps(verify_algebra()))
