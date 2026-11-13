"""Algebraic interface oracles; these do not certify real hand dynamics."""
from pathlib import Path
import json,hashlib
import numpy as np
from force_model import ForceResponse,verify_algebra

ROOT=Path(__file__).resolve().parent
def main():
    checks=verify_algebra()
    B=np.array([[-2.,1.,0.,0.,0.],[.3,3.,.2,0.,0.],
                [0.,-.4,2.5,.3,0.],[0.,0.,.2,2.,.2],[.2,0.,0.,-.3,1.8]])
    model=ForceResponse(B,.04);n=np.array([2.,1.,1.5,.7,.9]);d=.02;h=3
    u=np.array([[.01,-.02,.01,0.,.01],[0.,.01,-.01,.01,0.],[-.01,0.,.02,0.,-.01]])
    P,c=model.action_prediction(n,h,d);alpha=np.zeros(5);latent=np.zeros(5);actual=[]
    a=np.exp(-.005/model.tau)
    for action in u:
        for _ in range(4):alpha+=action/4;latent=a*latent+(1-a)*alpha
        actual.append(n+B@latent)
    checks['increment_history_independent_oracle']=bool(np.allclose(P@u.ravel()+c,np.ravel(actual),atol=1e-12))
    # With a negative gain, opening can INCREASE a force: no diagonal certificate.
    checks['opening_not_assumed_minimum_force']=bool((B@np.array([-.01,0,0,0,0]))[0]>0.)
    active=np.array([1,0,1,1,0],bool);search=np.array([0,.003,0,0,.002])
    r=np.arange(15).reshape(h,5)/20+1.
    P,c,U,uc=model.reference_prediction(n,h,d,.2,active,search)
    prev=n.copy();expected=[];expected_actions=[];alpha=np.zeros(5);latent=np.zeros(5)
    L=.2*model.inverse(active=active);effective_B=active[:,None]*B
    for ref in r:
        action=L@(ref-prev)+search
        for _ in range(4):alpha+=action/4;latent=a*latent+(1-a)*alpha
        prev=n+effective_B@latent
        expected_actions.append(action);expected.append(prev.copy())
    checks['partial_contact_search_prediction']=bool(np.allclose(P@r.ravel()+c,np.ravel(expected),atol=1e-12))
    checks['partial_contact_search_action']=bool(np.allclose(U@r.ravel()+uc,np.ravel(expected_actions),atol=1e-12))
    other=r.copy();other[:,~active]+=100.
    checks['unobserved_force_reference_has_no_effect']=bool(np.allclose(U@r.ravel()+uc,U@other.ravel()+uc,atol=1e-12))
    modeled=np.ones(5,bool);feedback=np.array([1,0,1,1,1],bool)
    q0=np.array([.8,.7,.9,1.,.8]);z0=q0-np.array([.002,.001,.003,.002,.001])
    P,c,U,uc=model.reference_prediction(n,h,d,.2,modeled,alpha=q0,latent=z0,feedback_active=feedback)
    first=(U@r.ravel()+uc)[:5]
    direct=.2*model.inverse(active=feedback)@(r[0]-n)
    AP,ac=model.action_prediction(n,h,d,modeled,alpha=q0,latent=z0)
    actions=np.zeros(15);actions[:5]=direct
    checks['partial_dropout_reference_and_executor_same_first_action']=bool(np.allclose(first,direct,atol=1e-12))
    checks['partial_dropout_modeled_missing_output_still_propagates']=bool(np.allclose((P@r.ravel()+c)[:5],(AP@actions+ac)[:5],atol=1e-12))
    for label,bad in [('nan',np.full((5,5),np.nan)),('shape',np.eye(4))]:
        try:ForceResponse(bad,.01);checks[f'reject_{label}']=False
        except ValueError:checks[f'reject_{label}']=True
    result={'passed':all(checks.values()),'checks':checks,
            'scope':'Synthetic matrix recurrence and input contracts only; physical admission is separate.',
            'source_sha256':hashlib.sha256((ROOT/'force_model.py').read_bytes()).hexdigest()}
    (ROOT/'review/force_model_algebra.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
