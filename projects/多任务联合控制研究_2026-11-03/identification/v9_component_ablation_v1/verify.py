"""Pure saved-public-stream endpoint equivalence; optional zero physics step."""
from pathlib import Path
from copy import deepcopy
from dataclasses import fields
from datetime import datetime,timezone
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from identification.v9_component_ablation_v1.ablation_runtime import ComponentSession,PACKAGE,read_plan,source_identity,sha
from identification.v9_component_ablation_v1.component_controller import ComponentController,VARIANTS
from v9.controller import JointTaskController
from v9.runtime import V9Session,PublicPreviewFollower
from controller_adapter import create_controller
from contracts import SensorFrame

def frame_from_json(text):
    d=json.loads(str(text));names={f.name for f in fields(SensorFrame)}
    def value(v):
        if isinstance(v,list):
            a=np.asarray(v)
            if a.dtype.kind in 'biuf':return a
        return v
    f=SensorFrame(**{k:value(v) for k,v in d.items() if k in names})
    for k,v in d.items():
        if k not in names:setattr(f,k,value(v))
    f.dt=.005;return f

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--zero-step',action='store_true');args=parser.parse_args()
    plan=read_plan();before=source_identity();checks=[];replays=[]
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    for scene in plan['environments']:
        for variant,method in [('P0H0','ratio_preview'),('P1H1','joint')]:
            folder=ROOT/'output/development_v9_transport_v1/runs'/f'{scene}__{method}__610212'
            m=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            with np.load(folder/'trace.npz',allow_pickle=False) as z:t=z['time'];obs=z['observation_json'];requests=z['task_request_json']
            g=m['actual_ratio_task_gains'] if method=='joint' else m['gains']
            c=ComponentController(m['calibration'],m['limits'],m['budget'],deepcopy(g),variant)
            ref=(JointTaskController(m['calibration'],m['limits'],m['budget'],deepcopy(g)) if method=='joint'
                 else create_controller('ratio',m['calibration'],m['limits'],m['budget'],deepcopy(g)))
            follower=PublicPreviewFollower();warm=np.flatnonzero((t>=7.3-1e-9)&(t<7.5-1e-9));hold=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9))
            for i in warm:
                f=frame_from_json(obs[i]);c.state.update(deepcopy(f),.005);ref.state.update(deepcopy(f),.005)
            mismatches=[];state_diffs=[];risk_seen=0
            for i in hold:
                f=frame_from_json(obs[i]);task=json.loads(str(requests[i]));c.set_task_request(task)
                if method=='joint':ref.set_task_request(task)
                a=ref.step(deepcopy(f),'HOLDING','hold');b=c.step(deepcopy(f),'HOLDING','hold')
                expected_motion=(ref.motion_reference if method=='joint' else follower.step(task,pause_transport=bool(a.pause_transport)))
                if not np.array_equal(a.delta_closure,b.delta_closure) or not np.array_equal(expected_motion,c.motion_reference) or a.pause_transport!=b.pause_transport:
                    mismatches.append(dict(time=float(t[i]),force_max=float(np.max(abs(a.delta_closure-b.delta_closure))),motion_max=float(np.max(abs(expected_motion-c.motion_reference)))))
                rs=ref.state.force_state if method=='joint' else ref.state
                for name in ('normal','valid','alpha','latent_alpha','force_uncertainty'):
                    if not np.array_equal(getattr(rs,name),getattr(c.state.force_state,name),equal_nan=True):state_diffs.append([float(t[i]),name])
                risk_seen+=bool(b.debug['component_ablation']['counterfactual_risk_hold'])
                if variant=='P0H0' and b.debug['component_ablation']['actual_additional_risk_hold']:raise AssertionError('H0 applied extra hold')
            ck(scene+'/'+variant+'/all900_public_action_motion_pause_exact',not mismatches,mismatches[:5])
            ck(scene+'/'+variant+'/unique_force_state_exact',not state_diffs,state_diffs[:5])
            ck(scene+'/'+variant+'/risk_evidence_preserved',risk_seen>0,risk_seen)
            if variant=='P0H0':
                ck(scene+'/projection0_no_highlevel_rollback',b.debug.get('reference_memory_rollback') is False)
            replays.append(dict(scenario=scene,variant=variant,reference_method=method,trace_sha256=sha(folder/'trace.npz'),warmup=len(warm),policy_ticks=len(hold),mismatches=mismatches,force_state_differences=state_diffs,risk_evidence_ticks=risk_seen))
    # Route-level falsification: P0 must not mask a novel positive proposal or
    # revert baseline reference memory, independent of the scenario signal.
    from types import MethodType
    from vendor.frozen_controllers import ControlAction
    raw=ComponentController(m['calibration'],m['limits'],m['budget'],deepcopy(g),'P0H0')
    f=frame_from_json(obs[int(hold[0])]);raw.latest_frame=f;raw.set_task_request(json.loads(str(requests[int(hold[0])])))
    raw._known=lambda:(np.zeros(5,bool),np.ones(5,bool));raw._risk=lambda known:(np.array(m['limits']),m['budget'],dict(material_risk=False,slip_risk=False))
    proposal=np.full(5,.001)
    def original_action(self,*a,**kw):self.integral=.321;self.boost=.456;return ControlAction(proposal.copy(),np.full(5,np.nan),'mock_original',False,{})
    raw.nominal.step=MethodType(original_action,raw.nominal);out=raw.step(f,'HOLDING','hold')
    ck('P0_no_novel_mask_minimum_or_reference_rollback',np.array_equal(out.delta_closure,proposal) and raw.nominal.integral==.321 and raw.nominal.boost==.456)
    if args.zero_step:
        for scene in plan['environments']:
            references={v:V9Session(method=method,scenario=scene,seed=610212,split='robustness',motion_mode='transport') for v,method in [('P0H0','ratio_preview'),('P1H1','joint')]}
            for variant in VARIANTS:
                s=ComponentSession(component_variant=variant,scenario=scene,seed=610212,split='robustness')
                ck(scene+'/'+variant+'/zero_step_route_joint',s.method=='joint' and s.time==0. and s.tick==0 and not s.trace and not s.plant.substep_records and s.research_sources==before)
                if variant in references:
                    ref=references[variant]
                    ck(scene+'/'+variant+'/zero_step_initial_physical_exact',all(np.array_equal(getattr(s.plant.data,k),getattr(ref.plant.data,k)) for k in ('qpos','qvel','ctrl')) and s.config==ref.config and s.sensors.rng.bit_generator.state==ref.sensors.rng.bit_generator.state)
    ck('all_source_identities_unchanged',source_identity()==before)
    result=dict(schema='v9_component_ablation_software_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(x['passed'] for x in checks),checks=checks,public_replays=replays,source_sha256=before,zero_step=args.zero_step,physics_steps_executed=0,
        causal_scope='projection package includes novel mask/antiwindup, hold package only disables additional supervisory freeze; baseline pause remains. No individual novel-mask causal inference. Full16s physical endpoint exactness still requires actual8 execution.')
    dest=PACKAGE/'qa';dest.mkdir(exist_ok=True);out=dest/('verification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],physics_steps=0),ensure_ascii=False));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
