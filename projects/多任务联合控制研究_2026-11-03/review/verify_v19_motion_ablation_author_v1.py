"""Finite zero-tick and archived public task-package author check; no physics."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch
import argparse,ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from v19_contact_reference.adapter import ContactMotionAdapter
from v19_contact_reference.runtime import source_identity as full_identity
from v19_motion_ablation.adapter import NoMotionAdapter,ABLATION_ROLE
from v19_motion_ablation.runtime import ReleaseSession,source_identity,source_files
import mujoco

def equal(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        x=np.asarray(a);y=np.asarray(b);return x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b

class NormalizeRunner(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module=='v19_motion_ablation.runtime':node.module='v19_contact_reference.runtime'
        return node
    def visit_Constant(self,node):
        if not isinstance(node.value,str):return node
        if node.value=='Assigned FullV19 combined motion-effort-task ablation; same holding, progress, normal/reference QP, governor, branch, devices, plant, permissions, rates, metrics and16s schedule; actual observations retained and no force prediction':
            node.value='Frozen holding; new current5ms explicit-reference and native-branch contact-transfer task only for coordinated release. Original devices, plant, permissions, rates, metrics and16s schedule; no contact-force prediction'
        elif node.value=='Assigned motion-task ablation with complete frozen FullV19 source parent plus four independent namespace files':
            node.value='V19ContactReference full21 current-state task coordinator plus frozen A1/native parent closure and foundation assets/evidence'
        else:
            node.value=node.value.replace('v19_motion_ablation_runtime_or_evaluation_failure','v19_release_runtime_or_evaluation_failure')
            node.value=node.value.replace('v19_motion_ablation','v19_contact_reference').replace('V19MotionAblation','V19ContactReference')
        return node

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    parent_before=full_identity(['ratio_effort']);identity_before=source_identity(['ratio_effort']);checks=[];details={}
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    ck('parent152 exact subset plus only four assigned namespace files',len(parent_before)==152 and len(identity_before)==156 and
       all(identity_before[k]==v for k,v in parent_before.items()) and
       set(identity_before)-set(parent_before)=={'v19_motion_ablation/'+n for n in ('__init__.py','adapter.py','runtime.py','run_development.py')})
    parent_ast=ast.parse((ROOT/'v19_contact_reference/run_development.py').read_text(encoding='utf-8'))
    child_ast=NormalizeRunner().visit(ast.parse((ROOT/'v19_motion_ablation/run_development.py').read_text(encoding='utf-8')))
    ck('runner control-flow assignment snapshot failures AST unchanged except explicit namespace and descriptive strings',
       ast.dump(parent_ast,include_attributes=False)==ast.dump(child_ast,include_attributes=False))
    # The one new branch must alter task coefficients, not sensor validity.
    fixture=dict(J=np.arange(50,dtype=float).reshape(2,25),y=np.array([1.,2.]),digits=np.array([0,1]),
        tau_motion25=np.linspace(-.2,.3,25),diagnostics=dict(motion_records=[dict(external_projection_available=False,relative_slip2_m_s=None)],
            robot_motion_records=[dict(current_robot_tangent_world_m_s=[.1,.2,.3])],motion_unknown=[dict(digit=0,reason='missing')]))
    frozen=deepcopy(fixture)
    with patch.object(ContactMotionAdapter,'current_contact_intents',return_value=deepcopy(fixture)) as mocked:
        result=NoMotionAdapter.__new__(NoMotionAdapter).current_contact_intents(SimpleNamespace())
    ck('real parent invoked once rather than replacing observer/geometry',mocked.call_count==1)
    ck('assigned disabled RHS zero25 and original real tau logged',np.array_equal(result['tau_motion25'],np.zeros(25)) and
       result['diagnostics']['original_motion_generalized_effort25']==fixture['tau_motion25'].tolist() and result['diagnostics']['motion_tasks_enabled'] is False)
    ck('unknown measured motion remains unknown and zero task not claimed zero speed',result['diagnostics']['motion_records'][0]['external_projection_available'] is False and
       result['diagnostics']['motion_records'][0]['relative_slip2_m_s'] is None and result['diagnostics']['zero_task_is_not_observed_zero_velocity'])
    ck('mock parent data and all original J-pressure-mask diagnostics untouched',equal(fixture,frozen) and
       all(equal(result[k],fixture[k]) for k in ('J','y','digits')) and
       all(equal(result['diagnostics'][k],v) for k,v in fixture['diagnostics'].items()))
    search_frame=SimpleNamespace()
    with patch.object(ContactMotionAdapter,'current_contact_intents',return_value=deepcopy(fixture)) as forwarded:
        search=NoMotionAdapter.__new__(NoMotionAdapter).current_contact_intents(search_frame,desired_lift_velocity_m_s=-.03272727272727272)
    ck('SEARCH governor velocity task target forwarded unchanged before task disable',
       forwarded.call_args.args==(search_frame,) and forwarded.call_args.kwargs==dict(desired_lift_velocity_m_s=-.03272727272727272) and
       search['diagnostics']['original_motion_generalized_effort25']==fixture['tau_motion25'].tolist())
    ck('CONTACT default zero velocity forwarded rather than deleting task metadata',mocked.call_args.kwargs==dict(desired_lift_velocity_m_s=0.))
    steps=[];forward_calls=[];original_forward=mujoco.mj_forward
    def forbidden(*a,**kw):steps.append(True);raise RuntimeError('forbidden integration or contact read')
    def initial_forward(*a,**kw):forward_calls.append(True);return original_forward(*a,**kw)
    guards=[patch('mujoco.'+name,side_effect=forbidden) for name in ('mj_step','mj_collision','mj_contactForce')]
    guards.append(patch('mujoco.mj_forward',side_effect=initial_forward))
    for guard in guards:guard.start()
    try:
        sessions=[]
        for method in ('ratio_effort','fixed_effort','fusion_effort'):
            s=ReleaseSession(method=method,scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated')
            meta=s.metadata();sessions.append(s)
            ck(method+'/zero actual ticks samples records and no integration',s.tick==0 and s.time==0. and len(s.trace)==0 and s.sensors.samples==0)
            ck(method+'/replacement adapter plus full parent provenance metadata',type(s.release_adapter) is NoMotionAdapter and
               meta['ablation_role']==ABLATION_ROLE and meta['motion_tasks_enabled'] is False and
               meta['frozen_full_v19_source_sha256']==parent_before and meta['research_source_sha256']==identity_before)
        ck('construction forward only unchanged parent initialization',len(forward_calls)==3 and len(steps)==0,
           dict(initial_parent_forward_calls=len(forward_calls),physics_steps=len(steps)))
        run=ROOT/'output/v18_small_nominal_transport_v1/runs/nominal__ratio_effort__619741'
        with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];encoded=z['observation_json']
        selection=np.flatnonzero((times>=13.9-1e-9)&(times<=13.925+1e-9))
        full=ContactMotionAdapter(sessions[0].plant.model,open18=np.concatenate(sessions[0].plant.open_pose),receiver_mount=sessions[0].receiver_mount)
        disabled=NoMotionAdapter(sessions[0].plant.model,open18=np.concatenate(sessions[0].plant.open_pose),receiver_mount=sessions[0].receiver_mount)
        with patch('mujoco.mj_forward',side_effect=RuntimeError('adapter must not call forward')):
            for i in selection:
                raw=json.loads(str(encoded[i]));values={k:(np.asarray(v) if isinstance(v,list) and k!='skin_patches' else v) for k,v in raw.items()}
                frame=SimpleNamespace(**values);full.observe(frame);disabled.observe(frame)
            expected=full.current_contact_intents(frame);actual=disabled.current_contact_intents(frame)
        ck('fixed real public packet produces nonzero measured parent motion task',float(frame.time)==13.925 and np.max(abs(expected['tau_motion25']))>0,
           dict(time_s=float(frame.time),parent_tau_norm=float(np.linalg.norm(expected['tau_motion25'])),public_history_packets=len(selection)))
        ck('fixed real packet ONLY RHS task changes with parent diagnostics exactly retained',
           set(expected)==set(actual) and all(equal(expected[k],actual[k]) for k in expected if k not in ('tau_motion25','diagnostics')) and
           all(equal(v,actual['diagnostics'][k]) for k,v in expected['diagnostics'].items()) and
           actual['diagnostics']['original_motion_generalized_effort25']==expected['tau_motion25'].tolist() and
           np.array_equal(actual['tau_motion25'],np.zeros(25)))
        with patch('mujoco.mj_forward',side_effect=RuntimeError('adapter must not call forward')):
            expected_search=full.current_contact_intents(frame,desired_lift_velocity_m_s=-.03272727272727272)
            actual_search=disabled.current_contact_intents(frame,desired_lift_velocity_m_s=-.03272727272727272)
        ck('real SEARCH target and all Full diagnostics remain exact with only final tau disabled',
           all(equal(v,actual_search['diagnostics'][k]) for k,v in expected_search['diagnostics'].items()) and
           all(equal(expected_search[k],actual_search[k]) for k in expected_search if k not in ('tau_motion25','diagnostics')) and
           actual_search['diagnostics']['original_motion_generalized_effort25']==expected_search['tau_motion25'].tolist() and
           np.array_equal(actual_search['tau_motion25'],np.zeros(25)))
        details['public_fixture']=dict(trace=str(run/'trace.npz'),trace_sha256=hashlib.sha256((run/'trace.npz').read_bytes()).hexdigest(),
            history_times_s=times[selection].tolist(),current_time_s=float(frame.time),parent_tau25=expected['tau_motion25'].tolist(),
            unchanged_original_motion_diagnostic_keys=list(expected['diagnostics']))
        details['initialization']=dict(parent_forward_calls=len(forward_calls),physics_steps=len(steps),samples_all3=[s.sensors.samples for s in sessions])
    finally:
        for guard in reversed(guards):guard.stop()
    parent_after=full_identity(['ratio_effort']);identity_after=source_identity(['ratio_effort'])
    ck('all parent152 and ablation156 source bytes unchanged through checks',parent_before==parent_after and identity_before==identity_after)
    payload=dict(schema='v19_motion_task_ablation_author_software_v1',passed=all(x['passed'] for x in checks),check_count=len(checks),checks=checks,details=details,
        source_sha256=identity_after,parent152_source_sha256=parent_after,source_count=len(identity_after),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_steps=0,contact_force_reads=0,registration_or_dispatch_performed=False,
        scope='package task ablation wiring, zero-tick constructors and one fixed public packet; not closed-loop results or a new selected algorithm')
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],sources=len(identity_after),
                         out=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest()),ensure_ascii=False))
    return 0 if payload['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
