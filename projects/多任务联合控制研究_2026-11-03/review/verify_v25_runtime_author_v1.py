"""Finite author software gate; fake callbacks and synthetic archives, no plant."""
from pathlib import Path
from types import SimpleNamespace,MappingProxyType
from unittest.mock import patch
import argparse,hashlib,json,sys,tempfile,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out;assert not out.exists()
    checks=[];forbidden=[];error=None
    def ck(name,good):
        checks.append(dict(name=name,passed=bool(good)))
        if not good:raise AssertionError(name)
    def reject(name,fn):
        try:fn();good=False
        except (ValueError,TypeError):good=True
        ck(name,good)
    owned={n:sha(ROOT/'v25_response_identification'/n) for n in ('__init__.py','design.py','runtime.py','run_identification.py')}
    try:
        import research_imports,mujoco
        from v25_response_identification import runtime as mod
        from v25_response_identification.design import ProbeSpec,PROBE_ARMS
        from v13_a1.codec import Reference21,ACTUATORS21
        from v14_coord.design import reference_rates
        # This runner import must occur only once the fork author provides a
        # stable syntax/import window. Tests never call fork_session or Session.
        from v25_response_identification import run_identification as runner
        def deny(*a,**kw):forbidden.append('physics_or_FK');raise AssertionError(forbidden[-1])
        functions=('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce','MjData')
        with patch.multiple(mujoco,**{n:deny for n in functions if hasattr(mujoco,n)}):
            ck('exact_five_declared_arms',len(PROBE_ARMS)==5 and all(ProbeSpec(a,12.).arm==a for a in PROBE_ARMS))
            reject('no_hidden_21axis_expansion',lambda:ProbeSpec('thj2_plus',12.))
            reject('boolean_origin_rejected',lambda:ProbeSpec('zero',True))
            reject('offgrid_origin_rejected',lambda:ProbeSpec('zero',12.0001))
            reject('changed_horizon_rejected',lambda:ProbeSpec('zero',12.,hold_s=.3))
            reject('changed_amplitude_rejected',lambda:ProbeSpec('zero',12.,amplitude_fraction=.1))
            class Nominal:
                def __init__(self):self.decision_evidence=[];self.value=np.zeros(21);self.step=reference_rates('release')*.005;self.lo=-self.step;self.hi=self.step;self.receipts=[]
                def action(self,**kw):
                    self.decision_evidence.append(dict(actual_reference_lower21=self.lo.copy(),actual_reference_upper21=self.hi.copy(),geometry_proposal_accepted=True,emitted_hierarchy_score=dict(available=True)))
                    return Reference21(self.value,float(kw['frame'].time)),dict(original_nominal=True)
                def reconcile(self,receipt):self.receipts.append(receipt)
            def act(graph,t=12.,gen=0):return graph.action(frame=SimpleNamespace(time=t),generation=gen)
            nominal=Nominal();g=mod.ProbeCoordinator(nominal,ProbeSpec('lift_plus',12.));ref,_=act(g)
            expected=.5*reference_rates('release')[18]*.005
            ck('actual_domain_scaled_positive_lift_step',ref.values[18]==expected and np.count_nonzero(ref.values)==1)
            ck('changed_probe_does_not_borrow_nominal_certificate',g.records[-1]['nominal_task_certificate_applies_to_emitted_reference'] is False)
            receipt=SimpleNamespace(executed=ref,engine=SimpleNamespace(values=ref.values),mode='release',mode_generation=0,execution_changed=False,stale_plan_replaced=False)
            g.reconcile(receipt)
            ck('same_real_receipt_goes_to_nominal_memory_and_probe_record',nominal.receipts==[receipt] and g.records[-1]['actual_receipt']['actual_reference21']==ref.values.tolist())
            nominal.hi[18]=expected/2;ref,_=act(g,12.005)
            ck('fresh_domain_clip_retains_raw_target_and_reports_difference',ref.values[18]==expected/2 and g.records[-1]['raw_probe_target21'][18]==expected and g.records[-1]['requested_target_clipped_by_fresh_TASK'])
            g=mod.ProbeCoordinator(Nominal(),ProbeSpec('wrj1_minus',12.));ref,_=act(g)
            ck('wrist_sign_and_axis_not_lift',ref.values[20]<0 and np.count_nonzero(ref.values)==1)
            nominal=Nominal();g=mod.ProbeCoordinator(nominal,ProbeSpec('zero',12.));act(g);nominal.value[0]=nominal.step[0]/4;ref,_=act(g,12.005)
            ck('experimental_zero_is_fixed_carrier_not_adaptive_policy_bridge',ref.values[0]==0 and nominal.value[0]!=0 and not g.records[-1]['nominal_task_certificate_applies_to_emitted_reference'])
            act(g,12.010,1);ck('generation_change_cancels_not_reschedules',g.terminal_status=='CANCELLED_MODE_GENERATION_CHANGED' and not g.records[-1]['active'])
            nominal=Nominal();nominal.hi[18]=0;g=mod.ProbeCoordinator(nominal,ProbeSpec('lift_plus',12.));act(g)
            ck('zero_permission_is_explicit_unsupported',g.terminal_status=='UNSUPPORTED_ZERO_ACTUAL_ORIGIN_AUTHORITY' and not g.records[-1]['active'])
            class UnknownNominal(Nominal):
                def action(self,**kwargs):
                    result=super().action(**kwargs);self.decision_evidence[-1]['geometry_proposal_accepted']=False;return result
            g=mod.ProbeCoordinator(UnknownNominal(),ProbeSpec('zero',12.));act(g)
            ck('same_reference_does_not_invent_a_missing_nominal_certificate',not g.records[-1]['nominal_task_certificate_applies_to_emitted_reference'])
            g=mod.ProbeCoordinator(Nominal(),ProbeSpec('zero',12.));act(g);act(g,12.2);ck('fixed_window_finishes_without_wallclock_catchup',g.terminal_status=='PROBE_WINDOW_COMPLETED')
            ck('nonempty_mappingproxy_serializes_without_state_loss',mod.plain(MappingProxyType({'nested':MappingProxyType({'v':np.array([2.])})}))=={'nested':{'v':[2.]}})
            frame=SimpleNamespace(time=12.,robot_position25=np.zeros(25),robot_velocity25=np.zeros(25),robot_actuator_axes21=ACTUATORS21)
            calls=[];events=[];obj=mod.ResponseSession.__new__(mod.ResponseSession)
            obj.plant=SimpleNamespace(reference_snapshot=lambda:Reference21(np.zeros(21),12.),engine_snapshot=lambda:SimpleNamespace(values=np.zeros(21)),last_execution_receipt21=None)
            obj.response_observer=SimpleNamespace(on_receipt=lambda receipt,snapshot:events.append((receipt,snapshot)))
            def parent(self,obs,*args):
                calls.append('single_parent_execute');self.plant.last_execution_receipt21=SimpleNamespace(executed=Reference21(np.ones(21),12.));return 'same_parent_result'
            for name in ('_execute_joint','_lifecycle_step'):
                calls.clear();events.clear()
                with patch.object(mod.FrozenSession,name,parent):
                    answer=obj._execute_joint(frame,'hold') if name=='_execute_joint' else obj._lifecycle_step(frame)
                ck(name+'_one_parent_then_real_receipt_only',answer=='same_parent_result' and len(calls)==len(events)==1 and np.array_equal(events[0][1]['reference21'],np.zeros(21)))
            observed=[];adapter=SimpleNamespace(observe=lambda *a:deny());obj.release_adapter=adapter;obj.response_observer=SimpleNamespace(observe=lambda f,a:observed.append((f,a)))
            result=(object(),object(),frame,'original_common_json',2.)
            with patch.object(mod.FrozenSession,'observe_public',lambda self:result):answer=obj.observe_public()
            ck('passive_observer_no_resample_no_frame_replacement',observed==[(frame,adapter)] and all(a is b for a,b in zip(answer[:4],result[:4])) and answer[-1]>=2.)
            reject('blind_release_rejected_before_any_constructor',lambda:mod.ResponseSession(receiver_policy='blind'))
            assignment=dict(mode='bridge-fork',method='ratio_effort',scenario='nominal',seed=619741,motion_mode='hold',origin_time_s=12.)
            runner.validate_assignment(assignment,'bridge-fork');ck('registered_assignment_accepted',True)
            reject('confirmation_seed_split_rejected',lambda:runner.validate_assignment(dict(assignment,split='confirmation'),'bridge-fork'))
            reject('unexpected_hidden_assignment_parameter_rejected',lambda:runner.validate_assignment(dict(assignment,amplitude=1.),'bridge-fork'))
            reject('two_origin_definitions_rejected',lambda:runner.validate_assignment(dict(assignment,origin_trigger='pre_receiver'),'bridge-fork'))
            reject('boolean_registered_origin_rejected',lambda:runner.advance_to_origin(SimpleNamespace(),dict(origin_time_s=True)))
            with tempfile.TemporaryDirectory(prefix='v25_reader_pure_') as td:
                td=Path(td);n=3201;times=np.arange(n)*.005
                def save(path,padding=4,scalar=0.,timing=0.,char='same'):
                    debug=json.dumps(dict(solver_timing_ms=dict(whole_decision=timing),algorithm='unchanged'))
                    np.savez(path,time=times,normal=np.full(n,scalar),legacy_common_sensor_json=np.full(n,char,dtype=f'<U{padding}'),debug_json=np.full(n,debug),control_ms=np.full(n,timing))
                save(td/'a.npz');save(td/'b.npz',500,timing=1.)
                ck('bridge_only_declared_timing_and_unicode_padding_ignored',runner.compare_trace(td/'a.npz',td/'b.npz')['passed'])
                save(td/'b.npz',500,scalar=np.nextafter(0.,1.));ck('bridge_single_numeric_bit_fails',not runner.compare_trace(td/'a.npz',td/'b.npz')['passed'])
                save(td/'b.npz',500,char='sAme');ck('bridge_single_character_fails',not runner.compare_trace(td/'a.npz',td/'b.npz')['passed'])
            ck('no_pipeline_calls',not forbidden)
        ck('owned_production_unchanged_during_pure_gate',owned=={n:sha(ROOT/'v25_response_identification'/n) for n in owned})
    except Exception as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    result=dict(schema='v25_runtime_author_pure_v1',passed=error is None and all(c['passed'] for c in checks),checks=checks,error=error,
        source_sha256={f'v25_response_identification/{k}':v for k,v in owned.items()},physics_steps=0,model_constructions=0,
        optimizer_calls=0,forbidden_calls=forbidden,checker_sha256=sha(__file__),full198_identity_gate=False,
        scope='Integration-author synthetic protocol/receipt/bridge comparison checks only. Actual constructor and populated fork/full16 bridges remain untested here.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result['passed'],checks=len(checks),error=error),ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
