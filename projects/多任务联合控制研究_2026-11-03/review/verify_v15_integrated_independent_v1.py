"""Independent integrated task wiring review on archived PUBLIC packets.

Construct robot model/private robot FK only. Never integrate, collide, read
contacts or write an actual engine. This is off-policy software replay.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,hashlib,json,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
EXPECTED='3c1e1e3333644d9452c20035e8032a4498453a6680a2f262d74943808e5c8911'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=True).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    source=ROOT/'v15_support/controller.py';sources={};checks=[];failures=[];records=[];forbidden=[];FK={};truth=[];cache={};calls=[];taskcalls=[];shapecalls=[];history=[]
    def ck(name,yes,detail=None):
        checks.append(dict(name=name,passed=bool(yes),detail=detail))
        if not yes:failures.append(name)
    def close(name,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float);err=float(np.max(abs(a-b),initial=0.));bound=256*np.finfo(float).eps*max(1.,float(np.sum(abs(a))),float(np.sum(abs(b))))
        ck(name,a.shape==b.shape and err<=bound,dict(error=err,arithmetic_bound=bound))
    class Frame(SimpleNamespace):
        def __getattribute__(self,name):
            if name in ('object_position','object_velocity','object_pose','object_mass','object_friction','target_id','target_contact_id','is_target','raw_qpos','true_qpos','true_mass','true_friction'):
                truth.append(name);raise AssertionError('forbidden truth read:'+name)
            return super().__getattribute__(name)
    try:
        watched=['v15_support/controller.py','v15_support/task_qp.py','v15_support/fusion_tasks.py','v15_support/posture_tasks.py','v14_coord/effort_domain.py','v14_coord/effort_realization.py','v11/robot_servo_effort.py','v12/observation.py','v12/intents.py','v12_native/controller.py']
        sources={x:sha(ROOT/x) for x in watched};ck('controller_final_hash',sources['v15_support/controller.py']==EXPECTED)
        if not checks[-1]['passed']:raise RuntimeError('Wrong integrated candidate')
        import research_imports,mujoco,hand_plant
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):
                def deny(*a,_n=name,**kw):forbidden.append(_n);raise AssertionError('forbidden '+_n)
                setattr(mujoco,name,deny)
        for name in ('mj_kinematics','mj_comPos','mj_jac'):
            old=getattr(mujoco,name)
            def counted(*a,_n=name,_f=old,**kw):FK[_n]=FK.get(_n,0)+1;return _f(*a,**kw)
            setattr(mujoco,name,counted)
        hand_plant.HandPlant.__init__=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No plant'))
        from foundation_config import FoundationConfig,OPEN
        from v13_a1.codec import Reference21,ReferenceActor21
        from v13_joint.lifecycle import ReleaseEntry,release_nominal
        from v14_coord.design import reference_rates
        from v11.robot_servo_effort import RobotServoEffort
        import v15_support.controller as cm
        import v12_native.controller as oldcm
        ReferenceActor21.apply=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('No engine writer'))
        oldcm.FusionIntent.step=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Old nested FusionIntent forbidden'))
        model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model);oracle=RobotServoEffort(model)
        lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy();lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m
        lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
        for k,name in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
            j=model.joint(name).id
            if model.jnt_limited[j]:lo[k]=max(lo[k],model.jnt_range[j,0]);hi[k]=min(hi[k],model.jnt_range[j,1])
        open18=np.concatenate(OPEN)
        def load(rel):
            if rel not in cache:
                path=ROOT/rel
                with np.load(path/'trace.npz',allow_pickle=False) as z:cache[rel]=(z['time'],z['observation_json'],z['task_request_json'],json.loads((path/'metadata.json').read_text(encoding='utf-8')))
            return cache[rel]
        def frame(rel,t,release=False):
            ts,observations,tasks,meta=load(rel);i=int(np.argmin(abs(ts-t)));f=Frame(**json.loads(str(observations[i])))
            for k,v in list(vars(f).items()):
                if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21'):setattr(f,k,np.asarray(v))
            ref=np.r_[f.robot_actuator_target21[:18],f.executed_motion_reference]
            if actor.encode(Reference21(ref,f.time)).values.tobytes()!=np.asarray(f.robot_actuator_target21,float).tobytes():raise AssertionError('public command-memory reference encode mismatch')
            f.robot_reference21=ref;f.reference_lower21=lo.copy();f.reference_upper21=hi.copy();f.mode_generation=1 if release else 0
            f.release_entry_reference21=None;f.release_nominal_reference21=None
            if release:
                ef,_,_=frame(rel,12.);entry=ef.robot_reference21
                f.release_entry_reference21=entry.copy();f.release_nominal_reference21=release_nominal(ReleaseEntry(12.,Reference21(entry,12.),1),f.time+.020,open18,lo,hi).values.copy()
                for k in range(5,18):
                    if entry[k]>open18[k]:f.reference_lower21[k]=max(lo[k],open18[k]);f.reference_upper21[k]=min(hi[k],ref[k])
                    elif entry[k]<open18[k]:f.reference_lower21[k]=max(lo[k],ref[k]);f.reference_upper21[k]=min(hi[k],open18[k])
                    else:f.reference_lower21[k]=f.reference_upper21[k]=open18[k]
            return f,json.loads(str(tasks[i])),meta
        def build(method,rel,t):
            ts,_,_,meta=load(rel);c=cm.create_controller(method,dict(config=dict(shape=meta['shape'])),meta['limits'],meta['budget'],meta['actual_ratio_task_gains'],model)
            for tt in ts[(ts>=t-.2-1e-9)&(ts<t-1e-9)]:c.state.update(frame(rel,float(tt))[0],.005)
            return c
        original_qp=cm.solve_intent_qp;original_tasks=cm.build_fusion_tasks;original_shape=cm.build_holding_native_tasks
        def qp_spy(**kw):
            result=original_qp(**kw);calls.append((deepcopy(kw),result));return result
        def tasks_spy(**kw):
            result=original_tasks(**kw);taskcalls.append((deepcopy(kw),result));return result
        def shape_spy(**kw):
            result=original_shape(**kw);shapecalls.append((deepcopy(kw),result));return result
        def raw_value(p,d,z,include_extra=True):
            S=p['scales'];r=p['A21x25'].T@d-p['Jpx25'].T@(z-p['y_p']);short=max(0.,p['load_proxy_N']-float(p['support_coeff_p']@z))
            terms=[float(np.square(S['residual25']*r).sum()),float(np.square(S['intentP']*(z-p['z_intent_p'])).sum()),float(np.square(S['delta21']*d).sum()),float((S['load']*short)**2)]
            e=p.get('extra_task_rows')
            if include_extra and e is not None:terms.append(float(np.square(e['matrix']@np.r_[d,z]-e['target']).sum()))
            return sum(terms),terms
        cases=[('holding','output/development_v12_native_pilot_v1/runs/nominal__fusion_effort__617901',8.5,False),
            ('transport','output/development_v12_native_transport_v1/runs/friction_drop__ratio_effort__617902',10.,False),
            ('release','output/development_v12_native_pilot_v1/runs/nominal__fusion_effort__617901',12.020,True)]
        with patch.object(cm,'solve_intent_qp',qp_spy),patch.object(cm,'build_fusion_tasks',tasks_spy),patch.object(cm,'build_holding_native_tasks',shape_spy):
            for label,rel,t,releasing in cases:
                for method in cm.METHODS:
                    c=build(method,rel,t);f,task,_=frame(rel,t,releasing);c.set_task_request(task);startcalls=len(calls);starttasks=len(taskcalls);startshape=len(shapecalls)
                    action=c.step(f,'RELEASING' if releasing else 'HOLDING','release' if releasing else 'hold');e=c.decision_evidence[-1];tag=label+'/'+method
                    ck(tag+'/single_joint_qp',len(calls)==startcalls+1)
                    p,result=calls[-1];val,terms=raw_value(p,result.d21,result.zP)
                    close(tag+'/independent_original_objective_including_extras',val,result.objective)
                    accept=cm.check_same_problem_candidate(p,result);ck(tag+'/saved_same_problem_merit',accept==e['proposal_acceptance'] and accept['objective_bytes_equal'])
                    if c.fusion_spec is not None:
                        spec=c.fusion_spec;ck(tag+'/Sf_once_exact',p['scales']['intentP'].tobytes()==spec.intentP.tobytes())
                        ck(tag+'/base_prior_exact',p['z_intent_p'].tobytes()==spec.base_prior_p.tobytes())
                        ck(tag+'/fusion_extra_prefix_exact',p['extra_task_rows']['matrix'][:len(spec.labels)].tobytes()==spec.extra_matrix.tobytes() and p['extra_task_rows']['target'][:len(spec.labels)].tobytes()==spec.extra_target.tobytes())
                        if accept['accepted']:close(tag+'/accepted_conditional_memory_Bz',c.intent.previous,np.asarray(spec.metadata['conditional_digit_sum_map'])@result.zP)
                    if releasing:
                        ck(tag+'/release_no_holding_factories',len(taskcalls)==starttasks and len(shapecalls)==startshape and p['extra_task_rows'] is None)
                        ck(tag+'/release_holding_load_inactive',c.release_latched and e['task_activation']['holding_load_cancelled'] and np.array_equal(p['z_intent_p'],np.zeros(len(p['y_p']))))
                    else:
                        ck(tag+'/first_history_no_past_sample',shapecalls[-1][0]['previous_model_effort21'] is None and shapecalls[-1][1].metadata['history_task_status']=='no_past_public_model_sample_inactive')
                        ck(tag+'/history_explicitly_model_computed_not_measurement','not_independent_torque_measurement' in shapecalls[-1][1].metadata['history_role'])
                    maxstep=reference_rates(action.mode)*.005
                    ck(tag+'/typed_legal_reference_and_no_prediction',type(action.reference21) is Reference21 and np.all(abs(action.reference21.values-f.robot_reference21)<=maxstep+1e-15) and np.all(action.reference21.values>=lo-1e-15) and np.all(action.reference21.values<=hi+1e-15) and not c.debug['pressure_prediction_available'] and np.isnan(c.predicted).all())
                    records.append(dict(case=label,method=method,time_s=float(f.time),qp_status=result.status,accepted=accept['accepted'],fusion_tasks=c.fusion_spec is not None,patches=len(p['y_p']),extra_rows=0 if p['extra_task_rows'] is None else len(p['extra_task_rows']['target']),objective=result.objective,independent_terms=terms,nominal_native_task_admission=c.debug['native_task_metadata'],native_status=None if e['native'] is None else e['native']['status']))
            # Consecutive public frames, retaining controller task and model
            # memory; archived robot states are NOT generated by this policy.
            rel=cases[0][1];c=build('fusion_effort',rel,8.5);first=None;previous_intent=None
            for tt in (8.5,8.505,8.51,8.515,8.52):
                f,task,_=frame(rel,tt);c.set_task_request(task);beforelen=len(calls);c.step(f,'HOLDING','hold')
                if len(calls)==beforelen:continue
                p,result=calls[-1];shape_args,shape=shapecalls[-1]
                native_now=oracle.evaluate(np.r_[f.raw_joint,f.lift],np.r_[f.raw_joint_velocity,f.lift_velocity_m_s],np.asarray(f.robot_actuator_target21),time_s=f.time)
                ck('sequence/native_oracle_known/'+str(tt),native_now.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA')
                ck('sequence/stored_current_model_effort_bytes/'+str(tt),c.model_effort_history.tobytes()==native_now.predicted_effort21.tobytes())
                if first is not None:
                    close('sequence/previous_public_model_effort_not_current_or_nominal',shape_args['previous_model_effort21'],first['model_effort'])
                    ck('sequence/past_clock_exact',shape_args['previous_model_time_s']==first['time'] and shape.metadata['history_sample_time_s']==first['time'])
                    labels=list(shape.labels);start=labels.index('native_effort_history_0');expected=p['scales']['delta21']*(first['model_effort']-p['p0_21'])
                    close('sequence/history_target_absolute_goal_offset',shape.target[start:start+21],expected)
                    ck('sequence/history_rows_diagonal_once',shape.matrix[start:start+21,:21].tobytes()==np.diag(p['scales']['delta21']).tobytes())
                    if c.fusion_spec is not None and previous_intent is not None:
                        close('sequence/previous_conditional_intent_not_raw_pressure',taskcalls[-1][0]['previous_conditional_force5'],previous_intent)
                    history.append(dict(current_time_s=float(f.time),previous_time_s=first['time'],previous_model_effort21=first['model_effort'].tolist(),current_model_effort21=native_now.predicted_effort21.tolist(),history_target=expected.tolist()))
                first=dict(time=float(f.time),model_effort=native_now.predicted_effort21.copy())
                previous_intent=None if not isinstance(c.intent,cm.FusionTaskMemory) else deepcopy(c.intent.previous)
            ck('sequence_two_real_decisions',len(history)==1)
        # Direct merit counterexample: extra tasks must affect acceptance, not
        # only QP reporting. No optimizer is used for these fake iterates.
        p=deepcopy(calls[0][0]);P=len(p['y_p']);E=np.zeros((1,21+P));E[0,0]=3.;p['extra_task_rows']=dict(matrix=E,target=np.array([2.]))
        d=np.minimum(p['effort_upper21']-p['p0_21'],np.maximum(p['effort_lower21']-p['p0_21'],np.zeros(21)))
        z=cm.project_normal_intents(p['z_intent_p'],p['patch_digit_p'],p['caps5'],p['total_budget'])
        # Use the controller's authoritative arithmetic for an exact merit
        # identity; independently check omission must be rejected.
        original=deepcopy(calls[0][0]);original_result=calls[0][1]
        r=SimpleNamespace(d21=original_result.d21,zP=original_result.zP,objective=original_result.objective,available=original_result.available)
        changed=cm.check_same_problem_candidate(p,r)
        ck('merit_rejects_cost_from_different_extra_rows',not changed['accepted'] and not changed['objective_bytes_equal'])
        ck('all_nine_fixture_calls',len(records)==9)
        ck('no_nested_fusion_source_construction','FusionIntent(' not in source.read_text(encoding='utf-8'))
        ck('no_forbidden_calls_or_online_truth',not forbidden and not truth)
        ck('sources_unchanged',sources=={x:sha(ROOT/x) for x in sources})
    except Exception:failures.append('exception');records.append(dict(exception=traceback.format_exc()))
    inputs={name:dict(npz_column_loads=1,metadata_sha256=sha(ROOT/name/'metadata.json'),trace_sha256=sha(ROOT/name/'trace.npz')) for name in cache}
    report=dict(schema='v15_integrated_task_wiring_independent_v1',passed=not failures,checks=checks,failures=failures,records=records,sequential_history=history,
        source_sha256=sources,input_provenance=inputs,checker_sha256=sha(__file__),forbidden_calls=forbidden,forbidden_truth_reads=truth,robot_only_FK_calls=FK,
        physics_steps=0,scope='Nine off-policy archived public calls plus five consecutive public frames/two decisions and independent raw merit/history wiring oracles. Constructs known robot model/private FK; no contact, dynamics, engine write, runtime or closed-loop performance validation.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failures=failures,records=len(records),out=str(out)),ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
