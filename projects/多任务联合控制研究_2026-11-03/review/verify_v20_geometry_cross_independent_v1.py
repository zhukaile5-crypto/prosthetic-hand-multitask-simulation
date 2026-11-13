"""Finite V20 geometry/adapter/current-task cross review, zero physics.
Reviewer authored task_solver, not geometry/adapter/supervisor. One archived
current release packet plus its public .2s history; private robot FK/native
algebra allowed, no plant/session/forward/step/contact/writer/sensor.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy,copy
from contextlib import ExitStack
from unittest.mock import patch
import ast,gzip,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT.parent/'五指策略接入_2026-11-03';sys.path[:0]=[str(ROOT),str(F)];sys.dont_write_bytecode=True
import research_imports,mujoco,hand_plant
from foundation_config import FoundationConfig,OPEN
from v20_contact_exit.adapter import ContactExitAdapter
from v20_contact_exit.task_solver import solve_contact_exit
from v14_coord.design import reference_rates
from v13_a1.codec import Reference21,ReferenceActor21

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def main():
    out=ROOT/'review/v20_geometry_cross_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    files=('v20_contact_exit/geometry.py','v20_contact_exit/adapter.py','v20_contact_exit/task_solver.py','v20_contact_exit/design.py','v20_contact_exit/supervisor.py')
    pins={f:sha(ROOT/f) for f in files};calls={};forbidden=[];cases=[]
    def ck(name,cond,detail=None):cases.append(dict(name=name,passed=bool(cond),detail=plain(detail)))
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise RuntimeError('Forbidden '+name)
        return f
    def count(name,fn):
        def f(*a,**kw):calls[name]=calls.get(name,0)+1;return fn(*a,**kw)
        return f
    plan=read(ROOT/'review/v19_small_plan_v1.json');stage=next(s for s in plan['stages'] if s['name']=='nominal_transport');ass=stage['assignment'];run=ROOT/stage['output']/'runs'/f"{ass['scenarios'][0]}__{ass['methods'][0]}__{ass['seeds'][0]}"
    meta=read(run/'metadata.json')
    with np.load(run/'trace.npz',allow_pickle=False) as z:t=z['time'];public=z['observation_json']
    ids=np.flatnonzero((t>=15.8-1e-9)&(t<=16.+1e-9));frames=[SimpleNamespace(**json.loads(str(public[i]))) for i in ids];f=frames[-1]
    with ExitStack() as guards:
        for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_collision','mj_contactForce'):
            if hasattr(mujoco,n):guards.enter_context(patch.object(mujoco,n,deny(n)))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant')));guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor.apply')))
        for n in ('MjData','mj_resetData','mj_tendon','mj_transmission','mj_kinematics','mj_comPos','mj_jac'):
            if hasattr(mujoco,n):guards.enter_context(patch.object(mujoco,n,count(n,getattr(mujoco,n))))
        model,_=hand_plant._model(FoundationConfig(**meta['config']));adapter=ContactExitAdapter(model,open18=np.concatenate(OPEN))
        for frame in frames:adapter.observe(frame)
        current=np.asarray(f.robot_reference21);rates=reference_rates('release');scale=rates*.005
        lower=np.maximum(np.asarray(f.reference_lower21),current-scale);upper=np.minimum(np.asarray(f.reference_upper21),current+scale)
        params=dict(current21=current,reference_lower21=lower,reference_upper21=upper)
        got=adapter.current_contact_tasks(f,**params)
        ck('one_real_terminal_public_packet_has_known_current_geometry',got['available'] and len(got['pressure_P_N'])>0,got['diagnostics'])
        debt=current-np.asarray(got['encoder_reference21']);encoded=adapter.actor.encode(Reference21(current,f.time)).values
        ck('lift_reference_debt_has_no_gravity_bias_or_double_subtraction',np.array_equal(got['current_unbiased_debt21'],debt) and abs((encoded[18]-got['encoder_reference21'][18])-debt[18]-adapter.actor.lift_bias_m)<1e-16,dict(debt_lift_m=debt[18],bias_m=adapter.actor.lift_bias_m,engine_lift=encoded[18]))
        Jn=got['normal_jacobian_Px25'];B=got['proposal_map25x21'];N=got['Npx21'];v=np.asarray(f.robot_velocity25)
        ck('actual_rows_use_each_patch_Jn_pressure_and_full25_velocity_once',np.allclose(N,Jn@B,atol=1e-14,rtol=0) and np.allclose(got['vnormal_p'],Jn@v,atol=1e-14,rtol=0) and np.allclose(got['a_normal_p'],N@debt,atol=1e-14,rtol=0) and np.array_equal(got['pressure_P_N'],[f.skin_patches[p['patch_index']]['normal'] for p in got['diagnostics']['patches']]),dict(rows=len(N),digits=got['digits_p']))
        # Reproduce the prior source's actual false-known-empty classification.
        oldpath=ROOT/'review/revisions/v20_geometry_before_missing_patch_fix_v1/adapter_5a961.py'
        spec=importlib.util.spec_from_file_location('v20_contact_exit.adapter_prior_cross',oldpath);oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
        old=oldmodule.ContactExitAdapter(model,open18=np.concatenate(OPEN))
        for frame in frames:old.observe(frame)
        missing=deepcopy(f);missing.skin_patches=[];oldbad=old.current_contact_tasks(missing,**params);newbad=adapter.current_contact_tasks(missing,**params)
        ck('old_5a961_false_zero_contact_counterexample_preserved_and_current_rejects',oldbad['available'] and len(oldbad['pressure_P_N'])==0 and not newbad['available'] and any(f.contact_present),dict(old_source_sha256=sha(oldpath),public_contact_present=f.contact_present,public_normal=f.normal,old_available=oldbad['available'],new_status=newbad['diagnostics']))
        present=[i for i in range(5) if f.contact_present[i] and f.normal[i]>0];drop=present[0];partial=deepcopy(f);partial.skin_patches=[p for p in partial.skin_patches if p['digit']!=drop];pg=adapter.current_contact_tasks(partial,**params)
        ck('partial_missing_digit_retains_other_rows_and_marks_geometry_unknown',pg['available'] and pg['diagnostics']['partial_geometry'] and drop in pg['diagnostics']['missing_geometry_digits'] and drop not in pg['digits_p'],pg['diagnostics'])
        empty=deepcopy(missing);empty.contact_present=np.zeros(5,bool);eg=adapter.current_contact_tasks(empty,**params)
        ck('explicit_no_present_contact_differs_from_missing_geometry_without_force_threshold',eg['available'] and eg['normal_jacobian_Px25'].shape==(0,25) and eg['motion_matrix_Mx21'] is None,dict(status=eg['diagnostics'],normal_noise_retained=empty.normal))
        no_motion=deepcopy(f);no_motion.slip_domain_available=False;ng=adapter.current_contact_tasks(no_motion,**params)
        ck('unknown_motion_passes_None_triplet_not_fictitious_zero_error',ng['available'] and all(ng[k] is None for k in ('motion_matrix_Mx21','motion_error_M_m_s','motion_time_s')),ng['diagnostics']['motion'])
        wide=upper.copy();wide[18]=(adapter.actor.ctrl_range21[18,1]-adapter.actor.lift_bias_m)+.01
        bad=adapter.current_contact_tasks(f,**dict(params,reference_upper21=wide))
        ck('absolute_native_engine_domain_checks_lift_bias_once_without_permission_widening',not bad['available'],bad['diagnostics'])
        force_model=copy(model);aids=adapter.servo.aids;idx=int(aids[0]);force_model.actuator_forcelimited[idx]=True;force_model.actuator_forcerange[idx]=[-1e-12,1e-12]
        forced=ContactExitAdapter(force_model,open18=np.concatenate(OPEN))
        for frame in frames:forced.observe(frame)
        fg=forced.current_contact_tasks(f,**params)
        ck('single_force_clip_keeps_position_geometry_and_other_observations',fg['available'] and fg['native_state']['diagnostics']['force_clipped21'][0] and np.array_equal(fg['pressure_P_N'],got['pressure_P_N']) and np.array_equal(fg['Npx21'],got['Npx21']),dict(force_clipped=fg.get('native_state',{}).get('diagnostics',{}).get('force_clipped21'),geometry_available=fg['available']))
        # Execute ONLY the score helper AST, no controller/supervisor constructed.
        tree=ast.parse((ROOT/'v20_contact_exit/supervisor.py').read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_emitted_reference');scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'saved_score_helper','exec'),scope);score=scope['score_emitted_reference']
        target=np.zeros(21);target[:18]=np.asarray(meta['release_clearance_goal']['reference18'])-current[:18]
        solved=solve_contact_exit(time_s=f.time,dt_s=.005,current_reference21=current,encoder_reference21=got['encoder_reference21'],step_lower21=lower-current,step_upper21=upper-current,reference_rates21=rates,normal_jacobian_Px25=Jn,proposal_map25x21=B,robot_velocity25=v,pressure_P_N=got['pressure_P_N'],reference_target_delta21=target,motion_matrix_Mx21=got['motion_matrix_Mx21'],motion_error_M_m_s=got['motion_error_M_m_s'],motion_time_s=got['motion_time_s'])
        proposed=current+solved.x*scale;sc=score(solved,current21=current,proposed21=proposed,scale21=scale)
        ck('real_geometry_to_hierarchy_to_true_emitted_reference_reconstruction',solved.available and sc['available'],dict(status=solved.status,score=sc))
        native_x=(proposed-current)/scale;ind=[]
        for task in solved.diagnostics['task_matrices_and_targets']:
            A=np.asarray(task['matrix']);tar=np.asarray(task['target']);res=A@native_x-tar;ind.append(float(res@res))
        ck('emitted_score_uses_actual_reconstructed_x_and_all_original_layer_targets',all(np.float64(x).tobytes()==np.float64(y['squared_loss']).tobytes() for x,y in zip(ind,sc['levels'])) and np.array_equal(native_x,sc['normalized_emitted_delta21']),dict(independent_losses=ind))
        changed=proposed.copy();changed[0]+=.0001;badscore=score(solved,current21=current,proposed21=changed,scale21=scale)
        ck('changed_actual_reference_does_not_inherit_prior_attainment',not badscore['available'],badscore)
        ck('no_step_forward_contacts_or_actor_writer',not forbidden and adapter.actor.write_count==forced.actor.write_count==old.actor.write_count==0,dict(forbidden=forbidden,allowed_private_calls=calls))
    endpins={f:sha(ROOT/f) for f in files};ck('binding_actual_reviewed_sources_unchanged_during_cross_gate',pins==endpins,dict(before=pins,after=endpins))
    report=dict(schema='v20_geometry_and_emission_cross_independent_v1',passed=all(c['passed'] for c in cases),checks=len(cases),cases=cases,source_sha256=pins,reader_sha256=sha(__file__),
        archived_public_input=dict(trace_path=str(run/'trace.npz'),sha256=sha(run/'trace.npz'),history_range_s=[15.8,16.],public_frames=len(frames),current_packet_s=16.),
        role='task_solver author, independent of geometry/adapter/supervisor; cross-component gate, not own-math independent proof',
        scope=dict(static_model_compiles=1,private_robot_Data_allowed=True,allowed_private_calls=calls,step=0,forward=0,contact_queries=0,writer=0,complete_controller_or_session_constructions=0),
        limitations=['Old acquired normal planes/current robot geometry are experimental reference tasks, not future motion/pressure.', 'Only one archived current packet and finite negative fixtures; complete lifecycle integration is another reviewer scope.','Auxiliary local geometry is not implemented or certified by this finger-only adapter.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');out.with_suffix('.md').write_text('# V20几何与实际发出参考的跨组件审查\n\n'+('PASS' if report['passed'] else 'FAIL')+f"；{sum(c['passed'] for c in cases)}/{len(cases)}。完整绑定、真实旧公开包与失败修复证据见JSON。\n\n旧5a961的有接触但缺patch被误认knownP0已实际重现并保留；当前版本拒绝全缺几何，部分缺测保留其他真实行且标partial。参考债务排除lift actor bias一次；单force clip保留位置目标几何，缺motion传None。实际发出x以真正proposed-current重建，原各层matrix/target重评分，篡改参考不继承原attainment。\n\n审者是task_solver作者，未参与geometry/adapter/supervisor。静态编译与私有机器人FK/native代数，零step/forward/contact/writer，无Session/plant构造。不签完整运行准入/物理清离或安全。\n",encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(cases),failed=[c for c in cases if not c['passed']],out=str(out),sha256=sha(out),source_sha256=pins),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
