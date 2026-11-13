"""Peer clocks/authority/reference review; no plant, runner import or physics."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from identification.release_local_response_v3.decoder import coordinates,VARIANTS,TRAIN,ORIGINS,ReleasePath2D


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recorded_decoder():
    base=ROOT/'output/release_basis_minimal_v2';entry=base/'runs/capsule__native2d'
    state=read(entry/'entry_side_state.json')
    mapping=lambda value:dict(value.get('state',value)['mapping_items'])
    plant=mapping(mapping(state)['plant_side'])
    array=lambda value:np.frombuffer(base64.b64decode(value['bytes_b64']),dtype=value['array_dtype']).reshape(value['shape']).copy()
    constants={}
    for node in ast.parse((base/'sources/foundation/foundation_config.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id in ('OPEN','CLOSED') for v in node.targets):constants[node.targets[0].id]=ast.literal_eval(node.value)
    xml=ET.parse(base/'sources/foundation/assets/legacy_model.xml').getroot()
    bounds={node.get('class'):np.fromstring(node.find('position').get('ctrlrange'),sep=' ') for node in xml.iter('default') if node.find('position') is not None and node.find('position').get('ctrlrange')}
    actuators=list(xml.find('actuator'))[2:20]
    with np.load(entry/'entry_physics.npz',allow_pickle=False) as z:thumb=z['ctrl'][2:7].copy()
    decoder=ReleasePath2D(entry_alpha=array(plant['alpha']),entry_thumb=thumb,entry_lift=float.fromhex(plant['lift_ref']['float_hex']),entry_wrist=array(plant['wrist_ref']),
        open_pose=constants['OPEN'],closed_pose=constants['CLOSED'],finger_lower=[bounds[a.get('class')][0] for a in actuators],finger_upper=[bounds[a.get('class')][1] for a in actuators],release_open_rate=.8)
    return decoder,entry


def main():
    package=ROOT/'identification/release_local_response_v3';plan=read(package/'plan.json')
    sources={p.name:sha(p) for p in package.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ('plan.json','README.md'))}
    checks=[];records={}
    def check(name,value,**data):checks.append(dict(name=name,passed=bool(value),**data))
    check('expected_stable_plan_identity',sources['plan.json']=='ca531dbc220a5ae8961b580571c62c3ca862a80624ed642a77bc39ec158a77e5')
    check('expected_stable_decoder_identity',sources['decoder.py']=='04012acbfebf5402fe19ce751f277ac410af4296bac9afda815eea74fa2cc12e')
    check('expected_stable_runner_identity',sources['run_pilot.py']=='2ebf8b4e7b2fa50dced28c2df0817fdd78f8920cceaf404bba0d611177e34ff7')
    decoder,entry=recorded_decoder();ticks=np.arange(801);tau=(2400+ticks)*.005-12.
    native=np.array([decoder.at(float(t),'native2d').command21 for t in tau])
    step_cap=abs(np.diff(native,axis=0)).max(axis=0);variation_cap=abs(np.diff(native,axis=0)).sum(axis=0)
    with np.load(entry/'decoder_commands.npz',allow_pickle=False) as z:
        oldtau=z['path_time'].copy();oldticks=np.rint(z['control_time']/.005).astype(int)-2400;oldcommands=z['actual_command21'].copy()
    zero_traces=[]
    for stage,origin in ORIGINS.items():
        def coord(t,k,variant):return coordinates(decoder,float(t),int(k),variant,release_control_tick=2400,release_time_s=12.,probe_origin_tick=origin)
        reconstructed=np.array([decoder.decode(c['phi_s'],c['lift_reference_m']).command21 for c in (coord(t,k,'zero') for t,k in zip(oldtau,oldticks))])
        check(stage+'/all8801_actual_native_reference_commands_bitexact',np.array_equal(reconstructed,oldcommands));zero_traces.append(reconstructed)
        local={}
        for variant,(a,b) in VARIANTS.items():
            infos=[coord(t,k,variant) for t,k in zip(tau,ticks)]
            phi=np.array([x['phi_s'] for x in infos]);lift=np.array([x['lift_reference_m'] for x in infos])
            command=np.array([decoder.decode(x['phi_s'],x['lift_reference_m']).command21 for x in infos])
            elapsed_steps=np.maximum(0,np.minimum(4,ticks-origin));expected_phi_offset=-a*elapsed_steps*.005
            expected_lift_clock=tau-b*elapsed_steps*.005
            expected_lift=decoder.entry_lift*(1.-np.clip(expected_lift_clock/.6,0,1))
            label=stage+'/'+variant
            check(label+'/closed_form_phase_slowdown',np.max(abs((phi-tau)-expected_phi_offset))<1e-12)
            check(label+'/lift_delay_before_native_floor',np.max(abs(lift-expected_lift))<1e-12)
            check(label+'/native_prefix_through_origin_exact',np.array_equal(command[:origin+1],native[:origin+1]))
            check(label+'/held_four_quarter_weights',np.array_equal([infos[k]['ramp_weight'] for k in range(origin,origin+5)],[0,.25,.5,.75,1.]))
            check(label+'/monotonic_without_catchup',np.min(np.diff(phi))>=-1e-12 and np.max(np.diff(phi))<=.005+1e-12 and abs((phi[-1]-tau[-1])-(-a*.02))<1e-12)
            check(label+'/real21_native_authority_and_variation',np.all(command[:,:18]>=decoder.lower-1e-10) and np.all(command[:,:18]<=decoder.upper+1e-10)
                and np.all(abs(np.diff(command,axis=0))<=step_cap+1e-10) and np.all(abs(np.diff(command,axis=0)).sum(axis=0)<=variation_cap+1e-9))
            check(label+'/constant_actual_entry_wrists',np.array_equal(command[:,19:],native[:,19:]))
            check(label+'/eventually_floor_not_constant_positive_offset',lift[-1]==0. and np.min(np.diff(lift))>=-decoder.entry_lift/.6*.005-1e-12 and np.max(np.diff(lift))<=1e-12)
            check(label+'/actual_denominator_retained',np.array_equal(phi-tau,[x['actual_phi_offset_s'] for x in infos]) and np.array_equal(lift-native[:,18],[x['actual_lift_offset_m'] for x in infos]))
            tolerant=int(np.flatnonzero(lift<=1e-12)[0]);strict=int(np.flatnonzero(lift==0.)[0])
            local[variant]=dict(entry_control_s=float((2400+origin)*.005),entry_before_action_last_phi_s=float(tau[origin-1]),
                origin_command_phi_s=infos[origin]['phi_s'],full_bias_control_s=float((2400+origin+4)*.005),
                first_full_bias_physical_end_s=float((2400+origin+4)*.005+.0005),
                floor_within1e12_control_s=float((2400+tolerant)*.005),strict_zero_control_s=float((2400+strict)*.005),
                floor_at12p620=float(lift[124]),terminal_command21=command[-1].tolist())
        records[stage]=local
    check('two_origins_zero_same_actual_native_references',np.array_equal(*zero_traces))
    check('two_registered_origins_do_not_include_old020',set(ORIGINS.values())=={30,70})
    check('train_zero_plus_two_axes_no_mixed_leakage',tuple(plan['training'])==TRAIN and set(plan['strict_holdout'])==set(VARIANTS)-set(TRAIN)
        and set(TRAIN)=={'zero','phi_slow','lift_slow'} and plan['planned']==14)
    tree=ast.parse((package/'run_pilot.py').read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('planned_jobs','summarize')]
    scope=dict(Path=Path,ORIGINS=ORIGINS,VARIANTS=VARIANTS,TRAIN=TRAIN,identity=lambda:{'pure':'constant'},ROOT=ROOT)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'peer_ast_only_no_runner_import','exec'),scope)
    jobs=scope['planned_jobs'](Path('peer_fixture_no_output'))
    check('14_unique_jobs_with_independent_stage_roles',len(jobs)==14 and len({x['job_id'] for x in jobs})==14 and all(sum(x['stage']==s and x['training_role']=='train' for x in jobs)==3 and sum(x['stage']==s and x['training_role']=='heldout_mixed' for x in jobs)==4 for s in ORIGINS))
    failed=[dict(job_id=j['job_id'],stage=j['stage'],variant=j['variant'],status='failed',error='retained pure failure fixture') for j in jobs]
    summary=scope['summarize'](Path('peer_fixture_no_output'),failed,{'pure':'constant'})
    check('all14_failures_remain_not_passed',summary['execution_complete'] and summary['completed']==0 and len(summary['failures'])==14 and len(summary['pairs'])==14 and not summary['passed'])
    check('runner_has_preaction_entry_snapshots_and_same_stage_prefix_contract',all(value in (package/'run_pilot.py').read_text(encoding='utf-8') for value in ['old.entry_snapshot(s,directory)','s.plant.control_steps==tick','s.step_one_sensor_tick()','row[\'entries\']==zero[\'entries\']','pref[\'exact\']','cross_stage_zero_full16s']))
    for bad in [dict(native_tau=.1505,relative_control_tick=30,probe_origin_tick=30),dict(native_tau=.15,relative_control_tick=30,probe_origin_tick=4),dict(native_tau=.15,relative_control_tick=30,probe_origin_tick=np.bool_(True))]:
        try:coordinates(decoder,variant='zero',release_control_tick=2400,release_time_s=12.,**bad)
        except ValueError:check('reject_bad_clock_or_origin/'+str(bad),True)
        else:check('reject_bad_clock_or_origin/'+str(bad),False)
    after={p.name:sha(p) for p in package.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ('plan.json','README.md'))}
    check('seven_author_sources_stable_during_peer_review',sources==after and len(sources)==7)
    check('no_plant_runner_or_mujoco_import',not any(n in sys.modules for n in ('mujoco','hand_plant','identification.release_local_response_v3.run_pilot')))
    result=dict(schema='release_local_response_v3_peer_reference_review_v1',passed=all(c['passed'] for c in checks),checks=checks,records=records,author_source_sha256=sources,
        checker_sha256=sha(__file__),physics_steps=0,plant_constructions=0,
        conclusions=dict(reference_and_clock_contract_reviewed=True,actual_new_v3_full16s_zero_reproduction='not yet executed; root physical endpoint validation required',
            actual_new_v3_entry_prefix_pairing='runner checks reviewed; not yet a physical result',new_force_model_available=False,material_safety_certified=False),
        scope='independent numeric coordinates and actual oldnative reference commands, AST job/failure contract, source review; no new V3 physical outcomes')
    out=ROOT/'review/release_local_response_v3_peer_review_v1.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],output=str(out),sha256=sha(out),physics_steps=0),ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
