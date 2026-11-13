"""Native decoder oracle and source guard; no plant/physics construction."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,sys
import numpy as np
PILOT=Path(__file__).resolve().parent;ROOT=PILOT.parents[1]
sys.path.insert(0,str(PILOT));sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
from hand_plant import HandPlant
from foundation_config import OPEN,CLOSED,GROUPS
from decoder import ReleasePath2D,FINGER_AXES,COMMAND_AXES


def fixture(alpha,lift=.1):
    names=['rh_A_WRJ2','rh_A_WRJ1']+list(FINGER_AXES)+['lift_motor','integration_obstacle_motor']
    mapping={n:i for i,n in enumerate(names)}
    groups=[np.array([mapping['rh_A_'+n] for n in group]) for group in GROUPS]
    low=np.full(22,-10.);high=np.full(22,10.)
    low[groups[0]]=[-1.0472,0.,-.20944,-.698132,-.261799]
    high[groups[0]]=[1.0472,1.22173,.20944,.698132,1.5708]
    for g in groups[1:]:low[g]=0.;high[g]=3.1415
    # Actual native nonthumb metacarpal/knuckle directions include signed
    # bounds, but this fixture's held/open values are nonnegative.
    p=SimpleNamespace(time=12.,release_started=12.,release_alpha=np.asarray(alpha,float),alpha=np.asarray(alpha,float).copy(),
        release_thumb=np.asarray(OPEN[0])+alpha[0]*(np.asarray(CLOSED[0])-OPEN[0]),release_lift=lift,
        release_ctrl=np.zeros(22),lift_ref=lift,wrist_ref=np.array([.12,-.1]),
        open_pose=[np.array(x,float) for x in OPEN],closed_pose=[np.array(x,float) for x in CLOSED],
        model=SimpleNamespace(actuator_ctrlrange=np.column_stack((low,high)),body_subtreemass=np.array([3.794])),
        hand=0,config=SimpleNamespace(release_open_rate=.8),actuator_groups=groups,
        data=SimpleNamespace(ctrl=np.zeros(22)),max_hidden_clip=0.,lift_actuator=mapping['lift_motor'],
        obstacle_motor=mapping['integration_obstacle_motor'],aid=lambda n:mapping[n],env={'obstacle_x':0.})
    return p


def verify():
    checks={};comparisons=0;entries=[np.array(x) for x in ([.7,.8,.9,1.,.9],[1.,1.2,1.1,1.,1.3],[0.,0.,0.,0.,0.])]
    for e,alpha in enumerate(entries):
        p=fixture(alpha,.1-e*.01);decoder=ReleasePath2D.from_plant(p);aids=np.concatenate(p.actuator_groups)
        for tick in range(801):
            p.time=12.+tick*.005
            HandPlant._apply_ctrl(p) # native source evaluated directly on a robot-only algebra fixture
            elapsed=p.time-p.release_started;native=p.data.ctrl.copy();actual=decoder.at(elapsed,'native2d')
            expected=native.copy();expected[aids]=actual.finger_reference
            expected[:2]=actual.wrist_reference
            expected[20]=actual.lift_reference+p.model.body_subtreemass[p.hand]*9.81/1500.
            if not np.array_equal(expected,native) or not np.array_equal(actual.alpha,p.alpha):raise AssertionError((e,tick,'native mismatch'))
            fixed=decoder.at(elapsed,'fixed_height')
            if not np.array_equal(fixed.finger_reference,native[aids]) or not np.array_equal(fixed.wrist_reference,native[:2]):raise AssertionError('Fixed lift changed finger/wrist path')
            if fixed.lift_reference!=p.release_lift:raise AssertionError('Fixed lift does not hold entry')
            comparisons+=1
        checks[f'entry_{e}_801_native_and_fixed_lift_exact']=True
    d=ReleasePath2D.from_plant(fixture(np.full(5,.8)))
    checks['actual21_axes_18finger_plus_lift_and_2wrist']=bool(len(COMMAND_AXES)==21 and len(set(COMMAND_AXES))==21)
    checks['native_uses_separate_lift_coordinate']=bool(d.decode(.2,.1).lift_reference==.1 and d.at(.2,'native2d').lift_reference<.1)
    checks['finite_native_limits_and_full_open']=bool(np.array_equal(d.at(4.,'native2d').finger_reference[5:],np.concatenate([np.asarray(v) for v in OPEN[1:]])))
    for name,t,path in [('negative',-.1,'native2d'),('nonfinite',np.nan,'native2d'),('unknown',.1,'other')]:
        try:d.at(t,path)
        except ValueError:checks['reject_'+name]=True
        else:checks['reject_'+name]=False
    code=(PILOT/'run_pilot.py').read_text(encoding='utf-8');tree=ast.parse(code)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    checks['runner_has_explicit_execute_preflight_and_pin_modes']=all(x in code for x in ('--execute','--preflight','--pin-sources','add_mutually_exclusive_group(required=True)'))
    checks['no_direct_mj_step_call_in_new_runner_or_decoder']=not any(isinstance(n.func,ast.Attribute) and n.func.attr=='mj_step' for n in calls)
    checks['complete_job_exception_ledger_and_fresh_outputs']=all(x in code for x in ('worker_exception','partial_trace_retained','exist_ok=False','target.exists()','causal_lift_comparison_allowed'))
    checks['fixed_physics_and_full_duration_registered']=all(x in code for x in ('c.mass!=.12','c.friction!=.6','c.timestep!=.0005','s.plant.time-16.'))
    state_code=(PILOT/'state_snapshot.py').read_text(encoding='utf-8')
    checks['full_entry_clock_RNG_queue_and_prefix_compared']=all(x in code+state_code for x in ('mjSTATE_INTEGRATION','qacc_warmstart','bit_generator.state',"skin['fk_data']",'observation_json',"a['entry']==b['entry']"))
    checks['save_and_exception_partial_use_fresh_subdirectories']=all(x in code for x in ("record=target/'record';s.save(record)","partial=target/'partial_record';s.save(partial)"))
    checks['new_online_decoder_does_not_read_object_or_force']=not any(x in (PILOT/'decoder.py').read_text(encoding='utf-8') for x in ("p.object","p.truth(",'is_target','object_position','object_mass','object_friction'))
    if not all(checks.values()):raise AssertionError(checks)
    return dict(schema='release_basis_pure_validation_v2',passed=True,checks=checks,exact_comparisons=comparisons,
        physical_steps=0,plant_constructions=0,scope='native function oracle on algebra-only robot fixture; not physical equivalence/safety evidence',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PILOT.glob('*.py')})


if __name__=='__main__':
    report=verify();out=PILOT/'pure_validation_v2.json'
    if out.exists():raise FileExistsError(out)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
