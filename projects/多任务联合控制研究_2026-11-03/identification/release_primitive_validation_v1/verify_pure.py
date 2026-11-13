"""Pure assignment/clock/entry-permission/clearance tests; never a plant."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from numbers import Real
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from identification.release_primitive_validation_v1 import spec
from identification.release_primitive_validation_v1.clearance_diagnostics import clear_series
PACKAGE=Path(__file__).resolve().parent


@contextmanager
def fake_runtime_modules():
    modules={}
    class FakePlant:
        def __init__(self,config,variant='zero'):
            self.config=config;self.probe_variant=variant;self.model_sha256='same_robot'
            self.control_steps=0;self.release_started=None
            self.model=SimpleNamespace(geom=lambda _:SimpleNamespace(id=91))
        def truth(self):return dict(no_extra_sensor_sample=True)
        def begin_release(self):
            if self.release_started is None:self.release_started=self.control_steps*.005
    class FakeSession:
        def __init__(self,**kwargs):
            self.kwargs=kwargs;self.time=0.;self.tick=0;self.trace=[];self.release_driver=None
            self.config=SimpleNamespace(timestep=.0005,control_dt=.005)
            self.plant=FakePlant(self.config);self.observer_warmup_s=.2
            self.motion_reference_bounds=dict(lower=[0,-1,-1],upper=[.1,1,1])
            self.handoff_passed=False;self.sensors=object();self.original_sensor=self.sensors
            self.controller=object();self.original_controller=self.controller
            self.preview_follower=object();self.original_follower=self.preview_follower
            self.user_trajectory=object();self.original_trajectory=self.user_trajectory
        def metadata(self):return {}
        def step_one_sensor_tick(self):
            self.plant.control_steps=self.tick
            if (self.tick==1500 and not self.handoff_passed) or self.tick==2400:
                self.plant.begin_release()
    runtime=ModuleType('v10.runtime');runtime.V10Session=FakeSession
    runtime.public_motion_reference_bounds=lambda _:dict(lower=[0,-1,-1],upper=[.1,1,1])
    hand=ModuleType('hand_plant');hand.CommonPreparation=lambda p:SimpleNamespace(plant=p)
    skin=ModuleType('skin_geometry');skin.DelayedSkinGeometry=lambda p:SimpleNamespace(plant=p)
    modules.update({'v10.runtime':runtime,'hand_plant':hand,'skin_geometry':skin})
    previous={k:sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:yield FakePlant
    finally:
        for k,v in previous.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v


def verify():
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
    alljobs=spec.jobs(Path('not_executed'))
    check('exact32_unique_prospective_assignments',len(alljobs)==32 and len({j['job_id'] for j in alljobs})==32)
    check('exact16_pairs_with_both_branches',all(sum(j['seed']==seed and j['scenario']==scene and j['motion_mode']==motion for j in alljobs)==2
        for seed in spec.SEEDS for scene in spec.SCENARIOS for motion in spec.MOTION_MODES))
    check('same_preserved_ratio_preview_robustness_capsule_route',all(j['method']=='ratio_preview' and j['split']=='robustness' and j['shape']=='capsule' for j in alljobs))
    for seed in (True,np.bool_(False),'617901',610212,617903):
        try:spec.validate_context('nominal','hold',seed,'zero');ok=False
        except ValueError:ok=True
        check('reject_unregistered_seed_'+str(type(seed))+'_'+str(seed),ok)
    # Decode the unmodified frozen coordinate function directly; no driver/plant
    # module import and no replacement formula in these tests.
    decoder_path=ROOT/'identification/release_local_response_v3/decoder.py'
    tree=ast.parse(decoder_path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinates')
    env=dict(np=np,Real=Real,VARIANTS={'zero':(0.,0.),'lift_slow':(0.,1.)},
             ORIGINS={'phi_150ms':30,'phi_350ms':70},RAMP_TICKS=4,DT=.005)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(decoder_path),'exec'),env)
    for height in (.1,.084):
        d=SimpleNamespace(entry_lift=height,native_lift=lambda t,h=height:float(np.clip(h*(1-t/.6),0,h)))
        rows=[]
        for relative in (69,70,71,72,73,74,75,200):
            tau=(2400+relative)*.005-12.
            rows.append(env['coordinates'](d,tau,relative,'lift_slow',release_control_tick=2400,
                release_time_s=12.,probe_origin_tick=70))
        check(f'origin_offset0_height{height}',rows[1]['actual_lift_offset_m']==0.)
        check(f'entry_lift_scaled_full20ms_height{height}',abs(rows[5]['actual_lift_offset_m']-height/.6*.02)<1e-14)
        check(f'phi_never_changed_height{height}',all(r['actual_phi_offset_s']==0. for r in rows))
        check(f'floor_after_unclipped_clock_height{height}',rows[-1]['lift_reference_m']==0.)
    runner=PACKAGE/'run_validation.py'
    rt=ast.parse(runner.read_text(encoding='utf-8'))
    make=next(n for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=='make_session')
    with fake_runtime_modules() as FakePlant:
        e=dict(spec=spec,make_plant_class=lambda origin:FakePlant,np=np)
        exec(compile(ast.Module(body=[make],type_ignores=[]),str(runner),'exec'),e)
        for variant in spec.VARIANTS:
            j=next(j for j in alljobs if j['scenario']=='nominal' and j['motion_mode']=='transport' and j['variant']==variant)
            for passed in (False,True):
                s=e['make_session'](j,{'mock':'source'})
                check(f'{variant}/{passed}/same_sensor_controller_follower_trajectory',
                    s.sensors is s.original_sensor and s.controller is s.original_controller and
                    s.preview_follower is s.original_follower and s.user_trajectory is s.original_trajectory)
                check(f'{variant}/{passed}/rebound_all_plant_links',s.preparation.plant is s.plant and s.skin_geometry.plant is s.plant and s.obstacle_geom==91)
                s.tick=2400 if passed else 1500;s.plant.control_steps=s.tick;s.handoff_passed=passed
                s.step_one_sensor_tick()
                check(f'{variant}/{passed}/actual_release_clock',s.plant.release_started==(12. if passed else 7.5))
                check(f'{variant}/{passed}/entry_eligibility',s.plant.prospective_entry_eligible_at_release is passed)
                check(f'{variant}/{passed}/candidate_only_after_actual_handoff',s.plant.prospective_offset_enabled is (passed and variant=='lift_slow'))
                if not passed:
                    s.tick=2400;s.plant.control_steps=s.tick;s.handoff_passed=True;s.step_one_sensor_tick()
                    check(f'{variant}/early_native_never_reenabled_at12',s.plant.probe_variant=='zero' and not s.plant.prospective_offset_enabled)
    forbidden={'mass','friction','is_target','external_geom','object_position','object_velocity'}
    gate=[n for n in ast.walk(make) if isinstance(n,ast.FunctionDef) and n.name in ('step_one_sensor_tick','begin_release')]
    check('entry_gate_has_no_object_or_material_truth_inputs',not any(isinstance(n,ast.Attribute) and n.attr in forbidden for f in gate for n in ast.walk(f)))
    check('no_changed_legacy_method_route',not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='method' for t in n.targets) for n in ast.walk(make)))
    check('old_V3_decoder_used_without_new_response_model',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_plant_class' for n in ast.walk(make)))
    # First clear, subsequent recontact, terminal dwell and coarse missed touch.
    t=np.arange(201)*.005;f=np.zeros(len(t));f[(t>=.3)&(t<.4)]=1.
    c=clear_series(t,f,0.,.2,.2)
    check('first_clear_not_final_clear',abs(c['first_clear_confirmation_s']-.2)<1e-12 and abs(c['terminal_clear_confirmation_s']-.6)<1e-12)
    check('recontact_episode_retained',c['recontact_count']==1)
    f[-10:]=0.;f[-50:-10]=1.;c=clear_series(t,f,0.,.2,.2)
    check('short_terminal_clear_is_censored',c['terminal_clear_confirmation_s'] is None and c['terminal_clear_censored'])
    tf=np.arange(2001)*.0005;ff=np.zeros(len(tf));ff[205]=.3
    coarse=clear_series(tf[::10],ff[::10],0.,.2,.2);fine=clear_series(tf,ff,0.,.2,.2)
    check('200Hz_can_miss_touch_inside_first_dwell',abs(coarse['first_clear_confirmation_s']-.2)<1e-12 and fine['first_clear_confirmation_s']>.3 and coarse['recontact_count']==0)
    try:clear_series(np.delete(t,70),np.delete(f,70),0.,.2,.2);ok=False
    except ValueError:ok=True
    check('missing_raw_clock_never_implies_clearance',ok)
    late_t=np.arange(1001)*.005;late_f=np.where(late_t<3.6,1.,0.)
    late=clear_series(late_t,late_f,0.,.2,.2,3.5)
    check('late_any_clear_is_not_deadline_success',late['first_any_confirmation_s'] is not None and
        late['first_by_deadline_confirmation_s'] is None and late['not_observed_by_deadline'])
    plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'))
    check('selected_on_seen_V3_explicitly_disclosed','selected after seeing' in plan['selection_disclosure'])
    check('fresh_seed_whole_native_reference_not_fabricated','UNAVAILABLE_NEW_SEED' in plan['zero_native_validation'])
    check('one_bridge_separate_from32_scientific','total33 planned' in plan['bridge_regression'])
    check('frozen_V3_metrics_not_replaced',plan['model_admitted'] is False and 'Keep original V3first_dwell unchanged' in plan['additional_clearance_diagnostic'])
    check('first_dwell_raw_internal_touch_predeclared','INSIDE' in plan['additional_clearance_diagnostic'])
    check('deadline_limited_and_low_force_not_gap_predeclared','first_any' in plan['additional_clearance_diagnostic'] and
        'not geometric gap' in plan['additional_clearance_diagnostic'])
    check('all_required_source_files_exist',all((PACKAGE/name).is_file() for name in (
        '__init__.py','spec.py','provenance.py','run_validation.py','clearance_diagnostics.py',
        'verify_pure.py','plan.json','README.md')))
    return dict(schema='release_primitive_prospective_pure_qa_v1',passed=all(c['passed'] for c in checks),
        checks=checks,physics_steps=0,actual_plant_constructions=0,
        role='runner author pure QA; independent peer review and root real-model preflight still separate',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PACKAGE.iterdir() if p.suffix=='.py' or p.name in ('plan.json','README.md')})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--check-sources',action='store_true');args=a.parse_args()
    report=verify()
    if args.check_sources:
        from identification.release_primitive_validation_v1.provenance import identity
        report['actual_source_identity']=identity()
    path=PACKAGE/'qa'/('pure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    path.parent.mkdir(exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(path),passed=report['passed'],checks=len(report['checks']),
        failed=[c['name'] for c in report['checks'] if not c['passed']],physics_steps=0)))
    raise SystemExit(0 if report['passed'] else 2)
