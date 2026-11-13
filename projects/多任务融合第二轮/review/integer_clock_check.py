"""Regression for canonical control ticks and untouched raw physics time/state.

--prototype executes the prepared change in an isolated module, never editing
physics.py. --live verifies the parent-approved core after patch application.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import sys
import types
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scenarios import SCENARIOS,parameters

def sha(value):
    return hashlib.sha256(value).hexdigest()

def prepare(source):
    changes=[
        ('        self.data = mujoco.MjData(self.model)\n',
         '        self.data = mujoco.MjData(self.model)\n        self._control_tick = 0\n'),
        ('    def time(self):\n        return float(self.data.time)\n',
         '    def time(self):\n        """Canonical control clock; raw mjData.time remains the integrator clock."""\n        return self._control_tick * DT\n'),
        ('            solver_time=self.time\n','            solver_time=float(self.data.time)\n'),
        ('        self.record_contact_diagnostics(self.time,0.)\n',
         '        self.record_contact_diagnostics(float(self.data.time),0.)\n        self._control_tick += 1\n'),
    ]
    for old,new in changes:
        if source.count(old)!=1:
            raise AssertionError(f'Unexpected prototype source match count: {old!r}')
        source=source.replace(old,new)
    return source

def isolated_module(source,name):
    module=types.ModuleType(name)
    module.__file__=str(ROOT/'physics.py')
    sys.modules[name]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module

def event_table(times):
    windows=[('pulse1',9.,9.45),('pulse2',11.,11.45),('pulse3',13.,13.45),
        ('conflict',9.,9.65),('release_pull',11.6,12.6),
        ('dropout_partial',9.3,9.7),('dropout_all',12.,12.15),
        ('slip_noise',9.5,9.62),('force_outlier',11.,11.03)]
    rows=[]
    for name,begin,end in windows:
        active=np.flatnonzero((times>=begin)&(times<end))
        rows.append(dict(event=name,first_tick=int(active[0]),
            first_inactive_tick=int(active[-1]+1),duration_ticks=len(active),
            duration_s=len(active)*.005))
    rows.append(dict(event='friction_drop',first_tick=int(np.flatnonzero(times>=9.)[0])))
    return rows

def model_fields(model):
    names=('geom_type','geom_size','geom_pos','geom_quat','geom_solref','geom_solimp',
        'geom_friction','geom_contype','geom_conaffinity','body_mass','body_inertia',
        'actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange')
    fields={name:getattr(model,name).tolist() for name in names}
    fields['option']={name:getattr(model.opt,name) for name in
        ('timestep','integrator','cone','impratio','iterations','noslip_iterations','tolerance')}
    return fields

def run_check(legacy,new,dt):
    scenario=SCENARIOS[0];p=parameters(scenario,1104)
    a,b=legacy.Plant(scenario,p,dt),new.Plant(scenario,p,dt)
    assert model_fields(a.model)==model_fields(b.model),'Geometry/solver/actuator changed'
    env=dict(friction=p['friction'],obstacle=np.array([.28,0.,.49]),
             force=np.zeros(3),squeeze=0.)
    max_state_difference=0.;max_clock_difference=0.;clocks=[];raw=[]
    assert b._control_tick==0 and b.time==0.
    for tick in range(3401):
        assert b._control_tick==tick and b.time==tick*.005
        clocks.append(b.time);raw.append(float(b.data.time))
        max_clock_difference=max(max_clock_difference,abs(b.time-float(b.data.time)))
        max_state_difference=max(max_state_difference,
            float(np.max(abs(a.data.qpos-b.data.qpos))),
            float(np.max(abs(a.data.qvel-b.data.qvel))))
        assert a.data.time==b.data.time,'Raw mjData.time was changed'
        if tick<3400:
            a.advance(env);b.advance(env)
    assert max_state_difference<1e-9
    assert max_clock_difference<1e-9
    records=np.asarray(b.substep_records)
    dtrows=records[records[:,5]>0]
    endpoint=records[records[:,5]==0]
    assert len(dtrows)==3400*round(.005/dt)
    assert len(endpoint)==3400
    assert np.all(np.diff(dtrows[:,0])>0),'Substep timestamps collapsed to control clock'
    assert np.max(abs(np.diff(dtrows[:,0])-dt))<1e-9
    return dict(timestep=dt,control_ticks=3400,first_control_times=clocks[:3],
        raw_integrator_clock_first_times=raw[:3],
        max_state_difference=max_state_difference,
        max_control_vs_integrator_time_difference_s=max_clock_difference,
        event_windows=event_table(np.asarray(clocks)),
        substep_records=len(dtrows),endpoint_records=len(endpoint),
        model_fields_sha256=sha(json.dumps(model_fields(b.model),sort_keys=True).encode()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prototype',action='store_true')
    group.add_argument('--live',action='store_true')
    args=parser.parse_args()
    physics=ROOT/'physics.py'
    before_hash=sha(physics.read_bytes())
    if args.prototype:
        legacy_source=physics.read_text(encoding='utf-8')
        new_source=prepare(legacy_source)
        patch=''.join(difflib.unified_diff(legacy_source.splitlines(True),
            new_source.splitlines(True),fromfile='physics.py',tofile='physics.py'))
        (ROOT/'review'/'integer_clock.patch').write_text(patch,encoding='utf-8')
    else:
        snapshot=ROOT/'review'/'development_runtime_v2'/'physics.py'
        legacy_source=snapshot.read_text(encoding='utf-8')
        new_source=physics.read_text(encoding='utf-8')
    old=isolated_module(legacy_source,'legacy_physics_clock_test')
    new=isolated_module(new_source,'canonical_physics_clock_test')
    results=[run_check(old,new,dt) for dt in (.0005,.00025)]
    assert results[0]['event_windows']==results[1]['event_windows']
    assert before_hash==sha(physics.read_bytes()),'Core changed during clock check'
    report=dict(mode='prototype' if args.prototype else 'live',
        core_sha256=before_hash,legacy_source_sha256=sha(legacy_source.encode()),
        canonical_source_sha256=sha(new_source.encode()),
        raw_physics_clock_not_reset=True,models_and_solvers_unchanged=True,
        all_event_ticks_and_durations_match=True,results=results,
        scope='Identical constant inputs show clock implementation leaves raw integration/state unchanged; controlled event dynamics intentionally change after fixing timing.')
    target=ROOT/'review'/('integer_clock_prototype_validation.json' if args.prototype else 'integer_clock_validation.json')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
