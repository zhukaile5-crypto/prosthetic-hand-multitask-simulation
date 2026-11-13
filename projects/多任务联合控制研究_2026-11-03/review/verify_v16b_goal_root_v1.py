"""Actual archived public entry tests; no model construction or integration."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry,release_nominal
from v16b_release_goal.goal import release_clearance_goal
from v16b_release_goal.runtime import source_identity
from review.run_v16_release_small_v1 import previous_A
import research_imports
from foundation_config import OPEN

checks={};rows=[];preparation=np.concatenate(OPEN)
for scene,motion in [('fragile_conflict','hold'),('nominal','transport'),('friction_drop','transport')]:
    folder=previous_A(scene,motion,619732)
    with np.load(folder/'trace.npz',allow_pickle=False) as trace:
        i=int(np.flatnonzero(trace['time']>=12.-1e-9)[0])
        frame=json.loads(str(trace['observation_json'][i]));prior=json.loads(str(trace['observation_json'][i-1]))
    current=np.asarray(frame['release_entry_reference21'],float)
    entry=ReleaseEntry(12.,Reference21(current,12.),int(frame['mode_generation']))
    lo=np.asarray(prior['reference_lower21'],float);hi=np.asarray(prior['reference_upper21'],float)
    goal=release_clearance_goal(entry,preparation_open18=preparation,mechanical_lower21=lo,mechanical_upper21=hi)
    old=release_nominal(entry,16.,preparation,lo,hi)
    name=scene+'/'+motion
    checks[name+'_same_frozen_terminal']=bool(np.array_equal(goal.reference18,old.values[:18]))
    checks[name+'_different_preparation_thumb']=bool(not np.array_equal(goal.reference18[:5],preparation[:5]))
    checks[name+'_real_mechanical_box']=bool(np.all(goal.reference18>=lo[:18]) and np.all(goal.reference18<=hi[:18]))
    checks[name+'_entry_unmodified']=bool(np.array_equal(entry.reference21.values,current))
    checks[name+'_no_issued_future_command']=bool(goal.to_public()['extraction_is_not_a_command_or_future_measurement'])
    rows.append(dict(context=name,goal=goal.to_public(),source_trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest()))
pins=source_identity(['ratio_effort'])
frozen=json.loads((ROOT/'review/v16_release_small_plan_v1.json').read_text(encoding='utf-8'))['source_sha256']
checks['all_parent123_unchanged']=all(pins.get(k)==v for k,v in frozen.items())
checks['closure126']=len(pins)==126
record=dict(schema='v16b_goal_root_pure_v1',passed=all(checks.values()),checks=checks,rows=rows,
    source_sha256=pins,models_created=0,physics_steps=0,scope='Three archived public release entries; same saturated endpoint, not physical performance')
(ROOT/'review/v16b_goal_root_pure_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=record['passed'],checks=len(checks),parent123_unchanged=checks['all_parent123_unchanged'],source_count=len(pins)),ensure_ascii=True))
