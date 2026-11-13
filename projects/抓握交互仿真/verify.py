"""Deterministic physical on/off comparisons plus actual MuJoCo rendered evidence."""
from pathlib import Path
import json
import mujoco
from simulation import GraspSimulation,png_bytes

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'evidence';DEST.mkdir(exist_ok=True)
results=[]
for module,action in [('slip','disturb'),('force','squeeze'),('gate','release')]:
    for enabled in [True,False]:
        sim=GraspSimulation();sim.command(module,enabled)
        sim.command('grasp');sim.advance(3);baseline=sim.snapshot()
        sim.command(action);sim.advance(1.2)
        trace=[a for a in sim.history if a[0]>=3]
        metrics=dict(over_limit_s=round(sum(max(a[1:3])>sim.limit for a in trace)*.005,3),
            excess_integral_Ns=round(sum(max(0,max(a[1:3])-sim.limit) for a in trace)*.005,4),
            max_force_after_action_N=round(max(max(a[1:3]) for a in trace),3),
            minimum_height_m=min(a[4] for a in trace))
        results.append(dict(module=module,enabled=enabled,action=action,
            baseline={k:baseline[k] for k in ['state','normal','object_z','slip_events']},
            final=sim.snapshot(),metrics=metrics))
        print(module,enabled,metrics)
assert results[0]['final']['state']=='HOLDING' and results[1]['final']['state']=='DROPPED'
assert results[0]['baseline']['slip_events']==0
assert results[2]['final']['force_events']>0
assert results[2]['metrics']['excess_integral_Ns']<results[3]['metrics']['excess_integral_Ns']*.5
assert results[4]['final']['release_slip_events']==0 and results[4]['final']['state']=='RELEASED'
assert results[5]['final']['release_slip_events']>0 and results[5]['final']['object_z']>.19
(DEST/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

steady=GraspSimulation();steady.command('grasp');steady.advance(3)
start_height=steady.object_z;steady.advance(30)
stability=dict(duration_s=30,state=steady.state,height_drift_mm=(steady.object_z-start_height)*1000,
    slip_events=steady.slip_events,force_events=steady.force_events)
assert steady.state=='HOLDING' and abs(steady.object_z-start_height)<.002
(DEST/'stability.json').write_text(json.dumps(stability,indent=2),encoding='utf-8')

sim=GraspSimulation();renderer=mujoco.Renderer(sim.model,height=600,width=900)
frames=DEST/'frames';frames.mkdir(exist_ok=True)
frame_index=0;sequence=[]
def capture(duration,label):
    global frame_index
    for _ in range(round(duration*10)):
        sim.advance(.1)
        (frames/f'{frame_index:04d}.png').write_bytes(png_bytes(sim.render(renderer)))
        sequence.append(dict(frame=frame_index,label=label,time=round(float(sim.data.time),2),
            state=sim.state,normal=[round(float(f),2) for f in sim.normal],
            target=round(sim.force_ref,2),slip_events=sim.slip_events,force_events=sim.force_events))
        frame_index+=1
capture(.5,'READY');sim.command('grasp');capture(3,'GRASP / LIFT')
sim.command('disturb');capture(1.3,'SLIP REFLEX')
sim.command('release');capture(1.2,'RELEASE GATE')
sim.command('reset');sim.command('grasp');capture(2.7,'GRASP / LIFT')
sim.command('squeeze');capture(1.3,'FORCE PROTECTION')
renderer.close()
(DEST/'sequence.json').write_text(json.dumps(sequence,indent=2),encoding='utf-8')
print('PASS: all three comparisons; rendered',frame_index,'frames')
