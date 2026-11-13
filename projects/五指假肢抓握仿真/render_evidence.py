"""Render real trajectories separately from the physical acceptance comparisons."""
from pathlib import Path
import json,mujoco
from simulation import GraspSimulation,png_bytes
ROOT=Path(__file__).resolve().parent
def render_sequence():
    s=GraspSimulation();renderer=mujoco.Renderer(s.model,height=700,width=1000)
    dest=ROOT/'evidence';frames=dest/'frames';frames.mkdir(parents=True,exist_ok=True)
    sequence=[]
    def capture(seconds,label):
        for _ in range(round(seconds*8)):
            s.advance(.125);index=len(sequence)
            (frames/f'{index:04d}.png').write_bytes(png_bytes(s.render(renderer)))
            snap=s.snapshot()
            sequence.append(dict(frame=index,label=label,time=snap['time'],state=snap['state'],normal=snap['normal'],lift_mm=snap['lift_mm'],slip_events=snap['slip_events'],force_events=snap['force_events'],release_slip_events=snap['release_slip_events']))
    capture(.5,'OPEN HAND');s.command('grasp');capture(8,'FIVE-FINGER GRASP / LIFT')
    s.command('disturb');capture(2,'SLIP REFLEX');s.command('release');capture(3,'RELEASE GATE')
    s.command('grasp');s.advance(8);capture(.5,'FIVE-FINGER HOLD');s.command('squeeze');capture(2,'FORCE PROTECTION')
    renderer.close();(dest/'sequence.json').write_text(json.dumps(sequence,indent=2),encoding='utf-8')
    print('Rendered',len(sequence),'actual frames.',flush=True)
if __name__=='__main__':render_sequence()
