from pathlib import Path
import json,time,os
import numpy as np
import mujoco
from probe import write_png
OUT=Path(__file__).resolve().parent
path=OUT/'menagerie_allegro'/'scene_right.xml'
os.chdir(path.parent)
m=mujoco.MjModel.from_xml_path(path.name);d=mujoco.MjData(m)
mujoco.mj_forward(m,d);before=d.qpos.copy()
target=np.mean(m.actuator_ctrlrange,axis=1)
tic=time.perf_counter()
for _ in range(2000):
    d.ctrl[:]=target
    mujoco.mj_step(m,d)
wall=time.perf_counter()-tic
render='not attempted'
try:
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        cam=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,cam)
        renderer.update_scene(d,camera=cam);write_png(OUT/'allegro_scene.png',renderer.render())
    render='passed'
except Exception as e:render=f'{type(e).__name__}: {e}'
manifest=json.loads((OUT/'menagerie_allegro'/'download_manifest.json').read_text(encoding='utf-8'))
hand_addresses=m.jnt_qposadr[m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE]
result=dict(model='Menagerie Allegro Hand V3, right hand',scope='Load, execute 2000 dynamics steps under position actuator commands, check finite state and offscreen render. No grasp task or physical prosthesis validation.',commit=manifest['commit'],license='BSD-2-Clause',mujoco=mujoco.__version__,nq=m.nq,nv=m.nv,njnt=m.njnt,hand_hinge_joints=len(hand_addresses),nu=m.nu,nsensor=m.nsensor,timestep_s=m.opt.timestep,steps=2000,simulated_s=float(d.time),wall_s=wall,finite=bool(np.isfinite(d.qpos).all()),max_hand_hinge_change_rad=float(np.max(np.abs(d.qpos[hand_addresses]-before[hand_addresses]))),render=render)
result['passed']=bool(result['finite'] and result['max_hand_hinge_change_rad']>.01 and render=='passed')
(OUT/'allegro_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2));assert result['passed']
