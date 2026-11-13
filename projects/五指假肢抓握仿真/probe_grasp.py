from pathlib import Path
import os,json
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
m=mujoco.MjModel.from_xml_path('model.xml');d=mujoco.MjData(m)
print('nq nv nu',m.nq,m.nv,m.nu)
print('actuators',[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)])
obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
def aid(name):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'rh_A_'+name)
def forces():
    f=np.zeros(5)
    for i in range(d.ncon):
        c=d.contact[i]
        if gid not in [c.geom1,c.geom2]:continue
        other=c.geom2 if c.geom1==gid else c.geom1
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[other]) or ''
        for j,pre in enumerate(['rh_th','rh_ff','rh_mf','rh_rf','rh_lf']):
            if name.startswith(pre):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);f[j]+=abs(force[0])
    return f
targets={'THJ5':.8,'THJ4':.8,'THJ3':0.,'THJ2':-.2,'THJ1':.5,
         'FFJ4':0,'FFJ3':.6,'FFJ0':2.,'MFJ4':0,'MFJ3':.6,'MFJ0':2.,
         'RFJ4':0,'RFJ3':.6,'RFJ0':2.,'LFJ5':.1,'LFJ4':0,'LFJ3':.7,'LFJ0':2.}
for name,target in targets.items():d.ctrl[aid(name)]=target
lift=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'lift_motor')
fore=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'rh_forearm')
for i in range(6000):
    d.ctrl[lift]=m.body_subtreemass[fore]*9.81/1500+(.1*min(1,(i-2500)/1200) if i>2500 else 0)
    mujoco.mj_step(m,d)
    if i%500==499:print(round(d.time,2),'z',round(d.xpos[obj,2],4),'F',forces().round(2))
renderer=mujoco.Renderer(m,height=700,width=1000);cam=mujoco.MjvCamera();cam.lookat[:]=[.035,0,.30];cam.azimuth=35;cam.elevation=-15;cam.distance=.8
renderer.update_scene(d,camera=cam)
import sys
sys.path.insert(0,str(ROOT.parent/'抓握交互仿真'))
from simulation import png_bytes
(ROOT/'probe.png').write_bytes(png_bytes(renderer.render()));renderer.close()
