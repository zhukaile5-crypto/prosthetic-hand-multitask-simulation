from pathlib import Path
import os,sys
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT)
m=mujoco.MjModel.from_xml_path('model.xml');d=mujoco.MjData(m)
obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
prefix=['rh_th','rh_ff','rh_mf','rh_rf','rh_lf'];norm=np.zeros(5);filt=np.zeros(5)
groups=[['THJ5','THJ4','THJ3','THJ2','THJ1'],['FFJ4','FFJ3','FFJ0'],['MFJ4','MFJ3','MFJ0'],['RFJ4','RFJ3','RFJ0'],['LFJ5','LFJ4','LFJ3','LFJ0']]
op=[[.2,1.15,.1,.2,1.15],[0,0,0],[0,0,0],[0,0,0],[.1,0,0,0]]
cl=[[1.047,1.15,.1,.2,1.15],[0,.6,2],[0,.6,2],[0,.6,2],[.1,0,.7,2]]
ids=[[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'rh_A_'+key) for key in keys] for keys in groups]
alpha=np.array([.3,.5,.5,.5,.5]);
for group,values in zip(groups,op):
 for key,val in zip(group,values):
  if key.endswith('J0'):continue
  j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'rh_'+key);d.qpos[m.jnt_qposadr[j]]=val
mujoco.mj_forward(m,d)
ref=np.array([3.,1.8,1.8,1.8,1.8]);start=None
lift=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'lift_motor');fore=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'rh_forearm')
def observe():
 global norm,filt
 norm=np.zeros(5)
 for i in range(d.ncon):
  c=d.contact[i]
  if gid not in [c.geom1,c.geom2]:continue
  other=c.geom2 if c.geom1==gid else c.geom1
  name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[other]) or ''
  for j,pre in enumerate(prefix):
   if name.startswith(pre):
    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);norm[j]+=abs(force[0])
 filt=.7*filt+.3*norm
for i in range(16000):
 if i%5==0:
  observe();alpha=np.clip(alpha+np.clip((ref-filt)*.001,-.006,.004),[0,.45,.45,.45,.45],1.2)
  for j,indices in enumerate(ids):d.ctrl[indices]=np.array(op[j])+alpha[j]*(np.array(cl[j])-op[j])
  if start is None and min(filt)>.4:start=d.time
  d.ctrl[lift]=m.body_subtreemass[fore]*9.81/1500+(min(.10,max(0,(d.time-start-1)*.07)) if start else 0)
 mujoco.mj_step(m,d)
 if i%2000==1999:print(round(d.time,2),round(d.xpos[obj,2],4),norm.round(2),alpha.round(3),'start',start)
renderer=mujoco.Renderer(m,height=700,width=1000);cam=mujoco.MjvCamera();cam.lookat[:]=[.035,0,.30];cam.azimuth=35;cam.elevation=-15;cam.distance=.8
renderer.update_scene(d,camera=cam);sys.path.insert(0,str(ROOT.parent/'抓握交互仿真'));from simulation import png_bytes
(ROOT/'tune.png').write_bytes(png_bytes(renderer.render()));renderer.close()
