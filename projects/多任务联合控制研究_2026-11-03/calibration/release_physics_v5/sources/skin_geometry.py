"""Delayed virtual-skin geometry, shared by every method without object labels.

Known skin link/taxel coordinates are robot measurements. Per-patch pressures
share their finger's existing noisy pressure channel; no new random samples or
target/object state information are introduced.
"""
from collections import deque
from types import SimpleNamespace
import numpy as np
import research_imports


class DelayedSkinGeometry:
    def __init__(self,plant):
        import mujoco
        self.mj=mujoco;self.model=plant.model;self.data=mujoco.MjData(self.model)
        self.qadr=plant.raw_qadr.copy();self.lift_qadr=plant.lift_qadr;self.palm=plant.palm
        self.samples=deque(maxlen=150)

    def acquire(self,raw,frame):
        self.data.qpos[self.qadr]=np.asarray(raw['raw_joint'])
        self.data.qpos[self.lift_qadr]=float(raw['lift'])
        self.mj.mj_kinematics(self.model,self.data)
        R=self.data.xmat[self.palm].reshape(3,3).copy();origin=self.data.xpos[self.palm].copy()
        totals=np.zeros(5);positions=np.zeros((5,3));packets=[]
        for patch in raw['patches']:
            digit=int(patch['digit'])
            if not 0<=digit<5:continue
            pressure=max(0.,float(patch['normal']))
            if pressure<=1e-10:continue
            body=int(self.model.geom_bodyid[int(patch['hand_geom'])])
            if not self.model.body(body).name.startswith('rh_'):
                raise RuntimeError('Virtual skin link is not part of the robot')
            p=np.asarray(patch['position'],float);n=np.asarray(patch['normal_dir'],float)
            Rb=self.data.xmat[body].reshape(3,3).copy();pb=self.data.xpos[body].copy()
            packets.append(dict(digit=digit,robot_body=body,link_point=Rb.T@(origin+R@p-pb),
                link_normal=Rb.T@R@n,pressure=pressure,palm_to_link=Rb.T@R,
                body_rotation=Rb.copy(),body_position=pb.copy()))
            totals[digit]+=pressure;positions[digit]+=pressure*p
        positions/=np.maximum(totals[:,None],1e-12)
        self.samples.append((float(frame.time),packets,totals,positions))
        available=[x for x in self.samples if x[0]<=float(frame.sample_time)+1e-9]
        result=[]
        if available:
            sample,packets,totals,positions=available[-1]
            for patch in packets:
                digit=patch['digit']
                if not frame.valid[digit] or not np.isfinite(frame.normal[digit]):continue
                delta=np.asarray(frame.contact_pos[digit])-positions[digit]
                point=patch['link_point']+patch['palm_to_link']@delta
                result.append(dict(digit=digit,robot_body=patch['robot_body'],
                    link_point=point.tolist(),
                    link_normal=patch['link_normal'].tolist(),
                    reference_position=(patch['body_rotation']@point+patch['body_position']).tolist(),
                    reference_normal=(patch['body_rotation']@patch['link_normal']).tolist(),
                    normal=max(0.,float(frame.normal[digit]))*patch['pressure']/max(1e-12,totals[digit]),
                    sample_time=sample,age=float(frame.time)-sample))
        frame.skin_patches=result
        frame.skin_geometry_scope='known robot skin links; acquisition-time encoder FK; delayed sample; shared finger pressure noise; no external identity'
        return result
