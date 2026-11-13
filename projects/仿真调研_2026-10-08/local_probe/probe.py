"""Environment feasibility probe only; this is not a validated prosthetic model."""
from pathlib import Path
import json, time, platform, csv, zlib, struct
import numpy as np
import mujoco

OUT=Path(__file__).resolve().parent
DROP='''<mujoco model="drop_probe"><option timestep="0.001" gravity="0 0 -9.81"/><worldbody><geom name="ground" type="plane" size="1 1 .1"/><body name="ball" pos="0 0 .4"><freejoint/><geom type="sphere" size=".03" mass="1"/></body></worldbody></mujoco>'''
GRIP='''<mujoco model="two_jaw_environment_probe">
<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
<default><joint damping="1"/><geom friction=".8 .005 .0001" solref=".01 1" condim="3"/></default>
<worldbody><light pos="0 -1 2"/><geom name="floor" type="plane" size=".3 .3 .1" rgba=".86 .88 .9 1"/>
<body name="left" pos="-.035 0 .15"><joint name="left_slide" type="slide" axis="1 0 0" range="0 .025"/><geom name="left_pad" type="box" size=".005 .025 .035" mass=".1" rgba=".16 .4 .7 1"/></body>
<body name="right" pos=".035 0 .15"><joint name="right_slide" type="slide" axis="-1 0 0" range="0 .025"/><geom name="right_pad" type="box" size=".005 .025 .035" mass=".1" rgba=".16 .4 .7 1"/></body>
<body name="object" pos="0 0 .15"><freejoint name="object_free"/><geom name="object_geom" type="box" size=".015 .015 .018" mass=".1" rgba=".95 .55 .18 1"/></body>
</worldbody><actuator><motor joint="left_slide" ctrlrange="-4 4"/><motor joint="right_slide" ctrlrange="-4 4"/></actuator></mujoco>'''

def drop(dt):
    m=mujoco.MjModel.from_xml_string(DROP);m.opt.timestep=dt;d=mujoco.MjData(m)
    tic=time.perf_counter(); max_contact=0
    for _ in range(round(2/dt)):
        mujoco.mj_step(m,d);max_contact=max(max_contact,d.ncon)
    elapsed=time.perf_counter()-tic
    return dict(dt_s=dt,final_z_m=float(d.qpos[2]),finite=bool(np.isfinite(d.qpos).all()),max_contacts=max_contact,wall_s=elapsed,simulated_s=float(d.time)),d.qpos.copy()

def write_png(path,pixels):
    h,w,_=pixels.shape
    def chunk(t,data): return struct.pack('!I',len(data))+t+data+struct.pack('!I',zlib.crc32(t+data)&0xffffffff)
    raw=b''.join(b'\x00'+row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))

def grip():
    (OUT/'probe_model.xml').write_text(GRIP,encoding='utf-8')
    m=mujoco.MjModel.from_xml_string(GRIP);d=mujoco.MjData(m)
    d.qpos[:2]=.015
    mujoco.mj_forward(m,d)
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object')
    object_geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
    rows=[];snap=None;tic=time.perf_counter()
    for step in range(3000):
        d.ctrl[:]=2 if d.time<2 else -2
        mujoco.mj_step(m,d)
        fn=0.;contacts=0
        for i in range(d.ncon):
            c=d.contact[i]
            if object_geom in (c.geom1,c.geom2):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);fn+=abs(float(force[0]));contacts+=1
        rows.append([float(d.time),float(d.xpos[body,2]),float(d.qpos[0]),float(d.qpos[1]),fn,contacts])
        if step==1499:snap=d.qpos.copy()
    wall=time.perf_counter()-tic
    with (OUT/'probe_trace.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['time_s','object_z_m','left_slide_m','right_slide_m','sum_object_contact_normal_N','contact_points']);w.writerows(rows)
    held=min(r[1] for r in rows if .5<r[0]<1.9)
    render='not attempted'
    try:
        d.qpos[:]=snap;mujoco.mj_forward(m,d)
        with mujoco.Renderer(m,height=480,width=640) as renderer:
            cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.12];cam.distance=.45;cam.azimuth=100;cam.elevation=-25
            renderer.update_scene(d,camera=cam);write_png(OUT/'probe_scene.png',renderer.render())
        render='passed: offscreen PNG generated'
    except Exception as e:render=f'failed: {type(e).__name__}: {e}'
    return dict(min_hold_z_m=held,final_release_z_m=rows[-1][1],held=held>.12,released=rows[-1][1]<.04,finite=bool(np.isfinite(rows).all()),wall_s=wall,simulated_s=3,render=render)

if __name__=='__main__':
    a,qa=drop(.001);b,qb=drop(.001);fine,qf=drop(.0005)
    result=dict(scope='Environment smoke test: gravity, contacts, motor actuation, release, deterministic replay and offscreen rendering. Not a prosthesis validation.',python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,drop=a,deterministic_max_q_difference=float(np.max(np.abs(qa-qb))),drop_dt_halving_final_z_difference_m=abs(a['final_z_m']-fine['final_z_m']),grip=grip())
    result['passed']=bool(a['finite'] and a['max_contacts']>0 and abs(a['final_z_m']-.03)<.002 and result['deterministic_max_q_difference']==0 and result['grip']['held'] and result['grip']['released'] and result['grip']['finite'])
    (OUT/'probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['passed'], 'Environment probe did not pass'
