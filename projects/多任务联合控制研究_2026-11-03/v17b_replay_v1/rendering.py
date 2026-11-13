"""Lazy single-owner visual reconstruction; never steps or solves physics.

Independent camera/cache globals. The existing viewer is not patched. Only
the immutable-snapshot loader is reused, after checking its declared source.
"""
from collections import OrderedDict
from concurrent.futures import Future
from io import BytesIO
from queue import Queue
from threading import Thread
from pathlib import Path
from .reader import ROOT,sha

OLD_RENDERER_SHA='0f1ea693f8a90d3e063c626f770f259f323efcc6c81403ee9d2c88cb62694db0'
CAMERAS={'front':dict(azimuth=90.,elevation=-15.,distance=.68,lookat=(.045,0.,.36)),
         'side':dict(azimuth=0.,elevation=-15.,distance=.68,lookat=(.045,0.,.36)),
         'oblique':dict(azimuth=150.,elevation=-22.,distance=.68,lookat=(.045,0.,.36)),
         'wide':dict(azimuth=145.,elevation=-18.,distance=1.05,lookat=(.045,0.,.20))}

class ReplayRenderer:
    def __init__(self,store,*,width=560,height=420):
        self.store=store;self.width=width;self.height=height
        self.queue=Queue(maxsize=12);self.started=False
        self.thread=Thread(target=self._worker,name='v17b-recorded-render-owner',daemon=True)
    def start(self):
        if self.started:raise RuntimeError('Renderer already started')
        self.started=True;self.thread.start()
    def image(self,scenario,motion,seed,method,time_s,view):
        self.store.validate(scenario,motion,seed,method)
        if view not in CAMERAS:raise ValueError('Unknown view')
        if not self.started:raise RuntimeError('Renderer not explicitly started')
        future=Future();self.queue.put((future,(scenario,motion,seed,method,time_s,view)),timeout=2.)
        return future.result(timeout=60.)
    def close(self):
        if self.started:self.queue.put(None);self.thread.join(timeout=5.)
    def _worker(self):
        renderer=None;model=None;data=None;model_key=None;cache=OrderedDict()
        try:
            while True:
                item=self.queue.get()
                if item is None:break
                future,args=item
                try:
                    scenario,motion,seed,method,time_s,view=args
                    record=self.store.load_trace(scenario,motion,seed,method);i=self.store.index_at(record,time_s)
                    key=(*args[:4],i,view)
                    if key in cache:cache.move_to_end(key);future.set_result(cache[key]);continue
                    # Imports, model compile, and GL are reached only after an
                    # explicitly started server receives a frame request.
                    import mujoco
                    from PIL import Image
                    if sha(ROOT/'visual_replay_v1/rendering.py')!=OLD_RENDERER_SHA:raise ValueError('Snapshot-loader dependency changed')
                    from visual_replay_v1.rendering import load_snapshot_builder
                    config_module,plant_module=load_snapshot_builder(record['batch'],self.store.registration['source_sha256'])
                    config=config_module.FoundationConfig(**record['metadata']['config'])
                    next_key=(record['metadata']['model_sha256'],tuple(vars(config).items()))
                    if next_key!=model_key:
                        if renderer is not None:renderer.close();renderer=None
                        model,identity=plant_module._model(config)
                        if identity!=record['metadata']['model_sha256']:raise ValueError('Archived visual model differs')
                        data=mujoco.MjData(model);renderer=mujoco.Renderer(model,width=self.width,height=self.height);model_key=next_key
                    q=record['qpos'][i];v=record['qvel'][i];u=record['ctrl'][i]
                    if len(q)!=model.nq or len(v)!=model.nv or len(u)!=model.nu:raise ValueError('Model/state dimensions differ')
                    data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=u;data.time=float(record['time'][i])
                    # Only visible transforms/tendon paths; no forward/contact
                    # solve, actuator calculation, state integration or sensor.
                    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                    mujoco.mj_camlight(model,data);mujoco.mj_tendon(model,data)
                    camera=mujoco.MjvCamera();camera.lookat[:]=CAMERAS[view]['lookat']
                    for name,value in CAMERAS[view].items():
                        if name!='lookat':setattr(camera,name,value)
                    option=mujoco.MjvOption();option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
                    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    buffer=BytesIO();Image.fromarray(renderer.render()).save(buffer,format='PNG')
                    answer=(buffer.getvalue(),float(record['time'][i]));cache[key]=answer
                    while len(cache)>64:cache.popitem(last=False)
                    future.set_result(answer)
                except Exception as exc:future.set_exception(exc)
        finally:
            if renderer is not None:renderer.close()
