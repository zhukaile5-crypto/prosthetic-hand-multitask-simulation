"""Optional one-thread GL consumer of archived positions, never a controller.

Importing this file does not import MuJoCo, compile a model or create a renderer.
Only a later explicitly started server's frame request creates visual resources.
"""
from collections import OrderedDict
from concurrent.futures import Future
from io import BytesIO
from queue import Queue
from threading import Thread
import importlib.util,sys

from .reader import sha

CAMERAS={'front':dict(azimuth=90.,elevation=-15.,distance=.68),
         'side':dict(azimuth=0.,elevation=-15.,distance=.68),
         'oblique':dict(azimuth=150.,elevation=-22.,distance=.68)}


def load_snapshot_builder(batch,source):
    """Use the exact executed snapshot's model XML/assets and builder definitions."""
    folder=batch/'sources/foundation'
    for name in ('foundation_config.py','hand_plant.py'):
        if sha(folder/name)!=source['foundation/'+name]:raise ValueError('Executed model-builder snapshot changed')
    suffix=source['foundation/hand_plant.py'][:12]
    def load(name,path):
        if name in sys.modules:return sys.modules[name]
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);return module
    config=load('_replay_config_'+suffix,folder/'foundation_config.py')
    previous=sys.modules.get('foundation_config');sys.modules['foundation_config']=config
    try:plant=load('_replay_plant_'+suffix,folder/'hand_plant.py')
    finally:
        if previous is None:sys.modules.pop('foundation_config',None)
        else:sys.modules['foundation_config']=previous
    return config,plant


class ReplayRenderer:
    """GL/model creation, state writes and scene rendering have one owner."""
    def __init__(self,store,*,width=640,height=440):
        self.store=store;self.width=width;self.height=height;self.queue=Queue(maxsize=12)
        self.thread=Thread(target=self._worker,name='recorded-result-renderer',daemon=True)
        self.started=False

    def start(self):
        if self.started:raise RuntimeError('Renderer worker already started')
        self.started=True;self.thread.start()

    def image(self,scenario,motion,seed,method,time_s,view):
        if view not in CAMERAS:raise ValueError('Unknown view')
        if not self.started:raise RuntimeError('Rendering service has not been explicitly started')
        future=Future();self.queue.put((future,(scenario,motion,seed,method,time_s,view)),timeout=2.)
        return future.result(timeout=45.)

    def close(self):
        if self.started:self.queue.put(None);self.thread.join(timeout=5.)

    def _worker(self):
        renderer=None;model=None;data=None;model_key=None;cache=OrderedDict()
        try:
            while True:
                job=self.queue.get()
                if job is None:break
                future,args=job
                try:
                    scenario,motion,seed,method,time_s,view=args
                    record=self.store.load_trace(scenario,motion,seed,method)
                    index=self.store.index_at(record,time_s)
                    key=(scenario,motion,seed,method,index,view)
                    if key in cache:
                        cache.move_to_end(key);future.set_result(cache[key]);continue
                    # Everything below is visual reconstruction only. No
                    # HandPlant/TrialSession construction, sensor/controller,
                    # mj_step, mj_forward or force recomputation occurs here.
                    import mujoco
                    from PIL import Image
                    config_module,plant_module=load_snapshot_builder(record['batch'],self.store.registration['source_sha256'])
                    config=config_module.FoundationConfig(**record['metadata']['config'])
                    next_key=(record['metadata']['model_sha256'],tuple(vars(config).items()))
                    if next_key!=model_key:
                        if renderer is not None:renderer.close();renderer=None
                        model,identity=plant_module._model(config)
                        if identity!=record['metadata']['model_sha256']:raise ValueError('Recorded and reconstructed visual model differ')
                        data=mujoco.MjData(model)
                        renderer=mujoco.Renderer(model,width=self.width,height=self.height);model_key=next_key
                    q=record['qpos'][index];v=record['qvel'][index];ctrl=record['ctrl'][index]
                    if len(q)!=model.nq or len(v)!=model.nv or len(ctrl)!=model.nu:raise ValueError('Recorded model dimensions differ')
                    data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=ctrl;data.time=float(record['time'][index])
                    # Kinematics updates visible bodies/geometries/cameras;
                    # it does not integrate dynamics or calculate contacts.
                    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                    mujoco.mj_camlight(model,data);mujoco.mj_tendon(model,data)
                    camera=mujoco.MjvCamera();camera.lookat[:]=[.045,0.,.36]
                    for name,value in CAMERAS[view].items():setattr(camera,name,value)
                    option=mujoco.MjvOption()
                    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
                    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    pixels=renderer.render();buffer=BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
                    answer=(buffer.getvalue(),float(record['time'][index]));cache[key]=answer
                    while len(cache)>90:cache.popitem(last=False)
                    future.set_result(answer)
                except Exception as exc:future.set_exception(exc)
        finally:
            if renderer is not None:renderer.close()
