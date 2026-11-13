"""Fixed-snapshot transmission cache; F067 native value arithmetic is unchanged.

All finite candidates, native residual/regularizer operations and inverse
selection are inherited from frozen native_effort21. The private servo calls
the original RobotServoEffort.evaluate; only its pure robot transmission
subroutine reuses immutable A/length/speed for identical q/v/model. No beta
collapse, value-only surrogate, writer, world kinematics, or dynamics is added.
"""
from dataclasses import replace
import hashlib
import numpy as np
from v11.robot_servo_effort import RobotServoEffort,EffortSnapshot
from v13_a1.codec import ReferenceActor21
from .native_effort21 import NativeEffort21


def _selector(ids):
    ids=np.unique(np.asarray(ids,int))
    if len(ids) and np.array_equal(ids,np.arange(ids[0],ids[-1]+1)):return slice(int(ids[0]),int(ids[-1])+1)
    ids.setflags(write=False);return ids


class _FixedSnapshotServo(RobotServoEffort):
    """Private evaluate-compatible servo; no externally supplied MjData."""
    def __init__(self,model):
        super().__init__(model)
        self._bound_model=self.model
        self._guard_getters=[]
        m=self.model;a=_selector(self.aids);j=_selector(self.jids)
        ctrl=_selector(self.ctrladr);out=_selector(self.outadr)
        def bind(name,selection):
            self._guard_getters.append((name,lambda name=name,selection=selection:np.asarray(getattr(self.model,name))[selection]))
        for name in ('actuator_gainprm','actuator_biasprm','actuator_gaintype','actuator_biastype',
                     'actuator_ctrlnum','actuator_ctrladr','actuator_outnum','actuator_outadr',
                     'actuator_dyntype','actuator_actnum','actuator_actadr','actuator_delay','actuator_plugin','actuator_group'):
            bind(name,a)
        for name in ('actuator_ctrlrange','actuator_ctrllimited'):bind(name,ctrl)
        for name in ('actuator_forcerange','actuator_forcelimited'):bind(name,out)
        # Include the one external port's routing ONLY, to preserve F067's
        # admission boundary. Never inspect its command or environment state.
        for name in ('actuator_trntype','actuator_trnid','actuator_gear'):bind(name,slice(None))
        for name in ('jnt_type','jnt_actfrclimited','jnt_actgravcomp','jnt_qposadr','jnt_dofadr','jnt_limited','jnt_range'):bind(name,j)
        for name in ('tendon_adr','tendon_num','tendon_actfrclimited','wrap_type','wrap_objid','wrap_prm'):bind(name,slice(None))
        other=sorted(set(range(m.nactuator))-set(self.aids))
        other_targets=np.asarray([int(m.actuator_trnid[i,0]) for i in other],int)
        self._external_target_ids=other_targets.copy()
        self._hand_body_id=int(m.body('rh_forearm').id)
        self._cache_key=None;self._cache_value=None
        self.evaluation_calls=0;self.transmission_hits=0;self.transmission_misses=0
        self.qv_invalidations=0;self.model_invalidations=0;self.explicit_invalidations=0
        self._initial_guard=self.model_guard_signature();self._active_guard=self._initial_guard

    def model_guard_signature(self):
        if self.model is not self._bound_model:return 'foreign_model_instance'
        m=self.model;h=hashlib.sha256()
        for name,getter in self._guard_getters:
            array=getter();h.update(name.encode());h.update(array.tobytes())
        if len(self._external_target_ids):
            h.update(np.asarray(m.jnt_type[self._external_target_ids]).tobytes())
        h.update(repr((m.nu,m.nout,m.nactuator,m.ntendon,m.njnt,m.nv,
            int(m.opt.disableflags),int(m.opt.enableflags),int(m.opt.disableactuator),
            float(m.body_subtreemass[self._hand_body_id]))).encode())
        return h.hexdigest()

    def clear_transmission_cache(self,*,reason='explicit'):
        if self._cache_key is not None:
            if reason=='model':self.model_invalidations+=1
            elif reason=='qv':self.qv_invalidations+=1
            else:self.explicit_invalidations+=1
        self._cache_key=None;self._cache_value=None

    def _transmission(self,q,v):
        key=(self._active_guard,q.tobytes(),v.tobytes())
        if key==self._cache_key:
            self.transmission_hits+=1
            return self._cache_value
        if self._cache_key is not None:self.clear_transmission_cache(reason='qv')
        arrays=super()._transmission(q,v)
        # No receiver may mutate a cached value through a returned array view.
        frozen=[]
        for value in arrays:
            value=value.copy();value.setflags(write=False);frozen.append(value)
        self._cache_key=key;self._cache_value=tuple(frozen);self.transmission_misses+=1
        return self._cache_value

    def evaluate(self,q25,v25,actual_ctrl,*,control_layout='robot21',time_s=0.):
        self.evaluation_calls+=1;guard=self.model_guard_signature()
        if guard!=self._initial_guard:
            self.clear_transmission_cache(reason='model')
            return EffortSnapshot('UNAVAILABLE',('prepared_bound_robot_model_or_admission_changed',),robot_axes25=self.axes25)
        self._active_guard=guard;hits=self.transmission_hits
        # Native g*u+b0+b1*length+b2*speed, force clipping, transmission,
        # derivatives and immutable snapshot construction are the frozen body.
        sample=super().evaluate(q25,v25,actual_ctrl,control_layout=control_layout,time_s=time_s)
        hit=self.transmission_hits>hits
        diagnostics=dict(sample.diagnostics)
        diagnostics.update(native_value_arithmetic='unmodified_f067_evaluate',prepared_transmission_cache_hit=hit,
            prepared_cache_private_data_scope='Private cache data is not a live plant or an externally writable observation interface')
        if hit:
            diagnostics['pipeline']=['same_q_v_model_immutable_transmission_cache',
                'unchanged_F067_affine_force_clipping_derivative_and_snapshot_arithmetic']
        return replace(sample,diagnostics=diagnostics)

    def cache_diagnostics(self):
        return dict(evaluation_calls=self.evaluation_calls,transmission_hits=self.transmission_hits,
            transmission_misses=self.transmission_misses,qv_invalidations=self.qv_invalidations,
            model_invalidations=self.model_invalidations,explicit_invalidations=self.explicit_invalidations,
            cache_populated=self._cache_key is not None)


class PreparedNativeEffort21(NativeEffort21):
    """Same API/finite candidate algorithm as 6f097; transmission-only cache.

    Domains belong to one solver instance. Known model changes invalidate the
    cache and the domain, including actuator_group/disableactuator admission.
    No exact-zero shortcut is enabled in this initial isolated optimization.
    """
    def __init__(self,model):
        self.servo=_FixedSnapshotServo(model)
        self.actor=ReferenceActor21(self.servo.model)
        self.robot_axes25=tuple(self.servo.axes25)
        self._owner=object();self._initial_model_signature=self._model_signature()

    def _model_signature(self):
        previous=super()._model_signature();guard=self.servo.model_guard_signature()
        if guard!=self.servo._initial_guard:self.servo.clear_transmission_cache(reason='model')
        return hashlib.sha256((previous+'|prepared_f067_admission|'+guard).encode()).hexdigest()

    def cache_diagnostics(self):return self.servo.cache_diagnostics()
