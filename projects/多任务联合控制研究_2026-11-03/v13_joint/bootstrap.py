"""Independent copy of frozen TrialSession initialization with one plant factory change.

The base methods remain inherited. No original factory or module global is
patched. A1Plant is used at reset, before the first public physics sample.
"""
from dataclasses import fields
from collections import deque
from copy import deepcopy
import uuid
import numpy as np
import research_imports
from session import (TrialSession,SCENE_BY_NAME,check_admission,load_json,
    _evidence_path,physical_parameters,validate_radius_evidence,ROOT)
from foundation_config import FoundationConfig
from contracts import SensorNoise
from sensors import HandSensors
from hand_plant import CommonPreparation
from .plant import JointPlant

class JointBootstrap(TrialSession):
    def __init__(self,method='ratio',shape=None,scenario='nominal',seed=540101,
                 mode='scheduled',split='demonstration',admission_path=None):
        if method not in ('ratio','unified'):raise ValueError('unknown admitted method')
        if scenario not in SCENE_BY_NAME:raise ValueError('unknown scenario')
        if mode not in ('scheduled','manual'):raise ValueError('unknown run mode')
        if split not in ('demonstration','development','confirmation','robustness'):raise ValueError('unknown experiment split')
        if split!='demonstration' and mode!='scheduled':raise ValueError('manual demonstrations cannot enter a formal matrix')
        self.admission=check_admission(admission_path)
        if not self.admission['passed']:raise RuntimeError(' '.join(self.admission['problems']))
        self.method=method;self.scenario_name=scenario;self.seed=int(seed);self.mode=mode;self.split=split
        plan=load_json(_evidence_path(self.admission,'experiment_plan'))
        if split=='demonstration':
            reserved={int(seed) for phase in plan.get('phases',{}).values() for seed in phase.get('seeds',[])}
            reserved|={int(job['seed']) for phase in plan.get('phases',{}).values() for job in phase.get('jobs',[])}
            if self.seed in reserved:raise ValueError('该种子保留用于开发或确认，请使用独立演示种子540101或查看已有回放。')
        scene,values=physical_parameters(scenario,seed,split,shape)
        self.scene=scene;self.shape=scene['shape']
        all_calibration=load_json(_evidence_path(self.admission,'calibration'))
        self.calibration=deepcopy(all_calibration['shapes'][self.shape])
        if 'response' in self.calibration:
            self.calibration={**self.calibration,**self.calibration['response']}
        if 'nominal_force_weights' not in self.calibration:
            raise ValueError('Production hand calibration must declare shared nominal_force_weights')
        self.trust_radius=validate_radius_evidence(self.calibration)
        bare=all_calibration.get('bare_hand',{})
        self.calibration.update({k:v for k,v in bare.items() if k in ('effort_bias_N','load_error_bound_N')})
        defaults=all_calibration.get('foundation_defaults',{})
        allowed={f.name for f in fields(FoundationConfig)}
        model_values={**self.calibration.get('config',{}),**defaults,**values}
        self.config=FoundationConfig(**{k:v for k,v in model_values.items() if k in allowed})
        self.plant=JointPlant(self.config)
        self.obstacle_geom=int(self.plant.model.geom('integration_obstacle_geom').id)
        self.preparation=CommonPreparation(self.plant)
        noise=dict(force_sigma=.01,tangential_sigma=.005,position_sigma=.0001,
                   slip_sigma=.0002,effort_sigma=.02,delay=.010)
        if scenario=='sensor_noise':noise.update(force_sigma=.08,tangential_sigma=.04,slip_sigma=.002)
        if scenario=='sensor_dropout':noise['dropout_windows']=((9.3,9.7,(1,2)),(11.3,11.4,(0,1,2,3,4)))
        self.sensors=HandSensors(self.calibration,SensorNoise(**noise),seed=self.seed)
        settings=load_json(ROOT/'vendor/frozen_settings.json')
        self.gains=deepcopy(settings.get('methods',settings.get('gains',{})).get(method,{}))
        if not self.gains:
            self.gains=deepcopy(settings.get('controllers',{}).get(method,{}))
        self.gains['release_rate']=float(self.config.release_open_rate)
        from controller_adapter import create_controller
        self.limits=np.asarray(scene.get('limits',[5.,5.,5.,5.,5.]),float)
        self.budget=float(scene.get('budget',18.))
        self.controller=create_controller(method,self.calibration,self.limits,self.budget,self.gains)
        self.run_id=uuid.uuid4().hex[:12];self.tick=0
        self.running=True;self.paused=False;self.completed=False;self.saved=False
        self.handoff_attempted=False;self.handoff_passed=False;self.handoff_alpha=None
        self.reference_relative=None;self.release_at=None;self.release_supported=False
        self.preparation_failure=None;self.failure_reason=None;self.failure_at=None
        self.events=[];self.commands=[];self.trace=[];self.history=deque(maxlen=3300)
        self.camera=dict(azimuth=150.,elevation=-15.,distance=.68)
        self.manual_force_until=0.;self.manual_squeeze_until=0.;self.manual_release=False
        self.control_ms=0.;self.last_truth=self.plant.truth();self.last_obs=None
        self.last_desired=np.zeros(5);self.last_prediction=np.full(5,np.nan)
        self.last_action=np.zeros(5);self.last_debug={};self.last_command='hold'
        self.environment_signature=None
        self.event('共同准备开始；当前方法尚未接管。')
