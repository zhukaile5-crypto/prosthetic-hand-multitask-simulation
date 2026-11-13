"""Archived result catalog and cached recorded states. No MuJoCo import.

Rendering is a separate optional consumer. This module never changes a result,
re-evaluates success, interpolates physics, or replaces missing quality by zero.
"""
from pathlib import Path
from collections import OrderedDict
from numbers import Real
import hashlib,json,threading
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_REGISTRATION=ROOT/'review/v12_native_development_registration_v1.json'
METHODS={'fixed_effort':'固定力目标','ratio_effort':'摩擦比策略','fusion_effort':'多任务融合'}
SCENARIOS={'nominal':'名义条件','friction_drop':'摩擦下降','fragile_conflict':'易损冲突','collision':'接触碰撞'}
MOTIONS={'hold':'静态持握','transport':'动态运输'}
SEEDS={617901:'种子 A',617902:'种子 B'}
WINDOWS={'static':'static_quality','dynamic':'dynamic_quality','holding':'entire_holding_quality'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,np.integer)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def metric_card(result):
    m=result.get('research_metrics',{});s=m.get('safety',{})
    answer=dict(full_task=m.get('full_motion_mission_success'),force_guard=s.get('guard_pass'),
        force_guard_scope='完整0–16秒，包含准备和释放',initial_stability_wait_s=m.get('initial_stability_wait_s'),
        excess_Ns=s.get('local_excess_integral_Ns'),failure_reason=m.get('failure_reason'))
    peaks=s.get('finger_peak_N');answer['max_finger_peak_N']=max(peaks) if peaks else None
    for name,key in WINDOWS.items():
        q=m.get(key,{}) or {};eligible=q.get('eligible') is True
        # Whole-holding raw avoids attributing future release FIR samples to
        # the holding phase. Static/dynamic keep their original50Hz FIR fields.
        band='raw_200Hz' if name=='holding' else 'bandlimited'
        variation=q.get(band) or {}
        answer[name]=dict(eligible=eligible,mean_N=q.get('mean_actual_finger_force_N') if eligible else None,
            tv_N_per_s=variation.get('finger_TV_rate_N_s') if eligible else None,
            total_sd_N=variation.get('total_force_SD_N') if eligible else None,
            window_s=q.get('window_s'),coverage=q.get('coverage'),
            tv_label='逐指TV：200Hz原记录' if name=='holding' else '逐指TV：50Hz滤波',
            unit='N/s',reason=None if eligible else '原评价窗口不合格或覆盖不足，保留不可评')
    return clean(answer)


class ResultStore:
    def __init__(self,registration=DEFAULT_REGISTRATION,*,verify_trace_hash=True,cache_records=6):
        self.registration_path=Path(registration).resolve();self.registration=read(self.registration_path)
        self.verify_trace_hash=bool(verify_trace_hash);self.cache_records=int(cache_records)
        if self.cache_records<3:raise ValueError('Three-method cache capacity required')
        self.lock=threading.RLock();self.cache=OrderedDict();self.records={};self.pending=[];self.batches=[]
        self.refresh()

    def refresh(self):
        with self.lock:
            reg=self.registration;source=reg['source_sha256'];pending=[];batches=[]
            if len(source)!=80:raise ValueError('Expected the registered source80 archive')
            for group in reg['groups']:
                run=(ROOT/group['output']).resolve();marker=run/'execution_complete.json'
                if not marker.is_file():pending.append(group['output']);continue
                finish=read(marker)
                if finish.get('execution_complete') is not True:raise ValueError('Archived batch did not complete: '+run.name)
                if finish.get('sources_unchanged') is not True:raise ValueError('Archived source identity changed: '+run.name)
                batches.append((run,finish))
            self.pending=pending;self.batches=batches
            if pending:self.records={};self.cache.clear();return
            records={}
            for run,finish in batches:
                manifest=read(run/'manifest.json');trials=read(run/'trials.json')
                if manifest['research_source_sha256']!=source:raise ValueError('Source80 archive mismatch')
                if sha(run/'trials.json')!=finish['trials_sha256']:raise ValueError('Archived trial ledger changed')
                results={(r['scenario'],int(r['seed']),r['method']):r for r in trials}
                for job in manifest['jobs']:
                    key=(job['scenario'],job['motion_mode'],int(job['seed']),job['method'])
                    if key in records:raise ValueError('Duplicate archived context/method')
                    result=results.get((key[0],key[2],key[3]));folder=Path(job['destination']).resolve()
                    if not folder.is_relative_to((ROOT/'output').resolve()):raise ValueError('Archive outside research output')
                    exists=result is not None and (folder/'trace.npz').is_file() and (folder/'metadata.json').is_file()
                    records[key]=dict(batch=run,folder=folder,result=result,available=exists,
                        reason=None if exists else '该次试验没有完整可回放记录，失败仍保留')
            expected={(s,m,int(seed),method) for s in reg['scenarios'] for m in reg['motion_modes'] for seed in reg['seeds'] for method in reg['methods']}
            if set(records)!=expected or len(expected)!=48:raise ValueError('Registered48 assignment mismatch')
            self.records=records;self.cache.clear()

    def validate(self,scenario,motion,seed,method=None):
        if scenario not in SCENARIOS or motion not in MOTIONS:raise ValueError('Unknown recorded context')
        if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or int(seed) not in SEEDS:raise ValueError('Unknown recorded seed')
        if method is not None and method not in METHODS:raise ValueError('Unknown recorded method')
        return scenario,motion,int(seed)

    def catalog(self):
        with self.lock:
            contexts=[]
            for seed in self.registration['seeds']:
                for scenario in self.registration['scenarios']:
                    for motion in self.registration['motion_modes']:
                        methods={method:dict(available=bool(self.records.get((scenario,motion,int(seed),method),{}).get('available')),
                            status='pending' if self.pending else ('recorded' if self.records.get((scenario,motion,int(seed),method),{}).get('available') else 'unavailable')) for method in METHODS}
                        contexts.append(dict(scenario=scenario,motion=motion,seed=int(seed),methods=methods))
            return dict(title='已运行结果回放',phase='开发试验：16个情境、2个物理种子，非总体优势证明',
                status='pending' if self.pending else 'ready',pending_batches=self.pending,cohort_complete=not self.pending,
                methods=METHODS,scenarios=SCENARIOS,motions=MOTIONS,seeds=SEEDS,contexts=contexts,duration_s=16.)

    def context(self,scenario,motion,seed):
        context=self.validate(scenario,motion,seed);methods={}
        with self.lock:
            for method,label in METHODS.items():
                r=self.records.get((*context,method),{})
                available=bool(r.get('available'))
                methods[method]=dict(available=available,display_name=label,metrics=metric_card(r['result']) if available else None,
                    reason='整批记录尚未完成，暂不显示部分对照' if self.pending else r.get('reason'))
            return dict(context=dict(scenario=scenario,motion=motion,seed=int(seed)),methods=methods,duration_s=16.,
                scope='同一记录时钟；画面来自保存的关节/物体姿态，力来自原始记录；没有重新控制或重新仿真')

    def load_trace(self,scenario,motion,seed,method):
        context=self.validate(scenario,motion,seed,method);key=(*context,method)
        with self.lock:
            if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
            r=self.records.get(key)
            if not r or not r['available']:raise ValueError('Recorded trial unavailable')
            path=r['folder']/'trace.npz';metadata=read(r['folder']/'metadata.json')
            if metadata['research_source_sha256']!=self.registration['source_sha256']:raise ValueError('Recorded trial source mismatch')
            expected=metadata['trace_sha256']
            if expected!=r['result']['trace_sha256']:raise ValueError('Result/trace identity mismatch')
            if self.verify_trace_hash and sha(path)!=expected:raise ValueError('Recorded trace changed')
            # Each compressed column is read once, never z[column][frame].
            with np.load(path,allow_pickle=False) as z:
                data={n:z[n] for n in ('time','qpos','qvel','ctrl','skin_normal','phase')}
            t=data['time']
            if len(t)!=3201 or not np.allclose(t,np.arange(3201)*.005,rtol=0,atol=1e-8):raise ValueError('Incomplete recorded16s clock')
            for field in ('qpos','qvel','ctrl','skin_normal'):
                if len(data[field])!=len(t) or not np.isfinite(data[field]).all():raise ValueError('Unavailable/nonfinite recorded '+field)
            data.update(metadata=metadata,batch=r['batch'],folder=r['folder'],trace_sha256=expected)
            self.cache[key]=data
            while len(self.cache)>self.cache_records:self.cache.popitem(last=False)
            return data

    @staticmethod
    def index_at(data,time_s):
        if isinstance(time_s,(bool,np.bool_)) or not isinstance(time_s,Real) or not np.isfinite(time_s) or not 0<=time_s<=16:
            raise ValueError('Replay time must be finite0..16 seconds')
        times=data['time'];right=int(np.searchsorted(times,float(time_s),side='left'));right=min(right,len(times)-1)
        return right-1 if right and abs(times[right-1]-time_s)<=abs(times[right]-time_s) else right

    def sample(self,scenario,motion,seed,time_s):
        context=self.validate(scenario,motion,seed);methods={};actual_times=[]
        for method in METHODS:
            r=self.records.get((*context,method),{})
            if not r.get('available'):methods[method]=dict(available=False,reason=r.get('reason','整批记录暂不可用'));continue
            d=self.load_trace(*context,method);i=self.index_at(d,time_s);force=d['skin_normal'][i]
            actual_times.append(float(d['time'][i]));methods[method]=dict(available=True,phase=str(d['phase'][i]),
                finger_force_N=force.tolist(),total_force_N=float(force.sum()))
        if actual_times and max(actual_times)-min(actual_times)>1e-10:raise ValueError('Recorded method clocks are not synchronized')
        return dict(time_s=actual_times[0] if actual_times else float(time_s),methods=methods,
            force_scope='保存的200Hz全外接触逐指压力；不是渲染器重新求解的接触力')
