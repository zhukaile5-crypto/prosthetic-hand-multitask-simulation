"""V17b80 archive consumer: role-specific records, no model or simulation.

The plan is the only assignment source. Nothing below reads branch results
until all six registered batches are terminal. Numeric NPZ columns are each
decompressed once per cached record; no controllers or sensors are imported.
"""
from pathlib import Path
from collections import OrderedDict
from numbers import Real
import hashlib, importlib.util, json, threading
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PLAN=ROOT/'review/v17b_prospective80_plan_v1.json'
PLAN_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
DIAGNOSTIC_PATH=ROOT/'review/continuous_diagnostics_v1.py'
DIAGNOSTIC_SHA='05e86a5869f144f990dbb9883add20d002c4384f402cfb07b15db333e9bd8cfa'
METHODS={'full_V17':'完整多任务 V17','holding_A':'仅持握改进 A','baseline_ratio':'力比 PI 基线',
         'baseline_fixed':'固定力基线','legacy_fusion':'旧融合'}
SCENARIOS={'nominal':'常规环境','friction_drop':'摩擦变化','fragile_conflict':'受拉与挤压','collision':'障碍接触'}
MOTIONS={'hold':'原地持握','transport':'抬升与转腕'}
SEEDS={619741:'新条件 A',619742:'新条件 B'}
WINDOWS={'static':'static_quality','dynamic':'dynamic_quality','holding':'entire_holding_quality'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(bool,np.bool_)):return bool(value)
    if isinstance(value,(int,np.integer)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def role_for(arm,method):
    if arm=='full_V17' and method=='ratio_effort':return 'full_V17'
    if arm=='holding_A' and method=='ratio_effort':return 'holding_A'
    if arm=='baselines':
        roles={'ratio_effort':'baseline_ratio','fixed_effort':'baseline_fixed','fusion_effort':'legacy_fusion'}
        if method in roles:return roles[method]
    raise ValueError('Unregistered arm/method combination')

def original_metrics(result):
    """Original eligibility/labels; all three TV fields use original raw200Hz."""
    m=result.get('research_metrics',{}) or {};s=m.get('safety',{}) or {}
    peak=s.get('finger_peak_N')
    out=dict(full_task=m.get('full_motion_mission_success'),legacy_task=m.get('mission_success',result.get('task_success')),
        force_guard=s.get('guard_pass'),force_guard_scope='原评价：完整0–16秒，包含准备与释放',
        initial_stability_wait_s=m.get('initial_stability_wait_s'),excess_Ns=s.get('local_excess_integral_Ns'),
        max_finger_peak_N=max(peak) if isinstance(peak,list) and peak else None,failure_reason=m.get('failure_reason'))
    for window,key in WINDOWS.items():
        q=m.get(key,{}) or {};valid=q.get('eligible') is True;variation=q.get('raw_200Hz') or {}
        out[window]=dict(eligible=valid,mean_N=q.get('mean_actual_finger_force_N') if valid else None,
            tv_N_per_s=variation.get('finger_TV_rate_N_s') if valid else None,
            total_sd_N=variation.get('total_force_SD_N') if valid else None,window_s=q.get('window_s'),
            coverage=q.get('coverage'),tv_label='逐指TV：原始200Hz，未滤波',
            force_scope='原评价目标物体五指法向力，不是全外接触皮肤压力',
            reason=None if valid else '原窗口不合格或覆盖不足；不可评不等于零')
    return clean(out)

def diagnostic_metrics(report):
    """Keep the frozen all-assigned scope separate; never create success."""
    windows={};measure=report.get('measurements') or {}
    for window in WINDOWS:
        q=measure.get('whole_holding' if window=='holding' else window,{}) or {}
        f=q.get('unselected_force') or {};ok=f.get('available') is True
        windows[window]=dict(available=ok,mean_N=f.get('mean_total_force_N') if ok else None,
            tv_N_per_s=f.get('finger_TV_rate_N_s') if ok else None,total_sd_N=f.get('total_force_SD_N') if ok else None,
            reason=None if ok else f.get('reason','连续诊断不可用'),window_s=f.get('window_s'),
            physical_object=q.get('physical_object'),scope='全分配固定窗连续诊断；不是原合格指标或新成功分数')
    contact=(report.get('contact_windows') or {}).get('release_and_after',{}) or {}
    floor=(report.get('floor') or {}).get('release_and_after12to16',{}) or {}
    before=(report.get('floor') or {}).get('before_release_0_to12',{}) or {}
    end=(report.get('object_end_states') or {}).get('record_end16',{}) or {}
    return clean(dict(windows=windows,delivery=dict(
        floor_contact=floor.get('any_positive_contact') if floor.get('available') is True else None,
        floor_before_release=before.get('any_positive_contact') if before.get('available') is True else None,
        target_receiver_top=contact.get('receiver_top'),target_floor=floor,
        end_position_m=end.get('position_m') if end.get('available') is True else None,
        end_linear_speed_m_s=end.get('linear_speed_m_s') if end.get('available') is True else None,
        end_angular_speed_rad_s=end.get('angular_speed_rad_s') if end.get('available') is True else None,
        scope='离线真物体接触账本；0N以上接触事件，触台不等于交付成功；落地标签覆盖12–16秒'),
        execution_context=report.get('execution_context')))

class ResultStore:
    def __init__(self,plan=DEFAULT_PLAN,*,verify_trace_hash=True,cache_records=10):
        self.plan_path=Path(plan).resolve()
        if sha(self.plan_path)!=PLAN_SHA:raise ValueError('Viewer binds only the registered V17b80 plan')
        self.registration=read(self.plan_path);self.verify_trace_hash=bool(verify_trace_hash)
        if isinstance(cache_records,bool) or not isinstance(cache_records,int) or not 5<=cache_records<=10:raise ValueError('Cache must contain5..10 records')
        if len(self.registration['source_sha256'])!=136:raise ValueError('Expected frozen source136')
        self.cache_records=cache_records;self.cache=OrderedDict();self.lock=threading.RLock()
        self.records={};self.pending=[];self.blocked=[];self.batches=[];self._diagnostic=None;self._placement_module=None
        self.refresh()

    def refresh(self):
        with self.lock:
            pending=[];blocked=[];batches=[]
            # Only completion markers may be read while any batch is incomplete.
            for stage in self.registration['stages']:
                folder=(ROOT/stage['output']).resolve();marker=folder/'execution_complete.json'
                if not marker.is_file():pending.append(stage['name']);continue
                finish=read(marker)
                if finish.get('execution_complete') is not True:pending.append(stage['name']);continue
                if finish.get('sources_unchanged') is not True or finish.get('runtime_failures'):
                    blocked.append(stage['name']);continue
                if finish.get('trials')!=stage['planned'] or finish.get('planned')!=stage['planned']:raise ValueError('Registered batch count mismatch')
                batches.append((stage,folder,finish))
            self.pending=pending;self.blocked=blocked;self.batches=batches;self.cache.clear();self.records={}
            if pending or blocked:return
            records={};source=self.registration['source_sha256']
            for stage,batch,finish in batches:
                manifest=read(batch/'manifest.json')
                if manifest.get('research_source_sha256')!=source:raise ValueError('Registered source136 differs')
                if finish.get('manifest_sha256')!=sha(batch/'manifest.json') or finish.get('trials_sha256')!=sha(batch/'trials.json'):raise ValueError('Archived ledger identity differs')
                trials=read(batch/'trials.json');results={}
                for result in trials:
                    k=(result['scenario'],int(result['seed']),result['method'])
                    if k in results:raise ValueError('Duplicate trial result')
                    results[k]=result
                if len(manifest['jobs'])!=stage['planned'] or len(trials)!=stage['planned']:raise ValueError('Incomplete batch assignment')
                for job in manifest['jobs']:
                    if job['method'] not in stage['methods'] or job['scenario'] not in stage['scenarios'] or job['seed'] not in stage['seeds'] or job['motion_mode']!=stage['motion_mode'] or job.get('task_policy')!=stage['task_policy'] or job.get('receiver_policy')!=stage['receiver_policy']:raise ValueError('Job differs from registered role')
                    role=role_for(stage['arm'],job['method']);key=(job['scenario'],job['motion_mode'],int(job['seed']),role)
                    folder=Path(job['destination']).resolve()
                    if not folder.is_relative_to((batch/'runs').resolve()):raise ValueError('Record outside registered batch')
                    result=results.get((job['scenario'],int(job['seed']),job['method']))
                    if key in records or result is None:raise ValueError('Role assignment missing/duplicated')
                    if read(folder/'result.json')!=result:raise ValueError('Trial/run result differs')
                    available=(folder/'trace.npz').is_file() and (folder/'metadata.json').is_file()
                    records[key]=dict(batch=batch,folder=folder,result=result,stage=stage,method=job['method'],role=role,available=available,
                        reason=None if available else '已分配记录缺少完整轨迹；保留不可用')
            expected={(s,m,seed,role) for s in SCENARIOS for m in MOTIONS for seed in SEEDS for role in METHODS}
            if set(records)!=expected or len(records)!=80:raise ValueError('Expected16 contexts×5 distinct roles')
            self.records=records

    def validate(self,scenario,motion,seed,method=None):
        if scenario not in SCENARIOS or motion not in MOTIONS:raise ValueError('Unknown registered context')
        if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed) not in SEEDS:raise ValueError('Unknown registered physical seed')
        if method is not None and method not in METHODS:raise ValueError('Use a role key, not a shared underlying method')
        return scenario,motion,int(seed)

    def catalog(self):
        contexts=[dict(scenario=s,motion=m,seed=seed,methods={role:dict(available=bool(self.records.get((s,m,seed,role),{}).get('available'))) for role in METHODS}) for s in SCENARIOS for m in MOTIONS for seed in SEEDS]
        return dict(title='已运行结果回放',status='blocked' if self.blocked else 'pending' if self.pending else 'ready',
            phase='前瞻开发对照：16个情境、2个物理种子；不代表独立确认或总体优势',
            cohort_complete=not self.pending and not self.blocked,pending_batches=self.pending,blocked_batches=self.blocked,
            complete_records=len(self.records),methods=METHODS,scenarios=SCENARIOS,motions=MOTIONS,seeds=SEEDS,contexts=contexts,duration_s=16.)

    def _aux(self):
        if self._diagnostic is None:
            if sha(DIAGNOSTIC_PATH)!=DIAGNOSTIC_SHA:raise ValueError('Frozen continuous diagnostic reader changed')
            spec=importlib.util.spec_from_file_location('_v17b_replay_frozen_diagnostic',DIAGNOSTIC_PATH)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);self._diagnostic=module
        return self._diagnostic

    def _placement(self):
        if self._placement_module is None:
            p=self.registration['auxiliary_metrics'];path=ROOT/p['reader']
            if sha(path)!=p['reader_sha256'] or sha(ROOT/p['contract'])!=p['contract_sha256']:
                raise ValueError('Registered placement calculator/contract changed')
            spec=importlib.util.spec_from_file_location('_v17b_replay_auxiliary_v3',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self._placement_module=module
        return self._placement_module

    def load_trace(self,scenario,motion,seed,method):
        context=self.validate(scenario,motion,seed,method);key=(*context,method)
        with self.lock:
            if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
            r=self.records.get(key)
            if not r or not r['available']:raise ValueError('Full cohort/record unavailable')
            metadata=read(r['folder']/'metadata.json');path=r['folder']/'trace.npz';expected=metadata.get('trace_sha256')
            if metadata.get('research_source_sha256')!=self.registration['source_sha256'] or expected!=r['result'].get('trace_sha256'):raise ValueError('Trace provenance mismatch')
            if metadata.get('method')!=r['method'] or metadata.get('task_policy')!=r['stage']['task_policy'] or metadata.get('receiver_policy')!=r['stage']['receiver_policy']:raise ValueError('Role metadata mismatch')
            if self.verify_trace_hash and sha(path)!=expected:raise ValueError('Recorded trace changed')
            names=('time','qpos','qvel','ctrl','normal','skin_normal','phase','position','relative','relative_velocity','rotation','angular_velocity')
            with np.load(path,allow_pickle=False) as z:data={name:z[name] for name in names if name in z.files}
            if not all(name in data for name in ('time','qpos','qvel','ctrl','normal','skin_normal','phase')):raise ValueError('Missing recorded visual/force column')
            t=data['time']
            if t.shape!=(3201,) or not np.allclose(t,np.arange(3201)*.005,rtol=0,atol=1e-8):raise ValueError('Incomplete recorded16s clock')
            for name in ('qpos','qvel','ctrl','normal','skin_normal'):
                if len(data[name])!=len(t) or not np.isfinite(data[name]).all():raise ValueError('Nonfinite recorded '+name)
            ledger={};lp=r['folder']/'delivery_intervals.npz';aux=self._aux()
            if lp.is_file():
                wanted=('interval_start_s','interval_end_s','contact_valid','object_state_valid','object_free_qpos7_end','object_free_qvel6_end','receiver_any_actor_top_normal_N',*('target_'+c+'_normal_N' for c in aux.CATEGORIES))
                with np.load(lp,allow_pickle=False) as z:ledger={name:z[name] for name in wanted if name in z.files}
            dp=r['folder']/'delivery_summary.json';delivery=read(dp) if dp.is_file() else {}
            diag=aux.measure_arrays(data,ledger,metadata,original_result=None,delivery_metadata=delivery)
            placement=None
            paux=self._placement()
            if all(key in ledger for key in paux.C['input_fields']):
                pa=paux.calculate(ledger,release_at_s=metadata.get('mechanical_release_at',metadata.get('release_at')),
                    episode_end_s=metadata['duration'],physical_period_s=metadata['physics_timestep'],original_result=r['result'])
                placement={key:pa[key] for key in ('status','official_first_stable_receiver',
                    'official_first_stable_by_release_deadline','official_final_stable_suffix',
                    'no_stability_reason','ever_floor_contact','stable_after_previous_floor_event',
                    'first_receiver_impact20ms','receiver_event_provenance')}
            data['placement']=placement
            data.update(metadata=metadata,batch=r['batch'],folder=r['folder'],trace_sha256=expected,
                diagnostics=diagnostic_metrics(aux.plain(diag)))
            for a in data.values():
                if isinstance(a,np.ndarray):a.flags.writeable=False
            self.cache[key]=data
            while len(self.cache)>self.cache_records:self.cache.popitem(last=False)
            return data

    def context(self,scenario,motion,seed):
        context=self.validate(scenario,motion,seed);methods={}
        for role in METHODS:
            r=self.records.get((*context,role),{});available=bool(r.get('available'))
            d=self.load_trace(*context,role) if available else None
            methods[role]=dict(available=available,display_name=METHODS[role],
                metrics=original_metrics(r['result']) if available else None,diagnostics=d['diagnostics'] if available else None,
                placement=d['placement'] if available else None,
                reason=r.get('reason','整批80条尚未完成，暂不显示部分对照'))
        return dict(context=dict(scenario=scenario,motion=motion,seed=int(seed)),methods=methods,duration_s=16.,
            scope='画面恢复保存的qpos/qvel/ctrl；不重新控制或求解接触。当前指力为全外接触皮肤；原目标物体指标另列。')

    @staticmethod
    def index_at(data,time_s):
        if isinstance(time_s,(bool,np.bool_)) or not isinstance(time_s,Real) or not np.isfinite(time_s) or not 0<=time_s<=16:raise ValueError('Finite replay time0..16 required')
        t=data['time'];right=min(int(np.searchsorted(t,float(time_s))),len(t)-1)
        return right-1 if right and abs(t[right-1]-time_s)<=abs(t[right]-time_s) else right

    def sample(self,scenario,motion,seed,time_s):
        context=self.validate(scenario,motion,seed);methods={};actual=[]
        for role in METHODS:
            if not self.records.get((*context,role),{}).get('available'):
                methods[role]=dict(available=False);continue
            d=self.load_trace(*context,role);i=self.index_at(d,time_s);actual.append(float(d['time'][i]))
            force=d['skin_normal'][i];target=d['normal'][i]
            methods[role]=dict(available=True,phase=str(d['phase'][i]),finger_force_N=force.tolist(),total_force_N=float(force.sum()),
                target_finger_force_N=target.tolist(),target_total_force_N=float(target.sum()))
        if actual and max(actual)-min(actual)>1e-10:raise ValueError('Record clocks not synchronized')
        return dict(time_s=actual[0] if actual else float(time_s),methods=methods,
            force_scope='当前力条：保存的200Hz全外接触皮肤压力；原指标：目标物体normal；均未重新求力')
