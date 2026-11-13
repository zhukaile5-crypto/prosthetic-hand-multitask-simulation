"""Independent saved-force resolution audit. NumPy/files only; no simulator imports."""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def js(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def episodes(t, durations, flag, value):
    """Contiguous sampled violations; durations are recorded step/observation widths."""
    idx=np.flatnonzero(flag)
    if not len(idx): return []
    cuts=np.flatnonzero(np.diff(idx)>1)+1
    rows=[]
    for ids in np.split(idx,cuts):
        peak=ids[int(np.argmax(value[ids]))]
        rows.append(dict(first_sample_s=float(t[ids[0]]),last_sample_s=float(t[ids[-1]]),
            interval_end_s=float(t[ids[-1]]+durations[ids[-1]]),samples=len(ids),
            duration_s=float(durations[ids].sum()),peak_N=float(value[peak]),peak_at_s=float(t[peak])))
    return rows

def dwell(t, value, threshold, duration, release, end, widths):
    """Timestamp-difference dwell, independent of production first_dwell helper."""
    selected=np.flatnonzero((t>=release-EPS)&(t<=end+EPS))
    since=None;first=None;first_begin=None
    for i in selected:
        if value[i]>threshold:
            since=None
        else:
            if since is None: since=i
            if first is None and t[i]-t[since]>=duration-EPS:
                first=float(t[i]);first_begin=float(t[since])
    tail_begin=None;tail_confirm=None
    if len(selected) and value[selected[-1]]<=threshold:
        k=len(selected)-1
        while k>0 and value[selected[k-1]]<=threshold: k-=1
        tail_begin=float(t[selected[k]])
        eligible=selected[k:][t[selected[k:]]-tail_begin>=duration-EPS]
        if len(eligible): tail_confirm=float(t[eligible[0]])
    post=(t>first+EPS)&(t<=end+EPS)&(value>threshold) if first is not None else np.zeros(len(t),bool)
    return dict(first_below_threshold_s=float(t[selected[value[selected]<=threshold][0]])
                    if len(selected) and np.any(value[selected]<=threshold) else None,
        first_dwell_start_s=first_begin,first_dwell_confirmation_s=first,
        first_dwell_latency_s=None if first is None else first-release,
        final_clear_tail_start_s=tail_begin,final_clear_tail_confirmation_s=tail_confirm,
        final_clear_tail_latency_s=None if tail_confirm is None else tail_confirm-release,
        observed_until_s=float(t[selected[-1]]) if len(selected) else None,
        recontact_after_first_confirmation=episodes(t,widths,post,value))

def window_violations(rt, rd, force, threshold, start, finish):
    if start is None or finish is None:return None
    # Both point samples and overlap-weighted held numerical intervals are explicit.
    points=(rt>=start-EPS)&(rt<=finish+EPS)&(force>threshold)
    overlap=np.maximum(0.,np.minimum(rt+rd,finish)-np.maximum(rt,start))
    bad=(overlap>1e-12)&(force>threshold)
    return dict(window_start_s=start,window_end_s=finish,
        raw_point_violation_count=int(points.sum()),
        raw_violating_interval_count=int(bad.sum()),
        interval_overlap_duration_s=float(overlap[bad].sum()),
        peak_N=float(np.max(force[points|bad])) if np.any(points|bad) else None,
        episodes=episodes(rt,np.where(bad,overlap,rd),bad,force),
        duration_semantics='overlap of recorded [substep_time,substep_time+duration) with window; numerical interval attribution, not continuous force interpolation')

def self_test():
    t=np.arange(0.,1.0001,.05);v=np.zeros(len(t));v[6]=1.
    a=dwell(t,v,.2,.2,0.,1.,np.full(len(t),.05))
    assert abs(a['first_dwell_confirmation_s']-.2)<EPS
    assert len(a['recontact_after_first_confirmation'])==1
    assert abs(a['final_clear_tail_confirmation_s']-.55)<EPS
    v=np.zeros(len(t));v[3]=1.
    a=dwell(t,v,.2,.2,0.,1.,np.full(len(t),.05))
    assert abs(a['first_dwell_confirmation_s']-.4)<EPS
    a=window_violations(np.array([.1,.15,.2]),np.full(3,.05),np.array([0.,.3,0.]),.2,.125,.175)
    assert abs(a['interval_overlap_duration_s']-.025)<EPS and a['raw_point_violation_count']==1
    return dict(passed=True,checks=6)

def inspect_trace(run, cohort, check):
    p=run/'record/trace.npz'; meta_path=p.parent/'metadata.json';res_path=run/'result.json'
    paths=[p,meta_path,res_path,run/'robot_physical_stream.npz']
    before={str(x):sha(x) for x in paths}
    meta=js(meta_path);result=js(res_path)
    with np.load(p,allow_pickle=False) as z:
        t=z['time'];sparse=z['all_hand_object'];rt=z['substep_time'];rd=z['substep_duration'];raw=z['substep_all_hand_object']
        phase=z['phase'];raw_n=z['substep_normal'];raw_p=z['substep_palm'];raw_o=z['substep_hand_other']
    with np.load(paths[-1],allow_pickle=False) as z:
        robot_t=z['physical_time_s'];robot_force=z['all_hand_target_pressure_N']
    protocol_path=cohort/'sources/research/protocol.json';protocol=js(protocol_path)
    th=protocol['thresholds'];cap=float(th['all_hand_clearance_N']);duration=float(th['release_clearance_dwell_s'])
    release=float(meta['release_at']);deadline=float(th['release_deadline_s']);end=16.
    check(meta.get('completed') is True and len(t)==3201 and len(rt)==32000,run.name+' full records')
    check(abs(release-12.)<EPS and cap==.2 and duration==.2,run.name+' frozen threshold')
    check(np.isfinite(raw).all() and np.isfinite(sparse).all(),run.name+' finite force')
    check(np.all(raw>=0) and np.all(sparse>=0),run.name+' nonnegative force')
    check(np.max(abs(np.diff(t)-.005))<EPS and np.max(abs(np.diff(rt)-.0005))<EPS and np.max(abs(rd-.0005))<EPS,run.name+' clocks')
    check(abs(rt[0])<EPS and abs(rt[-1]+rd[-1]-16.)<EPS,run.name+' raw start clock')
    check(len(robot_t)==32001 and np.max(abs(robot_t[1:]-rt-rd))<EPS,run.name+' robot end clock')
    check(np.array_equal(robot_force[1:],raw),run.name+' cached contact force raw vs robot end packet exact')
    check(np.max(abs(raw_n.sum(axis=1)+raw_p+raw_o-raw))<1e-10,run.name+' all hand categories')
    check(str(phase[np.argmin(abs(t-release))])=='RELEASING',run.name+' release phase')
    sparse_result=dwell(t,sparse,cap,duration,release,end,np.full(len(t),.005))
    raw_result=dwell(rt,raw,cap,duration,release,end,rd)
    raw_end_result=dwell(rt+rd,raw,cap,duration,release,end,rd)
    canonical=result['research_metrics']['release']['physical_clearance_latency_s']
    check(abs(sparse_result['first_dwell_latency_s']-canonical)<EPS,run.name+' canonical 200Hz first dwell')
    # Deliberately disclose that control-boundary forward forces are not raw subsampling.
    index=np.rint(t[:-1]/.0005).astype(int)
    boundary_difference=sparse[:-1]-raw[index]
    w=window_violations(rt,rd,raw,cap,sparse_result['first_dwell_start_s'],sparse_result['first_dwell_confirmation_s'])
    final_w=window_violations(rt,rd,raw,cap,sparse_result['final_clear_tail_start_s'],sparse_result['final_clear_tail_confirmation_s'])
    raw_deadline=dwell(rt,raw,cap,duration,release,min(end,release+deadline),rd)
    after={str(x):sha(x) for x in paths}
    check(before==after,run.name+' input unchanged')
    return dict(cohort=cohort.name,job_id=run.name,stage=meta.get('release_local_response_stage','phi_020ms'),
        variant=meta['release_local_response_variant'],trace_path=str(p.resolve()),input_sha256=before,
        frozen_protocol_path=str(protocol_path.resolve()),frozen_protocol_sha256=sha(protocol_path),
        threshold_N=cap,dwell_s=duration,release_at_s=release,deadline_s=deadline,
        sampled_200Hz=sparse_result,raw_recorded_start_labels=raw_result,raw_end_labels=raw_end_result,
        raw_first_dwell_within_original_deadline=raw_deadline['first_dwell_confirmation_s'] is not None,
        raw_minus_200Hz_first_latency_s=raw_result['first_dwell_latency_s']-sparse_result['first_dwell_latency_s'],
        raw_violations_inside_original_200Hz_first_dwell_window=w,
        raw_violations_inside_200Hz_final_clear_dwell_window=final_w,
        boundary_raw_not_simple_decimation=dict(max_abs_force_difference_N=float(np.max(abs(boundary_difference))),
            number_nonexact=int(np.count_nonzero(boundary_difference)),
            source_semantics='substep stored label=start before mj_step; force queried after mj_step, while q is integrated to start+dt; 200Hz truth follows an extra boundary mj_forward'),
        no_interpolation_continuity_claim=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    target=Path(args.out)
    if target.suffix!='.json' or target.exists() or target.with_suffix('.md').exists():raise ValueError('Fresh JSON and MD required')
    checks=[]
    def check(ok,name):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    rows=[];protocols={}
    for name,count in [('release_local_response_capsule_v2',7),('release_local_response_capsule_v3',14)]:
        cohort=ROOT/'output'/name;summary=js(cohort/'summary.json')
        check(summary['execution_complete'] and summary['completed']==count and summary['planned']==count and not summary['failures'],name+' complete ledger')
        runs=sorted((cohort/'runs').iterdir());check(len(runs)==count,name+' exact denominator')
        rows += [inspect_trace(run,cohort,check) for run in runs]
        protocols[name]=sha(cohort/'sources/research/protocol.json')
    check(len(rows)==21,'all21 retained')
    result=dict(schema='release_clearance_resolution_independent_v1',role='offline resolution diagnosis; original canonical result unchanged',
        passed=all(x['passed'] for x in checks),checks=len(checks),self_test=self_test(),script_sha256=sha(__file__),
        source_sha256={'script':sha(__file__)},protocol_sha256=protocols,records=rows,assertions=checks,
        physical_simulations_started=0,
        interpretation='Raw refers to every saved numerical physics-step force solve, not a proof of force between solver evaluations. Both recorded-start and integrated-end labels are reported; the latter shifts the confirmation by0.5ms. Coarse200Hz boundary forward forces are not asserted identical to raw decimation.')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# 释放清离判定的时间分辨率独立诊断','',
        '全部21条记录，原阈值0.2 N、持续0.2 s；只读分析，无新仿真。原200 Hz评价不覆盖或改写。',
        '','`substep_time`是求解步起始标签；同一保存力附在积分后机器人状态时，终点标签增加0.5 ms。200 Hz边界另有mj_forward，不能当作对子步列简单抽样。此处“持续”只指所有保存求解点均满足阈值，并非求解点之间连续时间证明。','',
        '|起点|路径|200Hz首次确认延迟(s)|raw首次确认延迟(s)|raw最终尾段确认延迟(s)|原200Hz窗raw违规峰(N)|违规总时长(ms)|违规段数|raw首次确认后再接触段数|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'NULL' if v is None else f'{v:.6f}'
    for x in rows:
        w=x['raw_violations_inside_original_200Hz_first_dwell_window'];a=x['raw_recorded_start_labels']
        lines.append('|'+ '|'.join([x['stage'],x['variant'],fmt(x['sampled_200Hz']['first_dwell_latency_s']),fmt(a['first_dwell_latency_s']),fmt(a['final_clear_tail_latency_s']),fmt(w['peak_N']),fmt(1000*w['interval_overlap_duration_s']),str(len(w['episodes'])),str(len(a['recontact_after_first_confirmation']))])+'|')
    lines+=['','详尽的首次/最终清离、再接触事件、违规子步数及事件时间见同名JSON。所有原输入SHA在读取前后相同；来源脚本和冻结协议SHA均在JSON。',f'','检查：{len(checks)}项通过；不改变原任务/释放结果。']
    with target.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),records=len(rows),out=str(target),sha256=sha(target)),ensure_ascii=False))

if __name__=='__main__':main()
