"""Frozen-pilot arithmetic diagnosis; no simulator imports or revised policy.

Reconstructs the single lift scalar from public robot state, own port command,
and source-pinned position-servo constants. All decisions retained. Candidate
native re-ranking is offline diagnostic only, never sent to an executor.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=('ratio_effort','fixed_effort','fusion_effort')
EXPECTED='eba6a670a6f4e013368f8ba70e268ac0d2ad02dd90cb01e9d10e0cdc6ee0658f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def summary(values):
    x=np.asarray(values,float)
    return None if not len(x) else dict(n=len(x),min=float(x.min()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))

def lift_rebuild(public,context,parameters):
    kp,kv,ctrl_lo,ctrl_hi,force_lo,force_hi=parameters
    q=float(public['robot_position25'][24]);v=float(public['robot_velocity25'][24]);actual=float(public['robot_actuator_target21'][18])
    bias=float(context['known_lift_engine_bias_m']);pref=float(context['nominal_reference8'][5]);pnom_port=float(context['nominal_port_target21'][18]);corr=float(context['requested_correction25'][24])
    lo=float(context['candidate_lower_reference8'][5]);hi=float(context['candidate_upper_reference8'][5])
    # Length/speed are the actual lift scalar joint coordinates, gear1. Use
    # one-element arrays to reproduce the frozen array-ufunc operation order.
    g=np.array([kp]);L=np.array([q]);V=np.array([v])
    def native_port(port):
        u=np.clip(np.array([port]),ctrl_lo,ctrl_hi)
        raw=g*u+np.array([0.])+(-g)*L+np.array([-kv])*V
        return float(np.clip(raw,force_lo,force_hi)[0])
    actual_effective=np.clip(np.array([actual]),ctrl_lo,ctrl_hi)
    raw_current=g*actual_effective+np.array([0.])+(-g)*L+np.array([-kv])*V
    beta=float((raw_current-g*actual_effective)[0]);pnom=native_port(pnom_port)
    def analytic_force(x):return float(np.clip(g*(np.array([bias])+x)+np.array([beta]),force_lo,force_hi)[0])
    def native_ref(x):return native_port(float(x+bias))
    def cost(force):
        residual=np.array([force-pnom])-np.array([corr]);return float(residual@residual)
    breaks=[lo,hi]
    for boundary in [force_lo,force_hi]:
        b=float((boundary-(kp*bias+beta))/kp)
        if lo<b<hi:breaks.append(b)
    breaks=np.unique(breaks)
    trial=[(float(x),'boundary') for x in breaks]+[(float(np.clip(pref,lo,hi)),'closest_nominal')]
    for left,right in zip(breaks[:-1],breaks[1:]):
        middle=float(left+(right-left)/2);slope=np.array([kp]);intercept=np.array([kp*bias+beta])
        raw=slope*middle+intercept
        if raw[0]<force_lo:slope[:]=0;intercept[:]=force_lo
        elif raw[0]>force_hi:slope[:]=0;intercept[:]=force_hi
        residual_intercept=(intercept-np.array([pnom]))-np.array([corr]);curvature=float(slope@slope)
        val=float(np.clip(-(slope@residual_intercept)/curvature,left,right)) if curvature>0 else float(np.clip(pref,left,right))
        trial.append((val,'segment_minimum'))
    candidates=[]
    for x,origin in trial:
        af=analytic_force(x);nf=native_ref(x)
        candidates.append(dict(reference_m=x,origin=origin,analytic_force_N=af,native_force_N=nf,
            analytic_loss_N2=cost(af),native_loss_N2=cost(nf),distance_to_nominal_m=abs(x-pref),
            native_engine_port_target_m=float(x+bias)))
    selected=min(candidates,key=lambda x:(x['analytic_loss_N2'],x['distance_to_nominal_m'],x['reference_m']))
    native_best=min(candidates,key=lambda x:(x['native_loss_N2'],x['distance_to_nominal_m'],x['reference_m']))
    # Same target, same finite candidate set; no changed gate/tolerance/policy.
    return dict(q_lift_m=q,v_lift_m_s=v,current_port_m=actual,known_robot_lift_bias_m=bias,
        actual_gain_times_current_port_N=float(kp*actual),nominal_gain_times_port_N=float(kp*pnom_port),
        position_term_N=float(-kp*q),damping_term_N=float(-kv*v),cached_beta_N=beta,
        nominal_native_force_N=pnom,requested_correction_N=corr,lo_m=lo,hi_m=hi,
        selected=selected,native_best_same_pool=native_best,candidates=candidates,
        selected_native_force_minus_analytic_N=selected['native_force_N']-selected['analytic_force_N'],
        native_ulp_at_selected_force_N=float(np.spacing(abs(selected['native_force_N']))),
        native_objective_regret_vs_same_pool_N2=selected['native_loss_N2']-native_best['native_loss_N2'],
        selected_at_upper=selected['reference_m']==hi,selected_at_nominal=selected['reference_m']==pref,
        force_clipped_at_selected=bool(abs(selected['native_force_N'])>=max(abs(force_lo),abs(force_hi))),
        absolute_large_operand_sum_N=abs(kp*pnom_port)+abs(kp*q)+abs(kv*v)+abs(pnom))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists() or out.with_suffix('.md').exists():raise FileExistsError('Fresh diagnosis required')
    pilot=ROOT/'output/development_v12_pilot_v1';execution=js(pilot/'execution_complete.json');manifest=js(pilot/'manifest.json')
    if not execution['execution_complete'] or execution['trials']!=3 or execution['runtime_failures']:raise ValueError('Complete3pilot required')
    core=ROOT/'v12/effort_tracker.py';core_before=sha(core)
    if core_before!=EXPECTED or manifest['research_source_sha256'].get('v12/effort_tracker.py')!=EXPECTED:raise ValueError('Frozen actual pilot identity required')
    xml=pilot/'sources/foundation/assets/legacy_model.xml';node=ET.parse(xml).getroot().find(".//position[@name='lift_motor']")
    cl=list(map(float,node.get('ctrlrange').split()));fl=list(map(float,node.get('forcerange').split()))
    parameters=(float(node.get('kp')),float(node.get('kv')),*cl,*fl)
    rows=[];checks=[]
    def check(name,x):
        checks.append(dict(name=name,passed=bool(x)))
        if not x:raise AssertionError(name)
    for method in METHODS:
        base=pilot/'runs'/f'nominal__{method}__617901';path=base/'trace.npz';meta=js(base/'metadata.json');result=js(base/'result.json');before=sha(path)
        check(method+' complete hash-bound trial',meta['completed'] and before==meta['trace_sha256'])
        with np.load(path,allow_pickle=False) as z:
            t=z['time'];debug=z['debug_json'];public=z['observation_json'];action=z['executed_action'];alpha=z['alpha'];motion=z['executed_motion_reference']
        ids=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7))
        check(method+' exact225decision denominator',len(ids)==225)
        records=[];axis_checks=[];status=Counter();reasons=Counter()
        for i in ids:
            d=json.loads(str(debug[i]));tracking=d['effort_tracking'];ctx=tracking['debug'];status[tracking['status']]+=1
            if tracking['status']=='NUMERICAL_FAILURE':reasons[ctx['reason']]+=1
            f=json.loads(str(public[i]));check(method+f'/{t[i]:.3f} actual robot clock',abs(f['robot_control_time_s']-t[i])<1e-9)
            rechecks=ctx.get('scalar_native_recheck',[])
            for r in rechecks:
                delta=r['native_actual_loss']-r['enumerated_min_loss'];tolerance=max(1e-24,r['roundoff_tolerance'])
                axis_checks.append(dict(time_s=float(t[i]),status=tracking['status'],axis=r['axis'],native_loss=r['native_actual_loss'],analytic_loss=r['enumerated_min_loss'],delta=delta,tolerance=tolerance,ratio=delta/tolerance))
            lift_recheck=next((r for r in rechecks if r['axis']==5),None)
            rebuilt=None
            if lift_recheck is not None:
                rebuilt=lift_rebuild(f,ctx,parameters);selected=rebuilt['selected']
                check(method+f'/{t[i]:.3f} analytic liftloss exact reproduced',selected['analytic_loss_N2']==lift_recheck['enumerated_min_loss'])
                check(method+f'/{t[i]:.3f} native liftloss exact reproduced',selected['native_loss_N2']==lift_recheck['native_actual_loss'])
            rejected=tracking['status']=='NUMERICAL_FAILURE'
            records.append(dict(time_s=float(t[i]),status=tracking['status'],reason=ctx.get('reason'),
                failed_axis=None if not rejected else (rechecks[-1]['axis'] if rechecks else None),
                scalar_recheck=rechecks,lift_rebuild=rebuilt,
                realised4tick_grip_allzero=bool(np.all(action[i:i+4]==0.)),
                actual4tick_alpha_change5=(alpha[i+3]-alpha[i-1]).tolist(),
                actual4tick_motion_end3=motion[i+3].tolist(),
                ratio_memory_rollback=bool(d.get('ratio_effort_saturation_memory_rollback',False))))
        failed=[r for r in records if r['status']=='NUMERICAL_FAILURE'];lift_failed=[r for r in failed if r['failed_axis']==5]
        check(method+' all failures traced to lift recheck',len(failed)==len(lift_failed))
        check(method+' original input unchanged',sha(path)==before)
        accepted=[r for r in records if r['status']!='NUMERICAL_FAILURE']
        all_lift=[x for x in axis_checks if x['axis']==5]
        by_axis={str(a):dict(n=sum(x['axis']==a for x in axis_checks),positive_delta=sum(x['axis']==a and x['delta']>0 for x in axis_checks),
            zero_delta=sum(x['axis']==a and x['delta']==0 for x in axis_checks),negative_delta=sum(x['axis']==a and x['delta']<0 for x in axis_checks),
            delta=summary([x['delta'] for x in axis_checks if x['axis']==a])) for a in range(8)}
        rows.append(dict(method=method,trace_path=str(path),trace_sha256=before,task_success=result['task_success'],
            decisions=225,status_counts=dict(status),failure_reasons=dict(reasons),failures=len(failed),
            effort_solution_decision_fraction=len(accepted)/225,numeric_fallback_decision_fraction=len(failed)/225,
            fallback4tick_sensor_ticks=4*len(failed),holding_sensor_tick_denominator=900,
            failure_times_s=[r['time_s'] for r in failed],failed_axis_counts=dict(Counter(r['failed_axis'] for r in failed)),
            failure_window_zero_grip_count=sum(r['realised4tick_grip_allzero'] for r in failed),
            failure_ratio_memory_rollback_count=sum(r['ratio_memory_rollback'] for r in failed),
            failed_lift_stats=dict(native_loss_N2=summary([r['scalar_recheck'][-1]['native_actual_loss'] for r in failed]),
                loss_delta_N2=summary([r['scalar_recheck'][-1]['loss_worsening'] for r in failed]),
                configured_tolerance_N2=summary([max(1e-24,r['scalar_recheck'][-1]['roundoff_tolerance']) for r in failed]),
                delta_over_tolerance=summary([(r['scalar_recheck'][-1]['loss_worsening']/max(1e-24,r['scalar_recheck'][-1]['roundoff_tolerance'])) for r in failed]),
                desired_correction_N=summary([r['lift_rebuild']['requested_correction_N'] for r in failed]),
                force_arithmetic_discrepancy_abs_N=summary([abs(r['lift_rebuild']['selected_native_force_minus_analytic_N']) for r in failed]),
                native_force_magnitude_N=summary([abs(r['lift_rebuild']['selected']['native_force_N']) for r in failed]),
                large_operand_sum_N=summary([r['lift_rebuild']['absolute_large_operand_sum_N'] for r in failed]),
                all_at_nominal_upper=all(r['lift_rebuild']['selected_at_nominal'] and r['lift_rebuild']['selected_at_upper'] for r in failed),
                same_pool_native_regret_N2=summary([r['lift_rebuild']['native_objective_regret_vs_same_pool_N2'] for r in failed]),
                any_force_clipped=any(r['lift_rebuild']['force_clipped_at_selected'] for r in failed)),
            scalar_check_by_axis=by_axis,records=records))
    check('frozen tracker source unchanged after diagnosis',sha(core)==core_before)
    output=dict(schema='v12_pilot_native_loss_arithmetic_diagnosis_v1',passed=all(x['passed'] for x in checks),checks=checks,new_physics=False,
        execution_source_sha256=EXPECTED,source79_unchanged_report=execution['sources_unchanged'],
        input_sha256={str(p):sha(p) for p in [pilot/'execution_complete.json',pilot/'manifest.json',pilot/'trials.json',xml,core]},
        script_sha256=sha(__file__),lift_constants=dict(kp_N_m=parameters[0],kv_N_s_m=parameters[1],ctrlrange_m=cl,forcerange_N=fl),methods=rows,
        scope='Completed3pilot only;45live source/records not read or changed. Same candidate set/goal/offline native ranking is diagnostic, not a revised controller or admitted action.',
        interpretation='All per-decision failures remain failures and physical fallback remains part of the actual trial. Numeric candidate ranking under one canonical native force arithmetic is a prospective algorithmic recommendation, not a posthoc tolerance change.')
    with out.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# V12首批实际pilot：逐决策数值失败诊断','',
        '只读完整3条nominal/seed617901，各225个20ms决策。冻结eba6执行源码与原始记录保持不变，不读取后续45条过程性能，不运行物理，不生成修订策略。',
        '', '|方法|任务成功|有效effort解/225|数值fallback/225|占比|故障轴|', '|---|---|---:|---:|---:|---|']
    for m in rows:lines.append(f'|{m["method"]}|{m["task_success"]}|{225-m["failures"]}|{m["failures"]}|{100*m["numeric_fallback_decision_fraction"]:.3f}%|lift input5 / output24|')
    lines+=['','全部拒绝原因均为 `combined native effort does not realize the scalar optimum`，并非输入失效、物理限位违规或非有限解。其他轴的完整/截断复核计数见JSON，未到达的后续轴不算已经检查。',
        '', '## 数值根因','',
        '升降E使用kp=1500N/m和kv=140N·s/m，engineport包含已知机器人重力bias。原生算式与枚举算式数学等价，但浮点运算次序不同：',
        '','```','native = clip(kp*(reference+bias) - kp*q - kv*v)',
        'cached_beta = native(current_port) - kp*current_port',
        'enumerated = clip(kp*(reference+bias) + cached_beta)',
        'residual = force(candidate) - native_force(nominal) - requested_correction','```','',
        '约百牛顿的乘积/位姿项先抵消为约几十牛顿内部力，再与名义力相减得到小修正残差。既定roundoff检查只按最终loss/修正平方缩放，未包含形成这些小量时的大操作数误差。按同一原始packet、同一候选和冻结XML常数，所有日志中的lift analytic/native loss已逐值重现。','',
        '## 不改变目标的后续数值算法建议','',
        '应保留原力矩目标与真实候选域，解析分段仍可负责生成候选；候选的排序、最终残差和最优性复核应使用同一个canonical native力值计算。当前把一种浮点枚举loss的最小值当作另一种浮点计算的硬真值，导致合法候选被拒。相同native输入重复评价应一致，且可以另作可达/分段/边界检查；不要仅把门放宽到本批观察到的误差。',
        '对无饱和且固定q/v的增量，可在数学上先消去共同PD/阻尼项，明确表达端口增量；若采用这种重表示，应有独立前向误差/高精度反例及饱和分支验证，并新版本冻结后再跑，而不是更改本批或追求指标翻转。',
        '以上仅是离线算法建议。物理任务成功不等于225次全部得到有效数值解；本批真实回退及其对ratio记忆的影响必须保留。细小内部算术差不能作为物理压力、稳定性或安全证明。','',
        '各失败时点、精确loss/容差、算术重建、原候选与同候选池native排序、fallback实际动作及全部来源SHA见同名JSON。']
    with out.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(out=str(out),sha256=sha(out),checks=len(checks),methods=[{k:m[k] for k in ['method','failures','status_counts','failed_axis_counts','failed_lift_stats']} for m in rows]),ensure_ascii=False))

if __name__=='__main__':main()
