"""Publish completed V23 evidence without changing sources or evaluation rules."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    d=read('review/v23_small_readout_v1.json')
    a=read('review/v23_small_execution_v1.json')
    plan=read('review/v23_small_plan_v1.json')
    old=read('review/v22_small_readout_v1.json')
    assert d['complete'] and not d['issues'] and a['passed'] and not a['issues']
    assert len(d['records'])==9 and len(a['checks'])==127 and a['all_assigned_checked']
    assert d['plan_sha256']==a['plan_sha256']==sha(ROOT/'review/v23_small_plan_v1.json')
    for name, digest in plan['source_sha256'].items():
        path=ROOT.parent/'五指策略接入_2026-11-03'/name[11:] if name.startswith('foundation/') else ROOT/name
        assert sha(path)==digest, name
    roles=('v21','backend','joint')
    idx={(r['variant'],*r['context_key']):r for r in d['records']}
    oldidx={(r['variant'],*r['context_key']):r for r in old['records']}
    labels=('名义运输','易碎冲突持握','摩擦变化运输')
    contexts=[tuple(k) for k in plan['contexts']]
    def count(role,key):
        return sum(idx[(role,*c)]['metrics'][key] is True for c in contexts)
    agg='\n'.join('|'+label+'|'+'|'.join(f'{count(v,key)}/3' for v in roles)+'|' for label,key in (
        ('原任务完成','original_task_success'),('物理清离','original_physical_clearance_success'),
        ('全程力限通过','original_full_guard_pass'),('释放窗曾落地↓','ever_floor'),
        ('末段低运动驻留','final_low_motion_suffix_found')))
    rows=[]; timing=[]; comparison=[]
    for label,c in zip(labels,contexts):
        for role in roles:
            r=idx[(role,*c)];m=r['metrics'];w=r['observed_control_walltime']
            rows.append(f'|{label}|{role}|{r["release_TV"]["all_actor_skin_TV_N"]:.3f}|{m["end_angular_speed_rad_s"]:.6f}|{m["original_200Hz_clearance_latency_s"]:.3f}|{m["receiver_first20ms_peak_N"]:.3f}|')
            timing.append(f'|{label}|{role}|{w["median_ms"]:.3f}|{w["p95_ms"]:.3f}|{w["maximum_ms"]:.3f}|{w["above_5ms_count"]}/{w["samples"]}|')
        new=idx[('joint',*c)];prior=oldidx[('joint',*c)]
        delta_tv=(new['release_TV']['all_actor_skin_TV_N']/prior['release_TV']['all_actor_skin_TV_N']-1)*100
        delta_omega=(new['metrics']['end_angular_speed_rad_s']/prior['metrics']['end_angular_speed_rad_s']-1)*100
        comparison.append(f'|{label}|{prior["metrics"]["end_angular_speed_rad_s"]:.6f}|{new["metrics"]["end_angular_speed_rad_s"]:.6f}|{delta_omega:+.2f}%|{delta_tv:+.2f}%|')
    branch=[]
    for role in roles:
        rr=[r for r in a['rows'] if r['variant']==role]
        n=lambda k:sum(r['counts'].get(k,0) for r in rr)
        branch.append(f'|{role}|{n("direct_search_capture")}|{n("accepted_full")}|{n("accepted_prefix")}|{n("fallback")}|')
    now=datetime.now(timezone.utc).isoformat()
    name='V23高效执行闭环九条结果与规范任务重构_20261106.md'
    assert not (ROOT/name).exists(), 'Do not overwrite a published stage report'
    body=f'''# 多任务融合：高效执行已验证，交付稳定性仍需重构

V23完成同条件九条物理运行及127项全执行审查。联合任务版的观察计算p95从V22约1.16–1.18秒降至约28.5–29.3毫秒，但末段低运动驻留由1/3降为0/3，摩擦变化情境末角速度明显恶化。本轮保留这个负结果，不扩大V23物理矩阵，也不以计算提速代替综合性能目标。

报告UTC：{now}。九条均已终态；原平台、旧186来源、全部历史数据及失败记录保留。下一版V24代码草稿已存在，但尚未以完成验收或物理改善对外表述。

## 全景方案与综合评价

总体链条是：公共触觉与机器人观测及其有效性→机械/速率/当前状态权限→法向与运动任务的统一表达→交付退出进度与全手路径→原问题证据及实际发出动作核验→物理反馈。持握、运输、噪声/延迟/丢包、材料与成功定义沿冻结版本。

评价同时报告原任务完成、全程力限、落地、接收冲击、清离延迟、物体末端运动、持续低运动驻留、逐指力变化总量与计算开销。沿用固定v3和12–16秒原始200Hz逐指TV，不重新调分数或阈值。低运动驻留要求接收面有载、手已低力清离、无其它支撑，线速度≤0.005m/s、角速度≤0.15rad/s持续0.2秒。它不是永久稳定放置或完整几何容纳证书。

此前完整16条件V21相对论文启发的力比PI迁移基线：原任务14/16对12/16，全程力限8/16对3/16，释放窗落地0/16对4/16；释放TV配对中位下降43.25%，末段低运动均5/16。这支持已有综合收益，不能外推后续新版本或未见情境。PI不是论文全系统复现，也不简化称为只会持续加力的策略。

## 本轮同条件九条

A=v21冻结策略；B=backend保持V21任务规则、换为高效可认证求解；C=joint与B同后端、使用V22联合任务。三条件为nominal/transport/619741、fragile_conflict/hold/619741、friction_drop/transport/619742。九条新运行仅对应三个已见开发条件和两个物理种子簇，不是九个独立重复。A三条完整16秒历史桥各92项通过。

|指标|A v21|B backend|C joint|
|---|---:|---:|---:|
{agg}

|条件|角色|释放逐指TV N↓|末角速度rad/s↓|清离延迟s↓|接收首20ms峰N↓|
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

|联合任务同角色对比|V22末角速|V23末角速|末角速变化|TV变化|
|---|---:|---:|---:|---:|
{chr(10).join(comparison)}

同V23后端相比，joint三条件末角速度都更高，TV在名义与易碎条件更高。摩擦变化虽然TV略低，末角速却达到0.818257rad/s。力总变差包含正常卸力，不等同纯振荡；力信号改善不能自动解释为物体放置改善。

## 计算和执行边界

|条件|角色|释放控制中位ms|p95 ms|最大ms|超过5ms次数|
|---|---|---:|---:|---:|---:|
{chr(10).join(timing)}

计时是同机批次观察值，包含直接状态阶段，不是受控独占硬件基准；新joint仍有约1.1–1.24秒极值，未达到200Hz实时要求。计算耗时不推进固定仿真时钟，所以不能把物理变化归因为算得更快、反馈更及时。

执行核查127/127通过，7209个释放决策完整保留，每臂2403。下表fallback由总数减已记录direct/full/prefix独立核对；若原键结构不同，不以缺键零冒充无fallback。

|角色|direct|全层|部分前缀|fallback|
|---|---:|---:|---:|---:|
|v21|1133|1152|118|0|
|backend|1133|1266|0|4|
|joint|1133|1261|0|9|

B/C共5630个实际提交有限证书在各自保存的原问题上通过KKT复核；206个joint法向witness另列，不冒充joint任务成功。实际21维发出/回执、原生当前状态权限、同式输出评分及源/依赖身份均通过，execution_changed与stale均0。这些证明执行和归档一致，不是性能或物理安全证书。

## 结构性原因与下一版任务定义

原法向任务允许多个同样达到最优纠偏量的21维参照点，后续运动预算却由所选参照点构造。因此更换求解实现能够改变预算，进而改变实际控制问题；不能把不同预算下较低的代价称为同一问题上的改进。

摩擦变化归档在13.630秒公共输入逐位一致时，法向最优值仅舍入差，参照点已不同；13.640秒出现约0.002968的实际轴参考差。固定保存输入分别重算两后端，复现参照最大差约0.35657、预算差约0.16105及同量级发出差。两者均有完整原问题证据。这定位了任务定义对非唯一参照的依赖，尚不能证明一个时刻独自解释全部终态恶化。

V24提出明确的整体合同：先保持原法向最优值，再在其达到集合中选择归一参考最小范数的唯一参照，专用于建立运动预算；随后实际联合任务在该预算内协调法向纠偏、可观测运动与退出进度。用于选参照的证据与实际执行任务证据分开，最小范数参照不得作为实际21维保持等式，否则会锁死退出自由度。参照两层不完整认证时，不声称预算已建立，转入原透明合法fallback。

这使预算来源与任务角色可明确陈述，但尚不保证物理稳定。当前运动表示仍是局部触觉投影，并非完整6D物体运动反馈；后续必须检验观测覆盖与实际接触响应，不能靠调权、场景专用规则或移动评价阈值掩盖缺口。

V24应依次完成作者数学、独立原问题与发出、三角色零步构造和六公共流验收，再预登记三情境×三角色对比。参考臂为冻结V23 joint，另两臂为canonical_backend/canonical_joint；保持PI、V21、V22、V23历史参考，避免只超过退步版本就宣称传统基线已被全面超越。小规模达标后再考虑完整16与未见物理种子。

## 可复核材料

- [完整九条读数](review/v23_small_readout_v1.json)
- [127项执行审查](review/v23_small_execution_v1.json)
- [三条完整参考桥](review/v23_original_V21_full16_bridges_v1.json)
- [三组完整时序与图注](report/v23_small_v1/release_curves_v1/figures.md)
- [物理前合同](review/V23高效执行同条件验证合同_v1.md)
- [首次分歧与固定输入反事实](review/V23摩擦变化joint首次参考分歧与局部反事实_v1.md)
- [唯一参照与证书角色边界](review/V23唯一法向witness任务合同候选与接线边界_v2.md)
- [此前完整16条件](V21完整16条件综合对照与数值重构依据_20261105.md)

![名义运输](report/v23_small_v1/release_curves_v1/01_nominal_transport_619741.png)

![易碎冲突持握](report/v23_small_v1/release_curves_v1/02_fragile_conflict_hold_619741.png)

![摩擦变化运输](report/v23_small_v1/release_curves_v1/03_friction_drop_transport_619742.png)

输入SHA256：readout `{sha(ROOT/'review/v23_small_readout_v1.json')}`；execution `{sha(ROOT/'review/v23_small_execution_v1.json')}`；plan `{sha(ROOT/'review/v23_small_plan_v1.json')}`。主目标尚未证明完成。
'''
    assert all((ROOT/n).is_file() for n in re.findall(r'\]\(([^)]+)\)',body))
    (ROOT/name).write_text(body,encoding='utf-8')
    hist=ROOT/'report/progress_history';hist.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for n in ('研究进度.json','当前进度与阶段汇报.md','review/v23_small_progress_v1.json'):
        shutil.copyfile(ROOT/n,hist/(stamp+'_'+Path(n).name))
    p=read('研究进度.json')
    p.update(updated_utc=now, goal_status='active', superiority_proven=False,
        latest_report=name, latest_stage_report=name, latest_algorithm_report=name,
        current_work='V23 all9 and127 execution complete; speed gain with joint physical regression. V24 canonical-witness task-contract draft under actual acceptance; no V24 physics dispatched.',
        current_physics_processes=[],current_physics_jobs=[],active_physics_jobs=[],
        previous_goal_turn_classification='no_progress: prior turn was requested read-only status summary; current continuation revalidated all9 terminal and V24 incomplete gates, then resumed authorized acceptance.')
    p['v23_current'].update(status='small9_complete_readout_execution127_passed_regression_retained',
        physics_active=False, physics_sessions=[], completed_physical_runs=9,
        execution_audit_pending=False, execution_audit='review/v23_small_execution_v1.json',
        execution_checks=127, report=name, superiority_proven=False,
        next='Retain negative result; verify V24 unique-witness task assembly and actual execution before new physical registration')
    p['v24_current']=dict(status='draft_under_author_independent_runtime_acceptance',
        source_namespace='v24_canonical_task_fusion', expected_source_count=192,
        source_count_verified=False, physical_runs=0, physics_active=False, physics_sessions=[],
        superiority_proven=False, mathematical_probe_only='review/v23_canonical_normal_witness_probe_v1.json')
    save('研究进度.json',p)
    progress=read('review/v23_small_progress_v1.json')
    progress.update(updated_utc=now, completed=9, all_terminal=True,
        phase='readout_and_execution_complete_regression_retained',physics_active=False,physics_sessions=[],
        execution_audit='review/v23_small_execution_v1.json',report=name)
    save('review/v23_small_progress_v1.json',progress)
    (ROOT/'当前进度与阶段汇报.md').write_text(f'''# V23九条完成，V24规范任务合同验收中

更新UTC：{now}。当前无物理仿真后台运行，主目标active，综合优势未证明完成。

V23任务3/3、清离3/3、力限2/3、落地0/3均保持；joint末段低运动由V22的1/3降至0/3。计算p95由约1.17秒降至约29毫秒，但尚非实时，物理回退完整保留。

九条全部终态；三条完整参考桥各92通过，127项/7209决策执行审查通过，原186来源及历史结果保留。V24规范唯一参照并将预算证据与实际任务分开；草稿已落盘，作者/独立/零步/公共流验收中，尚无V24物理结果。

[完整全景报告]({name})；[三组时序](report/v23_small_v1/release_curves_v1/figures.md)；[固定读数](review/v23_small_readout_v1.json)；[执行审查](review/v23_small_execution_v1.json)。
''',encoding='utf-8')
    print(json.dumps(dict(report=name,source186_unchanged=True,completed=9,execution_checks=127,v24_physics=0),ensure_ascii=False))

if __name__=='__main__':
    main()
