"""Publish the completed native48 snapshot from audited artifacts; starts no work."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def publish():
    read=lambda name:json.loads((ROOT/name).read_text(encoding="utf-8"))
    audit=read("review/v12_native_full48_readonly_acceptance_v1.json")
    plan=read("review/v12_native_development_registration_v1.json")
    summary=read("report/v12_native_full48_v1/summary.json")
    figure=read("report/v12_native_full48_v1/figure_data_and_provenance.json")
    assert audit["passed"] and len(audit["checks"])==13640 and not audit["issues"]
    assert plan["source_count"]==80 and all(x["n"]==16 for x in summary["records"])
    rows={x["method"]:x for x in summary["records"]};methods=figure["methods"]
    assert len(rows)==3 and sum(x["n"] for x in rows.values())==48
    groups=[]
    for g in plan["groups"]:
        d=ROOT/g["output"]; terminal=read(g["output"]+"/execution_complete.json")
        assert terminal["execution_complete"]
        assert len(list(d.glob("runs/*/result.json")))==g["records"]
        groups.append(dict(output=g["output"],completed=g["records"],planned=g["records"],execution_complete=True))
    now=datetime.now(timezone.utc).isoformat()
    lines=["# 当前进度与阶段汇报","",f"更新：{now}。","",
      "本文件是可直接查阅的正式阶段快照；每个重要阶段另在聊天中独立汇报。","",
      "**最新48/48条完整16秒物理记录及13,640项独立验收均完成。当前没有在运行的物理仿真。**",
      "原五指平台、旧V12的48条结果与来源均保留；新旧结果分别统计。","",
      "整体架构：公共触觉观测 → 多任务联合力目标 → 抓握与用户运动统一执行 → 反馈与释放优先。","",
      "四环境、静态/运输两任务、两个已见物理种子；三法各16条件，共用观测、执行和评价。摩擦比对照迁移论文机制，并非旧完整控制器逐轨迹复现。","",
      "|指标|摩擦比对照|固定力对照|融合策略|","|---|---:|---:|---:|"]
    def row(label,values):lines.append("|"+label+"|"+"|".join(values)+"|")
    row("完整任务完成",[f"{rows[m]['fullmotion']}/16" for m in methods])
    row("稳态平均总指力 N",[f"{rows[m]['static_force_N']:.2f}" for m in methods])
    row("稳态原200Hz逐指力变化率 N/s",[f"{rows[m]['static_raw200TV_N_s']:.2f}" for m in methods])
    row("全程局部超限积分合计 N·s",[f"{rows[m]['local_Ns_sum']:.3f}" for m in methods])
    row("全程力限通过",[f"{rows[m]['guard']}/16" for m in methods])
    row("最高单指瞬时峰 N",[f"{rows[m]['max_peak_N']:.2f}" for m in methods])
    row("控制分支耗时中位数 ms",[f"{x:.2f}" for x in figure['control_median_ms']])
    lines += ["","在已检验条件中，融合有明确多指标收益：稳态用力与绝对力变化率约减半，任务完成数达到16/16，超限剂量降低。初始稳定确认等待三法均为0.20秒，无加速收益。",
      "","仍有代价：融合12/16未通过全程力限，最高峰略高，控制计算更慢；全持握相对波动CV多数共同窗更差。控制耗时不含外层传感采样、几何采集、任务构建和物理积分，不能视为硬件实时保证。",
      "","稳态采用原8.2–8.9秒固定窗。扰动后共同可评价窗为9/16；不可评保留NULL。两个物理种子及共享前缀不能当成16次独立重复；尚不能外推总体或硬件优势。","",
      "![综合收益与计算代价](report/v12_native_full48_v1/综合收益与计算代价.png)","",
      "可视化三法同步回放已验收：[打开本机回放](http://127.0.0.1:8892/)。页面显示真实已保存物理记录，支持场景、任务、种子、视角及统一时间切换；不启动新仿真。服务与物理任务分别管理。","",
      "![三种真实结果的同步回放](report/v12_native_full48_v1/真实结果三方法回放验收.jpg)","",
      "后续优先验证碰撞/释放短峰的数值可信度，再推进接触转换与释放的整体协同，并扩大独立物理条件；不修改现有评价阈值或事后增设综合权重。","",
      "- [完整复验与收益代价报告](V12执行一致性修订与48组复验_20261105.md)",
      "- [逐条件完整指标](report/v12_native_full48_v1/metric_readout.json)",
      "- [独立原始记录验收](review/v12_native_full48_readonly_acceptance_v1.json)",
      "- [旧V12完整结果留档](V12多场景开发对比_20261105.md)",""]
    (ROOT/"当前进度与阶段汇报.md").write_text("\n".join(lines),encoding="utf-8")
    p=ROOT/"研究进度.json";d=read("研究进度.json")
    d.update(updated_utc=now,goal_status="active",superiority_proven=False,
      implementation="V12Native whole public observation / joint task-force intent / common native robot-effort execution;80 frozen source assets,48 complete development records and independent13640 checks. Originals preserved.",
      current_sources=plan["source_sha256"],current_sources_status="Latest V12Native80 immutable source set; original79 and older assets preserved.",
      last_completed_performance_matrix=[g["output"] for g in plan["groups"]],
      last_completed_performance_scope="48 unique completed records,16 per method,2 seen physical clusters; not independent confirmation.",
      latest_report="V12执行一致性修订与48组复验_20261105.md",
      current_work="Native48 audited and formally reported; replay UI verified. Numerical impact convergence prototype prepared, no physics started.",
      current_physics_processes=[],current_research_physics_process=None,current_physics=None,
      local_advantages_observed=True,
      claim_interpretation="Clear descriptive multi-metric development benefits, with peak, relative-variation and computation tradeoffs; no universal or hardware superiority claim.",
      previous_goal_turn_classification="progress: native48 completed,13640 raw/source checks passed, relative fluctuation and cost disclosed, real record replay verified.",
      next_required_work=["Pin and validate the fixed-input numerical-contact probe; require exact coarse full-trajectory bridges before finer integration.","Use verified transition/impact/release evidence to design whole task-force-motion coordination, retaining common baseline permissions and unchanged metrics.","Expand independent physical conditions with predeclared comparison and retain all failures; no posthoc utility weights."])
    native=d["v12_native_current"];native.update(status="complete48_audited",new48_completed=48,physics_active=False,
      full48_audit="review/v12_native_full48_readonly_acceptance_v1.json",full48_checks=13640,
      numerical_failures=0,decision_count=10800,latest_report=d["latest_report"],full48_summary=summary,
      descriptive_multi_metric_benefits=True,formal_general_superiority_proven=False)
    snap=dict(updated_utc=now,completed=48,planned=48,running_session=None,groups=groups,runtime_error_files=[],audit_passed=True,checks=13640)
    native["progress_snapshot"]=snap
    d["visual_replay_current"]=dict(url="http://127.0.0.1:8892/",verified=True,server_session=6234,kind="saved physics results only",physics_steps_started=0,screenshot="report/v12_native_full48_v1/真实结果三方法回放验收.jpg")
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (ROOT/"阶段进度记录.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(snap,ensure_ascii=False)+"\n")
    print(json.dumps(snap,ensure_ascii=False))
if __name__=="__main__":publish()
