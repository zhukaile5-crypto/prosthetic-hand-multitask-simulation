"""Synthetic acceptance cases for independent comprehensive evaluation."""
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import evaluate_trial, evaluate_state, force_quality_trace
from analyze import analyze_records, paired_cluster_estimate, holm_adjust, summarize_method


def fixture(duration=4., dt=.005):
    t = np.arange(round(duration / dt) + 1) * dt
    count = len(t)
    trace = dict(time=t, normal=np.full((count, 5), 2.),
                 position=np.tile([0., 0., .5], (count, 1)),
                 relative=np.zeros((count, 3)), relative_velocity=np.zeros((count, 3)),
                 rotation=np.tile(np.eye(3), (count, 1, 1)),
                 angular_velocity=np.zeros((count, 3)), support=np.zeros(count),
                 palm=np.zeros(count), validity=np.ones((count, 5), dtype=bool))
    meta = dict(scenario=dict(name="synthetic", title="独立评价验收合成数据", duration=duration,
                             limit=[10.] * 5, budget=50., release_at=None),
                reference=dict(relative=[0., 0., 0.], rotation=np.eye(3).tolist(),
                               absolute_z=.5, orientation_mode="full"),
                method="unified", seed=1, split="synthetic", command_at=0., events=[])
    config = dict(first_stability_horizon=4., establishment_deadline=2., static_window=[2.5, 3.5])
    return trace, meta, config


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(dict(name=name, passed=True))


checks = []
diagnostics = {}
trace, meta, cfg = fixture()
result = evaluate_trial(trace, meta, cfg)
check("首次稳定包含200ms确认窗口，不回填起点", abs(result["first_stable_time"] - .2) < 1e-9)
check("全部任务受限未稳定时间", abs(result["restricted_unstable_time"] - .2) < 1e-9)
check("有效窗完整采样201端点", result["static_window"]["metrics"]["samples"] == 201)
trace, meta, cfg = fixture()
trace["relative_velocity"][trace["time"] < .5, 0] = .02
result = evaluate_trial(trace, meta, cfg)
check("延迟进入稳定域按0.7s确认", abs(result["first_stable_time"] - .7) < 1e-9)
trace, meta, cfg = fixture()
trace["support"][:] = .1
result = evaluate_trial(trace, meta, cfg)
check("桌面托住不得记抬升稳定", result["first_stable_time"] is None and result["restricted_unstable_time"] == 4.)
check("建立失败保留全部任务分母", result["establishment_failure"] and not result["task_success"])
trace, meta, cfg = fixture()
trace["relative_velocity"][2] = [0., .006, 0.]
check("三维速度不是仅轴向代理", not evaluate_state(trace, meta, cfg)["stable"][2])
trace, meta, cfg = fixture()
wave = .5 * np.sin(2 * np.pi * 4 * trace["time"])
trace["normal"][:, 0] += wave
trace["normal"][:, 1] -= wave
result = evaluate_trial(trace, meta, cfg)
check("反向逐指涨落不会被总力平稳遮蔽", result["static_window"]["metrics"]["total_tv_rate"] < 1e-8 and
      result["static_window"]["metrics"]["finger_tv_rate"] > 10.)
trace, meta, cfg = fixture()
trace["normal"][trace["time"] >= 2.] = 0.
meta.update(failure_at=2., failure_reason="loss")
result = evaluate_trial(trace, meta, cfg)
check("失抓后零力不能成为平稳收益", not result["static_window"]["eligible"] and not result["task_success"])
check("先稳后掉落仍是完整任务失败", result["first_stability_attained"] and result["post_establishment_loss"])
trace, meta, cfg = fixture()
trace["relative_velocity"][:, 0] = .006
result = evaluate_trial(trace, meta, cfg)
check("静态窗不按整窗S筛选以免排除待测波动", result["static_window"]["eligible"] and not result["first_stability_attained"])
trace, meta, cfg = fixture()
trace["substep_time"] = np.arange(8001) * .0005
trace["substep_normal"] = np.full((8001, 5), 2.)
trace["substep_normal"][3001, 0] = 12.
result = evaluate_trial(trace, meta, cfg)
check("保留控制采样之间的瞬时峰值", result["force_safety"]["finger_peak"][0] == 12. and
      not result["force_safety"]["force_protection_pass"])
check("实际逐指超限积分独立于总力积分", abs(result["force_safety"]["local_excess_integral"] - .001) < 1e-10 and
      result["force_safety"]["total_excess_integral"] == 0.)
trace, meta, cfg = fixture()
trace["substep_time"] = np.arange(8000) * .0005
trace["substep_duration"] = np.full(8000, .0005)
trace["substep_normal"] = np.full((8000, 5), 2.)
trace["substep_normal"][-1, 0] = 12.
result = evaluate_trial(trace, meta, cfg)
check("完整末步超限按solver区间积分，不损半末步", abs(result["force_safety"]["local_excess_integral"] - .001) < 1e-10 and
      result["force_safety"]["integration_method"] == "solver_interval_sum" and
      abs(result["force_safety"]["covered_duration"] - 4.) < 1e-10)
trace, meta, cfg = fixture()
meta["events"] = [dict(id="pull", kind="temporary", onset=1., end=1.5, deadline=3.)]
trace["relative_velocity"][(trace["time"] >= 1) & (trace["time"] < 2), 0] = .02
result = evaluate_trial(trace, meta, cfg)
check("暂时扰动后恢复从结束到确认", result["events"][0]["outcome"] == "recovered_after_event" and
      abs(result["events"][0]["recovery_time"] - .7) < 1e-9)
trace, meta, cfg = fixture()
meta["events"] = [dict(id="pull", kind="temporary", onset=1., end=1.5, deadline=3.)]
trace["relative_velocity"][(trace["time"] >= 1) & (trace["time"] < 1.1), 0] = .02
result = evaluate_trial(trace, meta, cfg)
check("扰动内恢复单列，不生成负延迟", result["events"][0]["outcome"] == "recovered_during_event" and
      result["events"][0]["recovery_time"] is None)
trace, meta, cfg = fixture()
meta["events"] = [dict(id="pull", kind="temporary", onset=1., end=1.5, deadline=3.)]
result = evaluate_trial(trace, meta, cfg)
check("始终未失稳不伪作零延迟恢复", result["events"][0]["outcome"] == "never_unstable" and
      result["events"][0]["recovery_time"] is None)
trace["relative_velocity"][trace["time"] >= 1.8, 0] = .02
result = evaluate_trial(trace, meta, cfg)
check("扰动后才失稳不倒算为外扰内失稳", result["events"][0]["outcome"] == "never_unstable" and
      result["events"][0]["subsequent_instability"])
meta["failure_at"] = .8
result = evaluate_trial(trace, meta, cfg)
check("先掉落事件单列未暴露，不能删去失败", result["events"][0]["outcome"] == "not_exposed_failure" and not result["task_success"])
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
meta["events"] = [dict(id="pull", kind="temporary", onset=1., end=1.5, deadline=3.)]
trace["normal"][trace["time"] > 1.2 + 1e-10] = 0.
result = evaluate_trial(trace, meta, cfg)
check("用户释放中断恢复独立报告", result["events"][0]["outcome"] == "preempted_by_release" and result["release_success"])
trace["palm"][trace["time"] >= 1.2] = 1.
result = evaluate_trial(trace, meta, cfg)
check("五指低力但掌部仍接触不能记释放完成", not result["release_success"])
trace, meta, cfg = fixture()
cfg["palm_limit"] = 5.
trace["palm"][:] = np.nan
result = evaluate_trial(trace, meta, cfg)
check("保护范围内掌部真值缺失不可默认为合格", not result["force_safety"]["force_protection_pass"] and
      not result["truth_record_complete"] and not result["first_stability_attained"])
trace, meta, cfg = fixture()
trace["rotation"][:] = np.nan
result = evaluate_trial(trace, meta, cfg)
check("缺失功能姿态不默认为零误差", not result["first_stability_attained"] and not result["truth_record_complete"])
trace, meta, cfg = fixture()
trace = {key: value[trace["time"] <= 1.] for key, value in trace.items()}
meta["failure_at"] = .9
result = evaluate_trial(trace, meta, cfg)
check("提前中止不缩短失败首次稳定期限", result["horizon"] == 4. and not result["truth_record_complete"])
trace, meta, cfg = fixture()
valid = np.ones(len(trace["time"]), dtype=bool)
valid[(trace["time"] > .05) & (trace["time"] < .4)] = False
trace = {key: value[valid] for key, value in trace.items()}
result = evaluate_trial(trace, meta, cfg)
check("记录间隙不能凑够连续确认时长", abs(result["first_stable_time"] - .6) < 1e-9)
trace, meta, cfg = fixture()
trace["angular_velocity"][:, 1] = 2.
meta["reference"].update(orientation_mode="axis", axis=1)
result = evaluate_trial(trace, meta, cfg)
check("轴对称物体忽略无功能意义的绕轴自转", result["first_stability_attained"])

# Independent paired arithmetic oracle: bootstrap unit is seed across scenarios.
records = []
for seed in (1, 2, 3, 4):
    for scenario in ("s1", "s2"):
        for method in ("ratio", "unified"):
            trace, meta, cfg = fixture()
            meta.update(seed=seed, method=method, split="test")
            meta["scenario"]["name"] = scenario
            record = evaluate_trial(trace, meta, cfg)
            record["restricted_unstable_time"] += .1 * seed if method == "unified" else 0.
            records.append(record)
summary = analyze_records(records, bootstrap=5000)
comparison = next(r for r in summary["comparisons"] if r["endpoint"] == "restricted_unstable_time")
check("完整seed簇配对而非场景或时间点独立化", comparison["clusters"] == 4 and comparison["pairs"] == 8)
check("配对主端点均值独立算术校验", abs(comparison["difference"] - .25) < 1e-10)
check("两主要端点Holm校正", holm_adjust([.01, .03]) == [.02, .03])
check("门槛与品质不合成任意总分", "gate_margins" in summary and "score" not in summary)

# Full-record contract: finite endpoints alone do not prove a complete trial.
trace, meta, cfg = fixture()
keep = ~((trace["time"] > 2.) & (trace["time"] < 3.))
trace = {key: value[keep] for key, value in trace.items()}
result = evaluate_trial(trace, meta, cfg)
check("删中段但保末端仍必须判truth记录不完整", not result["truth_record_complete"] and
      not result["task_success"] and not result["static_window"]["eligible"])
trace, meta, cfg = fixture()
keep = trace["time"] >= .1
trace = {key: value[keep] for key, value in trace.items()}
result = evaluate_trial(trace, meta, cfg)
check("缺开头不能用已有首采样偷改任务起点", not result["truth_record_complete"] and
      not result["task_success"] and result["truth_coverage"]["start"] == 0.)
trace, meta, cfg = fixture()
cfg["require_full_substeps"] = True
trace["substep_time"] = np.arange(2000) * .0005
trace["substep_duration"] = np.full(2000, .0005)
trace["substep_normal"] = np.full((2000, 5), 2.)
result = evaluate_trial(trace, meta, cfg)
check("4s任务仅1s物理子步不能宣称完整或安全", not result["force_safety"]["full_substeps"] and
      not result["force_safety"]["force_protection_pass"] and not result["task_success"] and
      result["force_safety"]["total_excess_integral"] is None)
trace, meta, cfg = fixture()
cfg["require_full_substeps"] = True
result = evaluate_trial(trace, meta, cfg)
check("formal缺物理向量不降级成200Hz合格", not result["force_safety"]["force_protection_pass"] and
      not result["force_quality_contract"]["valid"] and
      result["static_window"]["metrics"]["finger_tv_rate"] is None)


def physical_fixture(physical_dt, mixed=True):
    trace, meta, cfg = fixture()
    meta["physics_timestep"] = physical_dt
    cfg.update(require_full_substeps=True, force_bandwidth_hz=50., force_filter_span=.05)
    source_t = np.arange(round(4. / physical_dt)) * physical_dt
    def waveform(t):
        force = np.full((len(t), 5), 2.)
        force[:, 0] += .4 * np.sin(2 * np.pi * 10 * t)
        if mixed:
            force[:, 0] += .6 * np.sin(2 * np.pi * 230 * t)
        return force
    trace["normal"] = waveform(trace["time"])
    trace["substep_time"] = source_t
    trace["substep_duration"] = np.full(len(source_t), physical_dt)
    trace["substep_normal"] = waveform(source_t)
    return trace, meta, cfg


trace, meta, cfg = physical_fixture(.0005)
quality = force_quality_trace(trace, meta, cfg)
result = evaluate_trial(trace, meta, cfg)
check("完整solver区间覆盖才记fullsubsteps", result["force_safety"]["full_substeps"] and
      abs(result["force_safety"]["coverage"]["covered_duration"] - 4.) < 1e-8)
check("离线滤波严格对齐既有200Hz任务时钟", np.array_equal(quality["time"], trace["time"]) and
      quality["contract"]["group_delay_seconds"] == 0. and
      not quality["contract"]["sensor_filter"])
interior = (quality["time"] >= .5) & (quality["time"] <= 3.5)
t = quality["time"][interior]
basis = np.column_stack([np.ones(len(t)), np.sin(2 * np.pi * 10 * t),
                         np.cos(2 * np.pi * 10 * t), np.sin(2 * np.pi * 30 * t),
                         np.cos(2 * np.pi * 30 * t)])
coefficient = np.linalg.lstsq(basis, quality["normal"][interior, 0], rcond=None)[0]
low_amplitude = float(np.hypot(coefficient[1], coefficient[2]))
alias_amplitude = float(np.hypot(coefficient[3], coefficient[4]))
phase_error = float(np.arctan2(coefficient[2], coefficient[1]))
diagnostics["fir_spectral_oracle"] = dict(low_frequency_hz=10, injected_high_hz=230,
                                        low_amplitude=low_amplitude, alias_amplitude=alias_amplitude,
                                        phase_error_radians=phase_error)
check("高频230Hz先抗混叠，降200Hz后30Hz伪波抑制", alias_amplitude < .003 and low_amplitude > .36)
check("对称FIR低频相位不滞后", abs(phase_error) < 1e-7)
metrics = result["static_window"]["metrics"]
check("主TV为固定band，raw200与子步TV诊断同时保留", metrics["finger_tv_rate"] < metrics["raw200_finger_tv_rate"] and
      metrics["substep_finger_tv_rate"] > metrics["raw200_finger_tv_rate"] and
      metrics["total_sd"] == metrics["raw200_total_sd"])
check("峰值与超限仍原始子步不受FIR过滤", result["force_safety"]["finger_peak"][0] > 2.9)
trace2, meta2, cfg2 = physical_fixture(.00025)
quality2 = force_quality_trace(trace2, meta2, cfg2)
result2 = evaluate_trial(trace2, meta2, cfg2)
clock_error = float(np.max(np.abs(quality["normal"][interior] - quality2["normal"][interior])))
tv_error = float(abs(result2["static_window"]["metrics"]["finger_tv_rate"] - metrics["finger_tv_rate"]))
diagnostics["physical_step_bandwidth_oracle"] = dict(steps=[.0005, .00025],
                                                     waveform_difference_max=clock_error,
                                                     primary_tv_difference=tv_error)
check("500/250us使用同50ms核并保持相同band与任务时钟", quality["contract"]["effective_kernel_span"] == .05 and
      quality2["contract"]["effective_kernel_span"] == .05 and clock_error < .001 and tv_error < .02)
trace, meta, cfg = physical_fixture(.0005, mixed=False)
trace["skin_normal"] = trace["normal"].copy()
trace["substep_skin_normal"] = trace["substep_normal"].copy()
cfg["require_skin_force"] = True
trace["skin_normal"][trace["time"] >= 2., 4] = 12.
trace["substep_skin_normal"][trace["substep_time"] >= 2., 4] = 12.
result = evaluate_trial(trace, meta, cfg)
check("外界指接触不得按目标身份排除限力", not result["force_safety"]["force_protection_pass"] and
      result["force_safety"]["force_signal_kind"] == "all_instrumented_finger_external_contacts" and
      result["force_safety"]["object_specific_finger_peak"][4] == 2.)
check("支持接触用grasp力而S限力用全skin压力", np.all(evaluate_state(trace, meta, cfg)["effective_support"]) and
      not np.any(evaluate_state(trace, meta, cfg)["stable"][trace["time"] >= 2.]))
del trace["substep_skin_normal"]
result = evaluate_trial(trace, meta, cfg)
check("formal skin物理记录缺失不退回目标物专属安全", not result["force_safety"]["valid"] and
      not result["force_safety"]["force_protection_pass"])
trace, meta, cfg = physical_fixture(.0005)
keep = ~((trace["substep_time"] >= 2.) & (trace["substep_time"] < 3.))
for key in ("substep_time", "substep_duration", "substep_normal"):
    trace[key] = trace[key][keep]
result = evaluate_trial(trace, meta, cfg)
check("物理子步删除中段即使两端保留也不完整", not result["force_safety"]["full_substeps"] and
      not result["force_safety"]["force_protection_pass"])
trace, meta, cfg = physical_fixture(.0005)
meta["physics_timestep"] = .00025
result = evaluate_trial(trace, meta, cfg)
check("solver每时长必须匹配明确声明的物理步长", not result["force_safety"]["coverage"]["intervals_match_declared_physics"] and
      not result["force_safety"]["force_protection_pass"])
trace, meta, cfg = fixture(duration=10.)
cfg["require_full_substeps"] = True
trace["substep_time"] = np.arange(2000) * .0005
trace["substep_duration"] = np.full(2000, .0005)
trace["substep_normal"] = np.full((2000, 5), 2.)
result = evaluate_trial(trace, meta, cfg)
check("独立审查10秒任务仅1秒子步反例已关闭", not result["force_safety"]["full_substeps"] and
      not result["force_safety"]["force_protection_pass"] and not result["task_success"])
trace, meta, cfg = physical_fixture(.0005, mixed=False)
trace["normal"][-1, 0] = 12.
result = evaluate_trial(trace, meta, cfg)
check("末端瞬时峰保留但不重复加正时长积分", result["force_safety"]["finger_peak"][0] == 12. and
      result["force_safety"]["terminal_endpoint_violation"] and
      result["force_safety"]["local_excess_integral"] == 0.)
trace, meta, cfg = fixture()
meta["reference"]["orientation_mode"] = "free"
trace["rotation"][:] = [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]
result = evaluate_trial(trace, meta, cfg)
check("无方向抓取任务不虚构初始朝向约束", result["first_stability_attained"] and
      np.all(evaluate_state(trace, meta, cfg)["orientation_error"] == 0.))
trace["angular_velocity"][:, 0] = .16
result = evaluate_trial(trace, meta, cfg)
check("free朝向仍要求完整三维角速度稳定", not result["first_stability_attained"])
meta["reference"]["orientation_mode"] = "full"
trace["angular_velocity"][:] = 0.
result = evaluate_trial(trace, meta, cfg)
check("明确full方向任务保持5度约束不随free放宽", not result["first_stability_attained"])
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
cfg["require_all_hand_release"] = True
trace["normal"][trace["time"] > 1.2 + 1e-10] = 0.
trace["all_hand_object"] = np.sum(trace["normal"], axis=1)
trace["all_hand_object"][trace["time"] > 1.2 + 1e-10] = .5
result = evaluate_trial(trace, meta, cfg)
check("目标物仍与其他handgeom接触不能记完全释放", not result["release_success"])
trace["all_hand_object"][trace["time"] > 1.2 + 1e-10] = 0.
trace["skin_normal"] = trace["normal"].copy()
trace["skin_normal"][trace["time"] >= 1.2, 2] = 1.
result = evaluate_trial(trace, meta, cfg)
check("已解除目标物但碰fixture不误判用户释放失败", result["release_success"] and
      result["release_contact_scope"] == "all hand-geometry contacts with target object")
del trace["all_hand_object"]
result = evaluate_trial(trace, meta, cfg)
check("正式allhand目标接触记录缺失不降级release合格", not result["release_success"])
trace, meta, cfg = fixture()
trace["normal"][trace["time"] >= 1.] = 0.
trace["palm"][trace["time"] >= 1.] = 2.
result = evaluate_trial(trace, meta, cfg)
check("无metadata失败时独立登记持续支持丧失", not result["task_success"] and
      abs(result["failure_at"] - 1.1) < 1e-9 and
      result["failure_reason"] == "support_contact_lost" and
      result["registered_failure_at"] is None and not result["static_window"]["eligible"])
trace, meta, cfg = fixture()
trace["normal"][(trace["time"] >= 1.) & (trace["time"] < 1.05)] = 0.
result = evaluate_trial(trace, meta, cfg)
check("仅50ms短暂丢接触恢复不登记任务支持失败", result["task_success"] and
      result["derived_support_failure_at"] is None)
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
trace["normal"][trace["time"] > 1.2 + 1e-10] = 0.
result = evaluate_trial(trace, meta, cfg)
check("主动释放后无支持属于任务动作不误判失败", result["task_success"] and
      result["release_success"] and result["derived_support_failure_at"] is None)
trace, meta, cfg = fixture()
trace["normal"][trace["time"] >= 1.] = 0.
meta.update(failure_at=2., failure_reason="gross_loss")
result = evaluate_trial(trace, meta, cfg)
check("独立支持失败与runner失败取最早时刻并导出原因", abs(result["failure_at"] - 1.1) < 1e-9 and
      result["registered_failure_at"] == 2. and result["failure_reason"] == "support_contact_lost")
trace, meta, cfg = fixture()
trace["desired"] = np.full_like(trace["normal"], 1000.)
trace["predicted_normal"] = np.full_like(trace["normal"], 1000.)
result = evaluate_trial(trace, meta, cfg)
check("方法语义不同desired不用于同时刻跨方法trackingRMSE", "force_tracking_rmse" not in result["static_window"]["metrics"] and
      result["static_window"]["metrics"]["mean_force"] == 10.)
trace, meta, cfg = fixture()
trace["motion_attained"] = np.ones(len(trace["time"]), dtype=bool)
trace["motion_attained"][trace["time"] >= 1.] = False
result = evaluate_trial(trace, meta, cfg)
check("已建立后连续300ms公共搬运目标未达成独立任务失败", not result["task_success"] and
      abs(result["failure_at"] - 1.3) < 1e-9 and result["failure_reason"] == "transport_reference_lost")
trace, meta, cfg = fixture()
trace["motion_attained"] = np.ones(len(trace["time"]), dtype=bool)
trace["motion_attained"][(trace["time"] >= 1.) & (trace["time"] < 1.1)] = False
result = evaluate_trial(trace, meta, cfg)
check("公共搬运目标100ms暂失后恢复不误任务失败", result["task_success"] and
      result["derived_transport_failure_at"] is None)
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
trace["normal"][trace["time"] > 1.2 + 1e-10] = 0.
trace["motion_attained"] = np.ones(len(trace["time"]), dtype=bool)
trace["motion_attained"][trace["time"] >= 1.2] = False
result = evaluate_trial(trace, meta, cfg)
check("释放指令后无需继续搬运不记公共路径失败", result["task_success"] and
      result["derived_transport_failure_at"] is None)
trace, meta, cfg = fixture()
cfg["static_window_sensitivity"] = [2.6, 3.4]
result = evaluate_trial(trace, meta, cfg)
check("较短固定敏感窗由共同配置给定而非方法终态挑选", result["static_window"]["start"] == 2.5 and
      result["static_window"]["end"] == 3.5 and
      result["static_window_sensitivity"]["start"] == 2.6 and
      result["static_window_sensitivity"]["end"] == 3.4 and
      result["static_window_sensitivity"]["metrics"]["samples"] == 161)
baseline = copy.deepcopy(result)
baseline.update(method="ratio", split="test")
trace["normal"][trace["time"] >= 3.8] = 0.
candidate = evaluate_trial(trace, meta, cfg)
candidate["split"] = "test"
window_estimate = paired_cluster_estimate([(candidate, baseline)], "static.finger_tv_rate", cohort="window")
completed_estimate = paired_cluster_estimate([(candidate, baseline)], "static.finger_tv_rate", cohort="completed_window")
check("全任务完成敏感队列不能保留窗口后失败任务", window_estimate["pairs"] == 1 and
      completed_estimate["pairs"] == 0 and not candidate["task_success"])
trace, meta, cfg = fixture()
cfg["static_window_sensitivity"] = [2.6, 3.4]
trace["normal"][trace["time"] > 3.4 + 1e-10] = 0.
candidate = evaluate_trial(trace, meta, cfg)
candidate["split"] = "test"
short_estimate = paired_cluster_estimate([(candidate, baseline)], "sensitivity.finger_tv_rate", cohort="sensitivity_window")
check("主窗与预声明短窗独立验收，短窗不替代主端点", not candidate["static_window"]["eligible"] and
      candidate["static_window_sensitivity"]["eligible"] and short_estimate["pairs"] == 1)
sensitivity_summary = analyze_records([candidate, baseline], bootstrap=5000)
sensitivity_rows = [r for r in sensitivity_summary["comparisons"] if r["role"] == "sensitivity"]
check("四个敏感性端点保全部预定分母且不进入主Holm", len(sensitivity_rows) == 4 and
      all(r["planned_pairs"] == 1 and "p_holm" not in r for r in sensitivity_rows))
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
cfg["require_all_hand_release"] = True
trace["normal"][trace["time"] >= 1.] = 0.
trace["all_hand_object"] = np.sum(trace["normal"], axis=1)
dropped_release = evaluate_trial(trace, meta, cfg)
check("指令前已掉落的无接触只算物理clearance不算释放", dropped_release["release_physical_clearance_success"] and
      not dropped_release["release_success"] and not dropped_release["release_exposed_support"] and
      dropped_release["release_outcome"] == "already_contact_free" and dropped_release["release_time"] is None)
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
cfg["require_all_hand_release"] = True
trace["normal"][trace["time"] > 1.2 + 1e-10] = 0.
trace["all_hand_object"] = np.sum(trace["normal"], axis=1)
held_release = evaluate_trial(trace, meta, cfg)
check("命令端点前仍支持持握然后解除才算可执行释放", held_release["release_success"] and
      held_release["release_exposed_support"] and held_release["release_outcome"] == "released" and
      abs(held_release["release_time"] - .205) < 1e-8)
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
cfg["require_all_hand_release"] = True
trace["normal"][(trace["time"] >= 1.) & (trace["time"] <= 1.2 + 1e-10)] = [0., 0., 2., 0., 0.]
trace["normal"][trace["time"] > 1.2 + 1e-10] = 0.
trace["all_hand_object"] = np.sum(trace["normal"], axis=1)
unsupported_release = evaluate_trial(trace, meta, cfg)
check("命令时仅残留目标接触但无有效支持不算释放暴露", unsupported_release["release_physical_clearance_success"] and
      not unsupported_release["release_success"] and unsupported_release["release_outcome"] == "unsupported_at_command")
trace, meta, cfg = fixture()
meta["scenario"]["release_at"] = 1.2
cfg["require_all_hand_release"] = True
trace["all_hand_object"] = np.sum(trace["normal"], axis=1)
deadline_release = evaluate_trial(trace, meta, cfg)
check("持握暴露但期限内不解除属于deadlinefailed", deadline_release["release_exposed_support"] and
      not deadline_release["release_physical_clearance_success"] and
      deadline_release["release_outcome"] == "deadline_failed")
del trace["all_hand_object"]
missing_release = evaluate_trial(trace, meta, cfg)
check("缺正式释放记录与无接触不同分类missingrecord", not missing_release["release_success"] and
      missing_release["release_outcome"] == "missing_record")
release_summary = summarize_method([dropped_release, held_release, deadline_release])
check("释放全部预定分母与暴露条件延迟分开不伪100percent", release_summary["release_trials"] == 3 and
      release_summary["release_successes"] == 1 and release_summary["release_exposed_trials"] == 2 and
      release_summary["release_physical_clearance_trials"] == 2 and
      abs(release_summary["release_time_conditional_mean"] - .205) < 1e-8)
out = ROOT / "review" / "evaluation_verification.json"
out.write_text(json.dumps(dict(passed=True, synthetic_only=True, checks=checks,
                               diagnostics=diagnostics, check_count=len(checks),
                               claims="metric implementation only; not controller efficacy"),
                          ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(passed=True, checks=len(checks), output=str(out)), ensure_ascii=False))
