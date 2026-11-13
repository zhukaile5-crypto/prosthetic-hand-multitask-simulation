"""Transparent readout of two new consumer-assumption diagnostics."""
from pathlib import Path
import json
import numpy as np
from verify_robot_servo_response_v1 import ROOT, Audit, read, sha, error_summary


def main():
    review = ROOT / "review"
    bridge_path = review / "robot_servo_runtime_bridge_v1.json"
    rolling_path = review / "robot_servo_rolling_prefix_v1.json"
    bridge = read(bridge_path); rolling = read(rolling_path); audit = Audit()
    audit.check("all15_120_mixed_3000calls_11160forecasts", len(rolling["contexts"]) == 15
        and sum(len(c["branches"]) for c in rolling["contexts"]) == 120
        and rolling["api_calls"] == 3000 and rolling["numerical_forecasts"] == 11160)
    audit.check("no_kernel_age_extension", rolling["maximum_kernel_age_s"] <= .22 + 1e-9)
    audit.check("rolling_prefix_slices_no_future_q", rolling["future_weight_violations"] == 0
        and all(1 <= p["observed_prefix_samples"] <= 25 and p["retained_start_s"] <= p["original_origin_s"]
            and p["actual_target_time_s"] <= .5 + 1e-9 for c in rolling["contexts"] for b in c["branches"] for p in b["predictions"]))
    split = {str(h): {part: dict(rolling=[], full=[], prediction_difference=[]) for part in ("before_reset", "after_reset")} for h in (.02, .04, .06, .1)}
    context_splits = []
    for context in rolling["contexts"]:
        directory = ROOT / "output/robot_servo_response_all15_v2" / context["context_id"]
        local = {h: {part: dict(rolling=[], full=[], prediction_difference=[]) for part in ("before_reset", "after_reset")} for h in split}
        with np.load(directory / "robot_response_raw.npz", allow_pickle=False) as original:
            for branch in context["branches"]:
                for p in branch["predictions"]:
                    h = str(p["horizon_s"]); k = round(p["original_origin_s"] / .02)
                    full_error = original[branch["branch"] + "_h" + str(round(p["horizon_s"] * 1000)) + "_error25"][k]
                    rolling_error = np.array(p["error25"])
                    part = "after_reset" if p["retained_start_s"] > 1e-9 else "before_reset"
                    difference = full_error - rolling_error
                    for target in (split[h][part], local[h][part]):
                        target["rolling"].append(rolling_error); target["full"].append(full_error); target["prediction_difference"].append(difference)
        context_splits.append(dict(context_id=context["context_id"], by_horizon={h: {
            part: {kind: error_summary(values) for kind, values in kinds.items()} for part, kinds in parts.items()} for h, parts in local.items()}))
    summaries = {h: {part: {kind: error_summary(values) for kind, values in kinds.items()} for part, kinds in parts.items()} for h, parts in split.items()}
    audit.check("before_reset_matches_full_history", all(max(summaries[h]["before_reset"]["prediction_difference"]["max_abs"]) < 1e-10 for h in summaries))
    result = dict(schema="robot_servo_consumer_bridge_readout_v1", diagnostic_contract_passed=all(c["passed"] for c in audit.checks), checks=audit.checks,
        input_sha256={str(p): sha(p) for p in (bridge_path, rolling_path)}, fullhistory_rolling_origin_splits=summaries,
        context_splits=context_splits, source_sha256=sha(__file__), new_physics=False, model_admitted=False,
        scope="predeclared.12s reset vs full-history on the same120 existing mixed sequences; before/after split describes whether history was actually dropped, not a selected performance window")
    path = review / "robot_servo_consumer_bridge_readout_v1.json"
    with path.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# 伺服模型接入：实际斜坡与滚动前缀诊断", "",
        "这两项是新的消费者假设核查，未改变冻结辨识器、API、原模型或旧结果，也未运行物理仿真。", "",
        "## 真实运行参考的斜坡形状", "",
        "8个V9 ratio_preview流取7.5–12.0秒公开观测，901点/225个20ms段。观测alpha等于同刻alpha_before；实际增量等于下一5ms执行记录。不能把动作后的alpha错配到动作前时刻。数值阈值仅用于描述偏差量级，JSON保留1e−12、1e−10、1e−9三种计数，不是控制或物理门限。", "",
        "| 任务/环境 | TH/FF/MF/RF/LF非直线段数（>1e−10） | 最大alpha偏差 | 相对训练micro .004 |", "|---|---|---:|---:|"]
    for r in bridge["records"]:
        maxdev = max(r["interval_deviation_max8"][:5]); axis = int(np.argmax(r["interval_deviation_max8"][:5]))
        amplitude = bridge["actual_training_micro_amplitude8"][axis]["max"]
        lines.append(f"| {r['task']}/{r['environment']} | {'/'.join(map(str,r['segment_counts_by_numeric_diagnostic_threshold']['1e-10'][:5]))} | {maxdev:.9f} | {maxdev/amplitude:.2%} |")
    lines += ["", "motion3在transport流的直线残差最多1.87e−16，处于浮点量级。手指最大偏差为真实5ms动作形状：nominal TH约.0007832，hold collision RF约.0011106。八流requested_action与实际delta alpha差异均约1.11e−16，所以主要非线性已经包含在逐tick反馈请求中，不能统一归因于plant再次裁剪。最坏TH例7.52–7.54s的四步增量为.00225、.0021875、.001640625、.001230469；抽取20ms端点会掩掉其形状。JSON另列逐段边界触碰与请求/兑现差，未把所有非线性都称为同一种裁剪。", "",
        "## 固定0.12秒滚动前缀", "",
        "所有15个context、120条原mixed留出均调用冻结API。每个原预测起点保留max(0,origin−.12)以后的真实5ms q及20ms边界参考并重置时间原点；未来参考仍为已声明任务路径。3000次API调用、11160个预测，最大新核龄.22s；没有传未来q/zero，没有复用.5s末端去外推到更晚时间。", "",
        "原点不超过.12秒时没有删除历史，此部分与full-history在数值精度内一致。下面只列确实删除历史的原点，用于诊断截断影响；完整原评分窗与所有25轴误差仍保留在JSON，未替换或挑选原指标。", "",
        "| 时域 | reset后样本数 | 全历史/滚动 最大角误差 mrad | 全历史/滚动 最大lift误差 mm | 两预测最大差：角 mrad / lift mm |", "|---|---:|---|---|---|"]
    for h, parts in summaries.items():
        s = parts["after_reset"]; f = s["full"]["max_abs"]; r = s["rolling"]["max_abs"]; d = s["prediction_difference"]["max_abs"]
        lines.append(f"| {float(h)*1000:.0f}ms | {s['rolling']['samples']} | {max(f[:24])*1000:.4f} / {max(r[:24])*1000:.4f} | {f[24]*1000:.4f} / {r[24]*1000:.4f} | {max(d[:24])*1000:.4f} / {d[24]*1000:.4f} |")
    lines += ["", "滚动重置在这些已见短路径上的误差可直接报告，但不构成任意长期滚动的准入。它删除旧命令记忆，并把原context核重新用于较晚的机器人状态，仍是新经验假设；跨状态/参考、方向和组合输入支持必须由消费者标注。真实baseline手指路径不是校准四步直线，因而本项mixed滚动误差也不能自动转移成baseline路径的预测误差。", "",
        "## 接入含义", "", "可继续实施明确标注experimental的局部nearest-anchor/directional提案。应保留实际5ms参考历史和bridge形状偏差、独立记录模型可用性与作用域，并用机器人exact FK复核有限候选几何。不能把20ms端点采样、最近anchor或分量box当作原已验证的任意组合、长期记忆或材料压力安全证明。", "",
        "公开状态到15anchors的补充坐标距离见review/v9_servo_anchor_proximity_independent_v1.json：角姿态、lift、alpha及速度分别报告，没有混合单位总分。不同坐标的最近anchor不同，不能拼成一条共同已准入状态。", "",
        "证据：review/robot_servo_runtime_bridge_v1.json、review/robot_servo_rolling_prefix_v1.json、review/robot_servo_consumer_bridge_readout_v1.json。原API和identify哈希均在运行前后固定核验，所有检查无physics。"]
    md = review / "robot_servo_consumer_bridge_readout_v1.md"
    with md.open("x", encoding="utf-8") as stream:stream.write("\n".join(lines) + "\n")
    print(json.dumps(dict(diagnostic_contract_passed=result["diagnostic_contract_passed"], checks=len(audit.checks), json_sha256=sha(path), md_sha256=sha(md), after_reset={h:{"n":s["after_reset"]["rolling"]["samples"],"max_prediction_difference_angle":max(s["after_reset"]["prediction_difference"]["max_abs"][:24]),"max_prediction_difference_lift":s["after_reset"]["prediction_difference"]["max_abs"][24]} for h,s in summaries.items()}),ensure_ascii=False))
    if not result["diagnostic_contract_passed"]:raise SystemExit(1)


if __name__ == "__main__":main()
