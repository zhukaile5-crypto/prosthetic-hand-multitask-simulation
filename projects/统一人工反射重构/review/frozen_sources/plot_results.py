"""Publication-exportable five-axis and paired tradeoff plots, with raw units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from analyze import METHOD_LABELS, load_records, analyze_records


ORDER = ("fixed", "ratio", "friction_ratio", "independent", "unified",
         "unified_no_continuity", "unified_no_geometry", "unified_fixed_stage")
COLORS = {"fixed": "#929aab", "ratio": "#375f87", "friction_ratio": "#375f87",
          "independent": "#b78b38", "unified": "#168275", "unified_no_continuity": "#8f68a8",
          "unified_no_geometry": "#b56965", "unified_fixed_stage": "#849438"}


def configure_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
                 if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.fonttype": "none", "figure.facecolor": "white"})


def _save(figure, out, name):
    figure.savefig(out / (name + ".png"), dpi=200, bbox_inches="tight")
    figure.savefig(out / (name + ".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_overview(result, out):
    splits = [s for s in ("test", "robustness", "development", "pilot") if s in result["summaries"]]
    splits += [s for s in result["splits"] if s not in splits]
    for split in splits:
        data = result["summaries"][split].get("main", {})
        if not data:
            continue
        methods = [m for m in ORDER if m in data] + [m for m in data if m not in ORDER]
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
        specifications = [
            ("completion_rate", "任务完成与建立后保持 ↑", "全部任务完成率 (%)", 100),
            ("force_protection_rate", "仪器覆盖指部限力 ↑", "完整原始物理子步均合格 (%)", 100),
            ("A_S", "首次稳定动态品质 ↓", "首次稳定受限等待 A_S (s)", 1),
            ("finger_tv_eligible_mean", "持握平稳 ↓", "固定带宽逐指 TV (N/s)；各自有效窗", 1),
            ("mean_force_eligible_mean", "施力适度 ↓", "平均法向握力 (N)；各自有效窗", 1),
            ("release_completion_rate", "意图与覆盖 ↑", "释放完成 / 固定窗口覆盖 (%)", 100),
        ]
        for axis, (key, title, ylabel, scale) in zip(axes.flat, specifications):
            x = np.arange(len(methods))
            values = [data[m].get(key) for m in methods]
            values = [v * scale if v is not None else np.nan for v in values]
            axis.bar(x, values, color=[COLORS.get(m, "#707070") for m in methods], width=.72)
            if key == "release_completion_rate":
                axis.scatter(x, [data[m]["static_coverage_rate"] * 100 for m in methods],
                             color="#111111", marker="D", s=22, label="静态窗覆盖")
                axis.legend(frameon=False, fontsize=8, loc="lower right")
            axis.set_title(title, loc="left", fontweight="bold")
            axis.set_ylabel(ylabel)
            axis.set_xticks(x, [METHOD_LABELS.get(m, m) for m in methods], rotation=35, ha="right")
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            if scale == 100:
                axis.set_ylim(0, 106)
            for i, value in enumerate(values):
                if np.isfinite(value):
                    axis.text(i, value, f"{value:.1f}" if scale == 100 else f"{value:.2f}",
                              ha="center", va="bottom", fontsize=7)
                else:
                    label = "无释放任务" if key == "release_completion_rate" else "无有效窗" if key.startswith("finger_tv") or key.startswith("mean_force") else "不可估计"
                    axis.text(i, .04, label, transform=axis.get_xaxis_transform(),
                              ha="center", va="bottom", fontsize=6, rotation=90)
            if key in ("finger_tv_eligible_mean", "mean_force_eligible_mean"):
                denominator = ", ".join(str(data[m]["static_eligible"]) for m in methods)
                axis.text(.01, .98, "有效窗 n=" + denominator, transform=axis.transAxes,
                          va="top", fontsize=7)
        figure.suptitle(f"统一人工反射：五轴综合评价（{split}，能力边界另列）", fontsize=16, fontweight="bold")
        figure.text(.015, .005, "TV按50 Hz/50 ms离线抗混叠后统一200 Hz计量；开发fallback另见合同。峰值/超限不滤。各自有效窗描述，正式差异用共同队列。", fontsize=8)
        figure.tight_layout(rect=(0, .04, 1, .94))
        _save(figure, out, "全景五轴_" + split)


def plot_paired(result, out):
    rows = [r for r in result["comparisons"] if r["method"] == result["candidate"] and
            r["domain"] == "main" and r["role"] in ("primary", "gate", "secondary")]
    specifications = [
        ("restricted_unstable_time", "首次稳定受限等待差异 (s) ↓"),
        ("static.finger_tv_rate", "固定带宽实际逐指涨落差异 (N/s) ↓"),
        ("static.mean_force", "平均握力差异 (N) ↓"),
        ("task_success", "任务完成差异 (百分点) ↑"),
        ("safety.force_protection_pass", "全子步力合格差异 (百分点) ↑"),
        ("static_coverage", "固定窗覆盖差异 (百分点) ↑"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 6.7))
    for axis, (endpoint, title) in zip(axes.flat, specifications):
        selected = [r for r in rows if r["endpoint"] == endpoint]
        for i, row in enumerate(selected):
            scale = 100 if row["unit"] == "fraction" else 1
            if row["difference"] is None:
                axis.text(.02, i, f"{row['split']}：无有效配对", transform=axis.get_yaxis_transform())
                continue
            point, low, high = [row[k] * scale for k in ("difference", "ci_low", "ci_high")]
            axis.errorbar(point, i, xerr=[[max(0, point - low)], [max(0, high - point)]],
                          fmt="o", color="#168275", capsize=4, linewidth=2)
            adjusted = row.get("p_holm")
            text = f"n={row['pairs']}/{row['planned_pairs']}, 簇={row['clusters']}"
            if adjusted is not None:
                text += f", Holm p={adjusted:.3f}"
            axis.text(.02, .05 + i * .09, text, transform=axis.transAxes, fontsize=7)
        axis.axvline(0, color="#999999", linestyle="--", linewidth=1)
        axis.set_yticks(range(len(selected)), [r["split"] for r in selected])
        axis.set_ylim(-.6, max(.6, len(selected) - .4))
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("完整方案 − 强基线：快、稳、轻与门槛的配对权衡", fontsize=15, fontweight="bold")
    figure.text(.02, .01, "点为平均差值，线为完整种子簇 bootstrap 95%区间；预定工程界限与各项门槛判断见结果表。", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .94))
    _save(figure, out, "响应与力涨落权衡")


def plot_failure_coverage(result, out):
    splits = [s for s in ("test", "robustness") if s in result["scenarios"]]
    if not splits:
        splits = result["splits"][:2]
    figure, axes = plt.subplots(len(splits), 2, figsize=(13, max(5.5, 5.3 * len(splits))), squeeze=False)
    for row_index, split in enumerate(splits):
        scenarios = result["scenarios"][split]
        names = list(scenarios)
        methods = [m for m in ORDER if m in result["methods"]] + [m for m in result["methods"] if m not in ORDER]
        for axis, metric, title in [(axes[row_index, 0], "completion_rate", "完整任务失败率"),
                                    (axes[row_index, 1], "static_coverage_rate", "固定静态窗口无效比例")]:
            matrix = np.array([[1 - scenarios[s][m][metric] if m in scenarios[s] else np.nan
                                for m in methods] for s in names]) * 100
            image = axis.imshow(matrix, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
            axis.set_xticks(range(len(methods)), [METHOD_LABELS.get(m, m) for m in methods], rotation=30, ha="right")
            axis.set_yticks(range(len(names)), [next(iter(scenarios[s].values())).get("title", s) +
                           (" [能力边界]" if next(iter(scenarios[s].values()))["boundary"] else "") for s in names])
            axis.set_title(split + " / " + title, loc="left", fontweight="bold")
            for i in range(len(names)):
                for j in range(len(methods)):
                    if np.isfinite(matrix[i, j]):
                        axis.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                                  color="white" if matrix[i, j] > 60 else "black", fontsize=7)
            figure.colorbar(image, ax=axis, fraction=.035, pad=.02, label="%（越低越好）")
    figure.suptitle("全部预定场景：失败和有效覆盖一起报告", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, .96))
    _save(figure, out, "失败与覆盖")


def plot_results(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_overview(result, out)
    plot_paired(result, out)
    plot_failure_coverage(result, out)
    return sorted(str(p) for p in out.glob("*.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--out")
    parser.add_argument("--baseline", default="ratio")
    parser.add_argument("--candidate", default="unified")
    args = parser.parse_args()
    source = Path(args.source)
    result = (json.loads(source.read_text(encoding="utf-8")) if source.name == "summary.json"
              else analyze_records(load_records(source), args.baseline, args.candidate))
    out = args.out or str((source if source.is_dir() else source.parent) / "figures")
    paths = plot_results(result, out)
    print(json.dumps(dict(figures=paths), ensure_ascii=False))


if __name__ == "__main__":
    main()
