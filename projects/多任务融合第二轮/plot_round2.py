"""Scientific figures from validated second-round analysis (PNG and SVG).

Usage: python plot_round2.py --analysis output/final_analysis --out output/figures
No experiment or parameter selection is performed here. The mandatory figures
use only the two analysis JSON files. The optional distribution figure reads
every original trial via the analysis provenance and verifies its SHA256.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = ("event", "ratio", "fusion", "coordinated",
           "coordinated_no_guard", "coordinated_no_redistribution")
COMPARATORS = ("event", "ratio", "fusion")
SUITES = ("test", "robustness")
SEEDS = {"test": tuple(range(8001, 8021)),
         "robustness": tuple(range(9001, 9011))}
LABELS = {
    "event": "事件规则", "ratio": "摩擦比基线", "fusion": "第一轮融合",
    "coordinated": "事件触发协调", "coordinated_no_guard": "协调去除力保护",
    "coordinated_no_redistribution": "协调去除接触重分配",
}
SUITE_LABELS = {"test": "正式留出", "robustness": "扰动留出"}
COLORS = {"event": "#667B8D", "ratio": "#31708F", "fusion": "#8B7694",
          "coordinated": "#BB6942", "coordinated_no_guard": "#A99477",
          "coordinated_no_redistribution": "#82918B"}
PHYSICS_METRICS = ("physics_peak_force_N", "physics_peak_penetration_mm")
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取有效JSON：{path}: {exc}") from exc


def finite_number(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and np.isfinite(value), f"缺失或非有限数值：{label}")
    return float(value)


def check_count(row, label, expected_n):
    require(isinstance(row, dict), f"无效汇总行：{label}")
    n = row.get("n")
    require(type(n) is int and n == expected_n and n > 0,
            f"空数据或样本数不符：{label}, n={n}, 预期={expected_n}")
    require(isinstance(row.get("counts"), dict), f"缺少计数：{label}")
    count = row["counts"].get("safe_completed")
    require(type(count) is int and 0 <= count <= n, f"无效合格完成数：{label}")
    rate = finite_number(row.get("rates_pct", {}).get("safe_completed"), label)
    require(np.isclose(rate, 100 * count / n, atol=1e-8, rtol=0),
            f"完成率与计数不一致：{label}")
    require(row.get("n_seed_clusters") == len(SEEDS[row["suite"]]),
            f"种子簇数不符：{label}")
    return count, n, rate


def load_analysis(directory):
    directory = Path(directory).resolve()
    input_paths = [directory / "summary.json", directory / "paired_effects.json"]
    hashes = {str(path): sha256(path) for path in input_paths if path.is_file()}
    require(len(hashes) == 2, "分析目录必须包含非空 summary.json 和 paired_effects.json")
    summary, paired = (read_json(path) for path in input_paths)
    require(isinstance(summary, dict) and isinstance(paired, dict), "分析JSON必须为对象")
    meta = summary.get("metadata")
    require(isinstance(meta, dict) and meta.get("analysis_version") == 2,
            "只接受第二轮 analysis_version=2；第一轮结果不可用于本图")
    require(paired.get("metadata") == meta, "两个分析文件的元数据不一致")
    require(isinstance(meta.get("methods"), list)
            and len(meta["methods"]) == len(METHODS)
            and set(meta["methods"]) == set(METHODS),
            "必须为第二轮声明的6个方法，不能混入第一轮8方法结果")
    require("coordinated" in meta.get("primary_hypothesis_family", ""),
            "主比较必须声明事件触发协调版本")
    definitions = meta.get("scenario_definitions")
    require(isinstance(definitions, list) and len(definitions) == 16,
            "第二轮必须完整保留16个场景")
    require(all(isinstance(item, dict) and type(item.get("boundary")) is bool
                and isinstance(item.get("name"), str) and item.get("title")
                for item in definitions), "无效场景定义")
    names = [item["name"] for item in definitions]
    require(len(set(names)) == 16 and sum(item["boundary"] for item in definitions) == 1,
            "场景重复或预定边界数量不符")
    main_names = [item["name"] for item in definitions if not item["boundary"]]
    definition_map = {item["name"]: item for item in definitions}
    domains, scenarios, effects = (summary.get("domains"), summary.get("scenarios"), paired.get("effects"))
    require(isinstance(domains, list) and domains and isinstance(scenarios, list)
            and scenarios and isinstance(effects, list) and effects,
            "分析数据为空，不能生成科研图")
    main_rows, scenario_rows, effect_rows = {}, {}, {}
    for row in domains:
        if row.get("domain") == "main":
            key = row.get("suite"), row.get("method")
            require(key[0] in SUITES and key[1] in METHODS and key not in main_rows,
                    f"重复或无效主域汇总：{key}")
            check_count(row, str(key), len(main_names) * len(SEEDS[key[0]]))
            require(row.get("scenarios") == main_names, f"主域场景顺序/内容不符：{key}")
            main_rows[key] = row
    require(set(main_rows) == {(suite, method) for suite in SUITES for method in METHODS},
            "主域缺少某个留出集或方法")
    for row in scenarios:
        key = row.get("suite"), row.get("scenario"), row.get("method")
        require(key[0] in SUITES and key[1] in definition_map and key[2] in METHODS
                and key not in scenario_rows, f"重复或无效场景汇总：{key}")
        check_count(row, str(key), len(SEEDS[key[0]]))
        require(row.get("domain") == ("boundary" if definition_map[key[1]]["boundary"] else "main"),
                f"场景边界标签不符：{key}")
        scenario_rows[key] = row
    require(set(scenario_rows) == {(suite, name, method) for suite in SUITES
                                  for name in names for method in METHODS},
            "未包含完整16×6×2场景汇总")
    for suite in SUITES:
        for method in METHODS:
            scenario_sum = sum(scenario_rows[(suite, name, method)]["counts"]["safe_completed"]
                               for name in main_names)
            require(scenario_sum == main_rows[(suite, method)]["counts"]["safe_completed"],
                    f"主域与场景完成数矛盾：{suite}/{method}")
    for row in effects:
        if (row.get("domain") == "main" and row.get("metric") == "safe_completed"
                and row.get("condition") == "all" and row.get("comparator") in COMPARATORS):
            key = row.get("suite"), row.get("comparator")
            require(key[0] in SUITES and key not in effect_rows
                    and row.get("method") == "coordinated", f"无效主域配对比较：{key}")
            require(row.get("unit") == "percentage_points", f"配对效应单位不符：{key}")
            expected_seed_keys = {str(seed) for seed in SEEDS[key[0]]}
            require(set(row.get("seed_difference_sums", {})) == expected_seed_keys
                    and set(row.get("seed_eligible_counts", {})) == expected_seed_keys,
                    f"配对效应种子不是第二轮预定新种子：{key}")
            require(row.get("n_pairs") == row.get("n_total_pairs")
                    == len(main_names) * len(SEEDS[key[0]]), f"主指标缺失配对：{key}")
            require(all(value == len(main_names) for value in row["seed_eligible_counts"].values()),
                    f"主指标种子簇中缺少场景：{key}")
            estimate = finite_number(row.get("estimate"), str(key))
            ci = row.get("ci95")
            require(isinstance(ci, list) and len(ci) == 2, f"缺少95%区间：{key}")
            low, high = (finite_number(value, str(key)) for value in ci)
            require(low <= high and -100 <= low <= 100 and -100 <= high <= 100,
                    f"无效95%区间：{key}")
            difference = (main_rows[(key[0], "coordinated")]["rates_pct"]["safe_completed"]
                          - main_rows[key]["rates_pct"]["safe_completed"])
            require(np.isclose(estimate, difference, atol=1e-8, rtol=0),
                    f"配对效应与主域计数不一致：{key}")
            if key[1] in ("event", "ratio"):
                p = finite_number(row.get("p_value_holm"), str(key))
                require(0 <= p <= 1 and row.get("holm_family")
                        == f"{key[0]}.main.safe_completed.two_baselines",
                        f"缺少两基线Holm校正：{key}")
            else:
                require(row.get("p_value_holm") is None, "旧融合比较属于探索性，不能显示为主Holm比较")
            effect_rows[key] = row
    require(set(effect_rows) == {(suite, method) for suite in SUITES for method in COMPARATORS},
            "缺少协调版本对事件、摩擦比或旧融合的配对效应")
    return {"directory": directory, "metadata": meta, "definitions": definitions,
            "main_names": main_names, "main_rows": main_rows,
            "scenario_rows": scenario_rows, "effect_rows": effect_rows,
            "input_hashes": hashes}


def configure_style():
    require(FONT_PATH.is_file(), f"缺少微软雅黑字体：{FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10.5,
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#26323D", "text.color": "#26323D",
                         "xtick.color": "#26323D", "ytick.color": "#26323D"})
    return family


def save_figure(fig, out, name, scope, records):
    fig.set_facecolor("white")
    for extension in ("png", "svg"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        records.append({"path": str(path), "sha256": sha256(path),
                        "bytes": path.stat().st_size, "format": extension,
                        "figure": name, "scope": scope})
    plt.close(fig)


def completion_bars(data, out, records):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    for ax, suite in zip(axes, SUITES):
        rows = [data["main_rows"][(suite, method)] for method in METHODS]
        rates = [row["rates_pct"]["safe_completed"] for row in rows]
        ax.bar(np.arange(6), rates, color=[COLORS[method] for method in METHODS], width=.68)
        for index, (rate, row) in enumerate(zip(rates, rows)):
            ax.text(index, rate + 1.6, f"{rate:.1f}%\n{row['counts']['safe_completed']}/{row['n']}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(np.arange(6), [LABELS[method] for method in METHODS], rotation=25, ha="right")
        ax.set_ylim(0, 113)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DEE3E7", linewidth=.7)
        ax.set_title(f"{SUITE_LABELS[suite]} · 15个非边界情景\n"
                     f"每方法{rows[0]['n']}次，{len(SEEDS[suite])}个配对种子簇", loc="left", fontsize=12)
    axes[0].set_ylabel("全程限力合格任务完成率（%）")
    fig.suptitle("第二轮：任务完成与五指限力同时合格", x=.08, ha="left", fontsize=16)
    fig.text(.08, .02, "包括建立抓握阶段；连续30 ms违规判据。短时力峰、穿透与真实材料损伤须另行评价。", fontsize=10)
    fig.subplots_adjust(left=.08, right=.985, bottom=.28, top=.81, wspace=.08)
    save_figure(fig, out, "completion_main", {
        "suites": list(SUITES), "domain": "main", "scenarios": data["main_names"],
        "methods": list(METHODS), "metric": "safe_completed",
        "error_bars": "none; paired effect intervals are in paired_effect_main"}, records)


def format_p(value):
    return f"{value:.3g}" if value >= .0001 else f"{value:.2e}"


def effect_forest(data, out, records):
    selected = [data["effect_rows"][(suite, comparator)]
                for suite in SUITES for comparator in COMPARATORS]
    lows = [row["ci95"][0] for row in selected] + [0]
    highs = [row["ci95"][1] for row in selected] + [0]
    left, right = min(lows), max(highs)
    padding = max(.5, .15 * (right - left))
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), sharex=True)
    for ax, suite in zip(axes, SUITES):
        ax.axvline(0, color="#6F7880", linewidth=1, linestyle="--", zorder=0)
        for y, comparator in zip((2, 1, 0), COMPARATORS):
            row = data["effect_rows"][(suite, comparator)]
            estimate, (low, high) = row["estimate"], row["ci95"]
            # Drawing the interval independently also permits percentile CIs
            # whose finite-sample endpoint does not surround the point estimate.
            ax.hlines(y, low, high, color="#31708F", linewidth=2.3)
            ax.vlines((low, high), y - .07, y + .07, color="#31708F", linewidth=1.3)
            ax.scatter(estimate, y, color="#31708F", s=47, zorder=3)
            ax.text(1.04, y, f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=10.5)
            p_text = format_p(row["p_value_holm"]) if comparator != "fusion" else "—（探索性）"
            ax.text(1.62, y, p_text, transform=ax.get_yaxis_transform(), va="center", fontsize=10.5)
        ax.text(1.04, 2.58, "差值及95%区间（百分点）", transform=ax.get_yaxis_transform(), fontsize=10.5)
        ax.text(1.62, 2.58, "Holm校正 p", transform=ax.get_yaxis_transform(), fontsize=10.5)
        ax.set_yticks((2, 1, 0), [f"协调 − {LABELS[method]}" for method in COMPARATORS])
        ax.set_ylim(-.55, 2.9)
        ax.set_xlim(left - padding, right + padding)
        ax.grid(axis="x", color="#E3E7EA", linewidth=.65)
        ax.set_title(f"{SUITE_LABELS[suite]} · 15个非边界情景", loc="left", fontsize=12)
    axes[1].set_xlabel("限力合格完成率的配对差值（百分点；正值表示协调更高）")
    fig.suptitle("第二轮：相对基线与第一轮融合的配对效应", x=.19, ha="left", fontsize=16)
    fig.text(.19, .025, "95%区间为10000次配对种子簇重采样，未作多重校正。事件/摩擦比两项比较在各留出集分别Holm校正。\n"
             "旧融合比较为探索性；没有标注显著性星号，也不根据点估计是否为正选择比较。", fontsize=10)
    fig.subplots_adjust(left=.19, right=.56, bottom=.15, top=.88, hspace=.42)
    save_figure(fig, out, "paired_effect_main", {
        "suites": list(SUITES), "domain": "main", "scenarios": data["main_names"],
        "method": "coordinated", "comparators": list(COMPARATORS),
        "metric": "safe_completed", "condition": "all", "units": "percentage_points",
        "ci": "marginal paired seed-cluster 95% bootstrap interval",
        "p_adjustment": "separate two-baseline Holm families in test and robustness; fusion exploratory"}, records)


def scenario_heatmaps(data, out, records):
    definitions = data["definitions"]
    for suite in SUITES:
        matrix = np.asarray([[data["scenario_rows"][(suite, item["name"], method)]
                              ["counts"]["safe_completed"] for method in METHODS]
                             for item in definitions])
        n = len(SEEDS[suite])
        fig, ax = plt.subplots(figsize=(12.8, 10.5))
        image = ax.imshow(matrix / n * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(6), [LABELS[method] for method in METHODS], rotation=18, ha="right")
        ax.set_yticks(np.arange(16), [item["title"] + ("（预定边界）" if item["boundary"] else "")
                                     for item in definitions])
        for i in range(16):
            for j in range(6):
                ax.text(j, i, f"{matrix[i, j]}/{n}", ha="center", va="center", fontsize=10.5,
                        color="white" if matrix[i, j] / n >= .6 else "#23313B")
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i, item in enumerate(definitions):
            if item["boundary"]:
                # A dotted enclosure marks scope without altering cell values.
                ax.axhline(i - .5, color="#756754", linestyle="--", linewidth=1.2)
                ax.axhline(i + .5, color="#756754", linestyle="--", linewidth=1.2)
        fig.colorbar(image, ax=ax, fraction=.028, pad=.035, label="限力合格完成率（%）")
        ax.set_title(f"{SUITE_LABELS[suite]}：16个情景的合格完成数", loc="left", fontsize=16, pad=18)
        fig.text(.29, .02, f"每格{n}次配对试验。边界保留展示，但不进入15情景主域比较；失败情景未删除。\n"
                 "颜色统一使用0–100%，不同留出集可直接对照；计数不代表独立人体样本。", fontsize=10)
        fig.subplots_adjust(left=.29, right=.91, bottom=.15, top=.92)
        save_figure(fig, out, f"scenario_completion_{suite}", {
            "suite": suite, "domain": "all", "scenarios": [item["name"] for item in definitions],
            "methods": list(METHODS), "metric": "safe_completed", "count_denominator": n,
            "color_scale_pct": [0, 100], "boundary": "included, visually marked, excluded from main comparison"}, records)


def original_trials(data):
    provenance = data["metadata"].get("inputs")
    require(isinstance(provenance, dict) and all(suite in provenance for suite in SUITES),
            "分析结果缺少原始试验来源")
    paths = {suite: Path(provenance[suite]["directory"]) / "trials.json" for suite in SUITES}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        return None, {}, "未生成物理极值分布图：分析来源的原始 trials.json 不在本机：" + "; ".join(absent)
    trial_sets, raw_hashes = {}, {}
    names = [item["name"] for item in data["definitions"]]
    for suite, path in paths.items():
        actual_hash = sha256(path)
        require(actual_hash == provenance[suite].get("trials_sha256"), f"原始试验哈希与分析来源不符：{path}")
        trials = read_json(path)
        require(isinstance(trials, list) and trials, f"原始试验为空：{path}")
        expected = {(name, method, seed) for name in names for method in METHODS for seed in SEEDS[suite]}
        indexed = {}
        for row in trials:
            key = row.get("scenario"), row.get("method"), row.get("seed")
            require(key in expected and key not in indexed, f"原始试验重复或非预定记录：{suite}/{key}")
            for metric in PHYSICS_METRICS:
                value = finite_number(row.get(metric), f"{suite}/{key}/{metric}")
                require(value >= 0, f"物理极值为负：{suite}/{key}/{metric}")
            indexed[key] = row
        require(set(indexed) == expected, f"原始试验缺少预定组合：{suite}")
        trial_sets[suite] = trials
        raw_hashes[str(path.resolve())] = actual_hash
    return trial_sets, raw_hashes, None


def physics_distributions(data, trial_sets, out, records):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.3))
    rng = np.random.default_rng(20261013)
    thresholds = {"physics_peak_force_N": .1, "physics_peak_penetration_mm": .05}
    metric_titles = {"physics_peak_force_N": "物理子步最大指力（N）",
                     "physics_peak_penetration_mm": "物理子步最大手物穿透（mm）"}
    for i, suite in enumerate(SUITES):
        for j, metric in enumerate(PHYSICS_METRICS):
            ax = axes[i, j]
            values = [np.asarray([row[metric] for row in trial_sets[suite] if row["method"] == method], float)
                      for method in METHODS]
            boxes = ax.boxplot(values, positions=np.arange(6), widths=.50, whis=(0, 100),
                               showfliers=False, patch_artist=True,
                               medianprops={"color": "#25333D", "linewidth": 1.4},
                               whiskerprops={"color": "#6F7880"}, capprops={"color": "#6F7880"})
            for method, patch in zip(METHODS, boxes["boxes"]):
                patch.set_facecolor(COLORS[method])
                patch.set_alpha(.23)
            for x, (method, samples) in enumerate(zip(METHODS, values)):
                jitter = rng.uniform(-.19, .19, len(samples))
                ax.scatter(x + jitter, samples, s=9, alpha=.43, color=COLORS[method],
                           edgecolors="none", rasterized=False, zorder=2)
            ax.set_yscale("symlog", linthresh=thresholds[metric])
            maximum = max(float(np.max(samples)) for samples in values)
            ax.set_ylim(0, max(maximum * 1.4, thresholds[metric] * 2))
            ax.set_xticks(np.arange(6), [LABELS[method] for method in METHODS], rotation=23, ha="right")
            ax.set_ylabel(metric_titles[metric])
            ax.set_title(f"{SUITE_LABELS[suite]} · 全16情景，每方法{len(values[0])}次", loc="left", fontsize=12)
            ax.grid(axis="y", which="both", color="#E3E7EA", linewidth=.6)
            ax.set_axisbelow(True)
    fig.suptitle("第二轮：保留全部试验的物理子步极值", x=.075, ha="left", fontsize=16)
    fig.text(.075, .02, "每点为一次完整试验；包含初始失败、掉落与极端边界，没有删去高值。箱须覆盖最小至最大值。\n"
             "纵轴为symlog：力在0.1 N内、穿透在0.05 mm内为线性，其外为对数；零值仍展示。点的横向抖动仅改善重叠。\n"
             "这是接触模型与数值可信度诊断，不是真实材料损伤概率；低力也可能来自失去接触。", fontsize=10)
    fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.88, hspace=.68, wspace=.19)
    save_figure(fig, out, "physics_extremes", {
        "suites": list(SUITES), "domain": "all", "methods": list(METHODS),
        "scenarios": [item["name"] for item in data["definitions"]],
        "metrics": list(PHYSICS_METRICS), "selection": "all trials, including failures and boundary",
        "axis": "symlog", "linear_thresholds": thresholds,
        "whiskers": "minimum and maximum; all trial values are also plotted",
        "cosmetic_jitter_seed": 20261013}, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="第二轮 analyze_results.py 输出目录")
    parser.add_argument("--out", type=Path, required=True, help="PNG、SVG和来源记录输出目录")
    args = parser.parse_args()
    try:
        script_hash = sha256(__file__)
        data = load_analysis(args.analysis)
        trials, raw_hashes, skipped = original_trials(data)
        font_family = configure_style()
        out = args.out.resolve()
        require(out != data["directory"], "图目录不能直接覆盖分析输入目录，请指定独立目录")
        out.mkdir(parents=True, exist_ok=True)
        records = []
        completion_bars(data, out, records)
        effect_forest(data, out, records)
        scenario_heatmaps(data, out, records)
        if trials is not None:
            physics_distributions(data, trials, out, records)
        all_hashes = data["input_hashes"] | raw_hashes
        require(sha256(__file__) == script_hash, "绘图脚本在运行期间被修改")
        require(all(sha256(path) == value for path, value in all_hashes.items()),
                "绘图期间输入数据发生变化；图不能视为冻结输出")
        source_record = {
            "figure_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "plot_script": {"path": str(Path(__file__).resolve()), "sha256": script_hash},
            "runtime": {"python": sys.version, "matplotlib": matplotlib.__version__, "numpy": np.__version__},
            "font": {"family": font_family, "path": str(FONT_PATH), "svg_fonttype": "path"},
            "analysis_directory": str(data["directory"]),
            "inputs": [{"path": path, "sha256": value} for path, value in all_hashes.items()],
            "analysis_script_sha256": data["metadata"].get("analysis_script_sha256"),
            "original_trial_provenance": data["metadata"]["inputs"],
            "methods": list(METHODS), "seeds": {suite: list(seeds) for suite, seeds in SEEDS.items()},
            "outputs": records, "skipped_optional_figures": [skipped] if skipped else [],
            "scope": "second-round simulation only; no first-round values or selected representative seeds",
            "caveats": ["safe_completed uses the frozen force-duration proxy, not material damage",
                        "confidence intervals are marginal; separate two-baseline Holm p-values",
                        "old fusion and ablations are exploratory", "seed clusters are not human participants"],
        }
        manifest_path = out / "figure_sources.json"
        manifest_path.write_text(json.dumps(source_record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"output": str(out), "figures": len(records) // 2,
                          "png": sum(item["format"] == "png" for item in records),
                          "svg": sum(item["format"] == "svg" for item in records),
                          "sources": str(manifest_path), "skipped": [skipped] if skipped else []}, ensure_ascii=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
