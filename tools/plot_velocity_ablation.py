"""Scientific plots of raw timing/work and separately qualified cloth diagnostics."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "builds/mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from report_velocity_ablation import read, States


def main():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "pdf.fonttype": 42})
    report = read(ROOT / "reports/VELOCITY_ABLATION_20261001_measure.json")
    labels = ["base", "full", "vel005", "vel100"]
    names = ["base", "TOI full", "TOI 0.05", "TOI 1.0"]
    colors = ["#687780", "#3379b7", "#dd9d2d", "#289478"]
    summaries = {s["variant"]: s for s in report["summaries"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
    ax = axes[0, 0]
    for i, label in enumerate(labels):
        times = summaries[label]["solver_seconds"]
        ax.bar(i, times["median"], color=colors[i], alpha=.8, width=.6)
        ax.scatter(i + np.array([-.13, 0, .13]), times["values"], color="#222222", s=22, zorder=3)
        ax.text(i, times["max"] + 1.1, f"{times['median']:.2f}", ha="center", fontsize=10)
    ax.set(xticks=range(4), xticklabels=names, ylabel="100 帧求解时间（秒）", ylim=(0, 50),
           title="(a) 原始计时：三次交错重复，中位数与各次样本")
    ax = axes[0, 1]
    x = np.arange(4)
    full = summaries["full"]
    for j, (key, name, color) in enumerate([
        ("directions", "方向求解", "#3379b7"),
        ("cg_iterations", "CG 迭代", "#dd9d2d"),
        ("outer", "TOI 外层", "#289478"),
    ]):
        values = [summaries[label][key]["median"] / full[key]["median"] * 100 for label in labels]
        ax.bar(x + (j - 1) * .23, values, width=.23, label=name, color=color)
    ax.axhline(100, color="#555555", linestyle="--", linewidth=.8)
    ax.set(xticks=x, xticklabels=names, ylabel="计算次数 / 默认 TOI 中位数（%）", ylim=(0, 130),
           title="(b) 1.0 减少内层工作，外层次数基本不变")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 0]
    for i, label in enumerate(labels):
        values = [r["vs_refined_base_dt4"]["cloth_only"]["max_mass_rms_over_cloth_scale"] * 100
                  for r in report["runs"] if r["variant"] == label]
        ax.scatter(i + np.array([-.13, 0, .13]), values, color=colors[i], s=38)
        ax.plot([i, i], [min(values), max(values)], color=colors[i], linewidth=2)
    reference = report["controls"]["strict_base_dt2_vs_dt4"]["cloth_only"]["max_mass_rms_over_cloth_scale"] * 100
    ax.axhline(reference, color="#777777", linestyle="--", label=f"base dt/2 vs dt/4：{reference:.2f}%")
    ax.axhline(1, color="#a33a3a", linestyle=":", label="计划中的 1% 建议预算")
    ax.set(xticks=x, xticklabels=names, ylabel="布料位置 RMS / 初始布料尺度（%）", ylim=(0, 8),
           title="(c) 全时段最大位置误差：dt/4 参考尚未收敛")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax = axes[1, 1]
    for label, name, color in zip(labels, names, colors):
        row = next(r for r in report["runs"] if r["variant"] == label and r["repeat"] == 1)
        states = States(ROOT / "runs/local" / row["name"])
        edges = states.edges
        stretch = np.linalg.norm(states.positions[:, edges[:, 0]] - states.positions[:, edges[:, 1]], axis=2) / states.edge_rest
        ax.plot(np.asarray(states.indices) * states.dt, stretch.max(axis=1), label=name, color=color, linewidth=1.3)
    ax.set(xlabel="物理时间（秒）", ylabel="最大布料边长 / 静止边长", ylim=(.99, 1.48),
           title="(d) 首次重复的最大边拉伸：物理表现仍有差异")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("本机布落球：velocity_tol 消融（host，100 帧 × 3；2026-10-01）\n原始速度收益与物理质量分开判断", fontsize=14)
    directory = ROOT / "reports/figures"
    directory.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(directory / f"velocity_ablation_20261001.{ext}", dpi=160)
    plt.close(fig)
    print(directory / "velocity_ablation_20261001.png")


if __name__ == "__main__":
    main()
