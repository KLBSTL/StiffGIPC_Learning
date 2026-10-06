"""Render measured v31 work, quality and contact classifications."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from report_velocity_ablation import States

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures"
OUT.mkdir(exist_ok=True)
data = json.loads((ROOT / "reports/TOI_WARM_HISTORY_V31_DIAGNOSE_20261001.json").read_text())
rows = [r for r in data["runs"] if "work" in r]
names = {"cold": "Reset all", "legacy": "Legacy all", "c_only": "C only", "c_lambda": "C + lambda",
         "c_gamma": "C + gamma", "c_lambda_gamma": "C + lambda + gamma", "c_friction": "C + friction", "all_independent": "All independent"}
labels = [names[r["variant"]] for r in rows]
x = np.arange(len(rows))
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), layout="constrained")
outer = np.array([r["work"]["outer"] for r in rows])
extra = np.array([r["work"]["directions"] - r["work"]["outer"] for r in rows])
axes[0, 0].bar(x, outer, color="#3979b7", label="Outer solves")
axes[0, 0].bar(x, extra, bottom=outer, color="#d99543", label="Extra inner solves")
axes[0, 0].set_ylabel("Global linear-system solves")
axes[0, 0].legend(fontsize=8)
for i, (o, e) in enumerate(zip(outer, extra)):
    axes[0, 0].text(i, o + e + 15, str(o + e), ha="center", fontsize=8)
axes[0, 0].set_ylim(0, (outer + extra).max() * 1.18)
alpha = [r["subsets"]["first5_outer"]["ccd_alpha"]["median"] for r in rows]
axes[0, 1].bar(x, alpha, color="#35916a")
axes[0, 1].set_yscale("log")
axes[0, 1].set_ylim(.003, 1)
axes[0, 1].set_ylabel("Median safe alpha in first 5 outers, frames 22–60")
classes = ["missing_after_active_update", "added_after_trial", "active_linear_violation", "active_linear_satisfied_geometric_block"]
class_labels = ["Still missing", "Added after trial", "Active, plane violated", "Active, plane satisfied"]
colors = ["#bb4f4f", "#e5a253", "#697aab", "#629985"]
bottom = np.zeros(len(rows))
for key, label, color in zip(classes, class_labels, colors):
    counts = np.array([r["subsets"]["all_first_min"]["classes"].get(key, 0) for r in rows])
    axes[1, 0].bar(x, counts, bottom=bottom, label=label, color=color)
    bottom += counts
axes[1, 0].set_ylim(0, bottom.max() * 1.25)
axes[1, 0].set_ylabel("Limited outer iterations, frames 22–60")
axes[1, 0].legend(fontsize=7, ncols=2, loc="upper center")
errors = [100 * r["vs_refined_base_dt4"]["cloth_only"]["max_mass_rms_over_cloth_scale"] for r in rows]
axes[1, 1].bar(x, errors, color="#846ca0")
axes[1, 1].axhline(1, color="#ad4141", linestyle="--", label="Suggested 1% diagnostic budget")
axes[1, 1].set_ylabel("Max cloth position RMS / initial cloth scale (%)")
axes[1, 1].legend(fontsize=7)
for i, value in enumerate(errors):
    axes[1, 1].text(i, value + .07, f"{value:.2f}", ha="center", fontsize=8)
axes[1, 1].set_ylim(0, max(errors) * 1.25)
for ax in axes.flat:
    ax.set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    for i, patch in enumerate(ax.patches):
        if rows[i % len(rows)].get("independent_ccd", {}).get("passed") is False:
            patch.set_edgecolor("#a12f2f")
            patch.set_linewidth(1.3)
            patch.set_hatch("//")
fig.suptitle("Local 60-frame warm-state ablation: one run per policy; reference is not converged; no speed qualification\nC + friction failed CCD (9 conservative flags, red hatching); other seven policies passed", fontsize=10)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"TOI_WARM_HISTORY_V31.{ext}", dpi=180)
plt.close(fig)

reference = States(ROOT / "runs/local/local_velocity_20261001_base_reference_r01")
states = [reference] + [States(ROOT / "runs/local" / r["run"]) for r in rows]
positions = [reference.positions[240]] + [s.positions[-1] for s in states[1:]]
combined = np.concatenate(positions)
low, high = combined.min(axis=0), combined.max(axis=0)
center, radius = (low + high) / 2, (high - low).max() * .55
fig = plt.figure(figsize=(10.5, 10.8), layout="constrained")
titles = ["Refined base dt/4 (unconverged)"] + labels
for i, (state, pos, title) in enumerate(zip(states, positions, titles), 1):
    ax = fig.add_subplot(3, 3, i, projection="3d")
    in_tet = np.zeros(len(pos), dtype=bool)
    in_tet[state.tets.ravel()] = True
    cloth = state.faces[~in_tet[state.faces].any(axis=1)]
    sphere = state.faces[in_tet[state.faces].all(axis=1)]
    view = pos[:, [0, 2, 1]]
    ax.add_collection3d(Poly3DCollection(view[sphere], facecolor="#c9cdd3", edgecolor="none", alpha=.55))
    ax.add_collection3d(Poly3DCollection(view[cloth], facecolor="#6898c1", edgecolor="#315773", linewidth=.06, alpha=.92))
    for setter, dim in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], [0, 2, 1]):
        setter(center[dim] - radius, center[dim] + radius)
    ax.view_init(elev=23, azim=-55)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    failed = i > 1 and rows[i - 2].get("independent_ccd", {}).get("passed") is False
    ax.set_title(title + (" (CCD failed)" if failed else ""), fontsize=9, color="#a12f2f" if failed else "black")
fig.suptitle("Actual exported geometry at t = 0.60 s, shared view/scale; one run per policy", fontsize=11)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"TOI_WARM_HISTORY_V31_GEOMETRY.{ext}", dpi=180)
plt.close(fig)
print("Exported warm-state work, quality and geometry PNG/PDF.")
