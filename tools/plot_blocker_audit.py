"""Export measured work and geometry plots for the v30 diagnostic report."""
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
data = json.loads((ROOT / "reports/TOI_BLOCKER_AUDIT_V30_20261001.json").read_text())
rows = data["runs"]
labels = ["Default TOI", "Velocity tol 1.0", "Persist C / lambda / gamma"]
colors = ["#3979b7", "#d38a28", "#35916a"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
x = np.arange(3)
outer = np.array([r["work"]["outer"] for r in rows])
extra = np.array([r["work"]["directions"] - r["work"]["outer"] for r in rows])
axes[0].bar(x, outer, color=colors, label="Outer solves")
axes[0].bar(x, extra, bottom=outer, color=colors, hatch="///", alpha=.5, label="Extra inner solves")
for i, (o, e) in enumerate(zip(outer, extra)):
    axes[0].text(i, o / 2, str(o), ha="center", color="white")
    axes[0].text(i, o + e / 2, str(e), ha="center")
axes[0].set_ylabel("Linear system solves in 60 frames")
axes[0].legend(fontsize=8)
classes = ["missing_after_active_update", "added_after_trial", "active_linear_violation", "active_linear_satisfied_geometric_block"]
class_labels = ["Still missing", "Added after trial", "Active, plane violated", "Active, plane satisfied"]
class_colors = ["#bb4f4f", "#e5a253", "#697aab", "#629985"]
bottom = np.zeros(3)
for cls, label, color in zip(classes, class_labels, class_colors):
    value = np.array([r["subsets"]["all_first_min"]["classes"].get(cls, 0) for r in rows])
    axes[1].bar(x, value, bottom=bottom, label=label, color=color)
    bottom += value
axes[1].set_ylabel("Limited outer iterations, frames 22–60")
axes[1].legend(fontsize=7, loc="upper center", ncols=2)
axes[1].set_ylim(0, 710)
alpha = [r["subsets"]["first5_outer"]["ccd_alpha"]["median"] for r in rows]
axes[2].bar(x, alpha, color=colors)
axes[2].set_yscale("log")
axes[2].set_ylim(.003, 1)
for i, value in enumerate(alpha):
    axes[2].text(i, value * 1.18, f"{value:.4f}", ha="center")
axes[2].set_ylabel("Median safe CCD alpha, first 5 outers")
for ax in axes:
    ax.set_xticks(x, ["Default", "Vel 1.0", "Persist"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle("Local cloth-on-sphere diagnosis: one 60-frame run per branch; no speed qualification", fontsize=11)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"TOI_BLOCKER_V30_WORK.{ext}", dpi=180)
plt.close(fig)

reference = States(ROOT / "runs/local/local_velocity_20261001_base_reference_r01")
states = [reference] + [States(ROOT / "runs/local" / r["run"]) for r in rows]
positions = [reference.positions[240]] + [s.positions[-1] for s in states[1:]]
all_positions = np.concatenate(positions)
low, high = all_positions.min(axis=0), all_positions.max(axis=0)
center = (low + high) / 2
radius = (high - low).max() / 2 * 1.05
fig = plt.figure(figsize=(14, 4.5), layout="constrained")
for i, (state, pos, title) in enumerate(zip(states, positions, ["Refined base dt/4"] + labels), 1):
    ax = fig.add_subplot(1, 4, i, projection="3d")
    in_tet = np.zeros(len(pos), dtype=bool)
    in_tet[state.tets.ravel()] = True
    cloth_faces = state.faces[~in_tet[state.faces].any(axis=1)]
    sphere_faces = state.faces[in_tet[state.faces].all(axis=1)]
    # Matplotlib uses z as vertical; the simulation's vertical axis is y.
    view = pos[:, [0, 2, 1]]
    ax.add_collection3d(Poly3DCollection(view[sphere_faces], facecolor="#c9cdd3", edgecolor="none", alpha=.55))
    ax.add_collection3d(Poly3DCollection(view[cloth_faces], facecolor="#6898c1", edgecolor="#315773", linewidth=.06, alpha=.92))
    for setter, dim in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], [0, 2, 1]):
        setter(center[dim] - radius, center[dim] + radius)
    ax.view_init(elev=23, azim=-55)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
fig.suptitle("Exported final geometry at t = 0.60 s, shared view and scale; refined reference is not converged", fontsize=11)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"TOI_BLOCKER_V30_GEOMETRY.{ext}", dpi=180)
plt.close(fig)
print("Exported work and geometry PNG/PDF figures.")
