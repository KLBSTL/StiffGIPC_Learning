"""Standalone scientific speed figure; failed groups never receive a bar."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'reports/AUTODL_V32_SMALL_SCENES_20261002.json').read_text())
arms = ['base', 'graph', 'toi005', 'toi005_graph', 'toi1', 'toi1_graph']
labels = ['Base', '+Graph', '+TOI .05', '+TOI .05\n+Graph', '+TOI 1.0', '+TOI 1.0\n+Graph']
scenes = ['cloth_hang_l', 'cloth_sphere7_l', 'bunny_cloth_bunny_l']
titles = ['Hanging cloth (1,939 vertices)', 'Cloth + sphere (3,084 vertices)', 'Bunny + cloth + bunny (39,475 vertices)']
lookup = {(r['scene'], r['arm']): r for r in data['summary']}
colors = ['#9299a1', '#417dac', '#bf7958', '#72a985', '#bd6558', '#698cb8']
fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
for ax, scene, title in zip(axes, scenes, titles):
    for i, arm in enumerate(arms):
        row = lookup.get((scene, arm))
        if row:
            speed = row['raw_speedup_vs_base']['total']
            ax.bar(i, speed, color=colors[i], width=.72)
            ax.text(i, speed + .035, f'{speed:.2f}x', ha='center', fontsize=9)
        else:
            failure = next(r for r in data['runs'] if r['scene'] == scene and r['arm'] == arm)
            ax.text(i, .08, f"PCG cap\n{failure['recorded_frames']}/100", ha='center', va='bottom', fontsize=8, color='#9b4545')
    ax.axhline(1, color='#555555', linewidth=.8, linestyle='--')
    ax.set_title(title, fontsize=11, pad=14)
    ax.set_xticks(np.arange(6), labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 2.03)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Raw solver speedup relative to base')
fig.suptitle('AutoDL RTX 4090 / CUDA 12.8: 100 frames, 3 repeats (median)', fontsize=14)
fig.text(.5, .025, 'TOI tolerances in m/s. Quality is not matched. Failed TOI bunny runs are excluded from speed comparisons.', ha='center', fontsize=10)
fig.tight_layout(rect=[0, .09, 1, .92])
out = ROOT / 'reports/figures/AUTODL_V32_SMALL_SCENES_20261002'
out.parent.mkdir(exist_ok=True)
fig.savefig(out.with_suffix('.png'), dpi=180)
fig.savefig(out.with_suffix('.pdf'))
print(out.with_suffix('.png'))
