"""Save verified v34 correctness evidence independently of pending timing."""
import json
from pathlib import Path
from report_local_perf_v33 import work
from report_v32_autodl import quality

ROOT = Path(__file__).resolve().parents[1]
smoke = json.loads((ROOT / 'reports/LOCAL_PERF_V34_SMOKE.json').read_text())
rows = []
for row in smoke['runs']:
    run = ROOT / row['run']
    frames, counts = work(run)
    directions = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    assert row['status'] == 'completed' and not counts['pcg_limit_hits']
    expected = row['label'] == 'audit'
    assert all(n['fused_diag_update'] == expected for n in directions)
    rows.append({**row, 'work': counts,
        'max_host_graph_solution_relative_difference': max((n.get('same_system_relative_solution_difference', 0) for n in directions), default=0),
        'max_host_repeat_solution_relative_difference': max((n.get('same_system_host_repeat_relative_difference', 0) for n in directions), default=0)})
export = json.loads((ROOT / 'reports/LOCAL_PERF_V34_PATHS.json').read_text())['runs'][0]
run = ROOT / export['run']
frames, counts = work(run)
assert export['status'] == 'completed' and len(frames) == 100
assert not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
for i, frame in enumerate(frames):
    states = sorted((run / 'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
    assert len(states) == 1 + sum('alpha' in t for t in frame['toi'])
    assert states[0].read_bytes() == (run / f'trace/state_{i:04d}.bin').read_bytes()
    assert states[-1].read_bytes() == (run / f'trace/state_{i+1:04d}.bin').read_bytes()
ccd = json.loads((ROOT / 'reports/LOCAL_PERF_V34_PATH_VALIDATION.json').read_text())
assert ccd['exit_code'] == 0 and ccd['validation']['passed']
q = quality(run, ROOT / 'runs/local/local_perf_v33_formal_cloth_sphere7_l_toi_diag_r01')
report = {'fixture': json.loads((ROOT / 'builds/local_perf_v34_fixture.json').read_text()),
          'smoke': rows, 'export': export, 'native_work': counts,
          'path_validation': ccd, 'frame_bridges_verified': True,
          'quality_vs_prior_v32_diag_r01': q,
          'performance_qualified': False, 'timing_status': 'pending matched interleaved matrix'}
(ROOT / 'reports/LOCAL_PERF_V34_PRECHECK.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
v = ccd['validation']
text = f'''# v34 正确性预检（2026-10-03）

已完成独立副本、Release 构建和冻结身份；新增 Graph PCG 的 x/r 更新与 3×3 块对角应用融合。默认开关为 0，使用 `--fused-diag-update 1` 显式启用。局部 ABD 12×12 预条件器仍按原顺序执行，MAS 自动回退。

- GPU fixture：28 组，108108 个 x/r/z 值逐位一致，CPU 最大相对差异 2.23977e-14。
- 实际布料–球：30 帧，116 个线性求解均使用融合；Graph/host 最大解相对差异 {rows[0]['max_host_graph_solution_relative_difference']:.6g}，host 自身重复最大差异 {rows[0]['max_host_repeat_solution_relative_difference']:.6g}。
- MAS 回退：10 帧，20 个线性求解均未使用融合。
- 额外导出：100 帧完成，无 PCG/outer 上限或原生不安全接受步；逐帧桥接、接受步数量、终点与 trace 完全对应。
- 独立 CPU Tight-Inclusion：{v['paths_checked']} 段全部通过，碰撞保守标志 {v['conservative_collision_flags']}。
- 此导出相对先前 v32 diag r01 的布料最大 RMS 差异 {q['max_cloth_mass_rms_vs_base_percent']:.4f}%；这是不同运行之间的轨迹差异，不是收敛物理解误差。

当前 GPU 背景占用约 34–41%，超过 30% 门槛。以上运行均标记 quality-only/diagnostic，不用于速度比。正式矩阵尚待运行，不能宣称已获得额外加速。

源代码：sources/stiff_perf_v34。构建：builds/local-fused-v34。冻结身份：manifests/perf_v34.json。完整证据：LOCAL_PERF_V34_PRECHECK.json。

预检命令：

```text
E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase smoke
E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase paths
E:/Anaconda/envs/DL/python.exe tools/validate_perf_v34.py
E:/Anaconda/envs/DL/python.exe tools/precheck_perf_v34.py
```
'''
(ROOT / 'reports/LOCAL_PERF_V34_PRECHECK.md').write_text(text, encoding='utf-8')
print(json.dumps({'passed': True, 'CCD_paths': v['paths_checked'], 'rms_vs_prior_percent': q['max_cloth_mass_rms_vs_base_percent']}))
