"""Verify the local fixed-tolerance preconditioner experiment and its geometry."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from report_v32_autodl import quality

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'base': 'base', 'graph': 'base+Graph',
          'toi_mas': 'TOI .05+Graph / MAS', 'toi_diag': 'TOI .05+Graph / block diagonal'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def work(run):
    frames = read(run / 'output/stats.json')['frames']
    directions = [n for f in frames for n in f['newton'] if 'pcg' in n]
    outer = [t for f in frames for t in f.get('toi', [])]
    accepted = [t for t in outer if 'alpha' in t]
    return frames, {
        'directions': len(directions),
        'cg_iterations': sum(n['pcg']['iterations'] for n in directions),
        'pcg_limit_hits': sum(n['pcg'].get('iteration_limit', False) for n in directions),
        'outer_limit_hits': sum(f.get('toi_exit') == 'iteration_limit' or f.get('newton_exit') == 'iteration_limit' for f in frames),
        'execution': dict(Counter(n['pcg'].get('execution', 'host') for n in directions)),
        'accepted_outer': len(accepted), 'unfinished_outer': len(outer) - len(accepted),
        'unsafe_native_steps': sum(not t.get('safe_state_verified', False) for t in accepted),
        'maximum_true_relative_residual': max((n['pcg']['true_relative_residual'] for n in directions if 'true_relative_residual' in n['pcg']), default=None),
        'newton_profile_ms': {key: sum(n.get('profile_ms', {}).get(key, 0) for n in directions)
                              for key in ['assembly', 'pcg', 'line_search']},
        'outer_profile_ms': {key: sum(t.get('profile_ms', {}).get(key, 0) for t in accepted)
                            for key in ['outer_total', 'active_update', 'safe_ccd', 'safe_update']}}


def main():
    matrix = read(ROOT / 'reports/LOCAL_PERF_V33_MATRIX_100.json')
    assert len(matrix['runs']) == 24, 'Wait for the full matrix'
    manifest = read(ROOT / 'manifests/robust_port_v32.json')
    assert matrix['source_digest'] == manifest['source_digest']
    assert matrix['runner_sha256'] == sha(ROOT / 'tools/run_robust_port.py')
    rows, keyed = [], {}
    for record in matrix['runs']:
        run = ROOT / record['run']
        req, result = read(run / 'requested.json'), read(run / 'result.json')
        assert req['source_digest'] == matrix['source_digest']
        assert req['runner_sha256'] == matrix['runner_sha256']
        binary = 'builds/local-base/Release/gipc.exe' if record['label'] == 'base' else 'builds/local-fused-v32/Release/gipc.exe'
        assert req['exe_sha256'] == manifest['binaries'][binary]['sha256'] == sha(ROOT / binary)
        for key, expected in {'steps': 100, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                              'suite': '0', 'platform': 'local', 'profile': False,
                              'audit_pcg': False, 'quality_only': False, 'substeps': False}.items():
            assert req[key] == expected, (run, key)
        frames, counts = work(run)
        times = list(csv.DictReader((run / 'trace/frames.csv').open()))
        graph_ok = req['execution'] != 'conditional_graph' or set(counts['execution']) == {'conditional_graph'}
        robust_ok = True
        if record['label'].startswith('toi_'):
            assert req['preconditioner'] == record['label'][4:]
            assert req['robust_velocity_tol'] == .05
            robust_ok = all(f['toi_policy'] == 'robust' and f['toi_penalty_scope'] == 'movable'
                and f['robust_trial_velocity_tol'] == .05
                and f['toi_frame_friction_snapshot']['captured_this_solve']
                and f['toi_frame_friction_snapshot']['physical_frame'] == i
                and all(t.get('candidate_query_policy') == 'current_trial_shared_full_ccd' for t in f['toi'] if 'alpha' in t)
                for i, f in enumerate(frames))
        usable = result['status'] == 'completed' and result['finite'] and len(times) == len(frames) == 100 and graph_ok and robust_ok and not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
        if usable:
            assert [int(t['frame']) for t in times] == list(range(1, 101))
            assert np.isclose(sum(float(t['solver_ms']) for t in times) / 1000, result['solver_seconds'])
        row = {**record, 'requested': req, 'work': counts, 'usable_timing': usable,
               'graph_verified': graph_ok, 'robust_protocol_verified': robust_ok,
               'stats_sha256': sha(run / 'output/stats.json'), 'frame_times': times}
        rows.append(row)
        keyed[(record['scene'], record['label'], record['repeat'])] = row

    summary, variability = [], {}
    for scene in ['cloth_hang_l', 'cloth_sphere7_l']:
        first = keyed[(scene, 'base', 1)]
        assert first['usable_timing']
        mask = np.asarray([int(t['candidate_pairs']) + int(t['ground_candidates']) > 0 for t in first['frame_times']])
        for row in [r for r in rows if r['scene'] == scene and r['usable_timing']]:
            ms = np.asarray([float(t['solver_ms']) for t in row['frame_times']])
            row['seconds'] = {'total': float(ms.sum() / 1000), 'noncontact': float(ms[~mask].sum() / 1000), 'contact': float(ms[mask].sum() / 1000)}
            base = keyed[(scene, 'base', row['repeat'])]
            row['quality_vs_base'] = quality(ROOT / row['run'], ROOT / base['run'])
            if row['label'] == 'base':
                row['base_repeat_quality'] = quality(ROOT / row['run'], ROOT / first['run'])
            if row['label'] == 'toi_diag':
                mas = keyed[(scene, 'toi_mas', row['repeat'])]
                row['quality_vs_mas'] = quality(ROOT / row['run'], ROOT / mas['run'])
        for label in LABELS:
            selected = [r for r in rows if r['scene'] == scene and r['label'] == label and r['usable_timing']]
            assert len(selected) == 3, (scene, label)
            seconds = {phase: statistics.median(r['seconds'][phase] for r in selected) for phase in ['total', 'noncontact', 'contact']}
            baselines = {arm: [r for r in rows if r['scene'] == scene and r['label'] == arm and r['usable_timing']]
                         for arm in ['base', 'graph', 'toi_mas']}
            ratios = {arm: {phase: statistics.median(r['seconds'][phase] for r in group) / seconds[phase] if seconds[phase] else None for phase in seconds}
                      for arm, group in baselines.items()}
            totals = [r['seconds']['total'] for r in selected]
            summary.append({'scene': scene, 'label': label, 'repeats': len(selected), 'seconds': seconds,
                'speed_vs': ratios, 'total_range_s': [min(totals), max(totals)],
                'total_cv_percent': statistics.stdev(totals) / statistics.mean(totals) * 100,
                'directions_median': statistics.median(r['work']['directions'] for r in selected),
                'cg_median': statistics.median(r['work']['cg_iterations'] for r in selected),
                'noncontact_frames': int((~mask).sum()), 'contact_frames': int(mask.sum()),
                'quality_vs_base_max_percent_range': [min(r['quality_vs_base']['max_cloth_mass_rms_vs_base_percent'] for r in selected), max(r['quality_vs_base']['max_cloth_mass_rms_vs_base_percent'] for r in selected)],
                'quality_vs_mas_max_percent_range': [min(r['quality_vs_mas']['max_cloth_mass_rms_vs_base_percent'] for r in selected), max(r['quality_vs_mas']['max_cloth_mass_rms_vs_base_percent'] for r in selected)] if label == 'toi_diag' else None,
                'ground_min_gap_m': min(r['quality_vs_base']['ground_min_gap_m'] for r in selected),
                'fixed_max_motion_m': max(r['quality_vs_base']['fixed_max_displacement_m'] for r in selected),
                'cloth_max_stretch': max(r['quality_vs_base']['cloth_max_stretch'] for r in selected),
                'cloth_p95_stretch_max': max(r['quality_vs_base']['cloth_p95_stretch_max'] for r in selected)})
        variability[scene] = max(r['base_repeat_quality']['max_cloth_mass_rms_vs_base_percent'] for r in rows if r['scene'] == scene and r['label'] == 'base')

    profiles = []
    for row in read(ROOT / 'reports/LOCAL_PERF_V33_PROFILE.json')['runs']:
        _, counts = work(ROOT / row['run'])
        profiles.append({**row, 'work': counts})
    bunny = []
    for row in read(ROOT / 'reports/LOCAL_PERF_V33_BUNNY.json')['runs']:
        run = ROOT / row['run']
        frames, counts = work(run)
        directions = [n for f in frames for n in f['newton'] if 'pcg' in n]
        bunny.append({**row, 'work': counts, 'last_saved_pcg': directions[-1]['pcg'],
                      'log_tail': (run / 'run.log').read_text(errors='replace').splitlines()[-12:]})
    for row in rows:
        row.pop('frame_times')
    validation_path = ROOT / 'reports/LOCAL_PERF_V33_ACCEPTED_PATHS.json'
    validation = read(validation_path) if validation_path.exists() else None
    if validation:
        for audit in validation['runs']:
            run = ROOT / audit['run']
            frames = read(run / 'output/stats.json')['frames']
            for i, frame in enumerate(frames):
                accepted = [t for t in frame['toi'] if 'alpha' in t]
                paths = sorted((run / 'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
                assert len(paths) == len(accepted) + 1
                assert paths[0].read_bytes() == (run / f'trace/state_{i:04d}.bin').read_bytes()
                assert paths[-1].read_bytes() == (run / f'trace/state_{i+1:04d}.bin').read_bytes()
            assert audit['validation']['paths_checked'] == audit['saved_path_states'] - 1
            audit['frame_bridges_and_endpoints_verified'] = True
    report = {'hardware': 'RTX 3070 Laptop 8 GiB; CUDA 13.0; Release sm86',
              'source_digest': matrix['source_digest'], 'binaries': matrix['binaries'],
              'protocol': matrix['protocol'], 'summary': summary, 'runs': rows,
              'profile_diagnostics': profiles, 'bunny_diagnostics': bunny,
              'base_repeat_max_rms_percent': variability,
              'quality_matched_speed_qualified': False,
              'phase_rule': 'Same base r01 native narrow-phase candidate_pairs + ground_candidates > 0 mask, all configurations.',
              'accepted_path_audit': validation,
              'selected_configuration': read(ROOT / 'manifests/local_perf_v33.json'),
              'numerical_source_changed': False}
    out = ROOT / 'reports/LOCAL_PERF_V33_20261003'
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with out.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['scene', 'label', 'total_s', 'speed_base', 'speed_graph', 'speed_mas', 'noncontact_s', 'contact_s', 'cv_percent', 'directions', 'cg', 'rms_base_percent', 'rms_mas_percent']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow(dict(zip(fields, [r['scene'], r['label'], r['seconds']['total'], r['speed_vs']['base']['total'], r['speed_vs']['graph']['total'], r['speed_vs']['toi_mas']['total'], r['seconds']['noncontact'], r['seconds']['contact'], r['total_cv_percent'], r['directions_median'], r['cg_median'], r['quality_vs_base_max_percent_range'], r['quality_vs_mas_max_percent_range']])))
    lines = ['# 本机 TOI 小幅加速实验 v33（2026-10-03）', '',
        'RTX 3070 Laptop 8 GiB，CUDA 13.0，sm86 Release。每组 100 帧 × 3 次，交错执行。dt=.01，Newton=.01，PCG rho 比阈值 1e-4，TOI velocity_tol=.05 m/s，suite=0，Graph=conditional_graph。', '',
        '本轮候选是已有 `--preconditioner diag` 配置。原 v32 可执行文件、源代码及 base 完整保留；没有修改停止阈值、PCG 上限、CCD 安全检查或物理模型。', '',
        '**这些是实测耗时比，未通过质量匹配认证。** 轨迹差异相对于同 dt base/MAS，不是收敛物理解误差。用户已暂停 GPU 应用；初始 12% 空载门槛未通过且尚未启动任何正式算例，之后使用 30% 门槛并保存背景样本。剩余桌面负载使结果适用于本次本机环境。', '',
        '计时为原生同步 solver_ms 之和，不含进程启动和求解器外轨迹导出。接触/非接触共用 base r01 的窄阶段候选帧掩码；候选存在不等于实际施力。不同阶段分别取中位数，和总中位数可能略有差异。', '']
    def fmt(v):
        return '—' if v is None else f'{v:.3f}'
    for scene in ['cloth_hang_l', 'cloth_sphere7_l']:
        selected = [r for r in summary if r['scene'] == scene]
        lines += [f'## {scene}', '', f"非接触 {selected[0]['noncontact_frames']} 帧，接触候选 {selected[0]['contact_frames']} 帧。", '',
            '| 配置 | 总秒 | /base | /base+Graph | /TOI MAS | 波动 CV | 布料最大 RMS/base |', '|---|---:|---:|---:|---:|---:|---:|']
        for r in selected:
            lo, hi = r['quality_vs_base_max_percent_range']
            lines.append(f"| {LABELS[r['label']]} | {r['seconds']['total']:.4f} | {r['speed_vs']['base']['total']:.3f}× | {r['speed_vs']['graph']['total']:.3f}× | {r['speed_vs']['toi_mas']['total']:.3f}× | {r['total_cv_percent']:.2f}% | {lo:.2f}–{hi:.2f}% |")
        lines += ['', '| 配置 | 非接触秒 | 非接触 /base | 接触秒 | 接触 /base | Newton 方向 | CG 次数 | 最大拉伸 | 最小地面间隙 m |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in selected:
            lines.append(f"| {LABELS[r['label']]} | {r['seconds']['noncontact']:.4f} | {fmt(r['speed_vs']['base']['noncontact'])} | {r['seconds']['contact']:.4f} | {fmt(r['speed_vs']['base']['contact'])} | {r['directions_median']} | {r['cg_median']} | {r['cloth_max_stretch']:.5f} | {r['ground_min_gap_m']:.3e} |")
        diag = next(r for r in selected if r['label'] == 'toi_diag')
        lo, hi = diag['quality_vs_mas_max_percent_range']
        lines += ['', f"块对角与相同 .05 m/s 的 MAS 轨迹最大 RMS 差异：{lo:.2f}–{hi:.2f}%（布料初始包围盒对角线归一化）。base 自身三次重复最大差异 {variability[scene]:.2f}%。", '']
    lines += ['## 采用的改进与结论', '',
        '- 布料–球采用已有块对角预条件器：TOI+Graph 中位数 7.0153 → 6.3268 秒，时间下降 9.81%，加速 1.109×；相对 base 为 1.724×，相对 base+Graph 为 1.292×。三轮成对 MAS/块对角速度比 1.170×、1.094×、1.164×。本轮尚不能外推到其他 GPU、分辨率或长帧数。',
        '- 悬挂布料继续采用 MAS：块对角中位数只改善 0.89%，三轮成对速度比 1.053×、0.989×、0.928×，低于本轮波动，且轨迹发生额外变化。',
        '- 以场景配置完成这次小幅优化，无需修改数值核心或重复编译同一源代码。`manifests/local_perf_v33.json` 固定参数、哈希和场景选择，`tools/run_local_perf_v33.py` 负责调用保留的 v32 runner。仅覆盖两个已测小场景；兔子尚未支持。',
        '- 诊断中块对角平均每次 CG 的摊销 PCG 成本约为 MAS 的 43%–46%（包括求解准备与审计开销）。虽然 CG 更多，总 PCG 仍下降；布料–球方向数也下降。预条件器改变 rho 停止尺度与方向，因此收益不能全部归因于每次 CG 更便宜。',
        '- 布料–球块对角相对 base 的最大轨迹 RMS 为 2.76%–3.07%，MAS 为 2.76%–3.24%；相对 MAS 的差异明显大于 base 重复波动。保持了原生安全策略并通过独立接受路径 CCD，但质量匹配标志仍为 false。', '',
        '后续优先：在布料–球上校准预条件器无关的真实残差停止标准，复核相同质量下是否保留约 10% 收益；兔子从失败帧的矩阵/试探位移与 CCD 候选增长联合定位，当前不扩大到长场景。', '']
    lines += ['## 耗时与失败诊断', '',
        '60 帧 profile/audit 用于定位成本，不纳入正式速度比。真实 L2 残差与 PCG 的预条件 rho 比是不同量，不能把 L2 残差直接与 1e-4 比较；预条件器更换也会改变相同 rho 阈值下的停止方向与轨迹。', '',
        '| 场景 | TOI 预条件器 | PCG ms | 组装 ms | 线搜索 ms | 活跃集 ms | 安全状态更新 ms | CG | 最大真实 L2 相对残差 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in profiles:
        if r['label'] == 'graph':
            continue
        w = r['work']; n = w['newton_profile_ms']; o = w['outer_profile_ms']
        lines.append(f"| {r['scene']} | {r['label']} | {n['pcg']:.1f} | {n['assembly']:.1f} | {n['line_search']:.1f} | {o['active_update']:.1f} | {o['safe_update']:.1f} | {w['cg_iterations']} | {w['maximum_true_relative_residual']:.4f} |")
    lines += ['', '兔子短测请求 40 帧，均未完成：', '']
    for r in bunny:
        lines.append(f"- {r['label']}：{r['status']}，完成 {r['recorded_frames']} 帧，墙钟 {r['wall_seconds']:.1f}s；PCG 上限次数 {r['work']['pcg_limit_hits']}。")
    lines += ['', 'MAS 最后线性系统达到 18257 次 PCG 上限；块对角下一帧日志出现 CCD 候选数 6,638,387 后超时。块对角日志中最后保存的已完成 PCG 不是超时位置的记录，不能据此认为当前帧线性求解正常。候选爆炸与过大试探轨迹有关的机制仍需定点诊断，尚未证实具体根因。本轮不为失败/部分帧报告有效加速比。', '',
        '## 安全与复现', '',
        f"- 完整计时组：{sum(r['usable_timing'] for r in rows)}/24。核对二进制/runner/源摘要、100 帧编号、Graph 实际执行、TOI 跨帧摩擦时序、停止参数、求解上限和原生安全标志。",
        '- 几何检查包含 0–100 帧有限值、固定点位移、地面间隙、布料边拉伸与质量加权轨迹/速度差异。',
        f"- 独立连续碰撞审计：{'两场景全部通过' if validation and validation['all_passed'] else '未通过或未运行'}；" + (f"共 {sum(r['validation']['paths_checked'] for r in validation['runs'])} 个保存路径段，碰撞保守标志 {sum(r['validation']['conservative_collision_flags'] for r in validation['runs'])}，逐帧桥接与终点已核对。覆盖额外 100 帧导出运行，不能替代所有正式计时轨迹的独立 CCD。" if validation else '详见 LOCAL_PERF_V33_ACCEPTED_PATHS.json。'), '',
        '```text', 'E:/Anaconda/envs/DL/python.exe tools/local_perf_v33.py --phase profile',
        'E:/Anaconda/envs/DL/python.exe tools/local_perf_v33.py --phase bunny',
        'E:/Anaconda/envs/DL/python.exe tools/benchmark_local_preconditioners_v33.py',
        'E:/Anaconda/envs/DL/python.exe tools/validate_local_perf_v33.py',
        'E:/Anaconda/envs/DL/python.exe tools/report_local_perf_v33.py', '```', '',
        '以上实验脚本拒绝覆盖已有运行。复现时需使用新的实验名称/结果路径。', '',
        '选择后的本机配置运行入口（已用布料–球 5 帧烟测确认实际 diag/Graph/robust 参数与冻结哈希）：', '',
        '```text', 'E:/Anaconda/envs/DL/python.exe tools/run_local_perf_v33.py --scene cloth_sphere7_l --name NEW_LOCAL_RUN --steps 100',
        'E:/Anaconda/envs/DL/python.exe tools/run_local_perf_v33.py --scene cloth_hang_l --name NEW_HANG_RUN --steps 100', '```', '']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(out.with_suffix('.md')), 'usable_runs': sum(r['usable_timing'] for r in rows),
                      'independent_ccd_passed': bool(validation and validation['all_passed'])}))


if __name__ == '__main__':
    main()
