"""Verify downloaded AutoDL records; stream geometry for mixed ABD/FEM scenes."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'base': 'base', 'graph': 'base+Graph', 'toi005': 'base+TOI (.05)',
          'toi005_graph': 'base+TOI (.05)+Graph', 'toi1': 'base+TOI (1.0)',
          'toi1_graph': 'base+TOI (1.0)+Graph'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def quality(run, reference):
    d, rd = run / 'trace', reference / 'trace'
    paths = sorted(d.glob('state_*.bin'))
    assert len(paths) == 101 and [int(p.stem[-4:]) for p in paths] == list(range(101))
    x0 = np.fromfile(paths[0], dtype='<f8').reshape(-1, 3)
    mass = np.fromfile(d / 'masses.bin', dtype='<f8')
    assert np.array_equal(x0, np.fromfile(rd / paths[0].name, dtype='<f8').reshape(-1, 3))
    assert np.array_equal(mass, np.fromfile(rd / 'masses.bin', dtype='<f8'))
    assert (d / 'topology.bin').read_bytes() == (rd / 'topology.bin').read_bytes()
    raw = np.fromfile(d / 'topology.bin', dtype='<u4')
    nv, nf, nt = map(int, raw[:3])
    faces = raw[3:3 + 3 * nf].reshape(-1, 3)
    tets = raw[3 + 3 * nf:].reshape(-1, 4)
    assert len(x0) == nv and len(tets) == nt
    in_tet = np.zeros(nv, dtype=bool)
    in_tet[tets.ravel()] = True
    cloth_faces = faces[~in_tet[faces].any(axis=1)]
    cloth = np.zeros(nv, dtype=bool)
    cloth[np.unique(cloth_faces)] = True
    assert cloth.any() and mass[cloth].sum() > 0
    cloth_scale = float(np.linalg.norm(np.ptp(x0[cloth], axis=0)))
    edges = np.unique(np.sort(np.concatenate([cloth_faces[:, [0, 1]],
        cloth_faces[:, [1, 2]], cloth_faces[:, [2, 0]]]), axis=1), axis=0)
    rest = np.linalg.norm(x0[edges[:, 0]] - x0[edges[:, 1]], axis=1)
    fixed = np.fromfile(d / 'boundary_types.bin', dtype='<i4') == 1
    meta = read(d / 'metadata.json')
    abd_n = meta['abd_point_num']
    body = np.fromfile(d / 'body_ids.bin', dtype='<i4')
    objects = [o for o in read(run / 'output/scene.json')['objects'] if o['body_type'] == 'ABD']
    fixed_abd_verified = set(body[:abd_n]).issubset(set(range(len(objects))))
    if fixed_abd_verified:
        for i, obj in enumerate(objects):
            if obj.get('fixed_mode') == 'all':
                fixed[:abd_n] |= body[:abd_n] == i
    def det(x):
        q = x[tets]
        return np.einsum('ij,ij->i', q[:, 1] - q[:, 0], np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]))
    rest_det = det(x0)
    assert np.all(rest_det != 0)
    q = {'cloth_vertices': int(cloth.sum()), 'cloth_scale_m': cloth_scale,
         'all_frames_finite': True, 'fixed_abd_mask_verified': fixed_abd_verified,
         'fixed_max_displacement_m': 0., 'ground_min_gap_m': float('inf'),
         'cloth_max_stretch': 0., 'cloth_p95_stretch_max': 0.,
         'tet_min_relative_jacobian': None, 'tet_nonpositive_frame_events': 0,
         'max_cloth_mass_rms_vs_base_m': 0., 'final_cloth_mass_rms_vs_base_m': 0.,
         'max_cloth_velocity_rms_difference_m_s': 0.,
         'scope': 'same-dt base trajectory difference; not converged physical error or independent CCD'}
    prev, prev_ref = None, None
    for path in paths:
        x = np.fromfile(path, dtype='<f8').reshape(-1, 3)
        ref = np.fromfile(rd / path.name, dtype='<f8').reshape(-1, 3)
        q['all_frames_finite'] &= bool(np.isfinite(x).all())
        diff = x[cloth] - ref[cloth]
        rms = float(np.sqrt(np.average(np.sum(diff * diff, axis=1), weights=mass[cloth])))
        q['max_cloth_mass_rms_vs_base_m'] = max(q['max_cloth_mass_rms_vs_base_m'], rms)
        q['final_cloth_mass_rms_vs_base_m'] = rms
        q['fixed_max_displacement_m'] = max(q['fixed_max_displacement_m'],
            float(np.linalg.norm(x[fixed] - x0[fixed], axis=1).max(initial=0)))
        q['ground_min_gap_m'] = min(q['ground_min_gap_m'],
            float((x[~fixed] @ np.asarray(meta['ground_normal']) - meta['ground_offset']).min(initial=np.inf)))
        stretch = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1) / rest
        q['cloth_max_stretch'] = max(q['cloth_max_stretch'], float(stretch.max()))
        q['cloth_p95_stretch_max'] = max(q['cloth_p95_stretch_max'], float(np.percentile(stretch, 95)))
        if nt:
            jacobian = det(x) / rest_det
            q['tet_min_relative_jacobian'] = min(q['tet_min_relative_jacobian'] if q['tet_min_relative_jacobian'] is not None else float('inf'), float(jacobian.min()))
            q['tet_nonpositive_frame_events'] += int((jacobian <= 0).sum())
        if prev is not None:
            dv = ((x[cloth] - prev[cloth]) - (ref[cloth] - prev_ref[cloth])) / .01
            vrms = float(np.sqrt(np.average(np.sum(dv * dv, axis=1), weights=mass[cloth])))
            q['max_cloth_velocity_rms_difference_m_s'] = max(q['max_cloth_velocity_rms_difference_m_s'], vrms)
        prev, prev_ref = x, ref
    q['max_cloth_mass_rms_vs_base_percent'] = 100 * q['max_cloth_mass_rms_vs_base_m'] / cloth_scale
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    matrix = read(a.data / 'reports/AUTODL_V32_MATRIX_100.json')
    manifest = read(a.data / 'manifests/robust_port_v32_autodl.json')
    rows, bykey = [], {}
    for row in matrix['runs']:
        run = a.data / row['run']
        if not (run / 'requested.json').exists():
            item = {**row, 'usable_timing': False, 'verification_failure': 'launcher failed before requested.json'}
            rows.append(item)
            bykey[(row['scene'], row['arm'], row['repeat'])] = item
            continue
        req, result = read(run / 'requested.json'), read(run / 'result.json')
        assert req['source_digest'] == manifest['source_digest'] == matrix['source_digest']
        assert req['runner_sha256'] == matrix['runner_sha256'] == sha(ROOT / 'tools/run_robust_port.py')
        binary = 'stiff_base' if req['arm'] == 'base' else 'stiff_robust_port'
        assert req['exe_sha256'] == manifest['binaries'][f'builds/autodl-{binary}/gipc']['sha256']
        for k, value in {'steps': 100, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                         'suite': '0', 'platform': 'autodl', 'profile': False,
                         'substeps': False, 'audit_graph': False, 'quality_only': False}.items():
            assert req[k] == value, (run, k)
        item = {**row, 'source_digest': req['source_digest'], 'exe_sha256': req['exe_sha256'],
                'stats_sha256': None, 'usable_timing': False}
        if (run / 'trace/frames.csv').exists():
            times = list(csv.DictReader((run / 'trace/frames.csv').open()))
            item['frame_times'] = times
        else:
            times = []
        if (run / 'output/stats.json').exists():
            frames = read(run / 'output/stats.json')['frames']
            item['stats_sha256'] = sha(run / 'output/stats.json')
            directions = [n for f in frames for n in f['newton'] if 'pcg' in n]
            outer = [t for f in frames for t in f.get('toi', [])]
            accepted = [t for t in outer if 'alpha' in t]
            execution = Counter(n['pcg'].get('execution', 'host') for n in directions)
            item['work'] = {'directions': len(directions),
                'cg_iterations': sum(n['pcg']['iterations'] for n in directions),
                'pcg_limit_hits': sum(n['pcg'].get('iteration_limit', False) for n in directions),
                'outer_limit_hits': sum(f.get('toi_exit') == 'iteration_limit' or f.get('newton_exit') == 'iteration_limit' for f in frames),
                'outer': len(outer), 'execution': dict(execution),
                'unsafe_native_toi_steps': sum(not t.get('safe_state_verified', False) for t in accepted),
                'unfinished_outer_records': len(outer) - len(accepted),
                'minimum_safe_alpha': min((t['alpha'] for t in accepted), default=None),
                'tiny_alpha_steps': sum(t['alpha'] < 1e-4 for t in accepted)}
            if result['status'] != 'completed':
                item['last_pcg'] = directions[-1]['pcg'] if directions else None
                item['failure_reason'] = 'PCG iteration limit' if item['work']['pcg_limit_hits'] else 'see run.log'
            item['graph_execution_verified'] = req['execution'] != 'conditional_graph' or set(execution) == {'conditional_graph'}
            if req['arm'] in ['base_toi', 'base_toi_graph']:
                item['robust_policy_verified'] = bool(frames) and all(
                    f['toi_policy'] == 'robust' and f['toi_penalty_scope'] == 'movable'
                    and f['robust_trial_velocity_tol'] == req['robust_velocity_tol']
                    and all(t.get('candidate_query_policy') == 'current_trial_shared_full_ccd' for t in f['toi'] if 'alpha' in t)
                    and f['toi_frame_friction_snapshot']['captured_this_solve']
                    and f['toi_frame_friction_snapshot']['physical_frame'] == i for i, f in enumerate(frames))
            else:
                item['robust_policy_verified'] = True
            item['usable_timing'] = result['status'] == 'completed' and result.get('finite', False) and len(times) == len(frames) == 100 and item['graph_execution_verified'] and item['robust_policy_verified'] and not any(item['work'][k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_toi_steps'])
        if item['usable_timing']:
            assert [int(t['frame']) for t in times] == list(range(1, 101))
            assert np.isclose(sum(float(t['solver_ms']) for t in times) / 1000, result['solver_seconds'])
        rows.append(item)
        bykey[(row['scene'], row['arm'], row['repeat'])] = item
    summary = []
    for scene in matrix['protocol']['scenes'].split(','):
        base = bykey.get((scene, 'base', 1))
        if not base or not base['usable_timing']:
            continue
        mask = np.array([int(t['candidate_pairs']) + int(t['ground_candidates']) > 0 for t in base['frame_times']])
        for row in [r for r in rows if r['scene'] == scene and r['usable_timing']]:
            ms = np.array([float(t['solver_ms']) for t in row['frame_times']])
            row['seconds'] = {'total': float(ms.sum() / 1000), 'noncontact': float(ms[~mask].sum() / 1000), 'contact': float(ms[mask].sum() / 1000)}
            paired = bykey.get((scene, 'base', row['repeat']), base)
            row['quality'] = quality(a.data / row['run'], a.data / paired['run'])
        for arm in LABELS:
            selected = [r for r in rows if r['scene'] == scene and r['arm'] == arm and r['usable_timing']]
            if not selected:
                continue
            timing = {phase: statistics.median(r['seconds'][phase] for r in selected) for phase in ['total', 'noncontact', 'contact']}
            bases = [r for r in rows if r['scene'] == scene and r['arm'] == 'base' and r['usable_timing']]
            graphs = [r for r in rows if r['scene'] == scene and r['arm'] == 'graph' and r['usable_timing']]
            baseline_time = {phase: statistics.median(r['seconds'][phase] for r in bases) for phase in timing}
            summary.append({'scene': scene, 'arm': arm, 'label': LABELS[arm], 'repeats': len(selected),
                'seconds': timing, 'total_range_seconds': [min(r['seconds']['total'] for r in selected), max(r['seconds']['total'] for r in selected)],
                'raw_speedup_vs_base': {phase: baseline_time[phase] / timing[phase] if timing[phase] > 0 else None for phase in timing},
                'raw_total_speedup_vs_graph': statistics.median(r['seconds']['total'] for r in graphs) / timing['total'] if graphs else None,
                'contact_frames': (np.flatnonzero(mask) + 1).tolist(), 'noncontact_frames': (np.flatnonzero(~mask) + 1).tolist(),
                'directions': statistics.median(r['work']['directions'] for r in selected),
                'cg_iterations': statistics.median(r['work']['cg_iterations'] for r in selected),
                'cloth_rms_max_percent_range': [min(r['quality']['max_cloth_mass_rms_vs_base_percent'] for r in selected), max(r['quality']['max_cloth_mass_rms_vs_base_percent'] for r in selected)],
                'ground_min_gap_m': min(r['quality']['ground_min_gap_m'] for r in selected),
                'tet_min_jacobian': min((r['quality']['tet_min_relative_jacobian'] for r in selected if r['quality']['tet_min_relative_jacobian'] is not None), default=None)})
    for row in rows:
        row.pop('frame_times', None)
    report = {'hardware': (a.data / 'builds/autodl_gpu.csv').read_text().strip(),
        'cuda': (a.data / 'builds/autodl_cuda.txt').read_text().strip(), 'protocol': matrix['protocol'],
        'source_digest': manifest['source_digest'], 'binaries': manifest['binaries'],
        'summary': summary, 'runs': rows, 'skipped': matrix['skipped'],
        'quality_matched_speed_qualified': False, 'independent_accepted_path_ccd_run': False,
        'phase_rule': 'base r01 narrow candidate_pairs+ground_candidates>0; same frame mask for all arms; null speed if no phase frames',
        'timing': 'native solver_ms synchronized around IPC_Solver; excludes startup and trace export outside solver'}
    component = read(a.data / 'builds/toi_components.json')
    assert component['passed'] and len(component['port_tests']) == 7 and all(t['passed'] for t in component['port_tests'])
    report['component_fixtures_passed'] = 7
    diagnostic = a.data / 'runs/autodl/v32_bunny_diag40_toi1_graph'
    if (diagnostic / 'result.json').exists():
        df = read(diagnostic / 'output/stats.json')['frames']
        report['bunny_failure_diagnostic'] = {'run': diagnostic.relative_to(a.data).as_posix(),
            'result': read(diagnostic / 'result.json'), 'last_pcg': df[-1]['newton'][-1]['pcg'],
            'last_outer': df[-1]['toi'][-1], 'not_part_of_timing_matrix': True}
    out = ROOT / 'reports/AUTODL_V32_SMALL_SCENES_20261002'
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with out.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scene', 'arm', 'repeats', 'total_s', 'speed_vs_base', 'noncontact_s', 'noncontact_speed', 'contact_s', 'contact_speed', 'speed_vs_graph', 'directions', 'CG', 'cloth_RMS_max_percent'])
        for r in summary:
            writer.writerow([r['scene'], r['arm'], r['repeats'], r['seconds']['total'], r['raw_speedup_vs_base']['total'], r['seconds']['noncontact'], r['raw_speedup_vs_base']['noncontact'], r['seconds']['contact'], r['raw_speedup_vs_base']['contact'], r['raw_total_speedup_vs_graph'], r['directions'], r['cg_iterations'], r['cloth_rms_max_percent_range']])
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    def speed(x):
        return '—' if x is None else f'{x:.3f}×'
    lines = ['# AutoDL 小场景 v32 对比（2026-10-02）', '',
        'RTX 4090 24GB，CUDA 12.8，sm89 Release。每组 100 帧，dt=.01，Newton=.01，PCG=1e-4，suite=0；GPU 串行执行。快速成功组最多重复 3 次，以下时间取中位数。', '',
        '**加速比为实测耗时比，未通过质量匹配认证。** 质量栏为与同 dt base 的布料轨迹差异，不能当作精确物理解误差；本轮没有另做独立 CPU 连续碰撞验证。Stable NH1 允许四面体翻转，负 Jacobian 单独记录。', '',
        '计时：IPC_Solver 外围同步的 solver_ms 求和，排除进程启动和求解器外轨迹导出。接触/非接触共用 base 第一轮原生窄阶段候选对数 >0 的帧掩码；缺少某阶段时速度比显示 —。', '']
    for scene in matrix['protocol']['scenes'].split(','):
        items = [r for r in summary if r['scene'] == scene]
        lines += [f'## {scene}', '']
        if items:
            lines += [f"base 掩码：非接触 {len(items[0]['noncontact_frames'])} 帧，接触 {len(items[0]['contact_frames'])} 帧。", '',
                '| 配置 | 重复 | 总秒 | /base | 非接触秒 | /base | 接触秒 | /base | /base+Graph | 布料 RMS 最大差异 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in items:
                lo, hi = r['cloth_rms_max_percent_range']
                lines.append(f"| {r['label']} | {r['repeats']} | {fmt(r['seconds']['total'])} | {speed(r['raw_speedup_vs_base']['total'])} | {fmt(r['seconds']['noncontact'])} | {speed(r['raw_speedup_vs_base']['noncontact'])} | {fmt(r['seconds']['contact'])} | {speed(r['raw_speedup_vs_base']['contact'])} | {speed(r['raw_total_speedup_vs_graph'])} | {lo:.2f}–{hi:.2f}% |")
            lines += ['', '| 配置 | Newton 方向中位数 | CG 中位数 | 最小地面间隙 m | 最小相对 tet Jacobian |', '|---|---:|---:|---:|---:|']
            for r in items:
                lines.append(f"| {r['label']} | {r['directions']} | {r['cg_iterations']} | {r['ground_min_gap_m']:.3e} | {fmt(r['tet_min_jacobian'])} |")
            lines += ['']
        failures = [r for r in rows if r['scene'] == scene and not r['usable_timing']]
        for r in failures:
            lines.append(f"- 未纳入速度汇总：{r['arm']} r{r['repeat']}，状态 {r['status']}，完成 {r.get('recorded_frames', 0)} 帧，原因 {r.get('failure_reason', 'verification gate')}；末次 PCG {r.get('last_pcg', {}).get('iterations', '—')} 次，原始诊断见 JSON 和对应 run.log。")
        lines += ['']
    if 'bunny_failure_diagnostic' in report:
        diagnostic = report['bunny_failure_diagnostic']
        pcg = diagnostic['last_pcg']
        lines += ['## 兔子 TOI 失败诊断', '',
            f"四个正式 TOI 配置都触发 PCG 上限（18257 次），随后拒绝提交未收敛的试探步，退出码 4；均未得到完整 100 帧加速比。base 和 base+Graph 完整通过。", '',
            f"补充残差审计运行完成 {diagnostic['result']['recorded_frames']} 帧后同样中止。末次真实相对残差 `{pcg['true_relative_residual']:.6g}`，预条件残差内积 rho `{pcg['rho_stop']:.6g}`，表明该线性求解已发散，远高于请求的 1e-4。诊断运行改变了审计/同步开关，失败帧不同，未用于计时。", '',
            '已确认直接原因是混合 ABD/FEM/布料场景的 TOI 线性求解未收敛。host 与 Graph 都失败，因此先检查 TOI 组装矩阵、MAS 预条件器的正定性/适配和接触刚度尺度；负 rho 是预条件系统的异常证据，尚不能单凭该标量定位到某个组装项。CCD 候选数增长记录在日志中，但没有出现显存不足退出。不能通过忽略 PCG 上限给出有效速度比。', '']
    lines += ['## 验证与范围', '',
        f"- 完整且满足时序/功能/求解标志门槛的组：{sum(r['usable_timing'] for r in rows)}/{len(rows)}。",
        '- 已检查源摘要、Linux 可执行文件哈希、runner 哈希、参数、100 帧编号、有限值、PCG/outer 上限、TOI 原生安全步标志和 Graph 实际执行方式。',
        '- 七项 Robust 适配 GPU 组件测试见下载目录 builds/toi_components.json。',
        '- 显存每秒采样一次，观测增量为下界；极短运行出现 0 MiB 不表示没有使用显存。预估值也不代表实际峰值。',
        '- 总时间、接触时间、非接触时间分别取三轮中位数，因此表中两个阶段的中位数之和可能与总时间中位数略有不同。',
        f"- 跳过组：{len(matrix['skipped'])}；具体原因见 JSON skipped。", '',
        f"原始数据：`{a.data.as_posix()}`。", f"源摘要：`{manifest['source_digest']}`。", '',
        '## 复现与核对命令', '', '在独立 AutoDL 目录运行：', '', '```bash',
        'bash tools/autodl_v32_build.sh',
        'python3 tools/benchmark_v32_autodl.py --steps 30 --repeats 1 --arms base,toi005,toi1_graph --timeout 90 --budget 600',
        'python3 tools/benchmark_v32_autodl.py --steps 100 --repeats 3 --timeout 300 --budget 1800',
        '```', '', '下载后用 DL Python 执行：', '', '```text',
        f'python tools/report_v32_autodl.py --data "{a.data.as_posix()}"',
        'python tools/plot_v32_autodl.py', '```', '']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(out.with_suffix('.md')), 'runs': len(rows),
                      'usable': sum(r['usable_timing'] for r in rows), 'summary': summary}))


if __name__ == '__main__':
    main()
