"""Verify the fixed-tolerance v34 Graph kernel fusion experiment."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import argparse

from report_v32_autodl import quality
from report_local_perf_v33 import work

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT)
    parser.add_argument('--platform', choices=['local', 'autodl'], default='local')
    args = parser.parse_args()
    data = args.data.resolve()
    prefix = 'AUTODL' if args.platform == 'autodl' else 'LOCAL'
    matrix = read(data / f'reports/{prefix}_PERF_V34_MATRIX.json')
    assert len(matrix['runs']) == 12
    fixture = read(data / f'builds/{args.platform}_perf_v34_fixture.json')
    assert fixture['passed'] and fixture['fixtures'] == 28
    rows, keyed = [], {}
    for record in matrix['runs']:
        run = data / record['run']
        req, result = read(run / 'requested.json'), read(run / 'result.json')
        frozen = read(data / 'manifests' / Path(req['manifest']).name)
        assert req['source_digest'] == frozen['source_digest']
        assert req['exe_sha256'] in {b['sha256'] for b in frozen['binaries'].values()}
        assert req['runner_sha256'] == record['runner_sha256']
        runner = 'run_perf_v34.py' if record['label'].startswith('v34') else 'run_robust_port.py'
        runner_candidates = [data / 'tools' / runner]
        if args.platform == 'local' and runner == 'run_perf_v34.py':
            runner_candidates.append(data / 'tools/run_perf_v34_local_frozen.py')
        assert req['runner_sha256'] in {hashlib.sha256(p.read_bytes()).hexdigest() for p in runner_candidates if p.is_file()}
        for key, value in {'steps': 100, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                           'suite': '0', 'profile': False, 'audit_pcg': False,
                           'audit_graph': False, 'substeps': False}.items():
            assert req[key] == value
        if args.platform == 'autodl':
            assert not req['quality_only'] and not req['audit_suite']
        frames, counts = work(run)
        if record['label'] != 'base':
            assert req['preconditioner'] == 'diag' and req['robust_velocity_tol'] == .05
            assert result['robust_frame_protocol_verified']
            assert counts['execution'] == {'conditional_graph': counts['directions']}
            assert all(f['toi_policy'] == 'robust' and f['toi_frame_friction_snapshot']['physical_frame'] == i
                       for i, f in enumerate(frames))
        if record['label'].startswith('v34'):
            expected = record['label'] == 'v34_on'
            assert req['fused_diag_update'] == str(int(expected))
            assert all(n['pcg']['fused_diag_update'] == expected for f in frames for n in f['newton'] if 'pcg' in n)
        assert result['status'] == 'completed' and result['finite'] and len(frames) == 100
        assert not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
        times = list(csv.DictReader((run / 'trace/frames.csv').open()))
        assert len(times) == 100 and [int(t['frame']) for t in times] == list(range(1, 101))
        assert abs(sum(float(t['solver_ms']) for t in times) / 1000 - result['solver_seconds']) < 1e-8
        row = {**record, 'requested': req, 'work': counts, 'times': times}
        rows.append(row); keyed[(record['label'], record['repeat'])] = row
    mask = [int(t['candidate_pairs']) + int(t['ground_candidates']) > 0 for t in keyed[('base', 1)]['times']]
    for row in rows:
        row['seconds'] = {phase: sum(float(t['solver_ms']) for t, m in zip(row['times'], mask) if phase == 'total' or m == (phase == 'contact')) / 1000
                          for phase in ['total', 'noncontact', 'contact']}
        row['quality_vs_base'] = quality(data / row['run'], data / keyed[('base', row['repeat'])]['run'])
        row['quality_vs_v32_diag'] = quality(data / row['run'], data / keyed[('v32_diag', row['repeat'])]['run'])
        row['quality_vs_v34_off'] = quality(data / row['run'], data / keyed[('v34_off', row['repeat'])]['run'])
        row['quality_vs_same_arm_r01'] = quality(data / row['run'], data / keyed[(row['label'], 1)]['run'])
        assert row['quality_vs_base']['all_frames_finite'] and row['quality_vs_base']['fixed_abd_mask_verified']
        row.pop('times')
    summary = []
    for label in ['base', 'v32_diag', 'v34_off', 'v34_on']:
        selected = [r for r in rows if r['label'] == label]
        totals = [r['seconds']['total'] for r in selected]
        seconds = {p: statistics.median(r['seconds'][p] for r in selected) for p in ['total', 'noncontact', 'contact']}
        baseline = {l: statistics.median(r['seconds']['total'] for r in rows if r['label'] == l) for l in ['base', 'v32_diag', 'v34_off']}
        phase_speed_vs = {l: {p: statistics.median(r['seconds'][p] for r in rows if r['label'] == l) / seconds[p]
                             if seconds[p] else None for p in seconds} for l in baseline}
        summary.append({'label': label, 'seconds': seconds, 'range_seconds': [min(totals), max(totals)],
            'cv_percent': statistics.stdev(totals) / statistics.mean(totals) * 100,
            'speed_vs': {l: value / seconds['total'] for l, value in baseline.items()},
            'phase_speed_vs': phase_speed_vs,
            'directions': statistics.median(r['work']['directions'] for r in selected),
            'cg_iterations': statistics.median(r['work']['cg_iterations'] for r in selected),
            'quality_vs_base_max_percent': max(r['quality_vs_base']['max_cloth_mass_rms_vs_base_percent'] for r in selected),
            'quality_vs_v34_off_max_percent': max(r['quality_vs_v34_off']['max_cloth_mass_rms_vs_base_percent'] for r in selected),
            'same_arm_repeat_max_rms_percent': max(r['quality_vs_same_arm_r01']['max_cloth_mass_rms_vs_base_percent'] for r in selected),
            'ground_min_gap_m': min(r['quality_vs_base']['ground_min_gap_m'] for r in selected),
            'cloth_max_stretch': max(r['quality_vs_base']['cloth_max_stretch'] for r in selected)})
    paired = [keyed[('v34_off', i)]['seconds']['total'] / keyed[('v34_on', i)]['seconds']['total'] for i in range(1, 4)]
    smoke = read(data / f'reports/{prefix}_PERF_V34_SMOKE.json')
    audits = []
    for record in smoke['runs']:
        assert record['status'] == 'completed'
        frames, counts = work(data / record['run'])
        directions = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
        assert directions and counts['execution'] == {'conditional_graph': len(directions)}
        assert not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
        if record['label'] == 'mas_fallback':
            assert not any(n['fused_diag_update'] for n in directions)
        else:
            assert all(n['fused_diag_update'] for n in directions)
            assert all('same_system_relative_solution_difference' in n and 'true_relative_residual' in n for n in directions)
        audits.append({**record, 'work': counts,
                       'max_host_graph_solution_relative_difference': max((n['same_system_relative_solution_difference'] for n in directions if 'same_system_relative_solution_difference' in n), default=None),
                       'max_host_repeat_solution_relative_difference': max((n['same_system_host_repeat_relative_difference'] for n in directions if 'same_system_host_repeat_relative_difference' in n), default=None)})
    path = ROOT / f'reports/{args.platform}_perf_v34_accepted_ccd.json' if args.platform == 'autodl' else ROOT / 'reports/local_perf_v34_accepted_ccd.json'
    paths = read(path) if path.exists() else None
    if paths:
        export = read(data / f'reports/{prefix}_PERF_V34_PATHS.json')['runs'][0]
        run = data / export['run']; frames, counts = work(run)
        for i, frame in enumerate(frames):
            states = sorted((run / 'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
            assert len(states) == 1 + sum('alpha' in t for t in frame['toi'])
            assert states[0].read_bytes() == (run / f'trace/state_{i:04d}.bin').read_bytes()
            assert states[-1].read_bytes() == (run / f'trace/state_{i+1:04d}.bin').read_bytes()
        paths = {'result': paths, 'run': export['run'], 'native_work': counts,
                 'frame_bridges_and_endpoints_verified': True}
    hardware = 'RTX 4090 24 GiB; CUDA 12.8; Release sm89' if args.platform == 'autodl' else 'RTX 3070 Laptop 8 GiB; CUDA 13.0; Release sm86'
    adoption = {'default_fused_diag_update': '0',
                'stable_fusion_gain_demonstrated': min(paired) > 1,
                'paired_median_speedup': statistics.median(paired),
                'reason': 'Fusion remains experimental; mixed paired timings do not establish a reliable improvement.'}
    report = {'hardware': hardware, 'protocol': matrix['protocol'],
              'phase_frames': {'noncontact': mask.count(False), 'contact': mask.count(True)},
              'fixture': fixture, 'summary': summary, 'runs': rows, 'smoke_audits': audits,
              'paired_fusion_speedup': paired, 'accepted_path_audit': paths,
              'adoption': adoption,
              'quality_matched_speed_qualified': False}
    out = ROOT / f'reports/{prefix}_PERF_V34_20261003'
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = [f'# {prefix} v34：Graph PCG 更新与块对角预条件融合', '',
        f'2026-10-03，{hardware}。布料–球小场景，每组 100 帧 × 3，交错执行；dt=.01，Newton=.01，PCG rho 比 1e-4，TOI velocity_tol=.05 m/s，suite=0。', '',
        '新增可选融合 kernel：同时更新 x/r 并应用全局 3×3 块对角预条件器；随后仍按原顺序应用局部 ABD 12×12 预条件器。Graph cache key 包含实际融合状态。MAS 自动回退原路径。没有修改收敛阈值、TOI、摩擦、CCD、安全检查或矩阵。', '',
        ('源代码独立保存在 sources/stiff_perf_v34_cuda128，Release 构建 builds/autodl-stiff_perf_v34_cuda128，冻结身份 manifests/perf_v34_cuda128_autodl.json。CUDA 12.8 修正仅是独立 fixture 的显式 C++ 数组类型；运行时数值源与 Windows v34 一致。base/v32 使用此前已验证的 Linux 二进制并复核哈希。'
         if args.platform == 'autodl' else '源代码独立保存在 sources/stiff_perf_v34，Release 构建 builds/local-fused-v34，冻结身份 manifests/perf_v34.json；base/v32 保留。'), '',
        '| 配置 | 总秒中位数 | /base | /v32 diag | /v34 关闭融合 | CV | Newton 方向 | CG | 最大 RMS/base |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['label']} | {r['seconds']['total']:.4f} | {r['speed_vs']['base']:.3f}× | {r['speed_vs']['v32_diag']:.3f}× | {r['speed_vs']['v34_off']:.3f}× | {r['cv_percent']:.2f}% | {r['directions']} | {r['cg_iterations']} | {r['quality_vs_base_max_percent']:.2f}% |")
    if args.platform == 'autodl':
        lines += ['', 'AutoDL 工作目录：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v34_20261003`。完整结果已下载到本任务的 `downloads/autodl_perf_v34_20261003`，压缩包 SHA256 校验通过。']
    lines += ['', '同一 v34 二进制融合开关的三轮成对速度比：' + '、'.join(f'{x:.3f}×' for x in paired) + '。', '',
        '| 配置 | 非接触秒 | 接触秒 | 最小地面间隙 m | 最大边拉伸 | 最大 RMS/v34 off |', '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['label']} | {r['seconds']['noncontact']:.4f} | {r['seconds']['contact']:.4f} | {r['ground_min_gap_m']:.3e} | {r['cloth_max_stretch']:.5f} | {r['quality_vs_v34_off_max_percent']:.2f}% |")
    lines += ['', '| 配置 | 非接触 /base | 接触 /base | 非接触 /v34 off | 接触 /v34 off | 同配置三次最大布料 RMS 差 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        b, o = r['phase_speed_vs']['base'], r['phase_speed_vs']['v34_off']
        lines.append(f"| {r['label']} | {b['noncontact']:.3f}× | {b['contact']:.3f}× | {o['noncontact']:.3f}× | {o['contact']:.3f}× | {r['same_arm_repeat_max_rms_percent']:.2f}% |")
    lines += ['', f"阶段掩码采用当前 base r01 原生窄阶段候选数 >0：非接触 {mask.count(False)} 帧，接触候选 {mask.count(True)} 帧，各配置使用相同帧掩码。候选存在不等于实际施力。分别取中位数，因此阶段和可能与总中位数略有差异。", '',
        '## 数值与安全验证', '',
        f"- GPU 融合 fixture：{fixture['fixtures']} 组，{fixture['bitwise_compared_values']} 个 x/r/z 值与原两 kernel 逐位一致；CPU 参考最大相对差异 {fixture['max_cpu_relative_error']:.3e}。包含非对角块、奇数尺寸、256 边界、多种 alpha 与越界哨兵。",
        '- 实际场景：30 帧融合 Graph/host 同一矩阵解审计及真实残差审计；10 帧 MAS 回退；正式 12/12 组完成，无求解上限、非有限值或原生不安全接受步。',
        f"- 30 帧固定系统审计记录：Graph/host 最大相对解差 {audits[0]['max_host_graph_solution_relative_difference']:.3e}，host 自身重复差 {audits[0]['max_host_repeat_solution_relative_difference']:.3e}，最大真相对残差 {audits[0]['work']['maximum_true_relative_residual']:.3e}。严格 1e-6 同解门槛仍未通过；原生 PCG rho 比停止与真残差不是同一指标，本轮没有放宽停止参数。",
        f"- 独立接受路径 CCD：{'通过，'+str(paths['result']['paths_checked'])+' 段，'+str(paths['result']['conservative_collision_flags'])+' 标志' if paths and paths['result']['passed'] else '待运行或未通过'}。逐帧桥接和终点核对；该检查覆盖额外导出运行。", '',
        '计时只统计原生 solver_ms，排除进程启动和求解器外轨迹导出。GPU 测试串行运行，保留背景占用/温度/频率样本。' + ('本轮背景占用持续约 34–41%，使用 --background-timing，所有运行均标记 quality-only；耗时只作共享负载下的诊断结果，不能认证几%的性能改善。' if matrix['protocol'].get('background_timing') else '每组运行前检查 GPU 利用率不高于 30%，并由 runner 拒绝与其他 CUDA 进程重叠。') + '实测耗时比不代表收敛物理解误差一致，质量匹配标志仍为 false。fixture 的逐位一致证明局部运算保持一致；整段仿真还受浮点并行规约顺序影响。', '',
        '## 本轮采用决策', '',
        f"成对速度比中位数为 {statistics.median(paired):.3f}×。默认继续使用 --fused-diag-update 0；只有三轮全部加速且质量验证通过时，才有依据考虑开启。当前小场景的数据不支持稳定融合收益。相对 base 的速度来自此前 TOI、Graph 与预条件器组合，不能算成此次 kernel 融合的新增收益。", '',
        '## 复现命令', '', '```text']
    if args.platform == 'autodl':
        lines += ['bash tools/autodl_v34_build.sh',
                  'python3 tools/benchmark_perf_v34.py --platform autodl --phase smoke',
                  'python3 tools/benchmark_perf_v34.py --platform autodl --phase matrix',
                  'python3 tools/benchmark_perf_v34.py --platform autodl --phase paths',
                  'python3 tools/export_perf_v34_autodl.py --version v34',
                  'E:/Anaconda/envs/DL/python.exe tools/validate_perf_v34.py --platform autodl --version v34 --data downloads/autodl_perf_v34_20261003',
                  'E:/Anaconda/envs/DL/python.exe tools/report_perf_v34.py --platform autodl --data downloads/autodl_perf_v34_20261003']
    else:
        lines += ['cmake --build builds/local-fused-v34 --config Release --target diag_fused_update_tests gipc -j 4',
                  'builds/local-fused-v34/Release/diag_fused_update_tests.exe',
                  'E:/Anaconda/envs/DL/python.exe tools/freeze_perf_v34.py',
                  'E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase smoke',
                  'E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase matrix',
                  'E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase paths',
                  'E:/Anaconda/envs/DL/python.exe tools/report_perf_v34.py']
    lines += ['```', '',
        '复现实验需要新输出名称；脚本拒绝覆盖已有记录。独立 CCD 的精确命令保存在报告补充中。', '']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(out.with_suffix('.md')), 'paired_fusion_speedup': paired,
                      'summary': [{k: r[k] for k in ['label', 'seconds', 'speed_vs']} for r in summary]}))


if __name__ == '__main__':
    main()
