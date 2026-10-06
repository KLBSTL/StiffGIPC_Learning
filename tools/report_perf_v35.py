"""Verify the AutoDL batching/refit configuration comparison on frozen code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from report_local_perf_v33 import work
from report_v32_autodl import quality

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base-data', type=Path, required=True)
    parser.add_argument('--profile-data', type=Path)
    args = parser.parse_args()
    data, base_data = args.data.resolve(), args.base_data.resolve()
    matrix = read(data / 'reports/AUTODL_PERF_V35_MATRIX.json')
    base_matrix = read(base_data / 'reports/AUTODL_PERF_V34_MATRIX.json')
    bases = {r['repeat']: r for r in base_matrix['runs'] if r['label'] == 'base'}
    assert len(matrix['runs']) == 6 and len(bases) == 3
    base_times = {i: list(csv.DictReader((base_data / r['run'] / 'trace/frames.csv').open())) for i, r in bases.items()}
    mask = [int(t['candidate_pairs']) + int(t['ground_candidates']) > 0 for t in base_times[1]]
    seconds = lambda times: {p: sum(float(t['solver_ms']) for t, m in zip(times, mask) if p == 'total' or m == (p == 'contact')) / 1000 for p in ['total', 'noncontact', 'contact']}
    base_seconds = {p: statistics.median(seconds(t)[p] for t in base_times.values()) for p in ['total', 'noncontact', 'contact']}
    frozen = read(data / 'manifests/perf_v34_cuda128_autodl.json')
    runner_sha = hashlib.sha256((data / 'tools/run_perf_v34.py').read_bytes()).hexdigest()
    rows, keyed = [], {}
    for r in matrix['runs']:
        run = data / r['run']
        req, result = read(run / 'requested.json'), read(run / 'result.json')
        assert req['source_digest'] == frozen['source_digest']
        assert req['exe_sha256'] == frozen['binaries']['builds/autodl-stiff_perf_v34_cuda128/gipc']['sha256']
        assert req['runner_sha256'] == r['runner_sha256'] == runner_sha
        suite = '1' if r['label'] == 'suite_on' else '0'
        for k, v in {'steps': 100, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                     'robust_velocity_tol': .05, 'preconditioner': 'diag',
                     'fused_diag_update': '0', 'suite': suite, 'platform': 'autodl',
                     'profile': False, 'audit_graph': False, 'audit_pcg': False,
                     'audit_suite': False, 'substeps': False, 'quality_only': False}.items():
            assert req[k] == v, (run, k)
        frames, counts = work(run)
        assert result['status'] == 'completed' and result['finite'] and len(frames) == 100
        assert result['robust_frame_protocol_verified']
        assert counts['execution'] == {'conditional_graph': counts['directions']}
        assert not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
        assert all(not n['pcg']['fused_diag_update'] for f in frames for n in f['newton'] if 'pcg' in n)
        times = list(csv.DictReader((run / 'trace/frames.csv').open()))
        assert len(times) == 100 and [int(t['frame']) for t in times] == list(range(1, 101))
        measured = seconds(times)
        assert abs(measured['total'] - result['solver_seconds']) < 1e-8
        row = {**r, 'requested': req, 'work': counts, 'seconds': measured}
        rows.append(row); keyed[(r['label'], r['repeat'])] = row
    for r in rows:
        r['quality_vs_suite_off'] = quality(data / r['run'], data / keyed[('suite_off', r['repeat'])]['run'])
        r['quality_vs_base'] = quality(data / r['run'], base_data / bases[r['repeat']]['run'])
        r['same_arm_repeat_quality'] = quality(data / r['run'], data / keyed[(r['label'], 1)]['run'])
        assert r['quality_vs_base']['all_frames_finite'] and r['quality_vs_base']['fixed_abd_mask_verified']
    summary = []
    off_seconds = {p: statistics.median(r['seconds'][p] for r in rows if r['label'] == 'suite_off') for p in base_seconds}
    for label in ['suite_off', 'suite_on']:
        selected = [r for r in rows if r['label'] == label]
        measured = {p: statistics.median(r['seconds'][p] for r in selected) for p in base_seconds}
        totals = [r['seconds']['total'] for r in selected]
        summary.append({'label': label, 'seconds': measured, 'range_seconds': [min(totals), max(totals)],
            'cv_percent': statistics.stdev(totals) / statistics.mean(totals) * 100,
            'speed_vs_suite_off': {p: off_seconds[p] / measured[p] if measured[p] else None for p in measured},
            'speed_vs_recent_base': {p: base_seconds[p] / measured[p] if measured[p] else None for p in measured},
            'directions': statistics.median(r['work']['directions'] for r in selected),
            'cg_iterations': statistics.median(r['work']['cg_iterations'] for r in selected),
            'max_cloth_rms_vs_off_percent': max(r['quality_vs_suite_off']['max_cloth_mass_rms_vs_base_percent'] for r in selected),
            'max_cloth_rms_vs_base_percent': max(r['quality_vs_base']['max_cloth_mass_rms_vs_base_percent'] for r in selected),
            'same_arm_repeat_max_rms_percent': max(r['same_arm_repeat_quality']['max_cloth_mass_rms_vs_base_percent'] for r in selected),
            'max_stretch': max(r['quality_vs_base']['cloth_max_stretch'] for r in selected),
            'ground_min_gap_m': min(r['quality_vs_base']['ground_min_gap_m'] for r in selected)})
    smoke = read(data / 'reports/AUTODL_PERF_V35_SMOKE.json')['runs'][0]
    assert smoke['status'] == 'completed' and smoke['recorded_frames'] == 100
    frames, counts = work(data / smoke['run'])
    assert all(f['bvh_refit_audit']['identical'] for f in frames)
    max_energy_error = max(f['energy_batch_audit']['max_relative_error'] for f in frames)
    assert max_energy_error <= 1e-10
    assert not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
    export = read(data / 'reports/AUTODL_PERF_V35_PATHS.json')['runs'][0]
    assert export['status'] == 'completed' and export['recorded_frames'] == 100
    run = data / export['run']; frames, counts = work(run)
    for i, frame in enumerate(frames):
        states = sorted((run / 'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
        assert len(states) == 1 + sum('alpha' in t for t in frame['toi'])
        assert states[0].read_bytes() == (run / f'trace/state_{i:04d}.bin').read_bytes()
        assert states[-1].read_bytes() == (run / f'trace/state_{i+1:04d}.bin').read_bytes()
    ccd = read(ROOT / 'reports/autodl_perf_v35_accepted_ccd.json')
    paired = [keyed[('suite_off', i)]['seconds']['total'] / keyed[('suite_on', i)]['seconds']['total'] for i in range(1, 4)]
    profiles = []
    if args.profile_data:
        profile_data = args.profile_data.resolve()
        stored = read(profile_data / 'reports/AUTODL_PERF_V35_PROFILE.json')
        assert len(stored['runs']) == 2
        for r in stored['runs']:
            run = profile_data / r['run']
            req, result = read(run / 'requested.json'), read(run / 'result.json')
            assert req == r['requested'] and result == r['result']
            assert req['exe_sha256'] == frozen['binaries']['builds/autodl-stiff_perf_v34_cuda128/gipc']['sha256']
            assert req['source_digest'] == frozen['source_digest']
            assert req['runner_sha256'] == hashlib.sha256((profile_data / 'tools/run_perf_v34.py').read_bytes()).hexdigest()
            for k, v in {'steps': 100, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                         'robust_velocity_tol': .05, 'preconditioner': 'diag',
                         'fused_diag_update': '0', 'suite': r['suite'], 'profile': True,
                         'audit_suite': False, 'quality_only': True}.items():
                assert req[k] == v
            frames, counts = work(run)
            assert result['status'] == 'completed' and result['finite'] and len(frames) == 100
            assert result['timing_is_diagnostic'] and result['robust_frame_protocol_verified']
            assert not any(counts[k] for k in ['pcg_limit_hits', 'outer_limit_hits', 'unsafe_native_steps', 'unfinished_outer'])
            assert counts['execution'] == {'conditional_graph': counts['directions']}
            assert r['directions'] == counts['directions'] == r['profiled_directions']
            assert r['cg_iterations'] == counts['cg_iterations'] and r['accepted_outer'] == counts['accepted_outer']
            raw = {**counts['newton_profile_ms'], **{k: v for k, v in counts['outer_profile_ms'].items() if k != 'outer_total'}}
            assert all(abs(r['profile_seconds'][k] - v / 1000) < 1e-9 for k, v in raw.items())
            phases = {k: r['profile_seconds'][k] for k in ['assembly', 'pcg', 'line_search', 'safe_update']}
            phases['full_ccd_and_active_update'] = r['profile_seconds']['active_update'] + r['profile_seconds']['safe_ccd']
            total = sum(phases.values())
            profiles.append({**r, 'nonoverlapping_sections_seconds': phases,
                             'section_shares_percent': {k: v / total * 100 for k, v in phases.items()},
                             'line_search_ms_per_direction': phases['line_search'] / r['directions'] * 1000,
                             'pcg_microseconds_per_iteration': phases['pcg'] / r['cg_iterations'] * 1e6})
    report = {'hardware': 'RTX 4090 24 GiB; CUDA 12.8; Release sm89', 'protocol': matrix['protocol'],
              'summary': summary, 'runs': rows, 'paired_suite_speedup': paired,
              'energy_max_relative_error': max_energy_error, 'refit_candidate_audited_frames': 100,
              'accepted_path_audit': {'result': ccd, 'run': export['run'], 'native_work': counts,
                                      'frame_bridges_and_endpoints_verified': True},
              'quality_matched_speed_qualified': False,
              'phase_frames': {'noncontact': mask.count(False), 'contact': mask.count(True)},
              'recent_base_source': str(base_data),
              'diagnostic_profiles': profiles,
              'decision': {'all_three_pairs_faster': min(paired) > 1,
                           'stable_gain_certified': False,
                           'scene_config_only': True, 'global_default_unchanged': True}}
    out = ROOT / 'reports/AUTODL_PERF_V35_20261003'
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# AutoDL v35：能量批处理与 BVH refit 组合', '',
             '使用冻结的 v34 CUDA 12.8 二进制，没有新增数值内核。布料–球小场景，100 帧 × 3，suite=0/1 交错运行；固定 dt=.01、Newton=.01、PCG rho 比 1e-4、velocity_tol=.05、diag + conditional Graph、融合关闭。', '',
             'AutoDL 工作目录：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v34_20261003`。完整结果已下载到本任务的 `downloads/autodl_perf_v35_20261003` 与 `downloads/autodl_perf_v35_profile_20261003`，两份压缩包 SHA256 校验通过。', '',
             'suite=1 在该配置中启用能量批处理和 CCD BVH refit。IPC 能量复用只用于 IPC 分支，本轮 TOI 不使用；MAS 静态拓扑不适用于本轮对角预条件器。', '',
             '| 配置 | 总秒中位数 | 非接触秒 | 接触秒 | /suite off | /近期 base | CV | 方向 | CG |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['label']} | {r['seconds']['total']:.4f} | {r['seconds']['noncontact']:.4f} | {r['seconds']['contact']:.4f} | {r['speed_vs_suite_off']['total']:.3f}× | {r['speed_vs_recent_base']['total']:.3f}× | {r['cv_percent']:.2f}% | {r['directions']} | {r['cg_iterations']} |")
    lines += ['', '三轮成对新增速度比：' + '、'.join(f'{v:.3f}×' for v in paired) + '。', '',
              '| 配置 | 非接触 /suite off | 接触 /suite off | 最大 RMS/off | 最大 RMS/base | 同配置重复 RMS | 最大边拉伸 | 最小地面间隙 m |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['label']} | {r['speed_vs_suite_off']['noncontact']:.3f}× | {r['speed_vs_suite_off']['contact']:.3f}× | {r['max_cloth_rms_vs_off_percent']:.2f}% | {r['max_cloth_rms_vs_base_percent']:.2f}% | {r['same_arm_repeat_max_rms_percent']:.2f}% | {r['max_stretch']:.5f} | {r['ground_min_gap_m']:.3e} |")
    lines += ['', '## 验证与范围', '',
              f'- 额外 100 帧逐帧 BVH refit/rebuild 第一次候选集合完全一致；所有能量查询批处理/原路径最大相对差 {max_energy_error:.3e}（门槛 1e-10）。',
              '- 正式六组均完成 100 帧，参数、来源/二进制/runner 哈希、Graph 实际执行和计时总和核对；无 PCG/外层上限、非有限值或原生不安全接受步。',
              f"- 额外 100 帧完整接受路径：{ccd['paths_checked']} 段独立 Tight-Inclusion 检查，{ccd['conservative_collision_flags']} 个保守标记，通过={ccd['passed']}。每帧起点/终点与 trace 逐字节核对。", '',
              '阶段按 v34 base r01 候选数共同掩码划分；候选存在不等于实际受力。总/阶段分别取中位数。base 来自同实例刚完成的 v34 三轮矩阵，未参与本次 suite 配对，因此 /base 是近期原始耗时对照。', '',
              'RMS 是同时间步轨迹差，不能视为对共同收敛物理解的误差；正式质量匹配速度资格仍为 false。组合只在本小场景评估，全局默认不变。Graph 融合开关仍为 0。v34 同系统审计的最大 Graph/host 相对解差 9.745e-5、真残差 0.107；严格 1e-6 同解门槛和完整物理参考仍未验收。', '',
              '## 采用结论', '',
              f"总时间中位数比为 {summary[1]['speed_vs_suite_off']['total']:.3f}×，成对比中位数为 {statistics.median(paired):.3f}×。三轮均略快，但幅度仅 1.3%–3.7%，与跨运行方向/CG 数变化和计时 CV 同量级。因此记录为小场景的正向候选配置，尚不认证稳定、质量匹配的通用收益；不改全局默认。当前组合没有减少 CG 总数，收益主要应从每次能量查询与 BVH 更新成本解释。", '']
    if profiles:
        lines += ['## 诊断分项计时', '',
                  '每配置额外单次 100 帧，开启同步分项计时；本表只定位成本，不能代替正式三轮速度比。不同运行方向/CG 数不同。', '',
                  '| 配置 | 组装秒 | PCG 秒 | 线搜索秒 | 完整 CCD+活动集秒 | 安全更新秒 | CG | 线搜索 ms/方向 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in profiles:
            p = r['nonoverlapping_sections_seconds']
            lines.append(f"| suite={r['suite']} | {p['assembly']:.4f} | {p['pcg']:.4f} | {p['line_search']:.4f} | {p['full_ccd_and_active_update']:.4f} | {p['safe_update']:.4f} | {r['cg_iterations']} | {r['line_search_ms_per_direction']:.3f} |")
        on = next(r for r in profiles if r['suite'] == '1')
        shares = on['section_shares_percent']
        lines += ['', f"开启组合后，已计时区段 PCG 占 {shares['pcg']:.1f}%，安全状态更新（含 BVH/候选与原生交叉检查）占 {shares['safe_update']:.1f}%，组装占 {shares['assembly']:.1f}%；线搜索只占 {shares['line_search']:.1f}%。因此只降低能量计算成本，对总时间的影响有限。下一优先项为固定 A/b 下 PCG 及安全更新成本，先分离算子吞吐和跨运行工作量变化，再选择优化。", '',
                  'Robust 分支把完整 CCD 与活动集更新共享在 active_update 区段，safe_ccd 单独字段主要是复用 alpha 的 bookkeeping；不能据其接近零推断 CCD 开销接近零。上表已将二者合并。区段和不含初始组装、帧初始化及其他未计时开销。', '']
    lines += ['## 复现', '', '```text',
              'python3 tools/benchmark_perf_v35.py --phase smoke',
              'python3 tools/benchmark_perf_v35.py --phase matrix',
              'python3 tools/benchmark_perf_v35.py --phase paths',
              'python3 tools/export_perf_v34_autodl.py --version v35',
              'python3 tools/profile_perf_v35.py',
              'E:/Anaconda/envs/DL/python.exe tools/validate_perf_v34.py --platform autodl --version v35 --data downloads/autodl_perf_v35_20261003',
              'E:/Anaconda/envs/DL/python.exe tools/report_perf_v35.py --data downloads/autodl_perf_v35_20261003 --base-data downloads/autodl_perf_v34_20261003 --profile-data downloads/autodl_perf_v35_profile_20261003',
              '```', '', '脚本拒绝覆盖已有记录；重新运行需新输出目录/名称。CPU CCD 精确命令和验证器哈希见 AUTODL_PERF_V35_PATH_VALIDATION.json。', '']
    out.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(out.with_suffix('.md')), 'paired_suite_speedup': paired, 'summary': summary}))


if __name__ == '__main__':
    main()
