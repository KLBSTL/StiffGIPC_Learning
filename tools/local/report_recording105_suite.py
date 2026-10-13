"""Measured tables, trajectory plots, and actual depth-tested simulation views."""
import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('tools/local', 'tools/paper_suite', 'tools/bench')]
from run_recording105_suite import SELECTED, ARMS, LABELS
from render import geometry, positions, export_obj
from render_combination_suite import bounds
from render_combination_depth import DepthRenderer
from config import read
from linux_runner import require, sha, write_new
from local_identity import record


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def tables(output, summary):
    spec = importlib.util.spec_from_file_location('table_format', ROOT / 'runs/autodl_20261011/format_tables.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = output / 'tables'
    folder.mkdir(exist_ok=False)
    headers = ['State', 'Outer rule', 'N', 'P', 't (s)', r'$e_x^{RMS}$ (mm)', r'$e_x^{max}$ (mm)']
    widths = [1.6, 3.15, .45, .85, .9, 1.5, 1.5]
    for kind, data in (('checkpoint', summary['matched100']), ('trajectory', summary['trajectory100'])):
        for scene in (None,) + SELECTED:
            selected = [r for r in data if scene is None or r['scene'] == scene]
            rows = [[r['state'], LABELS[r['method']], str(r['N']), str(r['P']), f'{r["t"]:.6f}',
                     f'{r["e_RMS_mm"]:.6f}', f'{r["e_max_mm"]:.6f}'] for r in selected]
            name = f'{kind}_frame100' + (f'_{scene.lower()}' if scene else '_all')
            caption = ('Identical cold physical input before step 100; all byte gates passed.' if kind == 'checkpoint'
                       else 'Actual step 100 from separate 105-step trajectories; same initial inputs, different preceding paths.')
            caption += '\nGraph, dt=.04, PCG rho=1e-4; medians over 3 rounds. Errors: worst of 9 pairings vs movement-only rho=1e-6.\nNumerical reference, not exact solution; all vertices; no physical-quality certification.'
            module.plot(folder, name, headers, rows, widths, group_key=0, caption=caption)
            with (folder / (name + '.csv')).open('x', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
                writer.writeheader()
                writer.writerows(selected)
            latex = [r'\begin{tabular}{llrrrrr}', r'\hline',
                     r'State & Outer rule & $N$ & $P$ & $t$ (s) & $e_x^{RMS}$ (mm) & $e_x^{max}$ (mm) \\', r'\hline']
            latex.extend(' & '.join(row) + r' \\' for row in rows)
            latex.extend([r'\hline', r'\end{tabular}'])
            (folder / (name + '.tex')).write_text('\n'.join(latex) + '\n', encoding='utf8')
    phase_headers = ['Scene', 'Outer rule', 'Hess', 'LSolver', 'CCD', 'LineS', 'Misc', 'TimeTot', r'$\bar P$', r'$\bar N$', r'$P_{tot}$', 'S']
    phase_rows = [[r['scene'], LABELS[r['method']], *[f'{r[k]:.3f}' for k in ('assembly', 'pcg', 'ccd', 'line_search', 'Misc', 'TimeTot')],
                   f'{r["P_bar"]:.2f}', f'{r["N_bar"]:.2f}', str(r['P']), f'{r["S_movement_over_rule"]:.3f}'] for r in summary['phases_first100']]
    module.plot(folder, 'phases_first100_all', phase_headers, phase_rows,
                [1.05, 3.15, .8, .9, .8, .8, .8, 1.0, .7, .7, .9, .7], group_key=0,
                caption='First 100 physical steps; dt=.04, PCG rho=1e-4, complete assembly including terminal.\nSeconds / counts: medians of three runs; column medians need not sum exactly. S = movement median / rule median.\nMisc includes state update and the remaining solver envelope; excludes trace export / initialization / final stats write.')


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = output / 'plots'
    folder.mkdir(exist_ok=False)
    frames = csv_rows(output / 'per_frame.csv')
    differences = csv_rows(output / 'trajectory_errors.csv')
    worst_error = {}
    for row in differences:
        if row['partition'] == 'all':
            key = row['scene'], row['method'], int(row['frame'])
            worst_error[key] = max(worst_error.get(key, 0), float(row['position_rms_mm']))
    reference_error = {}
    for row in csv_rows(output / 'reference_repeat_dispersion.csv'):
        if row['partition'] == 'all':
            key = row['scene'], int(row['frame'])
            reference_error[key] = max(reference_error.get(key, 0), float(row['position_rms_mm']))
    colors = {'original': '#377eb8', 'pure': '#e78a25', 'ours': '#3e9750'}
    for scene in SELECTED:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        keys = [('solver_ms', 'Solver (ms/step)'), ('assembly_ms', 'Assembly including terminal (ms)'),
                ('pcg_ms', 'Linear solve stage (ms)'), ('N', 'PCG directions N'), ('P', 'Total PCG iterations P')]
        for arm in ARMS[:3]:
            selected = [r for r in frames if r['scene'] == scene and r['method'] == arm]
            for ax, (key, label) in zip(axes.flat, keys):
                values = np.array([[float(next(r[key] for r in selected if int(r['repeat']) == repeat and int(r['physical_frame']) == frame))
                                    for frame in range(1, 106)] for repeat in (1, 2, 3)])
                x = np.arange(1, 106)
                ax.plot(x, np.median(values, axis=0), label=LABELS[arm], color=colors[arm], lw=1.3)
                ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[arm], alpha=.12)
                ax.set_ylabel(label)
            values = [worst_error[scene, arm, frame] for frame in range(1, 106)]
            axes.flat[5].semilogy(np.arange(1, 106), np.maximum(values, 1e-12), color=colors[arm], lw=1.3, label=LABELS[arm])
        axes.flat[5].semilogy(np.arange(1, 106), np.maximum([reference_error[scene, f] for f in range(1, 106)], 1e-12),
                             color='black', ls=':', lw=1.1, label='Reference repeat dispersion')
        axes.flat[5].set_ylabel('Position RMS vs numerical reference (mm)')
        axes.flat[5].legend(fontsize=6.5)
        for ax in axes.flat:
            ax.axvline(100, color='black', ls='--', lw=.8)
            ax.grid(alpha=.2)
            ax.set_xlabel('Physical step')
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{scene}: 105 steps, dt=.04; candidate medians / observed ranges over three runs; errors worst of nine pairs')
        fig.tight_layout()
        fig.savefig(folder / f'{scene.lower()}_performance.png', dpi=150)
        fig.savefig(folder / f'{scene.lower()}_performance.pdf')
        plt.close(fig)


def visuals(session, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = output / 'visuals'
    folder.mkdir(exist_ok=False)
    renderer = DepthRenderer(640)
    evidence, camera = [], []
    keyframes = (0, 25, 50, 75, 100, 105)
    renderer.verify_depth()
    try:
        for repeat in (1, 2, 3):
            data = {}
            for scene in SELECTED:
                states = {}
                for arm in ARMS[:3]:
                    source = session / f'performance_{scene.lower()}_{arm}_r{repeat}'
                    mesh = geometry(source)
                    xs = {f: positions(source, f, mesh['nv']) for f in keyframes}
                    states[arm] = dict(mesh=mesh, x=xs, folder=source)
                    for f, x in xs.items():
                        obj = folder / f'{scene.lower()}_{arm}_r{repeat}_f{f:04d}.obj'
                        export_obj(obj, mesh, x)
                        evidence.append(dict(scene=scene, method=arm, repeat=repeat, physical_frame=f,
                            state=str(source / f'trace/state_{f:04d}.bin'), state_sha256=sha(source / f'trace/state_{f:04d}.bin'),
                            topology_sha256=sha(source / 'trace/topology.bin'), obj=obj.name, obj_sha256=sha(obj)))
                common = bounds([x for arm in states.values() for x in arm['x'].values()])
                data[scene] = dict(states=states, bounds=common)
                camera.append(dict(scene=scene, repeat=repeat, bounds=[b.tolist() for b in common]))
                fig, axes = plt.subplots(3, 6, figsize=(16, 8), facecolor='white')
                for row, arm in enumerate(ARMS[:3]):
                    for col, f in enumerate(keyframes):
                        d = states[arm]
                        axes[row, col].imshow(renderer.mesh(d['mesh'], d['x'][f], scene, common))
                        axes[row, col].axis('off')
                        if row == 0:
                            axes[row, col].set_title(f'Step {f}', fontsize=10)
                        if col == 0:
                            axes[row, col].text(-.04, .5, LABELS[arm], rotation=90, va='center', ha='right', transform=axes[row, col].transAxes, fontsize=8)
                fig.suptitle(f'{scene}, repeat {repeat}: actual native vertices; dt=.04, rho=1e-4; shared camera; deformation scale 1')
                fig.tight_layout()
                fig.savefig(folder / f'{scene.lower()}_timeline_r{repeat}.png', dpi=150)
                fig.savefig(folder / f'{scene.lower()}_timeline_r{repeat}.pdf')
                plt.close(fig)
            fig, axes = plt.subplots(3, 6, figsize=(17, 9), facecolor='white')
            for row, arm in enumerate(ARMS[:3]):
                for col, scene in enumerate(SELECTED):
                    d = data[scene]['states'][arm]
                    axes[row, col].imshow(renderer.mesh(d['mesh'], d['x'][100], scene, data[scene]['bounds']))
                    axes[row, col].axis('off')
                    if row == 0:
                        axes[row, col].set_title(scene, fontsize=10)
                    axes[row, col].text(.5, -.04, LABELS[arm], transform=axes[row, col].transAxes, ha='center', fontsize=7)
            fig.suptitle(f'Actual physical step 100 from 105-step trajectories, repeat {repeat}; dt=.04, rho=1e-4; six scenes')
            fig.tight_layout()
            fig.savefig(folder / f'all_scenes_frame100_r{repeat}.png', dpi=160)
            fig.savefig(folder / f'all_scenes_frame100_r{repeat}.pdf')
            plt.close(fig)
        write_new(folder / 'manifest.json', dict(renderer=renderer.info, source_vertices=evidence, cameras=camera,
            report_controller=record(__file__),
            render_scope='Only rasterizer converts to float32; simulation exports/OBJ coordinates remain float64'))
    finally:
        renderer.release()


def report(output, summary):
    lines = ['# 六场景105帧详细记录与第100步三规则对照（2026-10-11）', '',
        '完整运行六场景、三规则和较严参考，每组105步、三轮。表格取第100物理步；Stack按用户要求继续暂缓。', '',
        '统一dt=.04、候选PCG rho=1e-4、movement=.01、累计beta=.01、门控相对残差.03、最少6次有效更新、原材料和FullCCD；三组均为当前源完整执行组合+conditional Graph K1。参考为movement-only、rho=1e-6，保持同一外层movement阈值，是较严数值参考而非精确解。', '',
        '最快版名称专指历史2.084×冻结程序；先导测试三种dt的100帧一次结果为1.628360×、2.173812×、1.876390×，因此选dt=.04记录。先导每dt比较Base和最快版，本详细记录用独立当前源观察副本；两者程序身份和用途不同。不同dt的100帧对应不同物理时长，movement终止绝对阈值也随dt变化，不构成时间精度等价认证。', '',
        '每帧同步host solver_ms含Graph捕获/准备/执行及同步；初始化、状态/实际速度导出、检查点保存和批量JSON写盘单独记录。输出包含7560条实际轨迹帧记录、所有逐Newton原始记录、每帧位置/速度差、动态布料拉伸、边界点位移和GPU采样。数据未减去推测的记录开销。', '',
        '## 同一第100步检查点', '',
        '各场景从movement-only第一轮在第99步后保存全部checkpoint状态。三规则和较严参考均统一冷恢复，误差取三候选×三参考九对最差值。完整设备字节、初态位置/实际速度、上下文和物理帧门槛通过；hot/cold差异保留在checkpoint_gate.json，不宣称冷恢复与热运行等价。', '',
        '| 场景 | 规则 | N | P | t(s) | RMS(mm) | max(mm) |', '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['matched100']:
        lines.append(f'| {r["scene"]} | {LABELS[r["method"]]} | {r["N"]} | {r["P"]} | {r["t"]:.6f} | {r["e_RMS_mm"]:.6f} | {r["e_max_mm"]:.6f} |')
    lines += ['', 'N为该单步实际PCG方向数，P为该单步全部PCG迭代数；可能还有只组装的终止Newton记录，另列在原始CSV。时间/N/P分别取三轮中位数，误差按全体顶点三维欧氏距离计算。每个场景另有独立CSV、TEX、PNG、PDF。', '',
        '## 各自完整轨迹的第100步', '',
        'trajectory_frame100_table.csv和对应六场景表取各规则从帧0独立运行至100步的状态；之前99步已经不同，因此与同输入检查点表分开解释。每帧1–105的误差含全体顶点、动态布料、FEM和ABD分区，以及实际速度差。参考维持相同外层movement=.01，未认证精确物理解。', '',
        '参考程序自身三轮的第100步差异如下（每场景三对最差值）。候选与参考的差异必须结合这个重复性范围解释；不能把参考轨迹当作唯一精确答案。逐帧、分区和实际速度重复性另存reference_repeat_dispersion.csv；冷检查点参考重复性保留在checkpoint_gate.json。', '',
        '| 场景 | 参考自身RMS差(mm) | 参考自身最大点差(mm) |', '|---|---:|---:|']
    for scene in SELECTED:
        repeat_errors = [r for r in summary['reference_dispersion_frame100'] if r['scene'] == scene]
        lines.append(f'| {scene} | {max(r["position_rms_mm"] for r in repeat_errors):.6f} | {max(r["position_max_mm"] for r in repeat_errors):.6f} |')
    lines += ['',
        '## 前100步性能与PCG工作', '',
        '下面的S相对同一当前观察程序的Movement规则；与先导测试冻结Base/最快版的S不同。各统计量取三轮中位数，阶段均摊成本包含准备和同步。N/P减少与每次PCG迭代变贵需要分别检查。', '',
        '| 场景 | 规则 | 总求解秒 | S | 平均N/帧 | 平均P/方向 | 总P | PCG阶段微秒/迭代 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['phases_first100']:
        lines.append(f'| {r["scene"]} | {LABELS[r["method"]]} | {r["TimeTot"]:.6f} | {r["S_movement_over_rule"]:.3f} | {r["N_bar"]:.2f} | {r["P_bar"]:.2f} | {r["P"]} | {r["pcg_stage_us_per_iteration"]:.2f} |')
    lines += ['',
        '完整轨迹的均摊PCG阶段成本还受不同接触数、矩阵规模/条件数、Graph捕获缓存及GPU频率影响；不能单凭该比值认定同一固定线性系统的执行退化。同检查点诊断提供同一物理初态的对照，后续Newton系统仍可能因各规则更新路径不同而不同。', '',
        '## 记录成本和统计口径', '',
        '新副本只补记已有CUDA间隔，没有增加逐PCG迭代的主机读回。旧日志的ipc_exit_assembly_ms未累加到旧assembly，新recording_phase_ms.assembly补齐这次终止组装；该新完整assembly已经包含terminal_assembly，不能再相加。CUDA间隔可能含主机提交导致的GPU空闲，不能称为纯kernel时长。', '',
        '原始phase_ms仍保留。状态更新阶段是原有postLineSearch间隔，帧末速度/预测状态更新和其他成本可落在unclassified_solver_envelope_ms。线性阶段/PCG迭代数是包含准备、MAS、同步和捕获的均摊代价，不是单次PCG kernel时间。', '',
        '逐Newton表的rho_ratio是原生日志rho_stop/rho_initial的预处理内积比，不能直接解释为真实L2相对残差或位置误差。native_json保留该次PCG、步长、beta、残差门控审计和Graph信息的完整原生字段。', '',
        'residual_observe_host_ms提取原生已有的残差范数/门控观察计时，与组装CUDA间隔和solver_ms重叠，不能再相加。代码在启用残差规则时每次正常组装后观察残差，然后检查Movement退出；即使第100步最后按Movement退出，也可能承担这些观察成本。原源码位置为GIPC.cu中的observe_residual和selected_exit调用。', '',
        'trace_export_host_ms包含GPU拷贝与文件输出；checkpoint_host_ms包含边界保存hook，只有第99步指定任务实际保存；其他帧仅检查hook。stats_write_ms是最终一次JSON写盘成本；帧CSV未把它摊回每帧。process_wall_seconds为runner监控包络，包含加载/导出和监控收尾粒度，不能用非solver差值冒充纯磁盘时间。', '',
        'recording_overhead.csv给出同一观察程序、同一冷检查点、三规则×三轮的记录开/关配对：开启时补记逐Newton和批量写JSON；关闭时保留旧记录方式。same_N_P仅筛选两个迭代计数；same_selected_counters还核对Newton记录数、能量/回退、Graph命中和执行计数。频率、调度等仍可能变化，时间差是配对观察值，不是已剔除全部干扰的纯记录开销。输出差异和GPU负载保留。残差门控计算是算法成本，两种记录模式都执行；单步配对不能推断105步批量写盘的完整收益。', '',
        'cost_diagnostics.csv记录第100步另行运行的CostScope原始嵌套区间，包括可观测的矩阵准备、MAS、Graph捕获/重放和残差归约等。嵌套区间不能相加，捕获图内部没有注入逐kernel计时；这些探针运行不混入正式时间。', '',
        '## 质量范围与证据', '',
        '布料拉伸统计覆盖至少含一个动态布料顶点的表面三角形边，包括固定锚点边。非零边界顶点位移不直接等于约束违反，可能含运动边界。ground gap只检查导出物理帧的动态顶点到y=-1平面距离；原生接触数/候选距离不是独立的全碰撞无穿透证明。', '',
        '共享Windows/WDDM桌面，保持本轮电源方案。所有仿真GPU串行，显存/时间保护保持；三次观测不构成稳定性能或物理质量认证。原源码、冻结Base/最快版、原程序及已有结果未改，日志-only副本的编译身份和原始数据校验信息分别封存。', '',
        '## 文件', '',
        '- [同检查点六场景合表](tables/checkpoint_frame100_all.pdf)',
        '- [独立轨迹第100步六场景合表](tables/trajectory_frame100_all.pdf)',
        '- [前100步耗时与迭代总表](tables/phases_first100_all.pdf)',
        '- [逐帧时间/计数](per_frame.csv)；[逐Newton全部记录](per_newton.csv)',
        '- [共同第100步原始逐帧记录/退出原因](checkpoint_per_frame.csv)；[共同第100步逐Newton记录](checkpoint_per_newton.csv)',
        '- [每帧位置/实际速度误差](trajectory_errors.csv)；[每帧质量观察](per_frame_quality.csv)',
        '- [较严参考自身重复差异](reference_repeat_dispersion.csv)',
        '- [初始化/导出/写盘/进程成本](run_costs.csv)；[记录开关配对](recording_overhead.csv)',
        '- [记录开关按场景/规则汇总](recording_overhead_summary.csv)',
        '- [前100步阶段表](phase_table_first100.csv)；[GPU采样](gpu_samples.csv)；[细分诊断](cost_diagnostics.csv)',
        '- [实际第100步六场景视觉图](visuals/all_scenes_frame100_r1.png)',
        '- [检查点输入门槛与hot/cold差异](checkpoint_gate.json)；[完整轨迹初态一致性](trajectory_input_gate.json)；[数据摘要和CSV SHA256](summary.json)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=ROOT / 'runs/local_recording105_dt0p04_20261011')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/local_recording105_dt0p04_20261011')
    args = parser.parse_args()
    summary = read(args.output / 'summary.json')
    require(summary['complete'], 'Complete analyzed experiment required')
    tables(args.output, summary)
    plots(args.output)
    visuals(args.session, args.output)
    report(args.output, summary)
    print(json.dumps(dict(tables=15, trajectory_plots=6, visual_panels=21, exported_obj=324, complete=True)))
