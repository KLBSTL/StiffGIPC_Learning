"""Write the requested concise report from measured data, without inventing gates."""
from config import ROOT,read

ARMS=('base','graph','toi','graph_toi')
LABELS={'base':'base','graph':'base+CUDA Graph','toi':'base+TOI','graph_toi':'base+CUDA Graph+TOI'}
SCENES={'hang':'悬挂布料 cloth_hang_l','fixed_bunny':'固定兔子布料 cloth_fixed_bunny_l'}

if __name__=='__main__':
    data=read(ROOT/'reports/active/FOURWAY_CLOTH_RESULTS_20261005.json')
    text=['# 两个布料场景的四配置加速比（2026-10-05）','',
        '本机 **RTX 3070 Laptop**，每组从零连续 **100帧、dt=.01**，每场景四组各三轮交错，共24次计时、2400物理帧；另8次两帧启动检查。加速比取每轮 `base秒数/该组秒数` 的三轮中位值，时间列为三轮求解耗时中位，二者不是强制用中位时间相除。启动和求解区间外导出不计入，重型诊断/分析/渲染在计时组中关闭。',
        '', '这是共享桌面诊断结果，不是受控GPU、七对统计下界或同质量性能认证。此次没有使用AutoDL，不能与先前4090的21/45帧结果直接比较。', '',
        '|场景|base|base+CUDA Graph|base+TOI|base+CUDA Graph+TOI|', '|---|---:|---:|---:|---:|']
    for key,s in data['scenes'].items():
        text.append('|'+SCENES[key]+'|'+ '|'.join(f'{s["summary"][a]["median_paired_speedup_over_base"]:.3f}×' for a in ARMS)+'|')
    text+=['','## 配置定义','',
        'base使用冻结的原版StiffGIPC；Graph-only为当前活动程序的IPC＋conditional PCG Graph，仍用legacy MAS。TOI host/Graph两组保留当前稳定Cholesky MAS、默认factor_inverse、默认world_block罚尺度、初值选择与safe重启守卫。restriction均为当前默认serial，**未开启warp**。四组关闭refit、能量批处理和energy reuse，Graph不是整个TOI外循环的捕获。', '',
        '因此TOI相对原版base的总收益包含所需稳定MAS差异，不能称纯TOI算法收益；TOI host与Graph使用同一程序、同一算法配置，只有PCG执行方式不同。IPC Graph与冻结base是两个程序，使用一致输入和物理参数并核验legacy/host或Graph实际执行，未将软件身份差异隐藏。所有材料、ABD/FEM/布料、完整CCD和原停止条件保留：IPC/TOI剩余量=.01、trial速度=.05 m/s、PCG rho=1e-4。', '',
        '当前活动程序SHA256：`57782cfeabdc35eb3151187a0f466911ef7b2ecf831b13ecc60ef221385b75f6`。每次运行保留requested/resolved配置、实际执行模式、源程序/链接身份及GPU采样。原版没有新resolved接口，其清单、有效场景参数和观测host执行单独核验。', '',
        '## 耗时、工作量与组件收益','']
    for key,s in data['scenes'].items():
        text+=['### '+SCENES[key],'','|配置|三次耗时 s|时间中位 s|配对加速比|PCG中位|方向数中位|','|---|---|---:|---:|---:|---:|']
        for a in ARMS:
            r=s['summary'][a]
            text.append(f'|{LABELS[a]}|'+', '.join(f'{v:.3f}' for v in r['seconds'])+f'|{r["median_seconds"]:.3f}|{r["median_paired_speedup_over_base"]:.3f}×|{r["median_pcg_iterations"]:.0f}|{r["median_directions"]:.0f}|')
        text+=['',f'Graph在TOI内的配对中位收益：**{s["median_graph_gain_with_toi"]:.3f}×**。在已经有Graph时，TOI相对IPC的配置收益：**{s["median_toi_gain_with_graph"]:.3f}×**（小于1即更慢）。二者不以不同轨迹的迭代总量冒充同一线性系统的算子收益。','']
    text+=['![耗时与布料拉伸](figures/fourway_cloth_timing_quality_20261005.png)','','## 质量与限制','',
        '24组计时均完整完成100帧，无非有限值、PCG触顶或breakdown。各组初态、拓扑、质量、固定点及场景/材料参数核对一致。只有执行与输入检查通过，完整物理质量尚未认证。以下为接受的物理帧端点指标；纯布料无体积FEM，不把其占位J=1当成体积质量证明。','']
    for key,s in data['scenes'].items():
        text+=['### '+SCENES[key],'','|配置|三轮最大布料边长比范围|最大固定点/物体漂移 m|严格端点重复范围通过次数|','|---|---|---:|---:|']
        for a in ARMS:
            r=s['summary'][a];v=r['cloth_max_stretch_range'];qs=[q for q in s['candidate_quality'] if q['arm']==a]
            passes=f'{sum(q["endpoint_envelope_passes"] for q in qs)}/3' if qs else '基线范围来源'
            text.append(f'|{LABELS[a]}|{v[0]:.8f}–{v[1]:.8f}|{r["fixed_max_drift_m"]:.3g}|{passes}|')
        text+=['',f'原版自身三对位置差的最大布料RMS：{s["base_repeat_max_position_cloth_rms_m"]*1000:.6f} mm。', '',
            '|配置|三轮对原版的最大布料位置RMS mm|超出原版位置重复范围的帧数（逐轮）|','|---|---|---|']
        for a in ARMS[1:]:
            qs=[q for q in s['candidate_quality'] if q['arm']==a]
            text.append('|'+LABELS[a]+'|'+', '.join(f'{q["max_position_cloth_rms_against_base_m"]*1000:.6f}' for q in qs)+'|'+', '.join(str(len(q['position_outside_base_repeat_frames'])) for q in qs)+'|')
        text+=['',f'![实际网格：{key}](figures/fourway_cloth_{key}_meshes_20261005.png)','']
    text+=['位置差不是物理真值误差。严格重复范围外的微小数值差异记为待确认；显著拉伸变化单列，不自动放宽门槛。图像采用第一轮真实网格、共同最终/峰值帧、相同相机/尺度/拉伸颜色；视觉相似也不能代替质量门禁。', '',
        '本轮没有导出/认证实际速度，没有追加接受子步的独立CPU CCD；原版历史CCD结果不能继承给新轨迹。因此这些数字只能称“该配置的诊断加速比”，不能称同质量加速或最终2×验收。', '',
        '## 复现与证据','',
        '在 `E:/university_class/ComputerGraphics/GIPC`，使用 `E:/Anaconda/envs/DL/python.exe`：','',
        '```text',
        'stiff_toi_cudagraph_20260929/tools/active/prepare_fourway_cloth.py',
        'stiff_toi_cudagraph_20260929/tools/active/batch.py --plan configs/active/fourway_cloth_smoke_20261005.json',
        'stiff_toi_cudagraph_20260929/tools/active/batch.py --plan configs/active/fourway_cloth_timing_20261005.json',
        'stiff_toi_cudagraph_20260929/tools/active/analyze_fourway_cloth.py --output reports/active/FOURWAY_CLOTH_RESULTS_20261005.json',
        'stiff_toi_cudagraph_20260929/tools/active/visualize_fourway_cloth.py',
        'stiff_toi_cudagraph_20260929/tools/active/report_fourway_cloth.py',
        '```','',
        '已有运行名/输出不可覆盖，复现须修改新plan中的运行名及报告名。完整记录：[预声明协议](FOURWAY_CLOTH_PROTOCOL_20261005.md)、[启动检查](FOURWAY_CLOTH_SMOKE_GATES_20261005.json)、[24组batch](FOURWAY_CLOTH_TIMING_BATCH_20261005.json)、[全部配置/几何/逐帧统计](FOURWAY_CLOTH_RESULTS_20261005.json)、[视觉记录](FOURWAY_CLOTH_VISUAL_20261005.json)。', '']
    path=ROOT/'reports/active/FOURWAY_CLOTH_RESULTS_20261005.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(text))
    print(path)
