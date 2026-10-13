"""Chinese delivery report from measured local combination-suite artifacts."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from config import read
from linux_runner import require,sha,write_new


def link(label,path):return f'[{label}]({path.as_posix()})'


def outer_exit_counts(session,scenes):
    rows=[]
    for scene in scenes:
        for arm in ('stiff','old','ours'):
            counts=Counter(f.get('newton_exit','missing') for repeat in (1,2,3)
                for f in read(session/f'performance_{scene.lower()}_{arm}_r{repeat}'/'output/stats.json')['frames'])
            require(sum(counts.values())==300 and not set(counts)-{'movement','cumulative_toi'},'Unexpected outer exit')
            rows.append(dict(scene=scene,method=arm,movement=counts['movement'],cumulative_toi=counts['cumulative_toi'],total=300))
    return rows


def deliver(session,tables,figures,verification,report):
    result=read(tables/'summary.json');images=read(figures/'render_manifest.json');ids=read(session/'identities.json')
    require(result['complete'] and images['complete'],'Incomplete experiment or rendering')
    verified=read(verification/'verification.json');reviewed=read(verification/'visual_review.json')
    require(verified['passed'] and reviewed['passed'],'Final verification or visual review incomplete')
    require(verified['summary_sha256']==sha(tables/'summary.json') and verified['render_sha256']==sha(figures/'render_manifest.json'),
            'Verified data changed')
    require(reviewed['render_sha256']==sha(figures/'render_manifest.json'),'Reviewed visual source changed')
    by={(r['scene'],r['method']):r for r in result['groups']}
    scenes=result['selected_scenes'];pairs=result['pairs']
    exits=outer_exit_counts(session,scenes)
    exit_table=tables/'outer_exit_counts.csv'
    if not exit_table.exists():
        with exit_table.open('x',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(exits[0]));writer.writeheader();writer.writerows(exits)
    n_runs=result['formal_runs'];n_frames=result['formal_frames']
    aggregate=result['aggregate_speedups']
    lines=[f'# {len(scenes)}场景本机执行组合完整复测（2026-10-11）','',
        f'全部{n_runs}次正式运行完成，每次100帧、三轮，共{n_frames}帧。所选场景总时间的配对中位速度比：2.084×冻结组合 **{aggregate["old"]["median"]:.6f}×**，当前同配置组合 **{aggregate["ours"]["median"]:.6f}×**（均相对原版StiffGIPC）。',
        '', '用户已明确选择“Stack暂缓，先交付六个场景”。Stack原版第61帧触发显存预算保护，末次仍余1505MiB，并非已确认OOM。该失败及原105次计划/ledger保留；本报告仅涵盖其余六个完整场景，没有补入Stack数据或图。',
        '', '## 配置与方法','',
        '用户确定继续本机，使用原历史low输入，所有方法统一dt=.01、PCG rho=1e-4、movement系数.01、MAS legacy、完整CCD与原材料。Hang原JSON的dt=.04通过明确运行参数覆写为.01；JSON/几何/材质文件本身没有改动。Stack原计划也为.01但正式结果暂缓。未改变数值类型、native求解器或电源模式。',
        f'电源方案在smoke、checkpoint、performance阶段前后均为：{ids["power"]}。GPU为RTX3070 Laptop/WDDM，共享桌面负载未完全隔离，以下是本轮观测，不能称为稳定性能保证。',
        '', '主矩阵五方法：frozen Stiff；冻结组合Host/Graph；当前组合Host/Graph。两种组合均启用FullCCD refit、批量能量、能量复用、普通BVH refit、FEM direct pullback、MAS pivot-only sync，外层为legacy累计阈值.01。Graph为conditional K1，无自适应forcing。每个版本的Host/Graph仅execution字段改变；两版本组合的完整expanded_config相同。',
        '', '这里主表的Current对应代码内部performance arm=ours，仅表示当前执行组合，**没有启用残差门控**。检查点单步arm=ours才是残差门控诊断；图表使用不同的明确方法名。',
        '', '冻结Base本身已有legacy累计退出：源码在第6次及之后接受更新时使用beta=(1-alpha)*beta，beta低于Newton阈值即可退出。当前legacy保留该规则。本轮原始日志也直接记录Base的cumulative_toi退出；所以不能把本矩阵的组合加速比称为新增TOI的独立收益。Base与两组合三轮各300步的实际终止原因如下（movement/累计TOI）。',
        '', '| 场景 | Base | 冻结组合 | 当前组合 |', '|---|---:|---:|---:|']
    for scene in scenes:
        rows=[r for r in exits if r['scene']==scene]
        lines.append('| '+scene+' | '+' | '.join(f'{r["movement"]}/{r["cumulative_toi"]}' for r in rows)+' |')
    base_source=Path(next(r['path'] for r in ids['programs']['base']['sources'] if Path(r['path']).name=='GIPC.cu'))
    lines+=['',link('Base累计规则源码',base_source)+f'（Kmin在10985行，预算更新/退出在11109–11123行）；'+link('实际终止计数CSV',exit_table),
        '', '## 完整速度与工作量','',
        '| 场景 | Base秒 | 冻结组合秒 | 当前组合秒 | 冻结组合S | 当前组合S | 冻结/当前配对时间比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for scene in scenes:
        a,b,c=(by[scene,k] for k in ('stiff','old','ours'))
        cross=statistics.median(p['speedup'] for p in pairs if p['scene']==scene and p['reference']=='old' and p['candidate']=='ours')
        lines.append(f'| {scene} | {a["TimeTot"]:.6f} | {b["TimeTot"]:.6f} | {c["TimeTot"]:.6f} | {b["S"]:.6f}× | {c["S"]:.6f}× | {cross:.6f} |')
    lines+=['', '秒数各取三轮运行级中位数；S及跨版本时间比各取三个同轮配对比值的中位数，不能用表内两个时间中位数的商替代。总速度比先逐轮求六场景时间和，再计算配对比；不是六个场景倍率的平均。',
        '', '每次从帧0启动，方法顺序逐轮轮换、场景顺序第二轮反转；无额外warm-up。solver计时为同步host窗口，包含Graph构建/捕获/执行，排除初始化与trace导出。profile=none，未使用Nsight、GIPC_COST_TRACE或算子探针。',
        '', '三轮配对倍率的完整范围如下；仅三次观察，不由此估计置信区间。',
        '', '| 场景 | 冻结组合S范围 | 当前组合S范围 |', '|---|---:|---:|']
    for scene in scenes:
        ranges=[]
        for arm in ('old','ours'):
            values=[p['speedup'] for p in pairs if p['scene']==scene and p['reference']=='stiff' and p['candidate']==arm]
            ranges.append(f'{min(values):.6f}–{max(values):.6f}')
        lines.append(f'| {scene} | '+ ' | '.join(ranges)+' |')
    lines+=[
        '', '## Graph消融：保留其他组合优化','',
        '| 场景 | 冻结Host/Graph | 当前Host/Graph |', '|---|---:|---:|']
    graph={(r['scene'],r['version']):r for r in result['graph']}
    for scene in scenes:lines.append(f'| {scene} | {graph[scene,"old"]["T_B_over_T_G"]:.6f}× | {graph[scene,"current"]["T_B_over_T_G"]:.6f}× |')
    lines+=['', '这个消融中的Host仍启用其余全部执行优化，因而它的Base含义与主表的frozen Stiff不同。',
        '', '## Newton/PCG与线性求解代价','',
        '| 场景 | Base N/P | 冻结组合 N/P | 当前组合 N/P | 当前/Base PCG工作比 | 当前/Base均摊LSolver/PCG比 |',
        '|---|---:|---:|---:|---:|---:|']
    for scene in scenes:
        rows=[by[scene,k] for k in ('stiff','old','ours')]
        selected=[p for p in pairs if p['scene']==scene and p['reference']=='stiff' and p['candidate']=='ours']
        work=statistics.median(p['pcg_ratio'] for p in selected);cost=statistics.median(p['linear_per_pcg_ratio'] for p in selected)
        labels=[f'{r["N_tot"]:g}/{r["P_tot"]:g}' for r in rows]
        lines.append(f'| {scene} | {labels[0]} | {labels[1]} | {labels[2]} | {work:.6f} | {cost:.6f} |')
    lines+=['', 'N为记录了PCG的实际方向数（包含movement判断末次方向），P为PCG总迭代；阶段表的N̄=N/100，P̄=P/N。LSolver含准备、归约、同步和捕获，LSolver/P只能说明均摊代价，不能分离一个迭代核。分项列分别取中位数，列和不保证等于TimeTot中位数。',
        '', '## 完整轨迹位置差异','',
        '| 场景 | Base/当前最大帧RMS(mm) | Base/当前最大顶点差(mm) | 冻结/当前最大帧RMS(mm) | 冻结/当前最大顶点差(mm) |',
        '|---|---:|---:|---:|---:|']
    for scene in scenes:
        values=[]
        for reference in ('stiff','old'):
            chosen=[p['trajectory']['position']['cloth'] for p in pairs if p['scene']==scene and p['reference']==reference and p['candidate']=='ours']
            values.extend([1000*max(x['max_frame_rms'] for x in chosen),1000*max(x['max_vertex'] for x in chosen)])
        lines.append('| '+scene+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines+=['', '取三轮0–100帧的最差值，沿用原分区中cloth=非tet且非ABD顶点，可能包含固定二维表面。该量是两条完整轨迹的差异，不能当作对精确解的误差。实际速度分区差异及FEM/ABD差异在summary.json的pairs中；没有为不导出速度的原版推断速度。',
        '', '## 本轮证据支持的结论','',
        f'两种组合均未在这六个场景的任一正式配对中达到2×；六场景总时间配对中位倍率分别为{aggregate["old"]["median"]:.6f}×和{aggregate["ours"]["median"]:.6f}×。历史“2.084×”是程序版本的来源标记，不是这次100帧、dt=.01、rho=1e-4测得的结果。不能把不同帧数、参数、电源/桌面状态的历史倍率作为本轮性能。',
        '', '主矩阵当前组合的PCG工作量在前五场景与Base接近，Boxes的同轮配对中位增加约4.5%；六场景LSolver/P配对中位下降约24%–53%。因此本轮主要观察到执行代价下降，没有观察到整段轨迹PCG工作量显著减少。主矩阵没有单独关闭累计规则，不能从它归因出累计TOI的独立收益。',
        '', '冻结/当前的同轮时间比中位数在各场景约0.959–1.010之间，且完整轨迹已发生变化。Sphere/Bunny的Base与当前最差帧cloth分区RMS差约22.4/32.6mm，最大顶点差约123.7/195.4mm；冻结与当前的差异也不能忽略。因此这里交付的是速度和轨迹差异的观测，不构成同物理质量的加速认证。',
        '', '同输入单步才隔离了外层规则：纯累计在Table/Bunny确实减少方向与PCG，但相对较严参考的最大差增至约2.14/10.25mm；残差门控把它降至约0.028/5.51mm。Table的门控耗时没有优于movement-only；Bunny仍有收益但仍存在毫米级偏差。这两个单步不能推导出六场景全程2×且质量等价。',
        '', '## 同检查点三规则','',
        '当前程序同一冷恢复状态，Table第28步、Bunny第46步，各方法三次；比较movement-only、legacy累计TOI beta=.01、补偿预算beta=.01＋残差相对门槛.03。每行Graph及其他执行优化完全一致，PCG rho=1e-4。参考为movement-only、rho=1e-6，是较严数值参考而非精确解。',
        '', '| 状态 | 规则 | N | P | t(s) | RMS(mm) | max(mm) |','|---|---|---:|---:|---:|---:|---:|']
    labels={'original':'Movement only','pure':'Cumulative TOI .01','ours':'Compensated＋residual gate'}
    for r in result['matched']:lines.append(f'| {r["state"]} | {labels[r["method"]]} | {r["N"]:g} | {r["P"]:g} | {r["t"]:.6f} | {r["e_RMS_mm"]:.6f} | {r["e_max_mm"]:.6f} |')
    gate=result['checkpoint_gate']
    lines+=['', f'24组恢复初态、实际速度、全部设备缓冲区、binary/frame及物理上下文逐字节门槛通过；hot/cold 1e-8终点连续性检查为 **{gate["hot_cold_1e8_passed"]}**，原始差异保留。所有单步行统一冷缓存，未认证冷缓存与热连续运行等价。位置误差取三候选×三参考九对最差值，以全体顶点三维欧氏距离计算RMS及max，m转mm。',
        '', '通用Windows runner的requested.from_zero字段在matched任务中保留了旧默认true；该字段不代表单步实际起点。实际恢复由GIPC_CHECKPOINT_LOAD、native checkpoint_restore.json和逐字节输入门槛证明；没有重写旧请求记录。',
        '', '## 原始证据、质量观察与验证','',
        f'另完成35次两帧smoke（含Stack）、两个28/46帧seed、24次单步。成功运行总计{n_runs+61}次、{n_frames+168}个物理帧；正式矩阵为{n_runs}次{n_frames}帧。另有Stack的61帧失败记录不参与计时或完成统计。全部计时重新从frames.csv核算，全部方向/PCG及阶段从native stats.json核算，所有0–100位置有限；两组合的实际速度全帧导出，原版不支持速度导出。',
        f'主矩阵{len(pairs)}个输入配对通过，所有源文件、编译记录、exe、DLL和链接对象完成后的SHA256复核一致。测试控制器与原始run请求逐次封存。',
        '', 'quality_observations.csv保留拉伸、固定点位移和每帧顶点到ground的最小有符号距离。拉伸沿用历史non-volume surface指标，可能包含固定表面，不冒充纯cloth的独立材料认证。地面距离仅观察导出帧边界，不覆盖帧内、物体碰撞和自碰撞；本轮没有独立全碰撞穿透或同物理质量认证。',
        '', '| 程序身份 | SHA256复核文件数 | exe SHA256 |', '|---|---:|---|']
    for row in result['identity_recheck']:
        lines.append(f'| {row["program"]} | {row["verified_files"]} | {row["exe_sha256"]} |')
    lines+=['',f'独立交付检查再次从CSV/PCG记录计算90次耗时与工作量，检查48个PNG、48个PDF、六套表格和306个OBJ；所有OBJ顶点和连接关系与native原始导出逐项完全相等。{verified["verified_program_files"]}个程序身份文件、原控制器和续测控制器SHA256一致。六张数据表和覆盖全部48张正式视觉图的六张检查拼图已目视检查，主要图另以完整尺寸核查。',
        '',link('机器核验记录',verification/'verification.json')+'；'+link('视觉核验记录',verification/'visual_review.json')]
    lines+=['', '## 数据表与实际视觉图','']
    for stem,label in [('phase_table_current','Base/当前组合阶段表'),('phase_table_frozen','Base/冻结组合阶段表'),
                       ('phase_table_both','三方法阶段合表'),('graph_table_current','当前组合Graph消融'),
                       ('graph_table_old','冻结组合Graph消融'),('checkpoint_table','同检查点表')]:
        lines.append('- '+label+'：'+' / '.join(link(ext.upper(),tables/(stem+'.'+ext)) for ext in ('png','pdf','csv','tex')))
    lines+=['',link('六场景第100帧三方法总对照',figures/'all_scenes_f0100_r1.png'),'',
        f'![六场景Base/当前组合]({(figures/"scenes_ours_f0100_r1.png").as_posix()})','',
        link('六场景Base/冻结组合',figures/'scenes_old_f0100_r1.png')+'；'+link('检查点视觉对照',figures/'matched_checkpoints_r1.png'),
        '', f'三轮均导出实际关键帧0/25/50/75/100、额外Table28/Bunny46，及世界坐标OBJ。共{len(images["figures"])//2}组PNG/PDF、{len(images["evidence"])}个OBJ。共用正交相机和每场景范围，变形比例1，无几何平滑；视觉接近不能替代误差与碰撞检查。',
        '', '正式视觉交付使用独立offscreen OpenGL 3.3、LESS深度测试、4样本MSAA，关闭背面剔除与混合。交叠三角形顺序反转的图像逐像素相同，近面遮挡测试通过；相机不裁切导出的几何。替代旧Matplotlib表面排序的显示伪影，没有改变位置、拓扑、仿真代码或计时。仅光栅上传转float32，OBJ保持原double世界坐标；浅色面片光照无几何平滑。旧离线排序图单独保留，不作为正式视觉图。',
        '', '## 实际命令','',f'工作目录：{ROOT.as_posix()}；解释器：E:/Anaconda/envs/DL/python.exe -X utf8 -B。','', '```text',
        'tools/local/run_combination_suite.py smoke --session local_combination_seven_20261011',
        'tools/local/run_combination_suite.py checkpoint --session local_combination_seven_20261011',
        'tools/local/run_combination_suite.py performance --session local_combination_seven_20261011',
        'tools/local/continue_combination_suite.py --session runs/local_combination_seven_20261011',
        f'tools/local/analyze_combination_suite.py --session {session.as_posix()} --output {tables.as_posix()}',
        f'tools/local/render_combination_depth.py --session {session.as_posix()} --output {figures.as_posix()}',
        f'tools/local/verify_combination_delivery.py --session {session.as_posix()} --tables {tables.as_posix()} --figures {figures.as_posix()} --output {verification.as_posix()}',
        '```','', '复跑应更换session及输出目录，所有入口拒绝覆盖原始结果。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    write_new(report.with_suffix('.json'),dict(complete=True,report=str(report),summary_sha256=sha(tables/'summary.json'),
        render_sha256=sha(figures/'render_manifest.json'),formal_runs=n_runs,formal_frames=n_frames,
        verification_sha256=sha(verification/'verification.json'),visual_review_sha256=sha(verification/'visual_review.json'),
        selected_scenes=scenes,excluded_scenes=result['excluded_scenes'],scope='User-approved complete six-scene subset; Stack deferred',
        outer_exit_counts=exits,
        aggregate=aggregate,physical_quality_certified=False,performance_certified=False))
    print(f'Wrote {report}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--tables',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True)
    a=p.parse_args();deliver(a.session.resolve(),a.tables.resolve(),a.figures.resolve(),a.verification.resolve(),a.report.resolve())
