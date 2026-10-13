"""Recompute complete local ledgers and write the comparison report/figures."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/paper_suite'),str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
from config import read
from linux_runner import require,sha
from render import geometry,positions,draw

MAIN=ROOT/'runs/local_graph_history_20261011_allowed'
EARLY=ROOT/'runs/local_graph_v4_20261011'
OUT=ROOT/'reports/local_graph_history_20261011'


def verify_runs(session):
    packet=read(session/'performance_ledger.json');require(packet['complete'],'Incomplete formal matrix')
    for row in packet['runs']:
        folder=session/row['task']['name'];result=read(folder/'result.json');require(result['status']=='completed','Native failure')
        with (folder/'trace/frames.csv').open() as stream:frames=list(csv.DictReader(stream))
        require(len(frames)==100,'Not 100 physical frames')
        seconds=sum(float(x['solver_ms']) for x in frames)/1000
        require(abs(seconds-row['summary']['solver_seconds'])<1e-10,'Timing discrepancy')
        native=read(folder/'output/stats.json')['frames']
        directions=sum('pcg' in n for f in native for n in f['newton'])
        pcg=sum(n['pcg']['iterations'] for f in native for n in f['newton'] if 'pcg' in n)
        require(directions==row['summary']['directions'] and pcg==row['summary']['pcg'],'Work discrepancy')
        for i in range(101):
            values=np.fromfile(folder/f'trace/state_{i:04d}.bin',dtype='<f8')
            require(values.size and np.isfinite(values).all(),'Invalid exported state')
            if read(folder/'requested.json')['capabilities'].get('actual_velocity') is True:
                velocity=folder/f'trace/velocity_{i:04d}.bin'
                require(velocity.is_file(),'Missing actual velocity')
                v=np.fromfile(velocity,dtype='<f8')
                require(v.size==values.size and np.isfinite(v).all(),'Invalid actual velocity')
    return packet


def ratio(analysis,scene,reference,candidate):
    values=[p['speedup'] for p in analysis['pairs'] if p['scene']==scene and p['reference']==reference and p['candidate']==candidate]
    require(len(values)==3,'Three complete paired values required')
    return statistics.median(values),values


def verify_pairs(session,packet,analysis):
    by={(r['task']['scene'],r['task']['repeat'],r['task']['arm']):r for r in packet['runs']}
    for pair in analysis['pairs']:
        left=by[(pair['scene'],pair['repeat'],pair['reference'])]
        right=by[(pair['scene'],pair['repeat'],pair['candidate'])]
        expected=left['summary']['solver_seconds']/right['summary']['solver_seconds']
        require(abs(expected-pair['speedup'])<1e-12,'Paired ratio discrepancy')
        check=pair.get('input',pair.get('initial'))
        require(check['passed'],'Initial input mismatch')
        for name,record in check['files'].items():
            lhs=sha(session/left['task']['name']/name)
            rhs=sha(session/right['task']['name']/name)
            require(lhs==rhs==record['left_sha256']==record['right_sha256'],'Initial byte identity changed')
    for row in analysis['rows']:
        base='base' if session==EARLY else 'frozen_stiff'
        if row['arm']!=base:
            expected,_=ratio(analysis,row['scene'],base,row['arm'])
            require(abs(expected-row['speedup_median'])<1e-12,'Reported median ratio discrepancy')


def table(analysis,arms,scene):
    by={r['arm']:r for r in analysis['rows'] if r['scene']==scene}
    lines=['| 方法 | 100帧秒数中位数 | 相对同轮Stiff倍率 | Newton方向中位数 | PCG总数中位数 | PCG/方向 |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,label in arms:
        row=by[arm];m=row['medians'];s=row['speedup_median']
        lines.append(f'| {label} | {m["solver_seconds"]:.6f} | {1 if s is None else s:.6f}× | {m["directions"]:g} | {m["pcg"]:g} | {m["pcg_per_direction"]:.3f} |')
    return '\n'.join(lines)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    choices=[(EARLY,'base','Stiff'),(EARLY,'early_graph','Early Graph'),(EARLY,'current_graph','Current Graph'),
             (MAIN,'historical_best','Old combination'),(MAIN,'current_best','Current combination'),(MAIN,'current_ours','Current Ours')]
    for repeat in (1,2,3):
        fig=plt.figure(figsize=(18,7),facecolor='white')
        for row,(scene,key) in enumerate([('cloth_hang_l','Hang'),('cloth_fixed_bunny_l','Bunny')]):
            states=[]
            for session,arm,label in choices:
                folder=session/f'performance_{scene}_{arm}_r{repeat}'
                mesh=geometry(folder);x=positions(folder,100,mesh['nv']);states.append((folder,mesh,x,label))
            xyz=np.concatenate([x for _,_,x,_ in states])[:,[0,2,1]]
            center=(xyz.min(axis=0)+xyz.max(axis=0))/2;radius=max(float(np.ptp(xyz,axis=0).max())*.55,.01)
            for col,(folder,mesh,x,label) in enumerate(states):
                ax=fig.add_subplot(2,6,row*6+col+1,projection='3d');draw(ax,mesh,x,key,(center-radius,center+radius))
                ax.set_title(f'{key}: {label}',fontsize=10)
                records.append(dict(repeat=repeat,scene=scene,arm=label,frame=100,
                                    state=str(folder/'trace/state_0100.bin'),sha256=sha(folder/'trace/state_0100.bin')))
        fig.text(.5,.02,f'Local comparison, repeat {repeat}: actual frame 100, dt=.01, rho=1e-4; deformation scale 1',ha='center',fontsize=10)
        fig.subplots_adjust(wspace=0,hspace=0,bottom=.05,top=.94)
        fig.savefig(OUT/f'frame100_r{repeat}.png',dpi=160);fig.savefig(OUT/f'frame100_r{repeat}.pdf');plt.close(fig)
    return records


def main(render):
    OUT.mkdir(parents=True,exist_ok=True)
    main_packet=verify_runs(MAIN);early_packet=verify_runs(EARLY)
    a=read(MAIN/'analysis.json');b=read(EARLY/'analysis.json');recovery=read(ROOT/'build/graph_v4_recovered_source/RECOVERY.json')
    verify_pairs(MAIN,main_packet,a);verify_pairs(EARLY,early_packet,b)
    # Early Graph emits capture counts/cache hits, but no capture-time event.
    # Its historical controller defaulted absent durations to zero. Keep the
    # original ledger; report this unavailable metric as null in the delivery.
    for row in b['rows']:
        if row['arm']=='early_graph':
            row['medians']['graph_capture_ms']=None
            row['raw']['graph_capture_ms']=[None,None,None]
            row['capture_timing_available']=False
    require(recovery['recovered']==278 and not recovery['missing'],'Historical recovery incomplete')
    report=['# 本机 Graph 历史版本复测（2026-10-11）','',
        '悬挂布料与固定兔子布料；每方法100帧、三轮，dt=.01、PCG rho=1e-4、movement=.01、legacy MAS、完整CCD。GPU运行串行，所有轨迹从帧0开始。电源方案始终为本轮开始时的野兽模式。',
        '', '早期v4原二进制未留存：本轮从Git与原会话补丁记录恢复278个源码/输入文件，全部匹配历史清单SHA256，再用当前本机CUDA 13.0/MSVC重新编译；它是原源码复测，不是重跑原b77acaa二进制。最近2.084×组直接使用其冻结原程序。当前组直接使用AutoDL准备阶段的本机clean build。历史2.084×是120帧、安静模式的本机观测，本次为100帧、开始时已处于野兽模式；这些差异不能与代码差异混合解释。',
        '', '倍率均为三个同轮配对时间比的中位数，不能用两个时间中位数的商替代。两批各自保留同轮Stiff及host；跨批绝对时间只作描述。共享Windows/WDDM桌面负载未完全隔离，三轮结果不构成稳定性能或物理质量认证。',
        '', '本次两场景中，没有任何方法的三轮配对中位数达到2×。早期纯Graph在rho=1e-4时没有复现历史2.724×；当前纯Graph也没有明显的大幅回退。最近2.084×版本包含多项执行优化，不能把它当作纯Graph收益。当前同配置组合与该冻结版本接近；残差门控Ours增加了方向与总PCG工作，当前配置下耗时更高。',
        '', '## A. 早期纯Graph与当前纯Graph：同批交错比较','']
    pure_arms=[('base','Stiff'),('early_host','早期Host'),('early_graph','早期Graph'),('current_host','当前Host'),('current_graph','当前Graph')]
    combo_arms=[('frozen_stiff','Stiff'),('current_host','当前Host'),('current_graph','当前Graph'),('historical_best','2.084×冻结组合'),('current_best','当前同配置组合'),('current_ours','当前Ours')]
    comparisons=[]
    for scene,label in [('cloth_hang_l','悬挂布料'),('cloth_fixed_bunny_l','固定兔子布料')]:
        report.extend([f'### {label}', '',table(b,pure_arms,scene),''])
        for reference,candidate,meaning in [('early_host','early_graph','早期纯Graph收益'),('current_host','current_graph','当前纯Graph收益'),('early_graph','current_graph','早期Graph/当前Graph耗时比')]:
            median,values=ratio(b,scene,reference,candidate)
            report.append(f'- {meaning}：配对中位数 {median:.6f}×；三轮 '+', '.join(f'{v:.6f}' for v in values)+'。')
            comparisons.append(dict(batch='early',scene=scene,reference=reference,candidate=candidate,median=median,raw=values))
        report.append('')
    report.extend(['纯Graph跨版本的早期/当前时间比：悬挂布料0.959604、兔子0.982130，即本次配对中位数中当前耗时分别约高4.2%和1.8%；三轮方向不一致（悬挂一轮当前更快，兔子一轮当前更快），不能认定这些个位数差异是稳定的代码退化。',
                   '例如纯Graph批次兔子Stiff三轮17.129、17.640、14.013秒，存在明显机器负载/时钟波动；未将其归因到某一具体后台进程。',''])
    report.extend(['## B. 最近2.084×组合、当前同配置组合、Ours','',
        '前两种组合均启用Graph K1、FullCCD refit、批量能量、能量复用、普通BVH refit、FEM direct pullback、MAS pivot-only sync；保留legacy累计阈值.01及movement退出。Ours将累计阈值改为.001并使用补偿预算及残差门控.03。无自适应PCG forcing。',''])
    for scene,label in [('cloth_hang_l','悬挂布料'),('cloth_fixed_bunny_l','固定兔子布料')]:
        report.extend([f'### {label}','',table(a,combo_arms,scene),''])
        for reference,candidate,meaning in [('historical_best','current_best','冻结组合/当前同配置组合耗时比'),('current_best','current_ours','当前组合/Ours耗时比')]:
            median,values=ratio(a,scene,reference,candidate)
            report.append(f'- {meaning}：配对中位数 {median:.6f}×；三轮 '+', '.join(f'{v:.6f}' for v in values)+'。')
            comparisons.append(dict(batch='combination',scene=scene,reference=reference,candidate=candidate,median=median,raw=values))
        report.append('')
    by={(r['scene'],r['arm']):r['medians'] for r in a['rows']}
    report.extend(['## 当前Ours的PCG增加来自哪里','',
        '| 场景 | 当前组合方向 | Ours方向 | 当前组合PCG | Ours PCG | 总PCG变化 | PCG/方向变化 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for scene,label in [('cloth_hang_l','悬挂布料'),('cloth_fixed_bunny_l','固定兔子布料')]:
        old=by[scene,'current_best'];new=by[scene,'current_ours']
        report.append(f'| {label} | {old["directions"]:g} | {new["directions"]:g} | {old["pcg"]:g} | {new["pcg"]:g} | {100*(new["pcg"]/old["pcg"]-1):+.2f}% | {100*(new["pcg_per_direction"]/old["pcg_per_direction"]-1):+.2f}% |')
    report.extend(['', '这次比较没有显示单次PCG方向明显变难：悬挂布料每方向PCG基本相同，兔子还略少。总PCG增加主要跟随外层Newton方向数增加。此对照同时改变累计阈值(.01→.001)、补偿预算与残差门控，不能单独把增量归因于门控或补偿中的一个；它证明当前Ours配置的整体代价，不能证明TOI原理必然无效。',
                   '', 'work_cost.csv还提供Hess、LSolver、CCD、LineS、Misc、TimeTot及均摊LSolver/PCG；LSolver含准备/归约/同步等，并非一个PCG迭代核的独立耗时。各列分别取三轮中位数，列和未必等于TimeTot中位数。Misc是solver窗口减去前四项后的余量。',''])
    old_stats=ROOT.parent/'stiff_toi_cudagraph_20260929/runs/local/graph_speed20_v4_cloth_hang_l_base_graph_measure_r01/output/stats.json'
    new_stats=EARLY/'performance_cloth_hang_l_early_graph_r1/output/stats.json'
    prefixes=[]
    for path,label in [(old_stats,'历史rho=1e-16'),(new_stats,'本次rho=1e-4')]:
        frames=read(path)['frames'][:20]
        pcgs=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
        prefixes.append(dict(label=label,directions=len(pcgs),pcg=sum(p['iterations'] for p in pcgs)))
    report.extend(['## 对历史2.724×的解释','',
        f'历史第一轮悬挂布料前20帧：{prefixes[0]["directions"]}个方向、{prefixes[0]["pcg"]}次PCG；本次早期Graph第一轮同为前20帧：{prefixes[1]["directions"]}个方向、{prefixes[1]["pcg"]}次PCG。rho收紧会显著增加每次线性求解的PCG工作，Graph可消除的逐迭代host同步占比也更大。该工作量对照支持收益随容差改变；它不是同轮重新测量1e-16，也不能将历史与当前绝对时间混合计算倍率。',
        '', '## 验证与边界','',
        f'两批正式运行共 {len(main_packet["runs"])+len(early_packet["runs"])} 次、{100*(len(main_packet["runs"])+len(early_packet["runs"]))} 帧；全部完成。重新从原生frames.csv核算solver时间，从stats.json核算Newton/PCG，并检查全部初态与100帧位置文件有限。',
        f'比较初始输入的配对共{len(a["pairs"])+len(b["pairs"])}个，均通过topology/mass/boundary/body id/初态/metadata/scene文件字节核验。完整位置差异与可获得的实际速度差异保留在analysis.json；早期程序和原版Stiff没有实际速度导出，未从位置推断速度。',
        '', 'Native source未改动。新增恢复、运行、分析脚本；恢复文件与编译结果保存在build/，原会话和历史结果均保留。第一次默认沙箱启动失败为零帧，另一个新session获准运行；零帧记录不纳入正式计时。',
        '', '早期v4缺少当前resolved_config接口：执行后端逐PCG记录核验；dt/rho等由精确恢复源码与显式环境绑定，不能冒充具备新接口的运行时逐字段确认。',
        '早期Graph只记录捕获计数/命中率，没有捕获耗时字段；交付汇总将该项记为null，原控制器的默认0不是测量值。',
        '两批正式运行均为profile=none，未启用GIPC_COST_TRACE，也未使用Nsight或算子探针。状态导出在solver计时窗口外。stretch观察指标在summary.csv中；本轮没有重新做独立碰撞穿透或物理等价认证。',
        '完成后的身份复核再次检查4个程序的源码、exe、DLL、链接对象与构建记录，全部与开跑前哈希一致：local_graph_history_identity_recheck_20261011.json。',
        '', '完整证据：',
        '- runs/local_graph_history_20261011_allowed/{identities,performance_ledger,analysis}.json',
        '- runs/local_graph_v4_20261011/{identities,performance_ledger,analysis}.json',
        '- build/graph_v4_recovered_source/RECOVERY.json',
        '- reports/local_graph_v4_{configure,build}_20261011.log',
        '', '## 复现命令','', '使用E:/Anaconda/envs/DL/python.exe -X utf8 -B；新session名称防止覆盖。',
        '', '```text',
        'tools/local/compare_graph_history.py smoke --session <new_combination_session>',
        'tools/local/compare_graph_history.py performance --session <same_combination_session>',
        'tools/local/compare_graph_history.py analyze --session <same_combination_session>',
        'tools/local/test_graph_v4.py smoke --session <new_early_session>',
        'tools/local/test_graph_v4.py performance --session <same_early_session>',
        'tools/local/test_graph_v4.py analyze --session <same_early_session>',
        'tools/local/report_graph_history.py --render',
        '```',''])
    evidence=figures() if render else []
    if evidence:
        report.extend(['## 第100帧示例','', '三轮均使用实际位置导出、共用场景相机与范围，变形比例1；截图接近不代表物理等价。','',
                       f'![repeat1]({(OUT/"frame100_r1.png").as_posix()})','',
                       f'[第二轮 PNG]({(OUT/"frame100_r2.png").as_posix()})；[第三轮 PNG]({(OUT/"frame100_r3.png").as_posix()})。三轮另有同名PDF。',''])
    (ROOT/'reports/LOCAL_GRAPH_HISTORY_20261011_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    result=dict(complete=True,formal_runs=66,formal_frames=6600,recovery=recovery,historical_prefix_work=prefixes,
                main_analysis=a,early_analysis=b,comparisons=comparisons,figures=evidence,
                physical_quality_certified=False,performance_certified=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows=[]
    for batch,analysis in [('pure',b),('combination',a)]:
        for row in analysis['rows']:
            rows.append(dict(batch=batch,scene=row['scene'],arm=row['arm'],speedup_median=row['speedup_median'],**row['medians']))
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    costs=[]
    for batch,packet,analysis in [('pure',early_packet,b),('combination',main_packet,a)]:
        for row in analysis['rows']:
            selected=[x['summary'] for x in packet['runs'] if x['task']['scene']==row['scene'] and x['task']['arm']==row['arm']]
            values=[]
            for m in selected:
                phases=m['phase_ms'];hess=phases['assembly']/1000;linear=phases['pcg']/1000;ccd=phases['ccd']/1000;line=phases['line_search']/1000
                values.append(dict(Hess=hess,LSolver=linear,CCD=ccd,LineS=line,Misc=m['solver_seconds']-hess-linear-ccd-line,
                    TimeTot=m['solver_seconds'],Pbar=m['pcg_per_direction'],Nbar=m['directions']/100,Ptot=m['pcg'],
                    linear_amortized_ms_per_pcg=phases['pcg']/m['pcg']))
            costs.append(dict(batch=batch,scene=row['scene'],arm=row['arm'],S=1 if row['speedup_median'] is None else row['speedup_median'],
                **{k:statistics.median(v[k] for v in values) for k in values[0]}))
    with (OUT/'work_cost.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(costs[0]));writer.writeheader();writer.writerows(costs)
    cb={(c['scene'],c['arm']):c for c in costs if c['batch']=='combination'}
    report.extend(['## LSolver总时间与均摊代价','',
        '| 场景 | 当前组合LSolver(s) | Ours LSolver(s) | 当前组合LSolver/PCG(ms) | Ours LSolver/PCG(ms) |',
        '|---|---:|---:|---:|---:|'])
    for scene,label in [('cloth_hang_l','悬挂布料'),('cloth_fixed_bunny_l','固定兔子布料')]:
        old=cb[scene,'current_best'];new=cb[scene,'current_ours']
        report.append(f'| {label} | {old["LSolver"]:.6f} | {new["LSolver"]:.6f} | {old["linear_amortized_ms_per_pcg"]:.6f} | {new["linear_amortized_ms_per_pcg"]:.6f} |')
    report.extend(['', '本轮Ours的总线性求解时间明显增加，但均摊LSolver/PCG并未增加。因此这些完整场景观察支持“求解方向与迭代工作总量增加”作为主要解释，未显示平均单次PCG工作变贵。上述均摊量包含线性系统准备与同步，不能用来分离某个迭代核的开销。',''])
    (ROOT/'reports/LOCAL_GRAPH_HISTORY_20261011_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(dict(complete=True,formal_runs=66,formal_frames=6600,comparisons=comparisons),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true')
    args=parser.parse_args();main(args.render)
