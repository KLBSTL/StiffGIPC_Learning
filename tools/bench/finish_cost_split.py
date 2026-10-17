"""Seal measured stability, actual GPU activity attribution, and bounded conclusions."""
import ast
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cost_split import RAW,REPORT,BUILD,PROJECT,PARENT,read
from local_identity import record,verify,tree
from linux_runner import require,write_new
from analyze_cost_split import csv_new
import numpy as np

def main():
    require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    contracts=read(RAW/'contracts/audit.json');stable=read(RAW/'stability/audit.json');trace=read(RAW/'trace_analysis_retry1/summary.json')
    require(contracts['passed'] and stable['passed'] and stable['selected_calls']==2048 and trace['passed'] and trace['attribution_verified'],'Audit required')
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');m=read(RAW/'input_manifest.json');parent=read(REPORT/'PARENT.json')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe'],identity['nsys']['binary']]+packet['compiled']+packet['helpers'])
    verify([packet[k] for k in ['identity','manifest','protocol','runner','nsys_help']]+m['original_inputs']+[m['source']]+contracts['inputs']+stable['inputs']+trace['inputs'])
    verify(read(PARENT)['files']+[parent['parent']]+parent['original_source_files'])
    g=read(BUILD/'generation.json');verify([g[k] for k in ['template','continuation','generated']]+[parent['generation']])
    source=Path(g['template']['path']).read_text(encoding='utf-8');require(source.count(g['boundary'])==1,'Template boundary differs');prefix=source.split(g['boundary'])[0]
    for e in g['edits']:require(prefix.count(e['old'])==1,'Template edit differs');prefix=prefix.replace(e['old'],e['new'])
    e=g['qualification'];require(prefix.count(e['old'])==1,'Qualification edit differs');prefix=prefix.replace(e['old'],e['new'])
    require(prefix+Path(g['continuation']['path']).read_text(encoding='utf-8')==Path(g['generated']['path']).read_text(encoding='utf-8'),'Generated source differs')
    seeds=[ROOT/'tools/local'/n for n in ['prepare_cost_split.py','run_cost_split.py']]+[ROOT/'tools/bench'/n for n in ['analyze_cost_split.py','analyze_cost_trace.py','finish_cost_split.py']]
    search=[ROOT/'tools',ROOT/'tools/local',ROOT/'tools/bench'];pending=seeds[:];seen=set()
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                for base in search:
                    q=base/Path(*name.split('.')).with_suffix('.py')
                    if q.is_file():pending.append(q);break
    sources=[record(p) for p in seen]+tree(PROJECT)+tree(BUILD/'overlay')
    dest=REPORT/'sources';dest.mkdir()
    for r in sources:
        p=Path(r['path']);q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
    launches=list(csv.DictReader((REPORT/'tables/trace_launches.csv').open(encoding='utf-8-sig')))
    phase_stats=[]
    for name,method in dict.fromkeys((r['name'],r['method']) for r in launches):
        rows=[r for r in launches if (r['name'],r['method'])==(name,method)]
        values=lambda key:[float(r.get(key) or 0) for r in rows]
        target=np.array(values('target_union_us'))/np.array(values('span_us'))
        tiny=[float(r.get('first_8d_solve_us') or 0)+float(r.get('second_8d_solve_us') or 0) for r in rows]
        transpose=[float(r.get('ZT_projection_us') or 0)+float(r.get('WT_projection_us') or 0) for r in rows]
        row=dict(name=name,method=method,launches=len(rows),span_median_us=float(np.median(values('span_us'))),
            tiny_sum_median_us=float(np.median(tiny)),transpose_sum_median_us=float(np.median(transpose)),
            target_union_median_us=float(np.median(values('target_union_us'))),target_share_median=float(np.median(target)),
            target_share_p10=float(np.quantile(target,.1)),target_share_p90=float(np.quantile(target,.9)),
            interval_overlap_median_us=float(np.median(values('interval_overlap_us'))),interval_overlap_max_us=max(values('interval_overlap_us')))
        phase_stats.append(row)
    csv_new(REPORT/'tables/trace_share_statistics.csv',phase_stats)
    write_new(RAW/'trace_analysis_retry1/share_statistics.json',dict(rows=phase_stats,source=record(Path(__file__)),
        formula='median per-launch target interval union / same-launch span; not ratio of independent medians'))
    counted=contracts['checked_outputs']+stable['checked_outputs']+trace['normal_checked_outputs']+trace['traced_checked_outputs']
    require(counted==684,'Wrong independent output total')
    lines=['# P5：MAS稳定性与完整Graph成本拆分','',
        '2026-10-17。本轮完成固定原生诊断、稳定性测量和Nsight实际逐核归属，没有改生产算法/默认、没有跑100帧或上传。沿用父同一A/b/M/Z/W/L、原MAS/ABD、FP64投影和4MiB显式workspace。','',
        '**36合同、576个A/A窗口、684份完整输出独立核验通过；Nsight验证18窗口、576次Graph启动、5568个kernel、1920个copy/memset活动。预定最短全格通过的稳定批次为2048次。**','',
        '[预定协议](PLAN.md)；[逐段源码检查](SOURCE_AUDIT.md)。','',
        '## 主要发现','',
        '当前两次8维求解的固定开销值得优先做小原型。原balanced_tiny_solve每次以grid1/block1执行，实际trace为40寄存器；两次调用每apply合计中位约Bunny24.1µs、Mixed26.9µs。已确定的是该核活动区间的成本，尚未把L读取、FP64除法和指令依赖各自作用分开。','',
        '| fused完整apply，Nsight样本 | Bunny rhs | Mixed rhs |',
        '|---|---:|---:|']
    selected={r['name']:r for r in phase_stats if r['method']=='fused'}
    for title,key,percent in [('完整Graph活动跨度中位（µs）','span_median_us',False),('两次转置投影累计核时间中位（µs）','transpose_sum_median_us',False),
        ('两次8维求解累计核时间中位（µs）','tiny_sum_median_us',False),('两类目标核区间并集中位（µs）','target_union_median_us',False),
        ('同一次启动目标份额中位','target_share_median',True)]:
        values=[selected[n][key] for n in ['bunny_rhs','mixed_rhs']]
        lines.append('| '+title+' | '+' | '.join(f'{v*100:.2f}%' if percent else f'{v:.3f}' for v in values)+' |')
    lines += ['', '目标份额按每次启动的目标区间并集/同次跨度计算，再取中位，不能拿各列独立中位数相除。Bunny该份额10%–90%约56.1%–56.7%，Mixed约41.7%–42.2%。这里是带跟踪的固定动作样本，不是纯算术指令时间、PCG迭代时间或正式加速比。','',
        '## A/A稳定性：2048次批次通过','',
        '| 输入 | 每次图含apply数 | 128通过 | 512通过 | 2048通过 | 2048 B/A中位 | A/B相对MAD |',
        '|---|---:|---|---|---|---:|---:|']
    grid={(r['name'],r['capture_calls'],r['calls']):r for r in stable['summary']}
    for name in ['bunny_rhs','mixed_rhs']:
        for grain in [1,16]:
            r=grid[(name,grain,2048)]
            lines.append(f'| {name} | {grain} | '+ ' | '.join('通过' if grid[(name,grain,k)]['passed'] else '未通过' for k in [128,512,2048])+
                f' | {r["event_ratio_median"]:.6f} | {r["A_relative_MAD"]*100:.2f}% / {r["B_relative_MAD"]*100:.2f}% |')
    lines += ['', 'A和B是完全同一MAS Graph，标签只在CPU。三个长度一次测完，三轮、每格四个ABBA/BAAB四元组，576窗口/144四元组；未筛选最快批次。全格AND要求中位偏差≤2%、每轮≤5%、10%/90%分位在±10%、两臂MAD≤5%，完整逐格判定见表。2048的通过只支持这种长窗口条件，不会追溯认证前轮128次workspace结果，也不代表之后32次诊断窗口同样稳定。','',
        '## 实际Graph逐核分解（µs/一次apply，96次样本中位）','',
        '| 输入 | 方法 | ZT | 第一次8维解 | 输入投影 | 原MAS核 | WT | 第二次8维解 | 输出投影 | copy/memset累计 | 内部空隙并集 | 完整跨度 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in trace['summary']:
        keys=['ZT_projection_us','first_8d_solve_us','input_projection_us','native_MAS_us','WT_projection_us','second_8d_solve_us','output_projection_us','memory_busy_us','internal_gap_us','span_us']
        lines.append('| '+r['name']+' | '+r['method']+' | '+' | '.join(f'{r.get(k,0.):.3f}' for k in keys)+' |')
    lines += ['', '“原MAS核”含ABD、限制、FP32局部作用、延拓四核；global copy和两次零化在copy/memset列，未漏算。各核列是活动duration累计的中位，copy/memset同理；内部空隙=跨度−所有活动区间并集。各列独立中位数以及存在重叠的累计duration不能直接相加。','',
        '每次启动均核对Graph关联ID和kernel符号序列，MAS/reference/fused分别4/13/12个kernel。对齐后分别归入ZT、tiny、输入、MAS、WT、tiny、输出，不按名字模糊匹配。完整API启动/节点/符号/内存大小见原始SQLite与导出CSV。','',
        '## 跟踪误差与统计边界','',
        '- 初版CPU后处理假定GPU活动完全串行，实际时间戳有1002处相邻kernel区间重叠（最大352ns），检查停止。原脚本/日志和首个分析目录保留，没有输出伪造通过结果。',
        '- retry1只修正后处理，用区间并集计算空隙，同时保留累计duration和重叠量。数值、schema、18标签、API启动、5568符号/关联和1920内存活动检查都保留，没有重跑或改变GPU程序。完整单次启动累计重叠最大约.928µs；这不是单对区间的352ns。',
        '- 目标按启动份额中位为Bunny56.40%、Mixed41.91%；整个捕获的总目标并集/总跨度加权值分别约59.30%和38.40%，因尾部样本权重不同而有差异，二者均保留。未删离群或只报有利统计。',
        '- 下面普通/跟踪窗口各只有32次，刻意用于降低跟踪数据量，未作为正式性能测试。两个进程发生在不同时间，环境和工具影响混合，不能据比值估算纯CUPTI开销，更不能用小于1证明工具加速。','',
        '| 输入 | 方法 | trace/普通event窗口比中位（三次） |',
        '|---|---|---:|']
    for name in ['bunny_rhs','mixed_rhs']:
        for method in ['MAS','reference','fused']:
            rows=[r for r in trace['overhead'] if r['name']==name and r['method']==method]
            lines.append(f'| {name} | {method} | {float(np.median([r["event_ratio"] for r in rows])):.4f} |')
    lines += ['', '三次fused trace中位跨度分别约Bunny81.95/81.68/82.03µs、Mixed137.82/137.97/137.74µs；tiny合计也较一致。但不能将这些诊断一致性扩大为未插桩全程序净收益。GPU活动时间可能含调度/抢占，未采集硬件计数器，未声称测得纯global load/除法延迟或occupancy。','',
        '## 数值、身份与复现','',
        f'- 完整输出数：合同36，稳定性588（含12 warm），普通30（含12 warm），跟踪30（含12 warm），总684。最大独立CPU参考相对差为 {max(contracts["maximum_CPU_relative"],stable["maximum_CPU_relative"],trace["maximum_CPU_relative"]):.9g}。阈值仍≤1e-3，零输入精确零；CPU B_ref仍为存储局部系数对称后FP64作用，另有历史native(b)和Host/Graph校准。不是位置误差或PCG收敛实验。',
        '- readonly输入、MAS地址签名、scratch毒化/恢复、真实Graph后受控异常恢复、workspace哨兵均通过。所有GPU阶段Owned Job、显存、外来计算、UUID、电源检查通过；当前电源GUID52521609-efc9-4268-b9ba-67dea73f18b2（野兽模式）。共享WDDM桌面图形负载保留，未结束其他应用。',
        f'- 新exe SHA256 `{identity["exe"]["sha256"]}`。实际源码/对象/link/tlog、生成模板、cuBLAS DLL与编译器封存；Nsight版本 `{identity["nsys"]["version"].strip()}` 及二进制绑定。未完整封存外部SDK、driver/header和Nsight依赖DLL闭包。',
        '- Nsight以cuda/nvtx、node粒度、sample=none、cpuctxsw=none、CUDA profiler API capture运行。准备工作无cost标签，严格排除；18个cost NVTX窗口包括GPU完成同步，没有阶段内计时事件。完整nsys-rep/SQLite保留。',
        '- 原生产源码未改，未重跑旧29项全程序CTest或100帧，未上传。原材料、FullCCD、legacy .01/min6、rho1e-4及生产求解路径没有替换。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_cost_split.py prepare",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_cost_split.py build",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_cost_split.py seal",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_cost_split.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_cost_split.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_cost_split.py stability",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_cost_split.py stability",
        "# 稳定性通过后依次运行 run_cost_split.py profile、trace",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_cost_trace.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_cost_split.py",'```','',
        '入口拒绝覆盖旧目录；复测需新批次路径。表：[A/A判定](tables/stability_summary.csv)、[所有576窗口](tables/stability_windows.csv)、[逐核](tables/trace_kernels.csv)、[内存活动](tables/trace_memory.csv)、[逐启动/重叠](tables/trace_launches.csv)、[同次份额](tables/trace_share_statistics.csv)、[跟踪对照](tables/trace_overhead.csv)。','',
        '## 有限下一步','',
        '优先做一个小的8维求解原型：L协作预取到共享内存，保持原FP64代入公式、原求和顺序，不先换显式逆或降低精度。先独立合同和全apply动作校准，再以2048次长窗口与原核对照，同时保留MAS A/A和三轮。通过后才考虑把转置归约最终阶段与求解进一步融合。','',
        '本轮没有声称实现加速。两次tiny只占固定apply的一部分；即使理想消除该区间，也不能据此承诺整程序2倍。历史Mixed慢13.4%的唯一原因仍未证实，本轮定位的是可量化的下一优化目标。']
    with (REPORT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+sources
    records += [record(p) for p in (ROOT/'build').glob('cost_split*_controller.log') if 'finish' not in p.name]
    files=sorted({r['path']:r for r in records}.values(),key=lambda r:r['path']);verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='native_cost_split.v1',files=files,parent=parent['parent'],existing_source_edits=[],
        contracts=36,stability_windows=576,independently_checked_outputs=684,trace_launches=576,trace_kernels=5568,
        trace_memory_events=1920,selected_stable_calls=2048,production_solver_modified=False,performance_certified=False,
        next_finite_decision='FP64 8d L cooperative prefetch with original arithmetic ordering; 2048-call pairs and MAS control before integration'))
    verify(read(REPORT/'DELIVERY_SEAL.json')['files']);print(dict(sealed=True,files=len(files),stable_calls=2048,kernels=5568))
if __name__=='__main__':main()
