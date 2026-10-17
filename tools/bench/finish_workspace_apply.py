"""Preserve the workspace structural result and rejected performance signal."""
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_workspace_apply import RAW,REPORT,BUILD,PROJECT,PARENT,read
from local_identity import record,verify,tree
from linux_runner import require,write_new

def main():
    require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    d=read(RAW/'analysis/summary.json');require(d['implementation_checks_passed'] and d['node_gate_passed'],'Implementation audit failed')
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json');parent=read(REPORT/'PARENT.json')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]+packet['compiled']+packet['helpers'])
    verify([packet[k] for k in ['identity','manifest','protocol','runner']]+manifest['original_inputs']+[manifest['source']]+d['inputs'])
    verify(read(PARENT)['files']+[parent['parent']]+parent['original_source_files'])
    generation=read(BUILD/'generation.json');verify([generation['parent_fixture'],generation['continuation'],generation['generated'],parent['generation']])
    original=Path(generation['parent_fixture']['path']).read_text(encoding='utf-8')
    require(original.count(generation['prefix_boundary'])==1,'Parent prefix ambiguous')
    prefix=original.split(generation['prefix_boundary'])[0]
    for edit in generation['edits']:
        require(prefix.count(edit['old'])==1,'Template edit ambiguous');prefix=prefix.replace(edit['old'],edit['new'])
    require(prefix+Path(generation['continuation']['path']).read_text(encoding='utf-8')==Path(generation['generated']['path']).read_text(encoding='utf-8'),'Generated source differs')
    seeds=[ROOT/'tools/local'/n for n in ['prepare_workspace_apply.py','run_workspace_apply.py']]
    seeds += [ROOT/'tools/bench'/n for n in ['analyze_workspace_apply.py','finish_workspace_apply.py']]
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
    source_dir=REPORT/'sources';source_dir.mkdir()
    for row in sources:
        p=Path(row['path']);target=source_dir/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(p.read_bytes())
    aggregate={(r['name'],r['method'],r['execution']):r for r in d['aggregate']}
    names=[c['name'] for c in manifest['cases'] if c['probe']!='zero']
    lines=['# P5：cuBLAS显式工作区对照','',
        '2026-10-17。独立诊断原型已完成并本机验证。只改workspace选择，不改MAS、投影、PCG或物理代码，不运行100帧、不上传。使用父同一A/b/M/Z/W/L、同一原生MAS、同一cuBLAS handle、流和向量地址。','',
        f'**结构/数值门槛通过：108合同、1296组测量、1476份完整输出独立核验。最大CPU参考相对差 {d["maximum_CPU_relative"]:.9g}，最大paired两臂输出相对差 {d["maximum_paired_output_relative"]:.9g}。**','',
        f'**有限性能门槛={d["finite_performance_gate_passed"]}（候选成本信号={d["finite_cost_signal"]}，MAS负对照={d["negative_control_passed"]}）。保留原生产默认，不认证加速。**','',
        '[预定协议](PLAN.md)；[逐段检查](SOURCE_AUDIT.md)。','',
        '## 确定完成：Graph内工作区分配/释放节点消失','',
        '| 每apply | 默认总节点 | 显式总节点 | 默认MemAlloc/MemFree | 显式MemAlloc/MemFree | kernel数前/后 |',
        '|---|---:|---:|---:|---:|---:|',
        '| MAS（含global copy/ABD） | 7 | 7 | 0/0 | 0/0 | 4/4 |',
        '| balanced reference | 21 | 17 | 2/2 | 0/0 | 13/13 |',
        '| balanced fused | 19 | 15 | 2/2 | 0/0 | 12/12 |','',
        '六合同输入与四计时输入均验证；16次捕获节点数为每apply的16倍，显式臂仍无memory节点。原GPU计算kernel数量没有减少；不能按节点减少比例推算耗时。此结果证明本机cuBLAS捕获工作区节点可通过显式buffer移除，尚不能证明这些节点是历史Mixed慢13.4%的主因。','',
        '使用4MiB payload、前后各256字节哨兵。setter在同一handle/stream上于计时外选择，默认臂用SetStream重置pool，user臂随后SetWorkspace；没有用空指针/0大小冒充默认。按NVIDIA CUDA13.0文档，设置stream会重置workspace，显式buffer需满足对齐，Graph捕获可能产生memory节点。[官方工作区说明](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublassetworkspace)、[Graph支持](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cuda-graphs-support)。','',
        '## 主要对照：单apply Graph回放128次，fused显式/默认','',
        '| 输入 | 配对event比中位 | 耗时变化 | 独立中位数之比 | 三轮配对中位 | 改善轮数 |',
        '|---|---:|---:|---:|---|---:|']
    for r in d['gate']:
        lines.append(f'| {r["name"]} | {r["event_ratio_median"]:.4f} | {(r["event_ratio_median"]-1)*100:+.2f}% | {r["independent_event_medians_ratio"]:.4f} | '+
            f'{r["round1"]:.4f} / {r["round2"]:.4f} / {r["round3"]:.4f} | {r["improving_rounds"]}/3 |')
    lines += ['', '比值<1表示显式更省。每行18组邻接AB/BA对，均为128次完整apply的窗口。Bunny约降低7%，Mixed仅约1%–2%，未达到四输入均≤.95且各至少2/3轮改善的预定门槛。不是整帧、PCG或2倍加速结论。','',
        '## MAS负对照：没有cuBLAS调用也存在显著波动','',
        '| 输入，graph1x128 | 配对event比中位 | 三轮 | 是否在.95–1.05 |',
        '|---|---:|---|---|']
    for r in d['controls']:
        lines.append(f'| {r["name"]} | {r["event_ratio_median"]:.4f} | {r["round1"]:.4f} / {r["round2"]:.4f} / {r["round3"]:.4f} | '+('是' if .95<=r['event_ratio_median']<=1.05 else '否')+' |')
    lines += ['', 'Mixed的负对照偏差约6%–9%，大于候选1%–2%的变化，所以本轮不能将Mixed小幅改善归因于workspace。WDDM共享桌面、GPU图形负载和频率/缓存状态仍可能影响成本；没有结束其他应用或删除离群。Bunny观察到较大改善，但整个预定门槛仍未通过，不挑选Bunny单独认证普遍收益。','',
        '## 所有方法/执行的显式/默认配对event中位','',
        '| 输入 | 方法 | Host128 | Graph16×8 | Graph1×128 |',
        '|---|---|---:|---:|---:|']
    for name in names:
        for method in manifest['methods']:
            lines.append('| '+name+' | '+method+' | '+' | '.join(f'{aggregate[(name,method,e)]["event_ratio_median"]:.4f}' for e in manifest['executions'])+' |')
    lines += ['', '例如Bunny reference的Graph16×8反而变慢约7%–13%；移除内存节点并未带来普遍收益。其原因未单独隔离，不将所有时间差都归因于分配节点。','',
        '## graph1x128独立中位数：微秒/apply','',
        '| 输入 | 方法 | 默认event | 显式event | 默认host完成窗口 | 显式host完成窗口 |',
        '|---|---|---:|---:|---:|---:|']
    costs={(r['name'],r['method'],r['execution'],r['workspace']):r for r in d['costs']}
    for name in names:
        for method in manifest['methods']:
            a=costs[(name,method,'graph1x128','default')];b=costs[(name,method,'graph1x128','user4MiB')]
            lines.append(f'| {name} | {method} | {a["event_us_per_call_median"]:.3f} | {b["event_us_per_call_median"]:.3f} | {a["host_us_per_call_median"]:.3f} | {b["host_us_per_call_median"]:.3f} |')
    lines += ['', '独立中位数之比与配对比值中位数分别计算，不能互换。当前批次128次/窗口，不直接拿旧16次批次或旧exe的时间作加速比。没有用工作区节点数或单个最快值认证速度。','',
        '## 验证与边界','',
        '- 复用Bunny14 n2、Mixed15 n2的原rhs/父pair1真残差，并加零输入。独立重建原始CSR上的W=A Z、粗LLT和CPU局部参考，完整复核1476输出；两臂输出逐对额外互比。零输入全量精确零。',
        '- CPU B_ref对存储FP32局部系数对称化后以FP64作用，不是FP32原子MAS逐位模拟；因此同时核对历史native(b)、两workspace/Host/Graph。阈值仍≤1e-3，没有放宽原精度或停止条件。这里的动作差不是物理位置误差或PCG真残差。',
        '- workspace越界哨兵、同handle配置getter、readonly输入、MAS地址签名、scratch毒化/恢复，以及两种workspace真实Graph后异常恢复均通过。最终实际切回默认pool再执行Host动作，之后恢复所有scratch。',
        '- 默认和user共用同一Space/handle，捕获图共存但串行执行。图/exec先销毁，handle随后销毁，workspace最后释放。未验证并发、多线程、其他GPU架构或生产PCG回退。',
        '- 每对workspace紧邻并AB/BA平衡，但中间有窗口外输出校验和文件写入。每批128次完整动作，仍为同一rhs重复，不是实际每迭代变化的PCG依赖链。',
        '- select/reset、D2H/数值/guard/导出分别计时，均不进入apply窗口；Graph构造与首次回放也保留。workspace分配和初始化在窗口外但未单独计时，不能据本轮稳态结果声称冷启动净收益。总Owned Job wall见result.json。',
        '- 当前电源GUID52521609-efc9-4268-b9ba-67dea73f18b2（野兽模式）、GPU UUID、Owned Job、显存和外来计算检查通过；未放宽保护。原生probe运行合同约4.109s、计时约21.844s；这些是进程总wall，不能当apply窗口。',
        '', '## 构建与复现','',
        f'- 新exe SHA256 `{identity["exe"]["sha256"]}`。一次构建成功，35警告/0错误；父936文件及本轮全部编译前后身份核验。源码模板三处替换独立重建一致，实际编译对象/link/tlog、库DLL、编译器均绑定。外部SDK/header/driver完整闭包未封存。',
        '- 原生产文件/默认未改，未重跑旧29项全程序CTest，未重新推进场景、测100帧或上传GitHub。验收范围为新增workspace工程、节点和输出合同。',
        '', '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_workspace_apply.py prepare",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_workspace_apply.py build",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_workspace_apply.py seal",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_workspace_apply.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_workspace_apply.py measure",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_workspace_apply.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_workspace_apply.py",'```','',
        '固定入口拒绝覆盖本批结果；重测需使用新批次路径。数据：[逐批时间](tables/timings.csv)、[全部648对](tables/pairs.csv)、[三轮/离群范围](tables/aggregate.csv)、[完整输出核验](tables/output_CPU.csv)、[节点与冷启动](tables/graphs.csv)、[各窗口费用](tables/costs.csv)。全部原始输出保留在本地runs并纳入封存。','',
        '## 有限决定','',
        '显式workspace作为数值正确、可移除捕获内存节点的诊断选项保留；不把它当作已解决Mixed性能的生产优化。当前证据不支持继续优先围绕分配节点追求2倍速度。下一步先拆剩余转置rank8投影与8维求解的真实Graph成本，测清可优化份额，再决定是否做专用归约原型；不同时改MAS、粗空间和PCG外层，也不跳到100帧。']
    with (REPORT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+sources
    records += [record(p) for p in (ROOT/'build').glob('workspace_apply*_controller.log') if 'finish' not in p.name]
    files=sorted({r['path']:r for r in records}.values(),key=lambda r:r['path']);verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='explicit_workspace.v1',files=files,parent=parent['parent'],
        existing_source_edits=[],contracts=108,measurements=1296,independently_checked_outputs=1476,
        node_gate_passed=True,finite_performance_gate_passed=d['finite_performance_gate_passed'],
        production_solver_modified=False,performance_certified=False,
        next_finite_decision='Profile transpose rank8/coarse solve cost before a dedicated reduction prototype; preserve default; no online/100-frame promotion'))
    verify(read(REPORT/'DELIVERY_SEAL.json')['files']);print(json.dumps(dict(sealed=True,files=len(files),
        node_gate=True,performance_gate=d['finite_performance_gate_passed'])),flush=True)
if __name__=='__main__':main()
