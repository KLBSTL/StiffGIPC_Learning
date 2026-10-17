"""Seal the additive full native apply diagnostic, without promoting its default."""
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_full_apply_graph import RAW,REPORT,BUILD,PROJECT,FROZEN,read
from local_identity import record,verify,tree
from linux_runner import require,write_new

def main():
    require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Delivery already sealed')
    d=read(RAW/'analysis/summary.json');require(d['implementation_checks_passed'],'Numerical audit failed')
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify(identity['sources']+identity['build_evidence']+identity['runtime']+[identity['exe']]+packet['compiled']+packet['helpers'])
    verify([packet[k] for k in ['identity','manifest','protocol','runner']]+manifest['original_inputs']+manifest['generated_inputs']+d['inputs'])
    parent=read(REPORT/'PARENT.json')
    for p in parent['parents']:verify(read(p['seal']['path'])['files']+[p['seal']])
    original=(FROZEN/'solver/MASPreconditioner.cuh').read_bytes();overlay=(BUILD/'overlay/solver/MASPreconditioner.cuh').read_bytes()
    addition=b'    static int full_apply_graph_fixture(const char*,const char*,bool);\n'
    require(overlay.count(addition)==1 and overlay.replace(addition,b'')==original,'Overlay is not declaration-only')
    abd=(FROZEN/'linear_system/preconditioner/abd_preconditioner.cu').read_bytes();kernel=(BUILD/'overlay/abd_kernel.inc').read_bytes().rstrip(b'\n')
    require(kernel in abd and kernel.count(b'abd_preconditioner_apply_kernel')==1,'ABD original kernel differs')
    # Retain Python helper closure used by preparation, compilation, ownership, and independent formulas.
    seeds=[ROOT/'tools/local'/n for n in ['prepare_full_apply_graph.py','retry_full_apply_graph_build.py','run_full_apply_graph.py']]
    seeds += [ROOT/'tools/bench'/n for n in ['analyze_full_apply_graph.py','finish_full_apply_graph.py']]
    search=[ROOT/'tools',ROOT/'tools/local',ROOT/'tools/bench'];pending=list(seeds);seen=set()
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                rel=Path(*name.split('.')).with_suffix('.py')
                for base in search:
                    q=base/rel
                    if q.is_file():pending.append(q);break
    source_rows=[record(p) for p in seen]+tree(PROJECT)+tree(BUILD/'overlay')
    source_dir=REPORT/'sources';source_dir.mkdir()
    for row in source_rows:
        p=Path(row['path']);q=source_dir/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
    aggregate={(r['name'],r['comparison']):r for r in d['aggregate']}
    cases=[c for c in manifest['cases'] if c['probe']!='zero']
    lines=['# P5：固定原生完整 apply 的 Host / 私有 Graph 对照','',
        '2026-10-17。复用父融合阶段同一 A/b/M/Z/W/L；实际原生FP32 MAS、原ABD核、全局identity copy及FP64 balanced投影全部保留。未推进场景，未修改生产求解器，未运行100帧或上传。','',
        f'**54组输出合同、648组测量及742份完整输出独立核验通过；有限性能门槛={d["finite_performance_gate_passed"]}。最大CPU参考相对差 {d["maximum_CPU_relative"]:.9g}。**','',
        '[预定协议](PLAN.md)；[逐段实现检查](SOURCE_AUDIT.md)。','',
        '## 融合版 / cuBLAS reference：完整 apply 配对耗时','',
        '| 固定输入 | Host16 | Graph16 | Graph1×16 | Graph16三轮 | 改善轮数 |',
        '|---|---:|---:|---:|---|---:|']
    for c in cases:
        name=c['name'];rows=[aggregate[(name,'fused/reference '+m)] for m in manifest['executions']];g=rows[1]
        lines.append('| '+name+' | '+' | '.join(f'{r["event_ratio_median"]:.4f}' for r in rows)+
            f' | {g["round1"]:.4f} / {g["round2"]:.4f} / {g["round3"]:.4f} | {g["improving_rounds"]}/3 |')
    lines += ['', '比值<1表示融合更省，均为同输入18组配对的CUDA event比值中位数。Graph16下四输入约降低3.4%–5.4%；要求四者均≤.95且各至少2/3轮改善，实际未全部满足。保持原默认，不提升为生产优化。',
        '', '**上一轮Mixed完整apply慢13.4%的现象未在本轮配对中稳定重现。不能因此宣称问题已修复或锁定唯一原因。** 本轮输入/M/Z与父相同，原生动作校准通过；独立工程代码生成环境、WDDM负载、提交顺序及短窗口仍有区别。',
        '', '## Graph回放相对Host提交','',
        '| 输入 | Graph16/Host reference | Graph1×16/Host reference | Graph16/Host fused | Graph1×16/Host fused |',
        '|---|---:|---:|---:|---:|']
    for c in cases:
        name=c['name'];keys=['graph16/host16 reference','graph1x16/host16 reference','graph16/host16 fused','graph1x16/host16 fused']
        lines.append('| '+name+' | '+' | '.join(f'{aggregate[(name,k)]["event_ratio_median"]:.4f}' for k in keys)+' |')
    lines += ['', '这是固定完整apply的局部结果，包含全局copy、ABD及原生MAS。Graph对象为探针私有，使用原 cudaStreamPerThread；不是生产PCG Graph、整帧加速比或独立CUDA流。Host event含提交空隙，二者差不能精确等同纯API成本。',
        '', '## 关键发现：cuBLAS捕获产生额外工作区节点','',
        '| 每个完整apply | kernel | memcpy | memset | MemAlloc | MemFree | 总节点 |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| MAS（含全局copy与ABD） | 4 | 1 | 2 | 0 | 0 | 7 |',
        '| balanced reference | 13 | 2 | 2 | 2 | 2 | 21 |',
        '| balanced fused | 12 | 1 | 2 | 2 | 2 | 19 |','',
        '四输入均观察到相同每apply节点构成，16次链的节点数为以上16倍。CUDA13本地driver_types.h明确枚举10为MemAlloc、11为MemFree。原apply显式路径没有分配/释放，原生MAS捕获亦无此节点，新增节点由其cuBLAS调用内部产生；当前BalancedBlas没有显式用户workspace。',
        '', '因此前轮阶段探针仅覆盖非转置投影，没有覆盖完整链中这类开销；不能用两个非转置核的收益代替完整apply收益。节点存在是确定事实，其运行成本及对Mixed异常的因果贡献仍未隔离。下一轮优先单独验证预分配cuBLAS workspace，先要求图内分配/释放节点消失、数值动作不变，再做相同配对；不同时改投影或MAS算法。',
        '', '## 各臂独立中位数（微秒/apply）','',
        '| 输入 | 方法 | 执行 | event | host完成窗口 |',
        '|---|---|---|---:|---:|']
    for r in d['costs']:lines.append(f'| {r["name"]} | {r["method"]} | {r["execution"]} | {r["event_us_per_call_median"]:.3f} | {r["host_us_per_call_median"]:.3f} |')
    lines += ['', '**独立中位数之比不等于配对比值中位数。** 例如Bunny rhs Graph16融合/参考独立中位数约81.89/72.09，但配对比值中位数为.9662；Mixed rhs也出现方向差别。全部样本保留，不能只选有利统计量认证收益。该现象和大范围离群共同表明此桌面短测成本结论不够稳定。',
        '', '## 数值、恢复及计时范围','',
        '- Bunny14 n2/Mixed15 n2，rhs和父pair1 reference解的原始CSR真残差，另每场景一个零输入合同。材料、FullCCD、rho1e-4、原PCG/物理轨迹均未改，但本轮没有运行这些生产步骤。',
        '- 合同6输入×3方法×3执行=54输出；计时4输入×3轮×6组×9组合=648输出，另36份warm和4份仪器化MAS输出。独立Python重建原始CSR/局部系数公式，检查全部742份；没有仅采样首尾。',
        '- CPU B_ref对存储FP32局部逆对称化后以FP64作用；不是原生FP32逐位模拟。C++/Python阈值均≤1e-3；历史原生b动作与各执行方式互校准也≤1e-3。选定残差输入不是新的PCG收敛误差证明。',
        '- 所有只读MAS缓冲区、r/Z/W/L/ABD逆逐字节不变；legacy scratch和探针中间向量先主动破坏后恢复；Graph已执行后注入异常，再验证scratch字节与地址签名恢复。生产PCG没有对象参与，未夸称生产回退已验证。',
        '- 统一16次完整作用，三种执行分别逐次Host、一次Graph16、连续16次Graph1。每批使用单份输出与scratch，保持真实依赖，没有32输出银行。批内rhs固定，仍不同于PCG每迭代变化的残差依赖。',
        '- reset/同步、D2H、验证、导出在计时外，费用另记。host总窗口、host提交窗口、CUDA event逐批记录。所有分配/构图、首次回放成本保留在graphs表。',
        '- 五个balanced阶段及MAS内部六阶段仅作独立仪器化观察，均在配对之外，仅各一次。不能相加还原无仪器耗时，也不能凭一次观测认定最慢核。',
        '- WDDM共享桌面有明显图形负载；不结束其他应用、不删离群。当前电源GUID为52521609-efc9-4268-b9ba-67dea73f18b2（野兽模式），运行UUID、电源及Owned Job保护通过。当前有限门槛未通过，不认证正式性能。',
        '', '## 构建、复核及复现','',
        f'- Probe SHA256 `{identity["exe"]["sha256"]}`；包含父封存原生MAS原文，新增头仅静态入口声明，ABD核原文字节复核通过。实际编译源、对象、link追踪、编译器、cuBLAS DLL和源码前后绑定。外部SDK/driver/header闭包未完整封存。',
        '- 首轮新driver漏包含原ABD类型头，编译失败；补原gipc/type_define.h后显式retry1通过（35警告/0错误），保留首轮日志。未在失败构建下启动GPU。没有放宽验证或保护条件。',
        '- 父投影574文件、父融合2315文件交付前重新核验；本轮未重跑旧29项全程序CTest，因为生产源码/程序未改。验收范围为新增独立回放工程及本轮合同。',
        '', '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_full_apply_graph.py prepare",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_full_apply_graph.py build",
        '# 历史首轮失败与显式retry1均保留；当前源码含所需类型头，全新构建可直接成功。',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_full_apply_graph.py seal",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_full_apply_graph.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_full_apply_graph.py measure",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_full_apply_graph.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_full_apply_graph.py",'```','',
        '入口拒绝覆盖现有批次。数据：[逐批时间](tables/timings.csv)、[配对](tables/pairs.csv)、[轮次与离群范围](tables/aggregate.csv)、[独立输出校验](tables/output_CPU.csv)、[Graph节点与冷启动](tables/graphs.csv)、[仪器化阶段](tables/instrumented_stages.csv)。全部原始输出在本地runs目录并纳入封存。']
    with (REPORT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+source_rows
    records += [record(p) for p in (ROOT/'build').glob('full_apply_graph*_controller.log') if 'finish' not in p.name]
    files=sorted({r['path']:r for r in records}.values(),key=lambda r:r['path']);verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='native_full_apply_graph.v1',files=files,parents=parent['parents'],
        existing_source_edits=[],contracts=54,measurements=648,independently_checked_outputs=742,
        finite_performance_gate_passed=d['finite_performance_gate_passed'],production_solver_modified=False,performance_certified=False,
        next_finite_decision='Isolate explicit cuBLAS workspace: require removal of graph allocation/free nodes and unchanged action before timing; no online or 100-frame promotion'))
    verify(read(REPORT/'DELIVERY_SEAL.json')['files']);print(json.dumps(dict(sealed=True,files=len(files),
        finite_performance_gate=d['finite_performance_gate_passed'])),flush=True)
if __name__=='__main__':main()
