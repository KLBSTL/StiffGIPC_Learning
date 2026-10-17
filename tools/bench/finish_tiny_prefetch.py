"""Finite L-prefetch conclusion, complete evidence archive, no production promotion."""
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_tiny_prefetch import RAW,REPORT,BUILD,PROJECT,PARENT,read,edited
from local_identity import record,verify,tree
from linux_runner import require,write_new
from analyze_tiny_prefetch import csv_new
import numpy as np

def main():
    require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    actions=['tiny','contracts','tiny_measure','measure'];audits={a:read(RAW/a/'audit.json') for a in actions}
    require(all(a['passed'] for a in audits.values()),'Independent audits required')
    require(sum(a['checked_outputs'] for a in audits.values())==1056,'Wrong independently checked outputs')
    identity=read(BUILD/'identity.json');packet=read(RAW/'runtime_seal.json');manifest=read(RAW/'input_manifest.json');parent=read(REPORT/'PARENT.json')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]+packet['compiled']+packet['helpers'])
    verify([packet[k] for k in ['identity','manifest','protocol','amendment','inspector_amendment','runner','production_before']]+manifest['original_inputs']+manifest['generated_inputs']+[manifest['source']])
    verify(read(PARENT)['files']+[parent['parent'],parent['generation']]+parent['original_source_files']+read(RAW/'production_before.json'))
    for a in audits.values():verify(a['inputs'])
    g=read(BUILD/'generation.json');verify([g[k] for k in ['template','header_original','header_generated','kernel','continuation','generated']])
    source=Path(g['template']['path']).read_text(encoding='utf-8');require(source.count(g['boundary'])==1,'Ambiguous setup')
    reconstructed=edited(source.split(g['boundary'])[0],[(e['old'],e['new']) for e in g['edits']])+Path(g['continuation']['path']).read_text(encoding='utf-8')
    require(reconstructed==Path(g['generated']['path']).read_text(encoding='utf-8'),'Generated fixture differs')
    header=edited(Path(g['header_original']['path']).read_text(encoding='utf-8'),[(e['old'],e['new']) for e in g['header_edits']])
    require(header==Path(g['header_generated']['path']).read_text(encoding='utf-8'),'Generated header differs')
    math_original=Path(g['header_original']['path']).parent/'native_balanced_math.h'
    require(math_original.read_bytes()==(BUILD/'overlay/linear_system/solver/native_balanced_math.h').read_bytes(),'Original math sibling differs')
    require(g['arithmetic_loops_identical'],'Arithmetic identity missing')
    # Retain transitive local Python code, not just the entry scripts.
    sass=read(RAW/'sass_analysis.json');require(sass['passed'],'Static compiled audit required')
    verify([sass[k] for k in ['source','tool','exe']]+sass['artifacts'])
    seeds=[ROOT/'tools/local'/n for n in ['prepare_tiny_prefetch.py','run_tiny_prefetch.py']]+[ROOT/'tools/bench'/n for n in ['analyze_tiny_prefetch.py','analyze_tiny_sass.py','finish_tiny_prefetch.py']]
    pending=seeds[:];seen=set();search=[ROOT/'tools',ROOT/'tools/local',ROOT/'tools/bench']
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else ([n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for name in names:
                for base in search:
                    q=base/Path(*name.split('.')).with_suffix('.py')
                    if q.is_file():pending.append(q);break
    sources=[record(p) for p in seen]+tree(PROJECT)+tree(BUILD/'overlay')
    dest=REPORT/'sources';dest.mkdir()
    for r in sources:
        p=Path(r['path']);q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
    first=ROOT/'reports/innovation_p5_tiny_prefetch_20261017/FAILED_ATTEMPT_SEAL.json';failure=read(first)
    verify(failure['files'])
    for row in failure['archive_mapping']:
        verify([row['archive']]);require(row['original']['sha256']==row['archive']['sha256'] and row['original']['bytes']==row['archive']['bytes'],'Failed-attempt archive differs')
    controls=audits['measure']['all_controls_passed'];signal=audits['measure']['all_AB_signal']
    eligible=controls and signal
    tiny=audits['tiny_measure']['summary'];full=audits['measure']['summary']
    next_step=('固定A/b的完整PCG三臂配对验证，再考虑生产；不直接上100帧。' if eligible else
        '不接入生产。若A/A失败先复现长窗口稳定性；本轮A/A全过但Bunny未达收益门槛，下一小原型只缓存8维中间向量a，保持原序FP64代入，先检查是否寄存器溢出，再数值合同和2048次配对。')
    outcome='达到本轮固定apply候选门槛；尚未接入生产。' if eligible else '数值合同通过，但未达到本轮完整apply推广门槛；保留为诊断原型。'
    max_cpu=max(a['maximum_CPU_relative'] for a in audits.values())
    max_pair=max(a.get('maximum_original_relative',0) for a in audits.values())
    lines=['# P5：8维求解 L 协作预取原型实测','',
        '2026-10-17。'+outcome,'',
        '**独立检查1056份完整输出；70组小系统的原核/候选/Host/Graph逐字节相同。固定系统为Bunny与Mixed，2048次/窗口、三轮、ABBA/BAAB；没有运行真实帧或改生产默认。**','',
        '[协议](PLAN.md)、[数值门槛修正001](PLAN_AMENDMENT_001.md)、[检查工具修正002](PLAN_AMENDMENT_002.md)、[源码检查](SOURCE_AUDIT.md)。','',
        '## 改动与正确性','',
        '- 一个32线程块协作读取64个FP64系数到512字节共享内存，同步后只由线程0按原始两行循环前/回代入。原核保留1线程；公式、累加顺序、FP64除法、编译浮点设置均不变。没有显式逆或降精度。','',
        f'- 小核420份合同输出全部逐位一致；独立Decimal 80位十进制参考最大相对差{audits["tiny"]["maximum_CPU_relative"]:.9g}。本机NumPy longdouble只有52位尾数，未误用作高精度。另检查SciPy和归一化后向误差、upper NaN、缩放、零输入、readonly及两侧哨兵。','',
        f'- 完整apply合同36份、计时阶段396份；全部独立CPU误差≤1e-3，最大原动作相对差{max_pair:.9g}（门槛1e-5）。计时小核另204份。所有1056份输出的CPU最大相对差{max_cpu:.9g}。完整MAS包含FP32归约，不能要求逐位相同，实际差异逐项保留。','',
        '- 显式cuBLAS workspace、原ABD/MAS、CSR/W/L关系、Graph节点/函数/线程配置、readonly/MAS签名、scratch毒化和真实候选Graph后异常恢复、workspace哨兵均通过。两种完整apply每次同为15节点/12核，两处tiny以外不变；Graph16是240节点。','',
        '## 小核独立配对：包含同样的64字节RHS重置（µs/操作）','',
        '| 输入 | 图内操作数 | 原窗口中位 | 预取窗口中位 | 配对候选/原 | 配对加速比 | 三轮候选/原 |','|---|---:|---:|---:|---:|---:|---|']
    for r in tiny:
        lines.append(f'| {r["name"]} | {r["capture_calls"]} | {r["A_median_us"]:.3f} | {r["B_median_us"]:.3f} | {r["event_ratio_median"]:.6f} | {r["speedup"]:.4f} | '+ '/'.join(f'{r[k]:.4f}' for k in ['round1','round2','round3'])+' |')
    lines += ['', '每次kernel前D2D重置8个double，防止重复求解改变输入。上述收益包括同样copy、Graph/调度与kernel，不能称纯算术核延迟。独立窗口中位数与四元组均值比的中位数不同，配对加速比用后者的倒数。小核没有单独A/A，完整MAS控制的有效范围见下。','',
        '## 完整fused apply配对（µs/apply）','',
        '| 输入 | 图内apply数 | 原窗口中位 | 预取窗口中位 | 配对候选/原 | 配对加速比 | 三轮候选/原 | 对应MAS A/A通过 |','|---|---:|---:|---:|---:|---:|---|---|']
    for r in full:
        if r['comparison']!='AB':continue
        control=next(x for x in full if x['name']==r['name'] and x['capture_calls']==r['capture_calls'] and x['comparison']=='AA')
        lines.append(f'| {r["name"]} | {r["capture_calls"]} | {r["A_median_us"]:.3f} | {r["B_median_us"]:.3f} | {r["event_ratio_median"]:.6f} | {r["speedup"]:.4f} | '+ '/'.join(f'{r[k]:.4f}' for k in ['round1','round2','round3'])+f' | {control["AA_passed"]} |')
    lines += ['', '## 同一MAS Graph的A/A控制','',
        '| 输入 | 图内apply数 | 配对B/A中位 | 三轮B/A | p10/p90 | A/B相对MAD | 通过 |','|---|---:|---:|---|---|---|---|']
    for r in full:
        if r['comparison']!='AA':continue
        lines.append(f'| {r["name"]} | {r["capture_calls"]} | {r["event_ratio_median"]:.6f} | '+ '/'.join(f'{r[k]:.4f}' for k in ['round1','round2','round3'])+
            f' | {r["p10"]:.4f}/{r["p90"]:.4f} | {r["A_relative_MAD"]:.2%}/{r["B_relative_MAD"]:.2%} | {r["AA_passed"]} |')
    lines += ['',f'A/A全格通过={controls}；完整AB全格达到预定中位<.98且每轮<1={signal}。门槛没有在观察结果后调整，全部四元组、原始窗口、分位和host完成时间保留。未筛选快样本或只报更有利粒度。','',
        '## 成本解释与范围','',
        '此原型只尝试缩短L的串行读取；a仍在全局内存中逐步更新，16次有数据依赖的除法仍串行，两次转置/原MAS/其他kernel也保留。预取引入512字节共享、一个同步及32线程的调度，不能从“共享内存更快”直接推导总收益。没有采集本轮Nsight/硬件计数器，不能把窗口差异全部归因于某条load。','',
        '实际sm_86二进制反汇编另行核验：原/候选静态LDG.E.64指令位置136/70，候选新增LDS.64 68处、STS.64 2处和一个barrier；DFMA同为80处、STG.E.64同为2处。循环、分支和谓词会改变动态次数，这些是静态位置而不是运行时访存次数/事务/周期。原与候选均40寄存器/线程、local=0，候选另占512字节shared。它证明部分全局读取确实转成shared，并不证明剩余a读写或除法的独占时间。完整原SASS、实际编译静态符号、cuobjdump身份和命令保留；两次最初按未修饰名猜测的空dump也保留，未用于结论。','',
        '父逐核诊断的两次tiny合计约Bunny24.1µs、Mixed26.9µs，但那是另一次带跟踪运行；不能与本轮未插桩窗口直接相减计算节省。即使完整消除此区间，也仅优化预条件作用的一部分，不能保证整程序2倍。','',
        '本轮GPU为共享WDDM RTX3070 Laptop，保留图形应用负载。电源GUID52521609-efc9-4268-b9ba-67dea73f18b2（野兽模式），不是安静模式；各运行均记录电源/UUID/显存/外来计算/OwnedJob检查。event含提交供给不足或GPU调度空隙，host/submit分开保留。三轮在同一进程连续执行，不是三次独立物理模拟。','',
        '## 失败保留、身份与复现','',
        '最初独立构建漏带原数学头的相邻include，C1083后补原字节并成功编译。其小核420份先通过；随后完整合同在Graph Runtime参数检查处中止，没有计时。原二进制、原始输出、失败日志、211份源文件映射封存于前一不带_retry1批次。当前批次从头构建并重做所有合同，修正仅检查工具对cuBLAS Driver内部核的处理。','',
        'Runtime节点查询只有返回精确cudaErrorInvalidDeviceFunction才计opaque；完整fused每apply严格4个opaque，MAS/小核0；自定义tiny数量/函数/线程必须精确匹配，其他错误仍中止。这是API检查兼容，不是跳过数值或执行错误。官方接口参见[NVIDIA Runtime Graph API](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-runtime-api/group__CUDART__GRAPH.html)。','',
        f'当前exe SHA256 `{identity["exe"]["sha256"]}`。来源为封存父888文件与原184份源；本轮source/object/link/tlog/编译器/运行库/输入/protocol均记录。生产文件运行前后身份一致。未复制完整外部SDK/driver依赖闭包，未改生产、跑100帧、重跑生产29项CTest或上传。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_tiny_prefetch.py prepare",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_tiny_prefetch.py build",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_tiny_prefetch.py seal",
        '# 依次运行 run_tiny_prefetch.py tiny、contracts；每阶段后运行 analyze_tiny_prefetch.py 同名action',
        '# 独立合同通过后依次运行 tiny_measure、measure，并作同名独立分析',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_tiny_prefetch.py",
        '```','', '入口拒绝覆盖目录。复测需新路径；prepare所需协议文件与本轮报告一起保留。完整窗口、所有CPU误差、四元组、汇总表在tables/，完整向量和资源采样在runs/。','',
        '## 有限下一步','',next_step,'',
        '本轮不修改生产默认，不把固定动作或小核速度称为真实PCG/全帧加速。']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_new(REPORT/'DECISION.json',dict(numerical_passed=True,outputs=1056,tiny_bitwise_equal=True,
        full_AA_passed=controls,full_AB_signal=signal,fixed_apply_candidate_eligible=eligible,production_modified=False,
        performance_certified=False,next_step=next_step,source=record(Path(__file__))))
    records=tree(RAW)+tree(BUILD)+tree(REPORT)+sources+[record(first)]
    records += [record(p) for p in (ROOT/'build').glob('tiny_prefetch*_controller.log') if 'finish' not in p.name]
    files=sorted({r['path']:r for r in records}.values(),key=lambda r:r['path']);verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='tiny_L_prefetch.v1',files=files,parent=parent['parent'],
        failed_attempt=record(first),checked_outputs=1056,tiny_cases=70,windows=576,full_AA_passed=controls,
        fixed_apply_candidate_eligible=eligible,production_modified=False,performance_certified=False,next_step=next_step))
    verify(read(REPORT/'DELIVERY_SEAL.json')['files'])
    print(dict(sealed=True,files=len(files),outputs=1056,full_AA_passed=controls,fixed_apply_candidate_eligible=eligible),flush=True)

if __name__=='__main__':main()
