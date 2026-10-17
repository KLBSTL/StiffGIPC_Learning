"""Seal the finite source/validator/runtime audit before the next optimization."""
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from tiny_prefetch_review import RAW,REPORT,OLD_RAW,OLD_REPORT,BUILD,PARENT,read,strict_metadata,independent_statistics
from local_identity import record,verify,tree,text
from linux_runner import require,write_new
from prepare_tiny_prefetch import edited
from run_tiny_review import deadline_ok
import analyze_tiny_prefetch as v1

def main():
    require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    history=read(RAW/'history.json');require(history['passed'] and history['statistics_unchanged'],'History review required')
    verify(read(OLD_REPORT/'DELIVERY_SEAL.json')['files']);verify(read(PARENT)['files'])
    verify(read(OLD_RAW/'production_before.json'))
    manifest=read(OLD_RAW/'input_manifest.json');fresh=[]
    for folder_name,action in [('fresh_tiny','tiny'),('fresh_contracts','contracts')]+[('sanitizer_'+t,'tiny') for t in ['memcheck','racecheck','synccheck']]:
        folder=RAW/folder_name;result=read(folder/'result.json');audit=read(folder/'audit.json')
        require(result['passed'] and result['exit_code']==0 and result['cleanup_owned_job_empty'] and audit['passed'],'Fresh owned stage/audit failed')
        verify(audit['inputs']);metadata=strict_metadata(manifest,read(folder/'data.json'),folder,action)
        fresh.append(dict(stage=folder_name,outputs=audit['outputs'],wall_seconds=result['wall_seconds'],maximum_CPU_relative=audit['maximum_CPU_relative'],**{k:v for k,v in metadata.items() if k!='outputs'}))
    require(history['outputs']+sum(r['outputs'] for r in fresh)==2772,'Wrong total outputs')
    deadline_tests=[]
    for value,accept in [(240.0001,False),(0.,False),(-1.,False),(float('nan'),False),(float('inf'),False),(.0001,True),(55.,True),(240.,True)]:
        accepted=True
        try:deadline_ok(value)
        except ValueError:accepted=False
        require(accepted==accept,'Deadline boundary failed');deadline_tests.append(dict(input=repr(value),expected=accept,accepted=accepted))
    write_new(RAW/'deadline_tests.json',dict(passed=True,tests=deadline_tests))
    # Reconstruct exactly what was compiled, without editing any historical sources.
    generation=read(BUILD/'generation.json')
    original=Path(generation['template']['path']).read_text(encoding='utf-8');anchor=generation['boundary']
    require(original.count(anchor)==1,'Ambiguous original boundary')
    generated=edited(original.split(anchor)[0],[(e['old'],e['new']) for e in generation['edits']])+Path(generation['continuation']['path']).read_text(encoding='utf-8')
    require(generated==Path(generation['generated']['path']).read_text(encoding='utf-8'),'Compiled fixture differs')
    header=Path(generation['header_original']['path']).read_text(encoding='utf-8')
    require(edited(header,[(e['old'],e['new']) for e in generation['header_edits']])==Path(generation['header_generated']['path']).read_text(encoding='utf-8'),'Compiled header differs')
    loops=lambda source:'\n'.join(s.strip() for s in source.splitlines() if s.strip().startswith('for(int i='))
    require(loops(header.split('static __global__ void balanced_tiny_solve(')[1].split('static __global__ void balanced_difference')[0])==
        loops(Path(generation['kernel']['path']).read_text()).replace('lower[','L['),'Original operation order differs')
    identity=read(BUILD/'identity.json');log=text(BUILD/'build.log');cuda_lines=[s for s in log.splitlines() if 'nvcc.exe' in s.lower() and '--compile' in s and 'driver.cu' in s]
    require(cuda_lines,'Missing actual compile command')
    for flag in ['-rdc=true','-use_fast_math','--default-stream=per-thread','sm_86','-lineinfo']:
        require(any(flag in s for s in cuda_lines),'Missing original compile option: '+flag)
    tracking=BUILD/'tiny_prefetch.dir/Release/tiny_prefetch.tlog/CudaCompile.read.1u.tlog';tracked=text(tracking).replace('\\','/').casefold()
    expected=[BUILD/'overlay'/p for p in ['solver/MASPreconditioner.cuh','linear_system/solver/native_balanced_gpu.cuh','linear_system/solver/native_balanced_math.h','tiny_fixture.inl','abd_kernel.inc','load_buffers.inc']]
    expected += [ROOT/'tools/bench/tiny_prefetch'/p for p in ['driver.cu','tiny_contracts.inl','graph_tiny_info.inl']]
    write_new(RAW/'compile_tracking.json',dict(command_lines=cuda_lines,tracking=record(tracking),
        coverage=[dict(file=record(p),listed_in_tlog=str(p).replace('\\','/').casefold() in tracked) for p in expected],
        limitation='MSBuild read tlog does not enumerate every user include; no full external dependency closure claim',
        fixture_reconstructed=True,header_reconstructed=True,original_arithmetic_order_verified=True))
    # Each row is an explicit human-reviewed source range, not a blanket tree claim.
    entries=[]
    def reviewed(rel,start,end,note):
        p=ROOT/rel;count=len(p.read_text(encoding='utf-8-sig').splitlines());end=count if end is None else end
        require(1<=start<=end<=count,'Invalid reviewed range');entries.append(dict(file=str(p),start=start,end=end,total_file_lines=count,note=note,sha256=record(p)['sha256']))
    for rel,note in [
        ('tools/bench/tiny_prefetch/tiny_kernel.cuh','Each load, barrier, branch and arithmetic loop'),
        ('tools/bench/tiny_prefetch/graph_tiny_info.inl','Runtime query status, own function geometry and library opaque handling'),
        ('tools/bench/tiny_prefetch/tiny_contracts.inl','RHS reset, guards, host/Graph coverage, timers, export and lifetime'),
        ('tools/bench/tiny_prefetch/continuation.inl','Workspace, dispatch, whole apply comparison, paired schedule and restore'),
        ('tools/bench/tiny_prefetch/driver.cu','Includes, CLI dispatch and exception result'),
        ('tools/bench/tiny_prefetch/CMakeLists.txt','Compiler definitions, options, includes and linkage'),
        ('tools/local/prepare_tiny_prefetch.py','Generation edits, inputs, frozen lineage and obsolete special recovery branch'),
        ('tools/local/run_tiny_prefetch.py','Identity, validation gates, GPU/Job guards and completion deadline gap'),
        ('tools/bench/analyze_tiny_prefetch.py','Triangular reference, whole apply algebra, schema coverage gaps and paired stats'),
        ('tools/bench/analyze_tiny_sass.py','Actual symbol resolution and static instruction count scope'),
        ('tools/bench/finish_tiny_prefetch.py','Source archive, generated files, decision conditions and stated result scope'),
        ('build/fused_apply_snapshot_20261016/StiffGIPC/linear_system/solver/native_balanced_gpu.cuh','Original arithmetic, BLAS stream and balanced formula'),
        ('build/fused_apply_snapshot_20261016/StiffGIPC/linear_system/solver/native_balanced_math.h','Quality screening, coefficients and coarse factor guard; source build not re-executed'),
        ('build/tiny_prefetch_20261017_retry1/overlay/abd_kernel.inc','Original independent 12-DOF ABD block action'),
        ('tools/local/windows_owned_job.py','Windows Job ownership, spawn, handle races and cleanup')]:reviewed(rel,1,None,note)
    reviewed('tools/bench/full_apply_graph/fixture.inl',1,120,'Reused setup prefix, Graph RAII, original MAS loads, readonly/scratch and global layout')
    reviewed('tools/bench/m3_numeric.py',1,84,'Independent stored local coefficient B_ref reconstruction and norm definition; unrelated PCG code excluded')
    reviewed('tools/local/windows_runner.py',1,90,'GPU lock, WDDM process classification and ownership checks only')
    reviewed('build/fused_apply_snapshot_20261016/StiffGIPC/solver/MASPreconditioner.cu',2330,2412,'Original called restrict/local/prolong dispatch only')
    reviewed('build/fused_apply_snapshot_20261016/StiffGIPC/solver/MASPreconditioner.cu',2536,2633,'Original preconditioner branch guards, zeroing and called stages; other MAS kernels excluded')
    v1.csv_new(REPORT/'tables/reviewed_lines.csv',entries);v1.csv_new(REPORT/'tables/fresh_runs.csv',fresh)
    reviewed_lines=sum(r['end']-r['start']+1 for r in entries)
    # Persist the new audit's own complete helper closure.
    seeds=[ROOT/'tools/bench'/p for p in ['tiny_prefetch_review.py','finish_tiny_review.py']]+[ROOT/'tools/local'/p for p in ['run_tiny_review.py','run_tiny_sanitizer_review.py']]
    pending=seeds[:];seen=set()
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                for base in [ROOT/'tools',ROOT/'tools/local',ROOT/'tools/bench']:
                    q=base/Path(*name.split('.')).with_suffix('.py')
                    if q.is_file():pending.append(q);break
    sources=[record(p) for p in seen];dest=REPORT/'sources';dest.mkdir()
    for r in sources:
        p=Path(r['path']);q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
    report=['# 8维求解优化链复查','',
        '**复查完成：所列数值/CUDA作用链未发现未解决错误；发现并补齐两处验证覆盖缺口及退出时限检查。未开始下一轮缓存a实现。**','',
        f'逐行复查范围为{len(entries)}个文件/区段、共{reviewed_lines}行，精确范围和源码SHA在[tables/reviewed_lines.csv](tables/reviewed_lines.csv)。[逐项检查与发现](SOURCE_REVIEW.md)，[本轮协议](PLAN.md)。这不是StiffGIPC全部源码的形式正确性证明。','',
        '## 发现与修正','',
        '1. 旧Graph检查可接受重复original/Graph1记录，缺少方法×粒度唯一覆盖判断。','2. 旧warm检查可接受六次引用同一个original/host输出，缺少方法×执行矩阵和文件唯一归属判断。',
        '3. 旧运行循环未单独检查最后轮询后进程退出的超时边界。新复查运行入口补退出时限和结束资源检查；8个边界用例通过。','',
        '前两项由内存中故意损坏的副本复现；新增v2入口拒绝26个重复/缺项/错误几何/输出归属/非法元数据反例。旧源码、二进制、结果及报告完全保留，不改封存证据。旧build_retry1是历史特例，不作为通用重试入口。','',
        '## 验证结果','',
        '| 核验 | 输出份数 | 结果 |','|---|---:|---|',
        '| 历史输出重新检查 | 1056 | 方法/粒度/文件覆盖完整，数值通过 |',
        '| 本机原exe小核合同 | 420 | 原/候选/Host/Graph逐位一致 |',
        '| 本机原exe完整apply合同 | 36 | 独立CPU误差通过 |',
        '| NVIDIA memcheck | 420 | 0 errors，数值通过 |',
        '| NVIDIA racecheck | 420 | 0 errors / 0 warnings，数值通过 |',
        '| NVIDIA synccheck | 420 | 0 errors，数值通过 |','',
        '共2772份输出完成独立数值核验（包含1056份历史复核，不是2772次不同物理模拟）。新工具检查仅覆盖70组小核/Graph路径，未将其扩大为全MAS、完整模拟的sanitizer结论。5个新增OwnedJob均退出并清理完成，逐次显存/UUID/电源/外来compute检查通过。电源仍为野兽模式。','',
        f'新完整apply相对CPU B_ref最大误差为{next(r["maximum_CPU_relative"] for r in fresh if r["stage"]=="fresh_contracts"):.9g}；小核对Decimal 80位十进制参考最大差为{fresh[0]["maximum_CPU_relative"]:.9g}。原≤1e-3独立CPU和≤1e-5原动作校准门槛未改。','',
        '## 历史性能复核','',
        '以标准库重新计算576个窗口的配对均值比、中位、各轮、p10/p90和MAD，与旧NumPy结果一致。本轮不重新计时。','',
        '| 完整apply | Graph1 | Graph16 |','|---|---:|---:|',
        '| Bunny | 1.0169× | 1.0163× |','| Mixed | 1.0294× | 1.0257× |','',
        '四组A/A仍通过；Bunny增益不足2%，Graph16第二轮候选/原为1.003341，仍未达到全格推广门槛。两项验证器缺口没有出现在历史真实记录中，因此不推翻旧数据，也不能将小幅固定动作收益称为整程序加速。','',
        '## 复现与边界','',
        '本轮使用同一exe SHA256 `'+identity['exe']['sha256']+'`，没有编译/更改数值核或生产树。overlay和原序循环重新构造匹配。原1415文件、上游888文件及生产树身份复核。NVCC实际命令/对象/link可核对，但read tlog没有记录全部用户头，外部SDK/driver/动态DLL完整闭包仍未封存；不能声称该依赖闭包已被证明。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/tiny_prefetch_review.py history",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_tiny_review.py tiny",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_tiny_review.py contracts",
        '# 分别运行 run_tiny_sanitizer_review.py memcheck、racecheck、synccheck',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_tiny_review.py",'```','',
        '入口拒绝覆盖历史输出；复测使用新目录。新v2校验先检查覆盖再做数值，作为之后原型的必要入口；旧analyze_tiny_prefetch仍作为封存v1保留，不应独自用它证明覆盖完整。','',
        '## 下一步','',
        '本轮复查条件已满足，可继续只缓存8维a的FP64原序小原型；先看寄存器/local溢出，再用v2核验和配对测试。L预取版仍为未推广实验，不改默认、不重跑100帧、不上传。本轮止于复查及其必要修正。']
    (REPORT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write_new(REPORT/'DECISION.json',dict(review_passed=True,reviewed_source_lines=reviewed_lines,validation_gaps_fixed=2,completion_guard_fixed=True,
        historical_results_unchanged=True,outputs_checked=2772,negative_tests=26,deadline_tests=8,sanitizers=['memcheck','racecheck','synccheck'],
        next_optimization_started=False,production_modified=False,performance_remeasured=False))
    records=tree(RAW)+tree(REPORT)+sources+[record(OLD_REPORT/'DELIVERY_SEAL.json')]
    records += [record(p) for p in (ROOT/'build').glob('tiny_review*_controller.log') if 'finish' not in p.name]
    files=sorted({r['path']:r for r in records}.values(),key=lambda r:r['path']);verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='tiny_prefetch_review.v2',files=files,parent=record(OLD_REPORT/'DELIVERY_SEAL.json'),
        reviewed_source_lines=reviewed_lines,outputs_checked=2772,negative_tests=26,deadline_tests=8,next_optimization_started=False,production_modified=False))
    verify(read(REPORT/'DELIVERY_SEAL.json')['files']);print(dict(sealed=True,files=len(files),reviewed_lines=reviewed_lines,outputs=2772),flush=True)

if __name__=='__main__':main()
