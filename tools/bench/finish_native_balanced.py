"""Report and seal the finite native balanced experiment, preserving prior bytes."""
import ast
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
from m3_numeric import ROOT,need
from local_identity import record,verify,tree,freeze_active_identity
from linux_runner import write_new
import run_native_balanced
import analyze_native_balanced
import audit_native_balanced_cost
import prepare_native_balanced
import verify_native_balanced

RAW=ROOT/'runs/innovation_p5_native_balanced_20261016'
REPORT=ROOT/'reports/innovation_p5_native_balanced_20261016'
BUILD=ROOT/'build/native_balanced_v2_20261016'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    destination=REPORT/'DELIVERY_SEAL.json';need(not destination.exists(),'Balanced stage already sealed')
    summary=read(RAW/'analysis/summary.json');need(summary['implementation_checks_passed'],'Independent audit missing')
    verify(summary['inputs']+[summary['source']])
    costs=read(REPORT/'COST_AND_CROSS_BATCH_AUDIT.json');need(costs['passed'],'Cost audit missing')
    verify(costs['inputs']+[costs['source']])
    contracts=read(ROOT/'build/native_balanced_verification_contracts/verification.json')
    full=read(ROOT/'build/native_balanced_verification_full/verification.json')
    for evidence,count in ((contracts,2),(full,29)):
        need(evidence['passed'] and evidence['expected']==count and evidence['cleanup_owned_job_empty'],'CTest incomplete')
        verify([evidence['source']])
    contract_build=read(BUILD/'contracts_build_identity.json')
    verify(contract_build['sources']+[contract_build['source']])
    prior=read(REPORT/'PRIOR_SOURCES.json');parent=read(prior['parent']['path']);verify([prior['parent']])
    preserved={r['original']['path']:r for r in prior['files']};mappings=[]
    for row in parent['files']:
        if row['path'] in preserved:
            old=preserved[row['path']];need(row==old['original'],'Prior source mapping differs')
            verify([old['archive']]);mappings.append(old)
        else:verify([row])
    need(len(mappings)==4,'Unexpected parent mutation scope')
    packet=read(RAW/'seal.json');identity=packet['identity']
    verify(identity['sources']+[identity['exe']]+identity['dlls']+[packet['runner'],packet['protocol']])
    write_new(RAW/'compiled_identity.json',identity)
    frozen=freeze_active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log',
        ROOT/'build/native_balanced_snapshot_20261016',from_verified_identity=RAW/'compiled_identity.json')
    systems=[];runs=[]
    for phase in ('small','faults','representative'):
        ledger=read(RAW/(phase+'_ledger.json'));need(ledger['complete'],'Incomplete stage')
        for entry in ledger['runs']:
            need(entry['passed'] and entry['result']['cleanup_owned_job_empty'],'Owned run incomplete')
            verify(entry['studies']);runs.append(entry)
            systems += [read(r['path']) for r in entry['studies']]
    sources={Path(__file__).resolve(),ROOT/'tools/build_windows.py',
             ROOT/'tests/native_balanced_math_test.cpp',ROOT/'tests/native_balanced_gpu_test.cu'}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT/'tools'):sources.add(path)
    source_bindings=[]
    for path in sorted(sources):
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'))
        target=REPORT/'sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out)
        current,saved=record(path),record(target)
        need((current['bytes'],current['sha256'])==(saved['bytes'],saved['sha256']),'Python source copy differs')
        source_bindings.append(dict(source=current,snapshot=saved))
    lines=['# 固定 A、b 的原生 balanced PCG：实现及有限验证','',
        '2026-10-16。已接入默认关闭的受保护原生host诊断路径，保留生产PCG/Graph循环及全部物理配置。'
        '本轮用实际FP32 MAS，不把固定FP64系数参考的迭代数替代原生结果。','',
        f'**数值主门槛：{summary["primary_numerical_gate_passed"]}；'
        f'主重复的原生迭代下降：{summary["primary_iteration_reduction_passed"]}。** '
        '以下是固定系统的仪器化诊断，尚非整程序净加速验收。','',
        '协议：[PLAN.md](PLAN.md)；源码逐段复核：[SOURCE_AUDIT.md](SOURCE_AUDIT.md)。','',
        '## 原生目标迭代数','',
        '| 场景 n1→n2 | MAS三次 | balanced三次 | balanced实际使用 | CPU共同精度 |',
        '|---|---|---|---|---|']
    for label,scene in (('bunny','Bunny14'),('mixed','Mixed15')):
        rows=[r for r in summary['targets'] if r['case']==label]
        base=sorted((r for r in rows if not r['requested_balanced']),key=lambda r:r['pair'])
        aug=sorted((r for r in rows if r['requested_balanced']),key=lambda r:r['pair'])
        fmt=lambda rs:' / '.join(str(r['iterations']) for r in rs)
        lines.append(f'| {scene} | {fmt(base)} | {fmt(aug)} | {sum(r["used_balanced"] for r in aug)}/3 | '
                     f'{sum(r["common_CPU_quality_passed"] for r in rows)}/6通过 |')
    lines += ['', '预先指定pair1为主结果；所有重复与回退均保留。两臂从零初值/同一MAS scratch开始，三对AB/BA。'
        '源空间仅使用n1 pair1的普通原生停止历史，rank固定8，前16+尾112；n2解/历史不用于空间。','',
        '原生诊断在previous-rho候选之后，额外检查原A真L2及共享原生MAS加权真残差≤.0095；'
        '独立CPU固定B_ref要求两项≤.01。5%余量在运行前固定。生产rho1e-4及退出规则没有改变。'
        '目标上限min(max_iter−1,2000)，没有残差替换或灵活CG。','',
        '## 代数与数值核验','',
        '| 场景 | 源统计/直接Gram差 | GPU源Z/CPU差 | 原生MAS(b)/固定B_ref差 | 原生balanced(b)/同Z固定参考差 |',
        '|---|---:|---:|---:|---:|']
    for r in summary['sources']:
        if r['case'] not in ('bunny','mixed'):continue
        fmt=lambda key:format(r[key],'.6g') if key in r else '未建立'
        lines.append(f'| {r["scene"]} | {fmt("Gram_relative")} | {fmt("Z_relative")} | '
                     f'{fmt("native_MAS_b_vs_fixed_reference_relative")} | {fmt("native_balanced_b_vs_fixed_reference_relative")} |')
    lines += ['', '目标W=AZ、E=ZᵀW及L Lᵀ=sym(E)已用原始完整CSR独立复核；'
        '数值拒绝无jitter/补秩。目标b动作差只是一项有限探针校准，不能视作全算子误差界。'
        'B_ref显式对称FP32局部逆后以FP64施加；原生M含FP32转换及归并，两者不等价。','',
        'Bunny/Mixed pair1的两臂共4份原生解另用Decimal50复核。完整解、原生曲线、'
        '共同加权残差和独立CPU影子求解见[tables](tables/target_solutions.csv)。','',
        '### 同一A/b/Z的CPU影子对照','',
        '| 场景 | CPU固定B_ref MAS | CPU固定B_ref balanced |',
        '|---|---:|---:|']
    for scene in ('Bunny','Mixed'):
        rs=[r for r in summary['CPU_shadow'] if r['scene']==scene]
        lines.append(f'| {scene} | {next(r["iterations"] for r in rs if r["method"]=="MAS_reference")} | '
            f'{next(r["iterations"] for r in rs if r["method"]!="MAS_reference")} |')
    lines += ['', 'CPU影子也使用.0095共同真残差阈值；原生还要求previous-rho候选已触发。'
        '本批Bunny的CPU166与原生170接近，因此不能把与旧批约121的差距全部归因于FP32。'
        '独立重开旧/新系统，Bunny n2的A相对Frobenius差约4.449%、b相对L2差约73.511%；'
        'Mixed n2分别约7.246%、63.309%。坐标维数及active映射相同，但A/b不相同，'
        '跨批迭代数不能作同系统精度对照。未在本轮隔离这些轨迹差异的完整因果来源。'
        '全部n1/n2差异见[cross_batch_systems.csv](tables/cross_batch_systems.csv)。','',
        '## 配对成本（ms，三对中位数）','',
        '| 系统 | MAS solve | balanced请求solve | 配对solve差 | MAS额外质量检查 | balanced额外质量检查 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,scene in (('bunny','Bunny14'),('mixed','Mixed15')):
        rows=[r for r in summary['timings'] if r['case']==label]
        base=[r for r in rows if not r['requested_balanced']];aug=[r for r in rows if r['requested_balanced']]
        median=lambda rs,key:statistics.median(r[key] for r in rs)
        deltas=[next(r['solve_host_ms'] for r in aug if r['pair']==p)-next(r['solve_host_ms'] for r in base if r['pair']==p) for p in (1,2,3)]
        lines.append(f'| {scene} | {median(base,"solve_host_ms"):.4f} | {median(aug,"solve_host_ms"):.4f} | '
            f'{statistics.median(deltas):+.4f} | {median(base,"quality_host_ms"):.4f} | {median(aug,"quality_host_ms"):.4f} |')
    lines += ['', 'solve窗口含GPU提交/同步、JSON逐步曲线及额外验收；CUDA event也含host提交间隙。'
        '失败尝试和从零回退成本包含在balanced请求窗口中。不能从本表推算生产Graph或整程序加速。','',
        'source capture、CPU小Gram特征分解、BLAS/显存初始化、GPU Z形成、target coarse准备、'
        'reset、导出及恢复均在原生study JSON单列。另有4次整apply校准和一次按stage同步的分项校准；'
        '后者不混入配对solve，不能当作稳定纯kernel时间。生产MAS准备在study之前，未被测成零。','',
        '### 为何Bunny迭代减少，耗时却增加','',
        '| 场景，三对差值的算术平均 | 总solve差ms | 已测质量窗口差ms | 剩余窗口差ms |',
        '|---|---:|---:|---:|']
    for r in costs['additive_pair_means']:
        lines.append(f'| {r["scene"]} | {r["solve_delta_ms"]:+.4f} | {r["measured_quality_delta_ms"]:+.4f} | '
                     f'{r["remainder_delta_ms"]:+.4f} |')
    lines += ['', '这里用均值以保留加法关系；分项中位数不一定可相加。Bunny balanced从第59次更新就触发候选，'
        '当时真L2约.097、共同加权残差约.046，尚不合格，之后共做112次真质量检查；'
        'MAS从209–210次开始，仅做31–32次。Mixed balanced做12–14次、MAS仅1次。'
        '这解释了已测质量窗口的开销来源。Bunny扣除该已测窗口后仍平均多8.5471ms，'
        '因此不能把变慢完全归于验收。剩余项包含递推、原生动作、提交与记录，'
        '本轮未用独立干预将其全部归因到某一个GEMV。','',
        'target coarse只测了一次，Bunny/Mixed分别29.1090/29.3924ms，包含8次原生SpMV、'
        '首次小GEMM、D2H、CPU检查/LLT和L上传，没有分离冷/热调用。Mixed每个目标solve平均省8.3118ms，'
        '单个n2的这部分节省不足以覆盖本次coarse准备，更未计source构造/采集。'
        '不能据此宣称真实单步净加速。复用的诊断保护壳在n2也分配了不用作采集的history bank，'
        '这属于当前私有准备成本，不应隐藏。','',
        '详见[COST_AND_CROSS_BATCH_AUDIT.json](COST_AND_CROSS_BATCH_AUDIT.json)、'
        '[quality_schedule.csv](tables/quality_schedule.csv)、[paired_cost_accounting.csv](tables/paired_cost_accounting.csv)。','',
        '## 合同、回退和恢复','',
        '- 两项新增合同先通过，再完成29/29串行CTest。独立GPU合同覆盖真实源投影、balanced公式及零秩拒绝。',
        '- Bunny1帧检查不足rank8时的显式MAS回退；Bunny14的coarse_reject检查有效源空间被拒绝后的回退。',
        '- Bunny14的after_apply故障在真实balanced动作完成后抛出，预期非零退出；九项恢复证明保留。',
        '- 正常和故障两类都核对原主方向、全部PCG工作区、Graph句柄/计数、完整系统/MAS scratch、算子、config/info/context/cost trace。'
        '该覆盖不是穷举所有CUDA/IO中途失败。','',
        f'实际记录{len(systems)}个固定系统报告，{sum(len(s["runs"]) for s in systems)}个完成求解记录；'
        '注入故障的未完成attempt单独保留，不记成成功求解。','',
        '## 构建与复现','',
        f'- exe SHA256 `{identity["exe"]["sha256"]}`；全新构建37个native TU，构建日志和实际NVCC/CL/object/link身份完整封存。',
        '- 沙箱内CUDA编译器识别发生MSVC C1902，尚未编译新增实现；保留失败目录。沙箱外新目录配置/合同/原生构建完成。'
        '日志有代码页、模板等警告，不声称零警告，也没有更改生产精度/优化选项来避开错误。',
        f'- 电源GUID `{packet["power_guid"]}`；RTX3070 Laptop/WDDM共享桌面，逐次电源/显存监测与Owned Job清理证据保留。',
        '- dt=.04、rho1e-4、legacy累计.01/min6、FullCCD/材料与此前最快版执行组合保持。'
        '本地构建含既有未上传P3源码，精简公开提交不能单独重建本轮程序。本轮未上传GitHub。',
        '- 旧阶段1545文件通过原字节映射复核；四处重叠源码在prior_sources保留。新[DELIVERY_SEAL.json](DELIVERY_SEAL.json)'
        '绑定全部原始结果、源码/编译/二进制/配置及本文。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_native_balanced.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/verify_native_balanced.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/build_windows.py --kind active --label native_balanced_v2_20261016 --configured --jobs 2",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/verify_native_balanced.py full",
        "# run_native_balanced.py依次执行seal、small、faults、representative（GPU串行）",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_native_balanced.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_native_balanced.py",'```','',
        '所有入口拒绝隐式覆盖已有批次。原始A/b/M、source bank、Z/W/E/L、逐步曲线及逐次解位于'
        '`runs/innovation_p5_native_balanced_20261016`，详细表在[tables/native_timings.csv](tables/native_timings.csv)。','',
        '## 下一关','']
    if not summary['primary_numerical_gate_passed']:
        decision='停止扩大到Graph/在线回收。先定位原生FP32动作、粗空间及共同残差门槛失败，保留本轮否定结果。'
    elif not summary['primary_iteration_reduction_passed']:
        decision='原生数值可用，但当前固定系统没有在两场景都实现迭代下降；先分析原生递推/源空间失效，不能靠CPU参考收益继续扩展。'
    else:
        decision='原生数值和迭代下降通过，但Bunny变慢、Mixed单目标节省不能覆盖本次coarse准备。下一步保持同一固定A/b/Z，先分离质量/逐步记录对计时的影响并测清coarse冷/热成本，再优化投影/apply的提交开销；不直接扩展Graph、在线回收或100帧全场景。'
    lines.append(decision)
    report=REPORT/'REPORT.md';need(not report.exists(),'Report already exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        need((report.parent/link).exists() or report.parent/link==destination,'Report link missing')
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+tree(ROOT/'build/native_balanced_snapshot_20261016')
    records+=tree(ROOT/'build/native_balanced_20261016')
    for name in ('contracts','full'):records+=tree(ROOT/('build/native_balanced_verification_'+name))
    records += identity['sources']+[b['source'] for b in source_bindings]+[record(Path(prior['parent']['path']))]
    records += [record(p) for p in (ROOT/'build').glob('native_balanced*.log') if 'finish' not in p.name]
    unique={r['path']:r for r in records};files=[unique[k] for k in sorted(unique)];verify(files)
    write_new(destination,dict(schema='native_balanced_fixed.v1',files=files,parent=prior['parent'],
        parent_mutable_mapping=mappings,parent_files_verified=len(parent['files']),source_bindings=source_bindings,
        binary=frozen['exe'],CTest=29,new_contracts=2,native_systems=len(systems),
        native_completed_solves=sum(len(s['runs']) for s in systems),
        primary_numerical_gate_passed=summary['primary_numerical_gate_passed'],
        primary_iteration_reduction_passed=summary['primary_iteration_reduction_passed'],
        production_solver_modified=False,performance_certified=False,next_finite_decision=decision))
    verify(read(destination)['files'])
    print(json.dumps(dict(sealed=True,files=len(files),CTest=29,native_systems=len(systems),
        primary_numerical_gate_passed=summary['primary_numerical_gate_passed'],
        primary_iteration_reduction_passed=summary['primary_iteration_reduction_passed'])),flush=True)

if __name__=='__main__':main()
