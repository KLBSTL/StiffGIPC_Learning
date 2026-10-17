"""Report and seal the finite native fused-projection experiment."""
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
import analyze_fused_apply
import run_fused_apply
import prepare_fused_apply
import verify_fused_apply
import retry_fused_apply_ctest
import audit_fused_apply_cost

RAW=ROOT/'runs/innovation_p5_fused_apply_20261016'
REPORT=ROOT/'reports/innovation_p5_fused_apply_20261016'
BUILD=ROOT/'build/native_fused_apply_20261016'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def med(rows,key):return statistics.median(r[key] for r in rows)

def main():
    dest=REPORT/'DELIVERY_SEAL.json';need(not dest.exists(),'Already sealed')
    summary=read(RAW/'analysis/summary.json');need(summary['implementation_checks_passed'],'Independent audit required')
    verify(summary['inputs']+[summary['source']])
    context=read(REPORT/'COST_CONTEXT_AUDIT.json');need(context['passed'],'Cost context audit required')
    verify(context['inputs']+[context['source']])
    packet=read(RAW/'seal.json');verify([packet['full_verification']])
    for path,count in ((ROOT/'build/fused_apply_verification_contracts/verification.json',2),
                       (Path(packet['full_verification']['path']),29)):
        evidence=read(path)
        need(evidence['passed'] and evidence['expected']==count and evidence['cleanup_owned_job_empty'],'CTest incomplete')
        verify([evidence['source']])
    cb=read(BUILD/'contracts_identity.json');verify(cb['sources']+[cb['source']]+cb['executables'])
    prior=read(REPORT/'PRIOR_SOURCES.json');verify([prior['parent']]);parent=read(prior['parent']['path'])
    preserved={r['original']['path']:r for r in prior['files']};mappings=[]
    for row in parent['files']:
        if row['path'] in preserved:
            old=preserved[row['path']];need(row==old['original'],'Parent mapping differs')
            verify([old['archive']]);mappings.append(old)
        else:verify([row])
    need(len(mappings)==3,'Unexpected edit scope')
    identity=packet['identity']
    verify(identity['sources']+[identity['exe']]+identity['dlls']+[packet['runner'],packet['protocol']])
    write_new(RAW/'compiled_identity.json',identity)
    snapshot=ROOT/'build/fused_apply_snapshot_20261016'
    frozen=freeze_active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log',snapshot,
        from_verified_identity=RAW/'compiled_identity.json')
    systems=[]
    for phase in ('small','faults','representative'):
        ledger=read(RAW/(phase+'_ledger.json'));need(ledger['complete'],'Native stage incomplete')
        for row in ledger['runs']:
            need(row['passed'] and row['result']['cleanup_owned_job_empty'],'Owned run incomplete')
            verify(row['studies']);systems.extend(read(r['path']) for r in row['studies'])
    source_files={Path(__file__).resolve(),ROOT/'tools/build_windows.py',ROOT/'tests/native_balanced_gpu_test.cu',
        ROOT/'tests/native_balanced_math_test.cpp'}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT/'tools'):source_files.add(path)
    bindings=[]
    for path in sorted(source_files):
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'))
        saved=REPORT/'sources'/path.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as inp,saved.open('xb') as out:shutil.copyfileobj(inp,out)
        a,b=record(path),record(saved);need((a['bytes'],a['sha256'])==(b['bytes'],b['sha256']),'Source copy differs')
        bindings.append(dict(source=a,snapshot=b))
    calibrations=[];stages=[]
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        study=read(RAW/label/'fixed'/f'f{frame}_n2_native_balanced_study.json')
        calibrations.extend(dict(scene=scene,**r) for r in study['separate_apply_calibration'])
        for key,fused in (('separate_synchronized_apply_stages',False),('separate_fused_synchronized_apply_stages',True)):
            stages.extend(dict(scene=scene,fused=fused,**r) for r in study[key])
    analyze_fused_apply.csv_new(REPORT/'tables/apply_calibration.csv',calibrations)
    analyze_fused_apply.csv_new(REPORT/'tables/apply_stages.csv',stages)
    lines=['# P5：rank8非转置投影融合与原生实测','',
        '2026-10-16。只将t=r−Wa、z←z+Z(a−c)各融合成一个FP64核；保留cuBLAS转置投影、'
        '粗三角求解和实际原生MAS。Reference仍是默认，生产PCG/Graph和物理配置不改。','',
        f'**独立实现检查通过；有限数值门槛={summary["numerical_gate_passed"]}；'
        f'预先规定的探索成本门槛={summary["exploratory_cost_gate_passed"]}。** '
        '本轮仍是固定系统诊断，不认证完整轨迹净加速，未启动100帧、未上传GitHub。','',
        '[预定方案](PLAN.md)；[逐段源码检查](SOURCE_AUDIT.md)。','',
        '## 实际原生PCG（三次中位数）','',
        '| 系统 | 动作 | 三次迭代数 | solve ms | 真实检查次数 | 筛选次数 |',
        '|---|---|---|---:|---:|---:|']
    names={0:'MAS',1:'balanced reference',2:'balanced fused'}
    for label,scene in (('bunny','Bunny14 n2'),('mixed','Mixed15 n2')):
        for variant in (0,1,2):
            rows=sorted((r for r in summary['timings'] if r['case']==label and r['variant']==variant),key=lambda r:r['pair'])
            lines.append(f'| {scene} | {names[variant]} | {" / ".join(str(r["iterations"]) for r in rows)} | '
                f'{med(rows,"solve_host_ms"):.4f} | {med(rows,"quality_checks"):g} | {med(rows,"screen_checks"):g} |')
    lines += ['', '同一次A/b/M/Z快照内三臂分别零x、同MAS scratch；三次顺序M/R/F、F/R/M、R/F/M。'
        '三臂均使用上一轮screened策略；first previous-rho候选、最多间隔8次更新、预算末步与零rho仍强制完整检查。'
        '最终原A真L2与共同原生MAS加权相对残差均≤.0095，独立CPU B_ref均≤.01。'
        '没有残差替换/降低精度；所有回退及失败attempt保留在请求solve窗口。','',
        'solve含host提交、JSON曲线、筛选/质量归约与同步；CUDA event也可包含host提交间隙。'
        '不同批次A/b可能不同，因此上述比较只用本批同一系统，不将前批数字当配对基线。','',
        '## 连续16次apply（6对AB/BA）','',
        '| 场景 | Reference每apply中位ms | Fused每apply中位ms | 配对比值F/R中位 | PCG配对F/R中位 | 探索门槛 |',
        '|---|---:|---:|---:|---:|---|']
    for cost in summary['costs']:
        rows=[r for r in summary['batches'] if r['scene']==cost['scene']]
        ref=[r for r in rows if not r['fused']];fused=[r for r in rows if r['fused']]
        lines.append(f'| {cost["scene"]} | {med(ref,"per_apply_host_ms"):.6f} | {med(fused,"per_apply_host_ms"):.6f} | '
            f'{cost["apply_pair_ratio_median"]:.4f} | {cost["PCG_pair_ratio_median"]:.4f} | {cost["exploratory_cost_gate_passed"]} |')
    lines += ['', '批次窗口内仅事件/连续16次完整apply/末尾同步，无逐次JSON、导出或stage同步；'
        '输入固定b，每批恢复同scratch，输出在窗口外完整保存并对reference(b)和独立CPU公式核验。'
        'b/Z/W/E/L摘要保持不变。全部12个批次/场景均保留，event与host时间分别记录。'
        '每apply中位数和配对比值中位数计算口径不同，不能互相替代。','',
        '预定探索门槛为两场景每apply配对比值中位数≤.90、PCG比值中位数≤1.03；'
        '仅作为继续局部开发的信号。固定b探针不等价于真实PCG不同r，更不等价于帧净收益。','',
        '| 场景，三组配对差均值 | 总solve差ms | 完整验收差ms | 筛选差ms | 剩余差ms |',
        '|---|---:|---:|---:|---:|']
    for r in summary['costs']:
        need(abs(r['delta_ms']-r['quality_delta_ms']-r['screen_delta_ms']-r['remaining_delta_ms'])<1e-9,'Cost accounting mismatch')
        lines.append(f'| {r["scene"]} | {r["delta_ms"]:+.4f} | {r["quality_delta_ms"]:+.4f} | '
            f'{r["screen_delta_ms"]:+.4f} | {r["remaining_delta_ms"]:+.4f} |')
    lines += ['', '剩余窗口仍含递推/host/JSON等，不能全部归于某个GPU核。另测4次单apply和一次逐stage同步，'
        '数据见[apply_calibration](tables/apply_calibration.csv)、[apply_stages](tables/apply_stages.csv)；'
        '同步stage之和不替代真实solve或纯kernel时间。','',
        '### 这次不能混淆的两类变化','',
        '本批与上一批保存的系统确实不同，尺寸/active映射相同。下面分母为上一批原始A/b的范数；'
        'RHS相对差不等于物理位置误差。Bunny本批reference已约90次，上批为164次，'
        '不能把该跨批变化算作两处融合的收益；其完整因果来源本轮尚未隔离。','',
        '| 场景 | 方向 | A相对Frobenius差 | b相对L2差 |',
        '|---|---:|---:|---:|']
    for r in context['systems']:
        lines.append(f'| {r["scene"]} | n{r["direction"]} | {r.get("A_relative_frobenius",float("nan")):.6g} | '
            f'{r.get("b_relative_l2",float("nan")):.6g} |')
    lines += ['', '单次同步分项也有明显顺序/调度影响：Mixed未改实现的Zᵀr+粗求解在reference/fused探针中为'
        '0.3396/0.0473ms，未改的MAS为0.2440/0.0715ms。因此不能把分项探针差直接当作融合核的因果收益。'
        '完整数据与原输入绑定见[COST_CONTEXT_AUDIT.json](COST_CONTEXT_AUDIT.json)。','',
        '源码也有待单独验证的成本权衡：融合输出核每个输出都算8个a−c，而reference原先只算一次8维差量。'
        '可用独立核批次及共享系数原型检查这一代价；本轮未隔离它对Mixed变慢的贡献，不将它写成确定原因。'
        'Mixed的连续apply有5/6对变慢；完整PCG窗口下降与之不一致，目前不足以支持普遍采用融合。','',
        '## 数值与源空间核验','',
        '| 场景 | 源Gram相对差 | GPU Z/CPU差 | Reference(b)/CPU差 | Fused(b)/CPU差 | Fused(b)/Reference差 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['sources']:
        if r['case'] not in ('bunny','mixed'):continue
        keys=('Gram_relative','Z_relative','reference_b_CPU_relative','fused_b_CPU_relative','fused_b_reference_relative')
        lines.append('| '+r['scene']+' | '+' | '.join(format(r[k],'.6g') for k in keys)+' |')
    passed=sum(r['common_CPU_quality_passed'] for r in summary['targets'])
    lines += ['', f'全部完成目标解共同CPU精度通过{passed}/{len(summary["targets"])}份，包括注入故障前已完成对照。'
        'Bunny/Mixed pair1三臂共6份原生解另经Decimal50标量CSR复核；每次筛选/检查时机、实际模式和三臂顺序独立复核。'
        '源空间仅用前一n1普通legacy停止历史的first16+last112，rank8；n2数据不用于选空间。'
        'W=AZ、E=ZᵀW、L Lᵀ=sym(E)以完整原始CSR重算，warm校准后首次粗算子字节恢复。','',
        'B_ref将存储FP32局部逆显式对称并以FP64作用，原生MAS含FP32转换/归并，两者不等价。'
        '动作误差≤1e-3是有限探针门槛，不是全算子误差界。融合改变求和顺序，允许迭代轻微变化，'
        '数值主门槛要求fused实际使用、共同精度通过、迭代≤reference的1.10倍。','',
        '## 首次与热准备成本','',
        '| 场景 | 源构造ms | Target首次coarse ms | warm三次ms | n2私有准备ms |',
        '|---|---:|---:|---|---:|']
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        src=read(RAW/label/'fixed'/f'f{frame}_n1_native_balanced_study.json')
        target=read(RAW/label/'fixed'/f'f{frame}_n2_native_balanced_study.json')
        warm=' / '.join(format(r['host_ms'],'.4f') for r in target['coarse_warm_calibration'])
        lines.append(f'| {scene} | {src["source_build_host_ms"]:.4f} | {target["target_coarse_host_ms"]:.4f} | '
            f'{warm} | {target["private_setup_host_ms"]:.4f} |')
    lines += ['', 'source控制/采集只测一对，不能把其差值当稳定采集开销。采集/完整bank导出、源小特征分解、'
        'BLAS/显存分配、Z形成、coarse导出、reset/输出及恢复均在原始study单列。'
        'n2保护壳仍分配未使用的history bank，属于诊断准备成本。没有用warm值替代首次成本。','',
        '## 合同、异常及复现','',
        '- 新GPU合同覆盖8种尾块尺寸、全部输入逐位不变、一般/零/源列rhs的两路径CPU公式，以及原源投影/粗零秩拒绝。',
        '- 两项合同先通过，再完成29/29串行CTest。默认selector=0和新selector=1小例均显式MAS回退。',
        '- 首次全套CTest通过前16项后因Owned Job的Member image查询触发WinError5而中断；已清理并保留日志。'
        '没有修改或放宽Owned Job检查，另开目录同命令完整重跑29/29通过；该中断不算算法测试通过。',
        '- coarse拒绝与真实fused动作后异常覆盖九项恢复：完整系统/MAS scratch、主方向、工作区、Graph、算子、info、context、config及cost trace。',
        f'- 共{len(systems)}份固定系统报告、{sum(len(s["runs"]) for s in systems)}份完成solve；部分失败attempt另存。',
        f'- exe SHA256 `{identity["exe"]["sha256"]}`；37个native TU及实际编译/对象/链接证据封存；有编译警告。',
        f'- 电源GUID `{packet["power_guid"]}`；RTX3070 Laptop/WDDM共享桌面，逐次电源/显存/Owned Job证明保存。',
        '- 材料/FullCCD、dt=.04、rho1e-4、legacy .01/min6及此前最快执行组合不变。',
        f'- 父阶段{len(parent["files"])}文件通过原字节映射核验；三处旧源码保存在prior_sources。'
        '当前源码、原始解/批次输出、表、构建与本文绑定于[DELIVERY_SEAL.json](DELIVERY_SEAL.json)。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_fused_apply.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/verify_fused_apply.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/build_windows.py --kind active --label native_fused_apply_20261016 --configured --jobs 2",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/verify_fused_apply.py full",
        '# 本批首次监控中断，保留失败后显式执行retry_fused_apply_ctest.py，未覆盖原目录',
        '# run_fused_apply.py依次seal、small、faults、representative，GPU串行',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_fused_apply.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_fused_apply.py",'```','',
        '详细数据：[逐次时间](tables/native_timings.csv)、[全部解精度](tables/target_solutions.csv)、'
        '[批次输出/时间](tables/apply_batches.csv)、[配对成本](tables/paired_cost.csv)、[曲线](tables/native_curves.csv)。'
        '入口拒绝覆盖既有批次；原始数据在`runs/innovation_p5_fused_apply_20261016`。','',
        '## 有限下一步','']
    if not summary['numerical_gate_passed']:
        decision='停止扩大到真实帧/Graph；先定位浮点求和变化、动作或共同精度失败，不靠放松阈值推进。'
    elif not summary['exploratory_cost_gate_passed']:
        decision='数值门槛通过，但两处局部融合未同时达到两场景的探索成本门槛；保持Reference默认。下一步先独立测两处投影核及系数差共享原型，核对Mixed的额外算术/提交成本与窗口波动，再决定优化剩余转置投影/粗求解，不扩大到100帧。'
    else:
        decision='数值与有限成本信号通过；保留Reference默认与回退。下一步将投影融合与剩余提交成本一并做短程真实帧入口，并完整计入源采集/构造/首次准备及物理质量，再做100帧对照。当前不能宣称帧净加速或2倍提升。'
    lines.append(decision)
    report=REPORT/'REPORT.md';need(not report.exists(),'Report exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        need((report.parent/link).exists() or report.parent/link==dest,'Report link missing')
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+tree(snapshot)
    for phase in ('contracts','full','full_retry1'):
        folder=ROOT/f'build/fused_apply_verification_{phase}'
        if folder.exists():records+=tree(folder)
    records+=identity['sources']+[b['source'] for b in bindings]+[prior['parent']]
    records+=[record(p) for p in (ROOT/'build').glob('fused_apply*.log') if 'finish' not in p.name]
    unique={r['path']:r for r in records};files=[unique[k] for k in sorted(unique)];verify(files)
    write_new(dest,dict(schema='native_fused_apply.v1',files=files,parent=prior['parent'],parent_mutable_mapping=mappings,
        parent_files_verified=len(parent['files']),source_bindings=bindings,binary=frozen['exe'],CTest=29,
        native_systems=len(systems),native_completed_solves=sum(len(s['runs']) for s in systems),
        numerical_gate_passed=summary['numerical_gate_passed'],exploratory_cost_gate_passed=summary['exploratory_cost_gate_passed'],
        production_solver_modified=False,performance_certified=False,next_finite_decision=decision))
    verify(read(dest)['files']);print(json.dumps(dict(sealed=True,files=len(files),
        numerical_gate_passed=summary['numerical_gate_passed'],exploratory_cost_gate_passed=summary['exploratory_cost_gate_passed'])),flush=True)

if __name__=='__main__':main()
