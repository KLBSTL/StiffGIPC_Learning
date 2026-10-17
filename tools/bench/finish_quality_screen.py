"""Seal the bounded quality-screen experiment without rewriting parent evidence."""
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
import analyze_quality_screen
import run_quality_screen
import prepare_quality_screen
import verify_quality_screen

RAW=ROOT/'runs/innovation_p5_quality_screen_20261016'
REPORT=ROOT/'reports/innovation_p5_quality_screen_20261016'
BUILD=ROOT/'build/native_quality_screen_20261016'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def median(rows,key):return statistics.median(r[key] for r in rows)

def main():
    destination=REPORT/'DELIVERY_SEAL.json';need(not destination.exists(),'Stage already sealed')
    summary=read(RAW/'analysis/summary.json');need(summary['implementation_checks_passed'],'Independent audit required')
    verify(summary['inputs']+[summary['source']])
    for phase,count in (('contracts',2),('full',29)):
        evidence=read(ROOT/f'build/quality_screen_verification_{phase}/verification.json')
        need(evidence['passed'] and evidence['expected']==count and evidence['cleanup_owned_job_empty'],'CTest incomplete')
        verify([evidence['source']])
    cb=read(BUILD/'contracts_identity.json');verify(cb['sources']+[cb['source']]+cb['executables'])
    prior=read(REPORT/'PRIOR_SOURCES.json');verify([prior['parent']]);parent=read(prior['parent']['path'])
    preserved={r['original']['path']:r for r in prior['files']};mappings=[]
    for row in parent['files']:
        if row['path'] in preserved:
            old=preserved[row['path']];need(row==old['original'],'Prior mapping differs')
            verify([old['archive']]);mappings.append(old)
        else:verify([row])
    need(len(mappings)==4,'Unexpected parent edit scope')
    packet=read(RAW/'seal.json');identity=packet['identity']
    verify(identity['sources']+[identity['exe']]+identity['dlls']+[packet['runner'],packet['protocol']])
    write_new(RAW/'compiled_identity.json',identity)
    snapshot=ROOT/'build/quality_screen_snapshot_20261016'
    frozen=freeze_active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log',snapshot,
        from_verified_identity=RAW/'compiled_identity.json')
    systems=[];entries=[]
    for phase in ('small','faults','representative'):
        ledger=read(RAW/(phase+'_ledger.json'));need(ledger['complete'],'Incomplete native stage')
        for entry in ledger['runs']:
            need(entry['passed'] and entry['result']['cleanup_owned_job_empty'],'Owned native run incomplete')
            verify(entry['studies']);entries.append(entry);systems.extend(read(r['path']) for r in entry['studies'])
    source_files={Path(__file__).resolve(),ROOT/'tools/build_windows.py',ROOT/'tests/native_balanced_math_test.cpp',
                  ROOT/'tests/native_balanced_gpu_test.cu'}
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
    apply_rows=[];apply_stages=[]
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        target=read(RAW/label/'fixed'/f'f{frame}_n2_native_balanced_study.json')
        apply_rows.extend(dict(scene=scene,**r) for r in target['separate_apply_calibration'])
        apply_stages.extend(dict(scene=scene,**r) for r in target['separate_synchronized_apply_stages'])
    analyze_quality_screen.csv_new(REPORT/'tables/separate_apply_calibration.csv',apply_rows)
    analyze_quality_screen.csv_new(REPORT/'tables/separate_synchronized_apply_stages.csv',apply_stages)
    lines=['# 原生balanced：验收筛选与coarse冷/热实测','',
        '2026-10-16。本轮完成固定系统诊断路径的验收筛选，默认关闭，生产PCG/Graph循环保持。'
        '用户已将100帧对照推迟到改进和短测之后；本轮没有启动100帧或上传GitHub。','',
        f'**实现核验通过；预先规定的有限数值门槛：{summary["finite_gate_passed"]}。** '
        '以下时间来自共享桌面上的固定系统仪器化诊断，不作为真实帧净加速比。','',
        '协议见[PLAN.md](PLAN.md)，逐段实现检查见[SOURCE_AUDIT.md](SOURCE_AUDIT.md)。','',
        '## 同一系统四臂对照','',
        '| 场景 | 原生预处理 | 完整验收频率 | 三次迭代数 | solve中位数ms | 完整检查中位数 | 筛选归约中位数 |',
        '|---|---|---|---|---:|---:|---:|']
    for label,scene in (('bunny','Bunny14 n2'),('mixed','Mixed15 n2')):
        for aug,screen in ((False,False),(True,False),(False,True),(True,True)):
            rows=sorted((r for r in summary['timings'] if r['case']==label and r['requested_balanced']==aug
                         and r['screened']==screen),key=lambda r:r['pair'])
            lines.append(f'| {scene} | {"balanced" if aug else "MAS"} | {"screened" if screen else "dense"} | '
                f'{" / ".join(str(r["iterations"]) for r in rows)} | {median(rows,"solve_host_ms"):.4f} | '
                f'{median(rows,"quality_checks"):g} | {median(rows,"screen_checks"):g} |')
    lines += ['', '每个场景从零推进到目标帧；在同一次n2快照内，12次目标solve共享完全相同的A、b、'
        'MAS系数和rank8源空间Z，各自零x及相同scratch起步。重复1 MD/BD/MS/BS，重复2反序，'
        '重复3 MS/BS/MD/BD；pair1预先规定为数值主重复，全部重复与回退保留。','',
        'dense在previous-rho候选激活后每次更新检查真实质量。screened先做FP64递推r的L2归约，'
        '≤.012才做完整检查；首候选、距上次完整检查8次更新、预算最后一步和零rho强制完整检查。'
        '最终两项原A真L2/共同原生MAS加权残差仍须≤.0095；独立CPU B_ref两项须≤.01。'
        '没有残差替换或放松精度，额外筛选归约与同步已计入solve窗口。','',
        '## 筛选相对于dense的成本','',
        '| 场景 | 预处理 | 配对总solve差ms | 已测完整验收差ms | 新增筛选ms | 剩余窗口差ms |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary['additive_pair_means']:
        lines.append(f'| {r["scene"]} | {"balanced" if r["requested_balanced"] else "MAS"} | '
            f'{r["solve_delta_ms"]:+.4f} | {r["quality_delta_ms"]:+.4f} | {r["extra_screen_ms"]:.4f} | '
            f'{r["remaining_delta_ms"]:+.4f} |')
    lines += ['', 'Bunny balanced完整检查由每次92次降到17次，迭代均为164；但筛选后仍慢于同系统MAS。'
        'Mixed balanced仅由15次降到13次，新增归约15次；已测验收+筛选窗口平均反而增加约0.480ms，'
        '总solve的下降来自剩余窗口减少约2.333ms，不能据此确认筛选本身产生稳定收益。','',
        '本表用三组配对差的算术平均保留加法关系：总差=验收差+新增筛选+剩余差。'
        '负值表示screened更省。剩余窗口含递推、balanced投影、host提交和JSON逐步记录，'
        '不能全部解释为某个GPU kernel。完整验收调用数与筛选调用数均由trace独立重算，'
        '并检查首候选、最大间隔与末次验收。','',
        'solve_host_ms含诊断提交/同步、JSON曲线与完整验收；CUDA event也可能含host提交间隙。'
        'reset、解导出在窗口外单列，失败attempt加回退保留在请求solve总窗口。'
        '本轮没有关闭JSON记录的独立对照，故未量出其单独因果成本。','',
        '## Coarse首次与warm成本','',
        '| 场景 | 首次实际ms | warm三次ms | warm中位数ms | 首次/Warm中位数 | 恢复首次W/E/L原字节 |',
        '|---|---:|---|---:|---:|---|']
    for scene in ('Bunny','Mixed'):
        rows=[r for r in summary['coarse'] if r['scene']==scene and r['case'] in ('bunny','mixed')]
        first=next(r['host_ms'] for r in rows if r['kind']=='first');warm=[r for r in rows if r['kind']=='warm']
        wm=median(warm,'host_ms')
        lines.append(f'| {scene} | {first:.4f} | {" / ".join(format(r["host_ms"],".4f") for r in warm)} | '
            f'{wm:.4f} | {first/wm:.3f} | 通过 |')
    lines += ['', '每次coarse均包含8列原生SpMV、FP64 GEMM、D2H、CPU条件/LLT检查和L上传。'
        'warm样本是在同Z/同A上追加调用，W/E相对变化均≤1e-10；正式四臂前恢复首次W/E/L并检查原字节。'
        '首次与warm差值是同进程首次调用效应的观测，未进一步隔离库初始化/缓存的独立因果来源。'
        '不以warm值覆盖首次实际成本，也不把校准成本藏入正式solve。','',
        '| 场景 | 源control solve ms | 源capture solve ms | 源空间构造ms | n2私有准备ms | coarse验证/恢复ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        src=read(RAW/label/'fixed'/f'f{frame}_n1_native_balanced_study.json')
        target=read(RAW/label/'fixed'/f'f{frame}_n2_native_balanced_study.json')
        control=next(r['solve_host_ms'] for r in src['runs'] if not r['capture'])
        capture=next(r['solve_host_ms'] for r in src['runs'] if r['capture'])
        lines.append(f'| {scene} | {control:.4f} | {capture:.4f} | {src["source_build_host_ms"]:.4f} | {target["private_setup_host_ms"]:.4f} | '
            f'{target["coarse_warm_verification_restore_ms"]:.4f} |')
    lines += ['', '源历史采集/导出、源CPU特征分解、BLAS/显存分配、GPU Z形成、Z与解导出和最终恢复均在原始study中独立保存。'
        '当前保护壳n2也分配未使用的history bank，属于诊断准备开销。control/capture只测一对且存在顺序/调度波动；'
        'Bunny的负时间差不能当作负采集开销。上述分项不能直接相加成生产帧时间。','',
        '## 另测的预处理动作成本','',
        '| 场景 | 动作 | 四次同步完整apply ms |',
        '|---|---|---|']
    for scene in ('Bunny','Mixed'):
        for augmented in (False,True):
            rows=[r for r in apply_rows if r['scene']==scene and r['balanced']==augmented]
            lines.append(f'| {scene} | {"balanced" if augmented else "MAS"} | '
                f'{" / ".join(format(r["whole_apply_host_ms"],".4f") for r in rows)} |')
    lines += ['', '全部样本保留，前次与后次差异明显；这是同一b的有限动作探针，不能据此给出每个真实PCG残差向量的精确成本。'
        '另存[分阶段单次同步](tables/separate_synchronized_apply_stages.csv)，含两次转置投影/粗求解、'
        '输入复制/W投影、原生MAS及最终Z投影。逐stage同步会增加开销，其时间之和不替代配对solve或纯kernel时间。','',
        '## 独立数值核验','',
        '| 场景 | 源统计/直接Gram差 | GPU Z/CPU系数差 | 原生MAS(b)/B_ref差 | 原生balanced(b)/同Z参考差 |',
        '|---|---:|---:|---:|---:|']
    for r in summary['sources']:
        if r['case'] not in ('bunny','mixed'):continue
        keys=('Gram_relative','Z_relative','native_MAS_b_vs_fixed_reference_relative','native_balanced_b_vs_fixed_reference_relative')
        lines.append('| '+r['scene']+' | '+' | '.join(format(r[k],'.6g') for k in keys)+' |')
    zero_count=sum(r['zero_rho_override'] for r in summary['targets'])
    quality_count=sum(r['common_CPU_quality_passed'] for r in summary['targets'])
    lines += ['', '源空间只用前一真实n1的普通legacy原生PCG历史，取前16+尾112，rank固定8，'
        '目标n2数据不用于选空间。所有W=AZ、E=ZᵀW、L Lᵀ=sym(E)由完整原始CSR复核；'
        '全部目标解以原始CSR重算L2并用共同CPU B_ref重算加权残差。','',
        f'CPU共同精度通过{quality_count}/{len(summary["targets"])}份已完成目标解（包括故障前已完成的MAS对照）；'
        'Bunny/Mixed的pair1四臂共8份原生解另经Decimal50标量CSR复核。'
        f'本次实际出现zero-rho强制例外{zero_count}次；未出现的运行路径不声称已被场景测试覆盖。','',
        'B_ref将存储的FP32局部逆显式对称后以FP64作用，原生MAS含FP32转换与归并，两者不等价。'
        'b单向量动作校准属于有限探针，不是全算子误差界。不同批次的A/b可能不同，'
        '本轮所有频率结论只比较同一次快照内四臂，不用前批迭代数直接作配对基线。','',
        '## 合同与恢复','',
        '- 先通过2项CPU/GPU合同，再完成29/29串行CTest；新增筛选边界合同覆盖首候选、8次间隔、.012边界、预算末步、零rr及非法/非有限输入。',
        '- Bunny1分别跑selector=0原6次target与selector=1新12次target，均为不足rank后的显式MAS回退。',
        '- Bunny14的coarse_reject验证有效源空间拒绝后的四臂回退；after_apply在真实balanced动作后抛出预期异常/非零退出。',
        '- 正常/异常均检查九项恢复：完整系统/MAS scratch、原主方向、PCG工作区、Graph状态、算子、info、context、config及cost trace。',
        f'- 共{len(systems)}个固定系统报告、{sum(len(s["runs"]) for s in systems)}个完成solve记录；故障的未完成attempt另存。'
        '这些不是所有CUDA/IO失败的穷举。','',
        '## 复现与保存','',
        f'- exe SHA256：`{identity["exe"]["sha256"]}`；全新构建{len(identity["source_object_map"])}个native TU，'
        '实际编译、对象、链接、运行输入及resolved config绑定。编译日志保留警告，不声称零警告。',
        f'- 电源GUID：`{packet["power_guid"]}`；本机RTX3070 Laptop/WDDM共享桌面，逐次电源/显存和Owned Job清理记录保留。',
        '- dt=.04、rho=1e-4、legacy累计.01/min6、FullCCD/材料及此前最快执行组合保持。balanced仍为受保护host诊断。',
        '- 前阶段2173文件通过原字节映射复核；四处重叠源码在prior_sources保留；当前源码也另存。'
        '所有原始数据、表、源码、二进制及构建身份由[DELIVERY_SEAL.json](DELIVERY_SEAL.json)绑定。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_quality_screen.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/verify_quality_screen.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/build_windows.py --kind active --label native_quality_screen_20261016 --configured --jobs 2",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/verify_quality_screen.py full",
        '# run_quality_screen.py依次seal、small、faults、representative，GPU串行',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_quality_screen.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_quality_screen.py",'```','',
        '入口拒绝覆盖既有结果；原始A/b/M、history、Z/W/E/L、逐次解与曲线在'
        '`runs/innovation_p5_quality_screen_20261016`。关键表：'
        '[逐次时间](tables/native_timings.csv)、[逐次残差](tables/target_solutions.csv)、'
        '[筛选配对差](tables/screen_vs_dense.csv)、[完整曲线](tables/native_curves.csv)。','',
        '## 下一步判断','']
    if not summary['finite_gate_passed']:
        decision='有限数值门槛未全部通过，停止扩大到真实帧/Graph；先定位筛选延迟、共同精度或回退问题，保留否定数据。'
    else:
        faster=[]
        for scene in ('Bunny','Mixed'):
            rows=[r for r in summary['comparisons'] if r['scene']==scene and r['requested_balanced']]
            if median(rows,'solve_delta_ms')<0:faster.append(scene)
        decision=f'有限数值门槛通过；balanced筛选相对dense的配对solve中位差为负的场景：{", ".join(faster) or "无"}。'
        decision+='尚需在同一A/b/Z上降低balanced投影/提交成本，并把源空间采集、构造与首次准备成本计入，再接入默认关闭的真实帧路径做短程质量核验。之后再运行100帧对照；目前不能给出其净加速比。'
    lines.append(decision)
    report=REPORT/'REPORT.md';need(not report.exists(),'Report already exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        need((report.parent/link).exists() or report.parent/link==destination,'Report link missing: '+link)
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+tree(snapshot)
    for phase in ('contracts','full'):records+=tree(ROOT/f'build/quality_screen_verification_{phase}')
    records+=identity['sources']+[b['source'] for b in bindings]+[record(Path(prior['parent']['path']))]
    records+=[record(p) for p in (ROOT/'build').glob('quality_screen*.log') if 'finish' not in p.name]
    unique={r['path']:r for r in records};files=[unique[k] for k in sorted(unique)];verify(files)
    write_new(destination,dict(schema='native_quality_screen.v1',files=files,parent=prior['parent'],
        parent_mutable_mapping=mappings,parent_files_verified=len(parent['files']),source_bindings=bindings,
        binary=frozen['exe'],CTest=29,new_contracts=2,native_systems=len(systems),
        native_completed_solves=sum(len(s['runs']) for s in systems),finite_gate_passed=summary['finite_gate_passed'],
        production_solver_modified=False,performance_certified=False,next_finite_decision=decision))
    verify(read(destination)['files'])
    print(json.dumps(dict(sealed=True,files=len(files),native_systems=len(systems),
        finite_gate_passed=summary['finite_gate_passed'])),flush=True)

if __name__=='__main__':main()
