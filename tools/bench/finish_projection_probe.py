"""Report and seal the standalone fixed-input projection experiment."""
import ast
from pathlib import Path
import re
import shutil
import sys
from m3_numeric import ROOT,need
from local_identity import record,verify,tree
from linux_runner import write_new
import prepare_projection_probe as prep
import run_projection_probe
import analyze_projection_probe
RAW,REPORT,BUILD=prep.RAW,prep.REPORT,prep.BUILD
read=prep.read

def main():
    dest=REPORT/'DELIVERY_SEAL.json';need(not dest.exists(),'Already sealed')
    summary=read(RAW/'analysis/summary.json');need(summary['implementation_checks_passed'],'Independent audit required')
    verify(summary['inputs']+[summary['source']]);parent=read(REPORT/'PARENT.json')['parent'];verify([parent])
    old=read(parent['path']);verify(old['files'])
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify([packet[k] for k in ('identity','manifest','protocol','runner','parent')]+packet['compiled'])
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']])
    verify(manifest['original_inputs']+manifest['generated_inputs']+[manifest['source'],manifest['protocol']])
    for action in ('contracts','measure'):
        result=read(RAW/action/'result.json');need(result['passed'] and result['cleanup_owned_job_empty'],'Incomplete native run')
    sources={Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT/'tools'):sources.add(p)
    sources.update(Path(r['path']) for r in identity['sources'])
    bindings=[]
    for p in sorted(sources):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
        target=REPORT/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,target.open('xb') as out:shutil.copyfileobj(src,out)
        a,b=record(p),record(target);need((a['bytes'],a['sha256'])==(b['bytes'],b['sha256']),'Source copy differs')
        bindings.append(dict(source=a,snapshot=b))
    def ratio(case,graph,comparison):
        return next(r for r in summary['aggregates'] if r['case']==case and r['graph']==graph and r['comparison']==comparison)
    lines=['# P5：投影单核与块内共享系数诊断','',
        '2026-10-16。本轮只新增独立小工程，复用父阶段同一封存输入与旧核，验证共享8个系数差的原型。'
        '现有模拟源码/默认实现/二进制未改，没有重新推进场景、接入真实PCG或运行100帧。','',
        f'**80项投影合同、720组测量独立核验通过；共享原型有限门槛={summary["shared_prototype_gate_passed"]}。** '
        '该具体共享实现未解决Mixed，不作默认替换。','',
        '[预定协议](PLAN.md)；[逐段代码检查](SOURCE_AUDIT.md)。','',
        '## 共享输出核相对naive输出核','',
        '| 场景/输入 | host event配对比中位 | Graph event配对比中位 | Graph三轮比值中位 | Graph改善轮数 |',
        '|---|---:|---:|---|---:|']
    for case in ('bunny_rhs','bunny_residual','mixed_rhs','mixed_residual'):
        h=ratio(case,False,'output_shared/output_naive');g=ratio(case,True,'output_shared/output_naive')
        lines.append(f'| {case} | {h["event_ratio_median"]:.4f} | {g["event_ratio_median"]:.4f} | '
            f'{" / ".join(format(v,".4f") for v in g["event_ratio_rounds"])} | {g["improving_rounds"]}/3 |')
    lines += ['', '比值<1表示共享原型更省；每个值来自同一输入18组配对。Mixed的rhs/residual在Graph下'
        '分别约慢5.5%/6.4%，各三轮中位数都未改善，未达到预先规定的≤.95且至少2/3轮改善。'
        'Bunny的收益不支持把共享实现普遍替换到Mixed。','',
        '共享前后8组完整warm银行输出（4输入×2mode）逐位相同；这说明该样本中的数值一致，'
        '不能推断它已在真实FP32 MAS/PCG中通过验证。','',
        '## 父融合核相对cuBLAS阶段对照','',
        '| 输入，Graph模式 | Input fused/reference | Output naive/reference | Output shared/reference |',
        '|---|---:|---:|---:|']
    for case in ('bunny_rhs','bunny_residual','mixed_rhs','mixed_residual'):
        values=[ratio(case,True,comp)['event_ratio_median'] for comp in
            ('input_fused/input_reference','output_naive/output_reference','output_shared/output_reference')]
        lines.append('| '+case+' | '+' | '.join(format(v,'.4f') for v in values)+' |')
    lines += ['', '在这个银行式单阶段探针里，Mixed原融合输入/输出核相对阶段reference仍有收益。'
        '所以不能把前轮完整apply慢13.4%简单解释成“两个核在所有条件下都更慢”。'
        '但本轮不含MAS及完整依赖链，银行写入会改变缓存状态，亦不能据此排除它们在真实链中的成本问题。'
        '提交空隙、节点调度、原生MAS交互和缓存均需后续直接对照，尚未定位其完整因果来源。','',
        '### Graph节点与执行范围','',
        '| 每批32次 | 实际节点数 |',
        '|---|---:|',
        '| input reference（复制+GEMV） | 64 |',
        '| input fused | 32 |',
        '| output reference（差量+GEMV） | 64 |',
        '| output naive / shared | 32 / 32 |','',
        'host模式逐次提交32个阶段，Graph模式一次回放同一批次；两者均包含完整阶段动作，Graph构造单列在warm记录。'
        'CUDA event在host模式也含提交间隙，Graph仍有节点调度开销，二者的差值不是纯host调用成本。'
        '这也不是原生产PCG Graph的整帧加速比。所有host/event时间和两mode配对均保留。','',
        '## 数值核验与数据构造','',
        f'- 8种尺寸1/31/32/33/255/256/257/1025 ×5方法×host/Graph，共{summary["contract_checks"]}项合同，'
        '每项检查全部32个输出银行及系数工作区；覆盖尾块barrier与输入逐位不变。',
        f'- 真实固定输入共{summary["sample_count"]}组，每组原生读回全部32个输出做CPU公式检查；'
        f'Python另外核对40份完整warm银行、{summary["saved_output_checks"]}份首尾输出。',
        f'- {summary["Decimal50_checks"]}项选定边界/误差较大行的50位标量表达式核验通过。'
        '这是选定行检查，未把它描述为全量高精度残差或PCG验收。',
        '- 真实输入CPU公式要求相对L2≤1e-10，小合同≤2e-12；FP64投影及8列顺序保持，无低精度替代。',
        '- 四输入为Bunny14 n2、Mixed15 n2的b和父reference主重复的最终真实残差。原Z/W/L与系统A/B绑定；'
        'a/u/c由存储FP32 MAS局部逆显式对称后以FP64作用的B_ref构造。b探针另与父原生balanced(b)动作校准≤1e-3。',
        '- C++ Eigen公式、Python NumPy公式、选定Decimal50表达式交叉核验；不只依赖程序自身passed。'
        '这些投影输入不冒充原生MAS每个PCG迭代的完整向量回放。','',
        '## 编译资源与实现权衡','',
        '| 输出核 | 寄存器 | shared bytes | barrier |',
        '|---|---:|---:|---:|']
    for r in summary['resources']:
        if 'balanced_output_projection' in r['symbol']:name='父naive'
        elif 'shared_output_projection' in r['symbol']:name='shared prototype'
        else:continue
        lines.append(f'| {name} | {r["registers"]} | {r["shared_bytes"]} | {r["barriers"]} |')
    lines += ['', '共享原型减少重复差量计算，但同时引入共享访问/块同步，并使用size_t索引（父核为int）。'
        '因此本轮否定的是这个具体实现的Mixed性能门槛，不能孤立断言某一次barrier、索引或8个减法是唯一原因。'
        '寄存器减少也不等于实测occupancy或必然加速。','',
        '独立工程包含父封存源码，实际编译命令/ptxas报告保留；工程与完整程序的代码生成上下文/选项存在差别，'
        '不声称GPU机器码完全相同。','',
        '## 计时与适用范围','',
        '- 每种方法每组有独立32份输出/c工作区，统一reset在计时外。否则reference重复相减及输出累加会改变问题。',
        '- host/event窗口只含阶段提交或一次Graph launch及末尾同步；reset、全部输出D2H/CPU验证、选择性导出分别记录。',
        '- 三轮变换场景/mode顺序，输入AB/BA各3组，输出六种排列；独立脚本逐项检查720组顺序和调用数。',
        '- Mixed输出银行约14.86MiB，实际单向量约0.46MiB；银行式写入对缓存/带宽的影响必须保留在解释里。',
        '- WDDM共享桌面有明显离群；Mixed rhs Shared/Naive Graph配对值约0.45–2.81，'
        '各轮中位数仍都>1。没有删离群、挑最快、使用未完成/资源失败批次认证收益。',
        '- 本轮未重新运行既有29项全程序CTest、实际FP32 MAS或PCG/物理推进；既有程序完全未改，'
        '本轮验收范围是新增独立探针。','',
        '## 身份、进程及复现','',
        f'- Probe exe SHA256 `{identity["exe"]["sha256"]}`。编译器、实际CUDA命令、对象、链接输入、'
        'cuBLAS运行库与源码绑定。SDK/系统头文件和driver依赖未做完整闭包封存。',
        f'- 电源GUID `{packet["power_guid"]}`；物理GPU UUID与CUDA运行时UUID一致，Owned Job/GPU串行/显存/外来计算保护保留。',
        '- 封存脚本有两次运行前错误：工程相对路径基准和.tlog目录过滤。均在GPU启动前停止，'
        '修正后保留原日志，没有放宽任何编译身份或运行身份检查。',
        f'- 父阶段{len(old["files"])}文件复核未变；没有改现有源文件、材料/FullCCD/legacy/PCG精度或上传GitHub。',
        '- 新源码、输入、输出、720组原始时间、合同及本文见[DELIVERY_SEAL.json](DELIVERY_SEAL.json)。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_projection_probe.py build",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_projection_probe.py inputs",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_projection_probe.py seal",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_projection_probe.py contracts",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_projection_probe.py measure",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/analyze_projection_probe.py",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_projection_probe.py",'```','',
        '入口拒绝覆盖已存在输出。关键表：[原始时间](tables/timings.csv)、[18组配对](tables/paired_ratios.csv)、'
        '[轮次汇总](tables/aggregate_ratios.csv)、[host/Graph对照](tables/host_graph_pairs.csv)、'
        '[独立公式](tables/warm_CPU.csv)、[Decimal50](tables/Decimal50_selected_rows.csv)。','',
        '## 下一步','',
        '共享实现未通过Mixed门槛，保持原默认与父naive实现。下一步优先在同一固定A/b/Z上直接对照'
        '“真实原生MAS完整apply的host提交与私有Graph回放”，核验MAS scratch与完整动作/恢复，再拆完整链交互。'
        '本轮单核结果不能替代该实验；暂不接入真实帧或扩大到100帧。']
    report=REPORT/'REPORT.md';need(not report.exists(),'Report exists');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        need((report.parent/link).exists() or report.parent/link==dest,'Report link missing')
    files=tree(RAW)+tree(REPORT)+tree(BUILD)+identity['sources']+identity['runtime']+identity['build_evidence']
    files+=[record(p) for p in sources]+[parent]
    files+=[record(p) for p in (ROOT/'build').glob('projection_probe*.log') if 'finish' not in p.name]
    unique={r['path']:r for r in files};files=[unique[k] for k in sorted(unique)];verify(files)
    write_new(dest,dict(schema='projection_probe.v1',files=files,parent=parent,parent_files_verified=len(old['files']),
        existing_source_edits=[],source_bindings=bindings,binary=identity['exe'],contracts=80,measurements=720,
        shared_prototype_gate_passed=summary['shared_prototype_gate_passed'],production_solver_modified=False,
        performance_certified=False,next_finite_decision=lines[-1]))
    verify(read(dest)['files']);print(dict(sealed=True,files=len(files),shared_prototype_gate_passed=summary['shared_prototype_gate_passed']),flush=True)

if __name__=='__main__':main()
