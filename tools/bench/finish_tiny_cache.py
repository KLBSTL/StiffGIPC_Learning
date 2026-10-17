"""Seal finite three-arm cache experiment and make the predeclared decision."""
import ast
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_tiny_cache import RAW,REPORT,BUILD,PROJECT,PARENT,read,transform,functions
from tiny_cache_experiment import implementation,verify_identity,schedule
from local_identity import record,verify,tree
from linux_runner import require,write_new

def main():
    require(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed');packet,identity,manifest=verify_identity();mod=implementation()
    stages=['tiny','contracts','sanitizer_memcheck','sanitizer_racecheck','sanitizer_synccheck','tiny_measure','measure']
    audits={s:read(RAW/s/'audit.json') for s in stages};require(all(a['passed'] for a in audits.values()),'All stages required')
    require(sum(a['outputs'] for a in audits.values())==3754,'Wrong independent output total')
    resource=read(RAW/'compiler/audit.json');require(resource['passed'],'No-spill resource audit required');verify(resource['files']+[resource[k] for k in ['tool','exe','source']])
    resources={r['method']:r for r in resource['rows']}
    for stage in stages:
        audit=audits[stage];verify(audit['files']);result=read(RAW/stage/'result.json')
        require(result['passed'] and result['exit_code']==0 and result['cleanup_owned_job_empty'],'Owned stage/cleanup failed')
        data=read(RAW/stage/'data.json');action='tiny' if stage.startswith('sanitizer_') else stage
        mod.strict_metadata(manifest,data,RAW/stage,action)
        if data['samples']:schedule(data)
        for case in data['cases']:
            for g in case['graphs']:
                for k in g['tiny_launches']:
                    require(k['registers_per_thread']==resources[k['kernel']]['resources']['REG'] and k['local_bytes']==0,'Actual kernel resource differs')
    # Reconstruct every generated copy and its audited validation functions.
    generation=read(BUILD/'generation.json')
    for g in generation['files']:
        verify([g['source'],g['target']]);original=Path(g['source']['path']).read_text(encoding='utf-8')
        expected=transform(original,[(e['old'],e['new'],e['count']) for e in g['edits']])+g['suffix']
        require(expected==Path(g['target']['path']).read_text(encoding='utf-8'),'Generated source differs')
    f=read(BUILD/'fixture_generation.json');verify([f['template'],f['continuation'],f['target']]);s=Path(f['template']['path']).read_text(encoding='utf-8')
    require(s.count(f['boundary'])==1,'Ambiguous setup boundary')
    expected=transform(s.split(f['boundary'])[0],[(e['old'],e['new'],e['count']) for e in f['prefix_edits']])+Path(f['continuation']['path']).read_text(encoding='utf-8')
    require(expected==Path(f['target']['path']).read_text(encoding='utf-8'),'Fixture prefix differs')
    v=read(BUILD/'validation_generation.json');verify([v['numeric_source'],v['review_source'],v['target']])
    numeric=functions(Path(v['numeric_source']['path']),v['numeric_functions']);numeric=transform(numeric,[(e['old'],e['new'],e['count']) for e in v['numeric_edits']])
    numeric=numeric.replace("len(c['warm'])==6,'Wrong tiny warm count'","len(c['warm'])==9,'Wrong tiny warm count'").replace("len(c['warm'])==6,'Wrong full warm count'","len(c['warm'])==8,'Wrong full warm count'")
    strict=transform(functions(Path(v['review_source']['path']),v['review_functions']),[(e['old'],e['new'],e['count']) for e in v['strict_edits']])
    require(v['imports']+numeric+'\n'+strict==Path(v['target']['path']).read_text(encoding='utf-8'),'Strict/numeric generated validator differs')
    for parent in [PARENT,ROOT/'reports/innovation_p5_tiny_prefetch_20261017_retry1/DELIVERY_SEAL.json',ROOT/'reports/innovation_p5_cost_split_20261017/DELIVERY_SEAL.json']:verify(read(parent)['files'])
    verify(read(RAW/'production_before.json'))
    # Freeze this phase's complete local Python helper closure.
    seeds=[ROOT/'tools/local'/n for n in ['prepare_tiny_cache.py','run_tiny_cache.py']]+[ROOT/'tools/bench'/n for n in ['tiny_cache_experiment.py','finish_tiny_cache.py']]
    pending=seeds[:];seen=set()
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else ([n.module] if isinstance(n,ast.ImportFrom) and n.module else [])
            for name in names:
                for base in [ROOT/'tools',ROOT/'tools/local',ROOT/'tools/bench']:
                    q=base/Path(*name.split('.')).with_suffix('.py')
                    if q.is_file():pending.append(q);break
    sources=[record(p) for p in seen]+tree(PROJECT)+tree(BUILD/'project')+tree(BUILD/'overlay');dest=REPORT/'sources';dest.mkdir()
    for r in sources:
        p=Path(r['path']);q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
    small=audits['tiny_measure'];full=audits['measure'];aa=small['AA_passed'] and full['AA_passed'];eligible=aa and full['all_AB_signal']
    next_step=('进入同一固定A/b的原生PCG配对诊断，先验证最终解/显式残差、迭代数和总PCG时间；不直接接管物理100帧。' if eligible else
        '不接入生产；先审查未通过的A/A或收益格，若是稳定性问题则另立长窗口控制协议，不用本轮挑快样本推广。')
    lines=['# P5：缓存8维中间向量a＋固定循环展开','',
        ('**数值、资源与动态检查通过；达到预定固定apply候选门槛，可进入固定PCG诊断。**' if eligible else '**数值、资源与动态检查通过；尚未达到预定推广门槛，保留为诊断原型。**'),'',
        '[测量前协议](PLAN.md)；[增量源码检查](SOURCE_AUDIT.md)。生产默认、原材料/CCD/TOI/rho均未改，未运行真实100帧或上传。','',
        '## 实现与编译结果','',
        '候选继承L协作预取，由线程0把8个FP64 a读入局部work数组，保持i/j代入顺序后写回。固定循环展开让编译器标量化并复用系数。原核和L预取核同时保留，完整apply的其他投影/ABD/MAS动作不变。','',
        '| 核 | 寄存器/线程 | shared字节 | local字节 | stack字节 | 静态LDG.64位置 | 静态STG.64位置 |','|---|---:|---:|---:|---:|---:|---:|']
    for method in ['original','prefetch','cached']:
        r=resources[method];x=r['resources'];c=r['static_instruction_sites']
        lines.append(f'| {method} | {x["REG"]} | {x["SHARED"]} | {x["LOCAL"]} | {x["STACK"]} | {c.get("LDG.E.64",0)} | {c.get("STG.E.64",0)} |')
    lines += ['', '静态反汇编与实际Graph kernel属性相互核对，无local/stack spill。静态位置含分支/循环，不等于运行时访存事务/周期。cached的a写回8处，旧核仅2处静态STG位于循环中，不能据8>2推断写流量增加。展开还改变L复用和循环控制，性能归因限定为组合，不单独归因于a缓存。','',
        '## 正确性与动态检查','',
        '| 阶段 | 完整输出数 | 结果 |','|---|---:|---|']
    for stage in stages:
        a=audits[stage];tail='逐位一致，独立CPU通过' if stage.startswith('tiny') or stage.startswith('sanitizer') else '独立CPU与原动作校准通过'
        if stage.startswith('sanitizer'):tail='CUDA零错误；'+tail
        lines.append(f'| {stage} | {a["outputs"]} | {tail} |')
    negatives=sum(len(a.get('negative_tests_rejected',[])) for a in audits.values())
    lines += ['',f'独立检查总3754份输出。70组小系统三方法/Host/Graph1/16逐位相同；小核最大Decimal80参考相对差{max(a["maximum_CPU_relative"] for s,a in audits.items() if s.startswith("tiny") or s.startswith("sanitizer")):.9g}。所有完整输出CPU最大相对差{max(a["maximum_CPU_relative"] for a in audits.values()):.9g}。原门槛仍为tiny≤1e-9、后向误差≤5e-13、完整动作CPU≤1e-3且与原动作≤1e-5；零输入精确零。','',
        f'v2精确覆盖/路径唯一归属检查用于每个阶段；{negatives}个包含cached方法的重复/丢失/错几何/spill/输出别名反例全部拒绝。memcheck/racecheck/synccheck均零错误，racecheck零警告；工具只检查小核/Graph路径，不冒称整个MAS或模拟的sanitizer结果。','',
        '## 配对计时：小核（包含同样64字节RHS重置）','',
        '| 输入 | Graph内操作数 | 比较 | A中位µs | B中位µs | B/A配对中位 | 加速比A/B | 三轮B/A |','|---|---:|---|---:|---:|---:|---:|---|']
    for r in small['summary']:
        if r['comparison']=='AA':continue
        lines.append(f'| {r["name"]} | {r["capture_calls"]} | cached/{r["comparison"].removeprefix("vs_")} | {r["A_median_us"]:.3f} | {r["B_median_us"]:.3f} | {r["event_ratio_median"]:.6f} | {r["speedup"]:.4f} | '+ '/'.join(f'{r[k]:.4f}' for k in ['round1','round2','round3'])+' |')
    lines += ['', '## 配对计时：完整fused apply','',
        '| 输入 | Graph内apply数 | 比较 | A中位µs | B中位µs | B/A配对中位 | 加速比A/B | 三轮B/A |','|---|---:|---|---:|---:|---:|---:|---|']
    for r in full['summary']:
        if r['comparison']=='AA':continue
        lines.append(f'| {r["name"]} | {r["capture_calls"]} | cached/{r["comparison"].removeprefix("vs_")} | {r["A_median_us"]:.3f} | {r["B_median_us"]:.3f} | {r["event_ratio_median"]:.6f} | {r["speedup"]:.4f} | '+ '/'.join(f'{r[k]:.4f}' for k in ['round1','round2','round3'])+' |')
    lines += ['', '每格三轮、四个ABBA/BAAB四元组，每窗口2048次；比较顺序轮换。配对速度按四元组B均值/A均值的中位倒数计算，不等于两臂独立中位数之比。所有1152窗口/288四元组保留，无删尾部或重跑挑样本。tiny含copy，不称纯核指令延迟。','',
        '## A/A稳定性','',
        '| 范围 | 输入 | Graph粒度 | B/A中位 | p10/p90 | A/B相对MAD | 通过 |','|---|---|---:|---:|---|---|---|']
    for scope,audit in [('tiny original',small),('full MAS',full)]:
        for r in audit['summary']:
            if r['comparison']!='AA':continue
            lines.append(f'| {scope} | {r["name"]} | {r["capture_calls"]} | {r["event_ratio_median"]:.6f} | {r["p10"]:.4f}/{r["p90"]:.4f} | {r["A_relative_MAD"]:.2%}/{r["B_relative_MAD"]:.2%} | {r["AA_passed"]} |')
    lines += ['',f'全部tiny/full AA通过={aa}；完整两种AB比较全格中位<.98且每轮<1={full["all_AB_signal"]}；固定apply候选资格={eligible}。共享WDDM图形负载仍未隔离，GPU为RTX3070 Laptop，电源沿野兽模式；原始GPU/电源/显存/进程采样与退出检查保留。未插桩窗口的event/submit/host/reset/验证时间全部保存，不能将当前比值称为真实PCG或全帧速度。','',
        '## 身份、范围与复现','',
        f'新exe SHA256 `{identity["exe"]["sha256"]}`；实际编译器/命令、对象/link/tlog、输入、协议、源码和机械生成记录封存。父1814文件及前两代结果、生产树身份复核。完整external SDK/driver/DLL解析闭包未复制；不声称所有硬件/状态的形式正确性。编译保留原fast_math/per-thread/pedantic设置，同一次二进制包含全部对照。','',
        '```powershell',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_tiny_cache.py prepare",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/prepare_tiny_cache.py build",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/tiny_cache_experiment.py resources",
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/run_tiny_cache.py seal",
        '# 依次run_tiny_cache.py tiny、contracts、三个sanitizer_*、tiny_measure、measure；每阶段后tiny_cache_experiment.py audit 同名阶段',
        "& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/finish_tiny_cache.py",'```','',
        '所有入口拒绝覆盖旧目录。复测用新批次路径；原始输出向量与资源日志在runs，计时/数值表在tables，来源副本和生成记录均保留。','',
        '## 有限下一步','',next_step,'', '本轮只完成固定预条件作用优化；生产默认保持原状，不承诺整个StiffGIPC达到2倍。']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_new(REPORT/'DECISION.json',dict(numerical_passed=True,no_spill=True,outputs=3754,AA_passed=aa,full_AB_signal=full['all_AB_signal'],
        fixed_apply_eligible=eligible,production_modified=False,performance_certified=False,next_step=next_step))
    records=tree(RAW)+tree(BUILD)+tree(REPORT)+sources+[record(PARENT)]
    records += [record(p) for p in (ROOT/'build').glob('tiny_cache*_controller.log') if 'finish' not in p.name]
    files=sorted({r['path']:r for r in records}.values(),key=lambda r:r['path']);verify(files)
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='tiny_cache.v1',files=files,parent=record(PARENT),outputs=3754,windows=1152,negative_tests=negatives,
        no_spill=True,AA_passed=aa,fixed_apply_eligible=eligible,production_modified=False,next_step=next_step))
    verify(read(REPORT/'DELIVERY_SEAL.json')['files']);print(dict(sealed=True,files=len(files),outputs=3754,AA_passed=aa,eligible=eligible),flush=True)

if __name__=='__main__':main()
