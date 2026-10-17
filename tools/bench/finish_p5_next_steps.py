"""Seal roadmap evidence and the standalone CPU scheduling contracts."""
import ast
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,text
from linux_runner import require,write_new
import analyze_p5_next_steps as offline
OUT=offline.OUT
BUILD=ROOT/'build/weighted_check_contract_20261017'
PROJECT=ROOT/'tools/bench/weighted_check'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    require(not (OUT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    verify(read(offline.PARENT)['files']);seal=read(OUT/'SOURCE_SEAL.json');verify(seal['files']+[seal['source']])
    require(offline.contracts()==dict(passed=True,meaningful_cases=7),'Offline contracts regressed')
    verify(read(ROOT/'runs/innovation_p5_mas_cached_20261017/production_before.json'))
    testlog=text(BUILD/'ctest.log');last=text(BUILD/'Testing/Temporary/LastTest.log')
    require('100% tests passed, 0 tests failed out of 1' in testlog and 'weighted_check_schedule: 11 contract groups passed' in last,'Unexpected CPU test result')
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};xml=ET.parse(BUILD/'weighted_check_schedule_test.vcxproj')
    units=[Path(n.attrib['Include']).resolve() for n in xml.findall('.//m:ClCompile',ns) if n.attrib.get('Include')]
    require(units==[PROJECT/'weighted_check_schedule_test.cpp'],'Unexpected CPU compile unit')
    objects=list((BUILD/'weighted_check_schedule_test.dir/Release').glob('*.obj'))
    tracks=[p for p in BUILD.glob('weighted_check_schedule_test.dir/Release/**/*.tlog') if p.is_file()]
    links=[p for p in tracks if p.name.lower().startswith('link.read')];require(len(objects)==1 and links,'Missing CPU object/link record')
    linked='\n'.join(text(p).replace('\\','/').casefold() for p in links)
    require(all(str(p.resolve()).replace('\\','/').casefold() in linked for p in objects),'Object not linked')
    compilers=[]
    for path in (BUILD/'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'):
        found=re.search(r'set\(CMAKE_CXX_COMPILER\s+"([^"]+)"\)',text(path))
        if found:compilers.append(Path(found.group(1)))
    require(len(compilers)==1,'Missing compiler identity')
    sources=tree(PROJECT)+[record(ROOT/'tools/bench/analyze_p5_next_steps.py'),record(Path(__file__)),record(ROOT/'build/fused_apply_snapshot_20261016/StiffGIPC/linear_system/solver/native_balanced_math.h')]
    require(max(Path(r['path']).stat().st_mtime_ns for r in sources if Path(r['path']).suffix in ['.h','.cpp'])<=objects[0].stat().st_mtime_ns,'Source newer than CPU object')
    evidence=[BUILD/n for n in ['configure.log','build.log','ctest.log','Testing/Temporary/LastTest.log','CMakeCache.txt','weighted_check_schedule_test.vcxproj']]+objects+tracks
    for p in [ROOT/'tools/bench/analyze_p5_next_steps.py',Path(__file__)]:ast.parse(p.read_text(encoding='utf-8'))
    exe=record(BUILD/'Release/weighted_check_schedule_test.exe')
    write_new(OUT/'CPU_CONTRACT.json',dict(passed=True,ctest_tests=1,contract_groups=11,offline_contract_cases=7,exe=exe,
        compiler=record(compilers[0]),sources=sources,evidence=[record(p) for p in evidence],GPU_runs=0,production_modified=False))
    copies=OUT/'implementation_sources';copies.mkdir()
    for p in [PROJECT/'weighted_check_schedule.h',PROJECT/'weighted_check_schedule_test.cpp',PROJECT/'CMakeLists.txt',Path(__file__)]:
        (copies/p.name).write_bytes(p.read_bytes())
    analysis=read(OUT/'ANALYSIS.json');require(analysis['samples']==1152 and analysis['GPU_runs']==0,'Wrong offline workload')
    decision=dict(selected_next_candidate='conditional weighted-residual cooldown, period=2',offline_rows=1152,
        offline_contract_cases=7,CPU_contract_groups=11,CPU_prototype_completed=True,GPU_integration_completed=False,
        production_modified=False,performance_certified=False,next_step='Private GPU fixed-A/b replay of both MAS and cached balanced under the same new policy; keep original acceptance thresholds.')
    write_new(OUT/'DECISION.json',decision)
    report='''# 后续路线与本轮推进结果

详细分析见[ROADMAP.md](ROADMAP.md)。后续顺序：

1. 条件2步检查冷却：先同质量四臂固定系统验证，不直接改生产退出。
2. 细分Mixed的balanced apply，按实测占比优化一处投影/归约。
3. 计入历史采集、源空间构造、目标更新、Graph重建和失败回退。
4. 默认关闭地接入真实推进，先5–10帧和持出状态验证。
5. 最后五场景100帧×3轮，重新做数据表、视觉与PDF。

本轮已推进：1152条冻结轨迹筛选、7项离线边界合同，以及独立C++状态机的11组合同比较（CTest 1/1）均通过。原测量数据、生产代码和停止阈值未改；没有启动GPU。

选择“L2已通过、MAS加权未通过后才进入2步冷却”，保留正常阶段的原筛选。其Bunny的96条臂轨迹保持原停止点/检查数；全体192条中168条能观察到候选接受，另24条必须补实际求解，不伪造终点后的残差。不能保证最多增加1次迭代，因为真残差不保证单调。

Mixed仍需要单次apply优化：其他工作固定时，即使把所有质量检查及递推筛选全部删掉，乐观记账也仅约1.244/1.201倍（Host/Graph），不是实测速度或严格硬件上界。

下一项是私有GPU固定回放：MAS/缓存balanced都测新旧两套调度，保持最终原A/原MAS双显式验收，记录额外迭代、检查次数及包括全部检查的总PCG时间。当前CPU原型没有接管GPU或真实帧。

验证命令：

`D:/computer/cmake/bin/ctest.exe --test-dir build/weighted_check_contract_20261017 -C Release --output-on-failure -j 1`

完整构建身份、原始日志、编译器和二进制SHA256见CPU_CONTRACT.json；候选逐条推演见schedule_samples.csv。所有模型列仅表示已观察子集的冻结轨迹均摊估算，不能当候选实际性能。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    files=tree(OUT)+sources+[exe,record(compilers[0])]+[record(p) for p in evidence]
    verify(files);write_new(OUT/'DELIVERY_SEAL.json',dict(files=files,decision=decision));verify(read(OUT/'DELIVERY_SEAL.json')['files'])
    print(dict(delivery_rehashed=True,files=len(files),CPU_contract_groups=11,offline_rows=1152,GPU_runs=0),flush=True)
if __name__=='__main__':main()
