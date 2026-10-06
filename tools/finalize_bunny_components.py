"""Consolidate measured component gains and rejected numerical-preserving experiments."""
import csv,json,statistics
from run_bunny_components import ROOT,read,sha
a=read(ROOT/'reports/BUNNY_COMPONENTS_ANALYSIS.json');s=a['summary'];host=s['host']
target=ROOT/'reports/BUNNY_COMPONENTS_20261004.md';assert not target.exists()
v=read(ROOT/'reports/BUNNY_COMPONENTS_VERIFICATION.json')
assert len(v['runs'])==2 and all(r['ccd']['passed'] for r in v['runs'])
assert all(r['result']['status']=='completed' for r in read(ROOT/'reports/BUNNY_COMPONENTS_MATRIX.json')['runs'])
for version in [54,55,56]:
    m=read(ROOT/f'manifests/perf_v{version}_local.json')
    for f in m['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
    for path,f in m['binaries'].items():assert sha(ROOT/path)==f['sha256'],path
names={'base':'StiffGIPC 基线','host':'当前程序 IPC host','graph':'IPC + CUDA Graph','refit':'IPC host + BVH refit','batch':'IPC host + 批量能量',
       'reuse':'IPC host + 能量复用','combined':'IPC + Graph + 三项组件','cholesky':'IPC host + MAS Cholesky'}
lines=['# 混合兔子相对 StiffGIPC 的组件收益（2026-10-04）','',
'结论：原生 IPC 加四项执行优化的实测耗时比为 **1.101×**，耗时下降约 **9.2%**。完整 TOI 组合仍不合格。本轮没有修好 TOI 的收敛/质量问题，也没有把停用 TOI 当作 TOI 加速。','',
'## 协议与范围','',
'用户确认 `bunny_cloth_bunny_l`。RTX 3070 Laptop 8 GiB，本机 Release；Stiff 基线来自保留的官方 commit `bb2849a7b292099581907937860d96ecfdf42588` 仪表化构建，274 个源依赖；当前 v54 为294个依赖。所有配置初始顶点、拓扑、质量、边界和场景元数据逐字节相同。dt=.01、100帧=1秒物理时间、Newton=.01、PCG rho 比1e-4、MAS、CCD容量10000000。','',
'八组各三轮交错串行，共24次完整计时；另三组3帧smoke、两组100帧独立审计。计时只取原生 solver_ms 汇总，包含每次新进程首帧成本，不把启动/轨迹输出当求解收益；正式计时无额外PCG/Graph/能量/CCD重复审计。背景桌面GPU负载约40%，保留逐秒温度、频率、功率与占用。结果是此共享负载下的实测，不能认证几个百分点的普适提升。','',
'## 整场结果','',
'| 配置 | 100帧中位秒 | 三轮最小–最大秒 | 相对Stiff | 相对同程序host | PCG迭代中位数 |',
'|---|---:|---:|---:|---:|---:|']
for k,t in s.items():lines.append(f"| {names[k]} | {t['median_seconds']:.3f} | {t['min_seconds']:.3f}–{t['max_seconds']:.3f} | {t['speed_vs_base']:.3f}× | {t['speed_vs_host']:.3f}× | {t['median_pcg_iterations']:.0f} |")
lines+=['','单组件均相对同一 v54 二进制 IPC host，其余组件关闭；组合启用 Graph、refit、批量能量、能量复用。没有将单组件收益相乘。基线与host的0.6%差异不足以声称移植本身加速。','',
'## 组件阶段归因','',
'| 单项 | 对应阶段 host→单项秒 | 阶段耗时比 | 整场相对host |',
'|---|---:|---:|---:|']
for k,phase in [('graph','pcg'),('refit','ccd'),('batch','line_search'),('reuse','line_search')]:
    h=host['phase_medians_seconds'][phase];z=s[k]['phase_medians_seconds'][phase]
    lines.append(f"| {names[k]} | {h:.3f}→{z:.3f} | {h/z:.3f}× | {s[k]['speed_vs_host']:.3f}× |")
lines+=['','Graph是主要收益：PCG阶段约节省1.88秒；三项其余优化在组合里额外节省约0.80秒。refit阶段本身较快，但整场没有可确认收益；批量能量单项整场接近噪声。阶段统计随轨迹、调用数量而变，不是冻结相同工作量的kernel微基准。能量阶段包含线搜索；PCG阶段包含图准备。','',
'混合场景是 MAS，`fused_diag_update` 在所有Graph记录中均为false，块对角融合kernel的收益为“不适用”，不能算进加速。MAS Cholesky不是同等代价的开关：约1.96万次PCG迭代保持接近，但PCG阶段由11.12秒涨到45.09秒；它修复旧逆矩阵数值问题，同时产生显著成本，不能在TOI中为了速度直接关闭。','',
'## 质量与碰撞','',
'| 配置 | 全程最大布料边长比（三轮范围） | FEM最小J（三轮范围） |',
'|---|---:|---:|']
for k in ['base','combined','cholesky']:
    c=s[k]['cloth_max_stretch_range'];j=s[k]['fem_min_J_range'];lines.append(f"| {names[k]} | {c[0]:.6f}–{c[1]:.6f} | {j[0]:.6f}–{j[1]:.6f} |")
lines+=['','24组均100帧完成、有限值、无PCG触顶或breakdown。组合对基线r1的最大质量加权轨迹RMS/各物体尺度：布料0.622–0.710%、FEM0.237–0.452%、ABD0.061–0.160%；基线自身r1对另外两轮为布料0.560–0.612%、FEM0.408–0.505%、ABD0.093–0.176%。这接近既有浮点重复差异量级，但不是逐位相同，也不通过旧1e-6轨迹等价阈值，不能宣称严格同质量认证。','',
'独立审计：base705段、组合703段接受路径（含跨帧桥接）全部通过 Tight-Inclusion CCD，零标记；组合100帧refit/rebuild候选集合相同，100帧批量能量误差≤1e-10，能量缓存重算审计没有触发不一致。两者都有FEM翻转；使用既有Stable NH1容许翻转政策，不能用CCD通过替代正体积或物理真值。','',
'同相机/边界/时刻的真实网格已视检：初态与0.4秒相近，1秒布料褶皱存在细节差异，无空白/错场景；碰撞结论来自独立几何检查。图为 `BUNNY_COMPONENTS_MESH.png`。','',
'## TOI与本轮修改的判定','',
'v54完整TOI配置启用choose-start、restart guard、Cholesky和四项执行组件，固定120秒预算；**43/100帧后超时**，已完成帧求解116.681秒。没有延长预算，也不拿43帧直接除以基线100帧报告完整加速比。旧v53完整记录约442–499秒对Stiff约30秒属于历史诊断，不能当作本轮v54完整结果。','',
'本轮新增独立组件启动入口，保留IPC执行优化候选；另外实际实现并编译两个Cholesky kernel候选：v55寄存器/warp广播，v56加连续读取共享内存缓存。当前BANKSIZE=16，三角系统48维；每行除法与减法顺序不变。六份历史冻结系统（包括旧逆矩阵出错系统）×10个向量均与旧action逐位一致；Graph三次重放及缓冲区扩容一致性全部通过。','',
'| 固定系统 | v55 旧/新kernel耗时比 | v56 旧/新kernel耗时比 |','|---|---:|---:|']
fixture_summary={}
for version in [55,56]:
    cases=read(ROOT/f'reports/V{version}_FIXTURES.json')['cases'];assert len(cases)==6 and all(c['result']['passed'] for c in cases)
    fixture_summary[version]={c['case']:c['result']['legacy_action_us']/c['result']['warp_action_us'] for c in cases}
for case in fixture_summary[55]:lines.append(f"| {case} | {fixture_summary[55][case]:.3f}× | {fixture_summary[56][case]:.3f}× |")
lines+=['','两种候选都没有可靠收益，v56在所有fixture更慢；**均不采用**。因此取消依赖它们的100帧矩阵与TOI长测，不让已知更差候选继续消耗实验时间。微基准每种100次kernel调用，用CUDA事件计时；顺序固定、共享GPU，仅足以拒绝当前候选，不用于高精度性能认证。v55/v56只完成fixture层验证，没有宣称完整场景通过。旧v54、base、所有原始结果保留，数值默认开关未更改。','',
'构建记录：v55初次SDK目录访问受沙箱限制，随后编译断言纠正了按旧函数名称误判的96维假设；修正后成功。v56首次增量构建复用了同名对象，未用于任何实验；检查后用CUDA CompileOut指定独立对象，`V56_build_release_r3.log`确认重新编译才冻结测试。编译warning见日志，源/二进制身份末尾复核通过。','',
'## 下一步','',
'原任务若必须保留TOI，重点应转到两个可量化成本：稳定MAS action约4倍成本，以及TOI需要约十倍PCG总迭代。不要继续把Graph或退出次数当主要修复方向。下一项优先评估通过SPD/对称性验证的快速局部预条件作用（保留旧Cholesky作为参考），先冻结矩阵成本与残差通过，再恢复有限混合体质量对照；全程保持材料和收敛要求。','',
'## 复现与文件','',
'```text',
'E:/Anaconda/envs/DL/python.exe tools/benchmark_bunny_components.py --phase smoke',
'E:/Anaconda/envs/DL/python.exe tools/benchmark_bunny_components.py --phase matrix',
'E:/Anaconda/envs/DL/python.exe tools/benchmark_bunny_components.py --phase quality',
'E:/Anaconda/envs/DL/python.exe tools/benchmark_bunny_components.py --phase toi',
'E:/Anaconda/envs/DL/python.exe tools/analyze_bunny_components.py',
'E:/Anaconda/envs/DL/python.exe tools/verify_bunny_components.py',
'E:/Anaconda/envs/DL/python.exe tools/fixtures_local_v55.py',
'E:/Anaconda/envs/DL/python.exe tools/fixtures_local_v56.py',
'E:/Anaconda/envs/DL/python.exe tools/render_bunny_components.py',
'```','',
'上述脚本拒绝覆盖旧输出，复现需新名称；确切逐组命令、环境、源码/程序/runner SHA、GPU样本保存在矩阵JSON及每次requested/result.json。单独复测组合可用 `tools/run_bunny_components.py --preset combined --name <新名称> --steps 100 --timeout 120`。','',
'主数据：`BUNNY_COMPONENTS_MATRIX.json`、`BUNNY_COMPONENTS_ANALYSIS.json`、`BUNNY_COMPONENTS_VERIFICATION.json`、`BUNNY_COMPONENTS_TOI.json`、`V55_FIXTURES.json`、`V56_FIXTURES.json`。本轮未跑全仓库CTest，未将未采用的CUDA候选推广至其他场景。']
target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
final={'complete_matrix_runs':24,'complete_quality_runs':2,'smoke_runs':3,'toi_completed_frames':43,'toi_status':'timeout',
       'ccd_paths':sum(r['ccd']['paths_checked'] for r in v['runs']),'ccd_all_passed':True,'fixture_cases':12,'vectors_compared':120,
       'v55_adopted':False,'v56_adopted':False,'baseline_median_seconds':s['base']['median_seconds'],
       'combined_median_seconds':s['combined']['median_seconds'],'measured_speedup':s['combined']['speed_vs_base'],
       'physical_quality_certified':False,'performance_certified':False,'render_inspected':True,
       'report':str(target),'report_sha256':sha(target),'source_and_binary_identities_reverified':True}
out=ROOT/'reports/BUNNY_COMPONENTS_FINAL.json';assert not out.exists();out.write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))
