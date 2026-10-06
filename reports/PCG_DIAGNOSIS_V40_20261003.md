# v40：PCG 上限是否必然，以及 StiffGIPC 对照

## 本轮回答

PCG 达到上限不是普遍必然。当前冻结 StiffGIPC 基线的四次同场景兔子运行全部完成 100 帧，逐次 PCG 统计没有上限事件。但代码存在迭代上限，因此不能外推成“StiffGIPC 永远不会达到上限”。

本轮还定位出一个具有条件必然性的失效：若初始 rho<0，现有 `abs(rho) <= tol*rho_initial` 正常停止条件在数学上不可能成立。只要没有其他异常退出，就会到达预算。v39 默认失败系统、v40 Graph 重复以及 v40 host 失败快照均独立复现初始负 rho。其来源是 MAS 部分，并非 ABD 块或 Graph 独有问题。

阶段判断从“扩大 MAS 作用精度仍无法完成兔子”推进为“病态局部块的 float32 逆失去正定性，破坏 PCG 前提；部分失败还使原生停止条件不可满足”。仍不宣称所有失败仅由这一问题造成，也未证明完整 A 为 SPD。

## 1. 停止规则与不同的上限处理

源码：`sources/stiff_base/StiffGIPC/linear_system/solver/pcg_solver.h` 默认 `max_iter_ratio=.3`、`global_tol_rate=1e-4`；PCG 用 `rho=r^T P r`，其中 P 是应用的近似逆。兔子系统 60858 个标量自由度，预算 floor(.3*N)=18257。源码在更新 x/r 后用上一轮 rho 判停，本轮未修改这一顺序。

冻结 base 的 PCG 返回当前方向和迭代次数，IPC 外层继续 CCD/线搜索。TOI 分支在 `toi_solver.cu` 检查 iteration_limit 并拒绝未收敛 trial。因此完成进程与线性收敛是不同证据，必须读取逐次统计。本轮 base 属于项目内官方 StiffGIPC 派生的冻结诊断版，保留原生数值路径；没有以它代表未经核对的最新上游版本。

CG/PCG 的标准收敛分析以对称正定系统及正定预条件器为前提，浮点误差会影响递推关系和收敛。参考 [Netlib Templates: CG](https://www.netlib.org/linalg/old_html_templates/subsection2.6.3.1.html) 和 [Templates 全文](https://www.netlib.org/templates/templates.pdf)。这不提供在 0.3N 的人为预算内必然完成的保证。源码的 rho 比也不能直接当成欧氏真残差门槛。

## 2. 先制定计划，再执行整场景对照

运行前计划：`WORK_PLAN_V40_20261003.md`。AutoDL RTX4090/CUDA12.8，复用冻结 base 和 v39，dt=.01、Newton=.01、兔子同一输入场景，MAS（场景 preconditioner_type=1）。TOI wide 使用 suite=1、velocity_tol=.05。每组最长 240 秒，开启真残差审计，不作为性能计时。

| 配置 | 完成帧 | PCG 求解次数 | 单次最多迭代 | 上限事件 | 最大真相对残差 |
|---|---:|---:|---:|---:|---:|
| StiffGIPC base 默认 r1 | 100 | 602 | 60 | 0 | 3.101e-2 |
| StiffGIPC base 默认 r2 | 100 | 621 | 49 | 0 | 3.094e-2 |
| StiffGIPC base 默认 r3 | 100 | 612 | 49 | 0 | 7.104e-2 |
| StiffGIPC base rho=1e-14 | 100 | 582 | 185 | 0 | 4.127e-7 |
| TOI wide host 默认 | 47 | 935 | 18257 | 第 48 帧 | 6.587 |
| TOI wide Graph 默认重复 | 34 | 541 | 18257 | 第 35 帧 | 28.54 |

两个 TOI 最大残差来自全部求解统计，不等于最终失败系统残差；最终分别为 3.814e-3 / 8.635e-4。v39 同配置 Graph 默认先前完成31帧，重复完成34帧，说明失败帧号不是稳定常数。host 也失败，不能将根因归为 Graph。

base 与 TOI 接触模型和后续轨迹不同，形成的线性系统也不同。表中的 60 对 18257 不能当成相同 A/b 下的求解器性能比较。默认 base 的真残差并未达到 1e-4；严格基线只是精度对照，仍非完整物理真值。

证据：`downloads/pcg_v40_20261003/reports/PCG_V40_CONTROLS.json` 及对应每次 `requested.json`、`output/stats.json`、`trace/frames.csv`。完整逐帧二进制轨迹保留远端，本地下载统计与新失败系统。未对本轮新轨迹另作独立全路径 CCD，不声明新的安全或质量认证。

## 3. 独立定位负 rho

CPU 从快照构造 P^T B P 层级作用（此处层级延拓与上文整体近似逆 P 是不同记号），并分别计算 ABD、FEM 对 `b^T M^{-1}b` 的贡献。

- v39 默认失败：MAS 贡献约 **-0.0342085813**，ABD 约 1.29e-42；CPU 总 rho 与 GPU -0.0342085810 的绝对差 3.31e-10。
- v40 Graph 重复：CPU 初始 rho 约 -0.0476239206。
- v40 host 失败：CPU 初始 rho 约 -57.2278063。
- v39 严格失败：初始 rho 为正，约188.48；固定重放在后续迭代出现负 rho，属于另一种触发时机。

v39 默认的 float32 存储逆在块 2、3、389 有负特征值；块389在实际 RHS 上贡献 -0.1516576，超过其他块的正贡献，使总 rho 变负。该块输入最小特征值约0.01251，条件数约8.35e9；存储逆最小特征值约-8.82e-8。

同一块用 CPU 双精度重新求逆，该 RHS 二次型为 +0.0205114；仅将这个 CPU 逆舍入到 float32，二次型就变为 **-0.0155253**，最小特征值约-1.01e-8。这独立表明 float32 存储可破坏此块的正定性，不需要假设 Graph 错误。GPU 当前存储逆的乘积误差约4.87，而 CPU double 逆为3.25e-8；求逆算法误差也可能叠加。

v39 严格失败的块1002条件数约5.50e10，存储逆有约-1.87e-9的最小特征值。其初始 RHS 的二次型仍为正，说明初始 rho>0 本身不足以证明预条件器正定。

**对 v39 报告的补充解释：**此前筛选的是相对尺度低于 -1e-7 的“明显负特征值”，所以计数为0；原始记录同时已有3/1个非正块。不能把该阈值筛选结果解释成正定认证。本轮使用实际负二次型和独立精度对照确认了失效。

## 4. 同一失败 A/b 的 24 次固定重放

独立 `sources/mas_replay_v40` 复用 v38 原生内核和 v37 SpMV；新增配置参数与每250次的真/递推残差采样，模拟器无修改。两个 v39 失败系统 × 三种精度 × 两个 rho 容差 × 两次重复，共24次，均沿用0.3N预算。

该诊断继承 v38 的 **rho<=0 或 pAp<=0 时立即 breakdown** 防护，比实际 host/Graph 更早识别无效 PCG 前提。因此表中的1/8次 breakdown不是说原模拟器也在这一轮停止；原模拟器会继续，可能最终达到上限。该差异明确保留。

| 固定系统 | 原生 float | float32 逆 + wide 运算 | double 逆 + wide 运算 |
|---|---|---|---|
| v39 默认失败 A/b | 4/4 初始负 rho，第一轮防护退出 | 4/4 初始负 rho，第一轮防护退出 | 4/4 rho 判停；默认678–690次，严格5909–5914次 |
| v39 严格失败 A/b | 4/4 第5–7轮负 rho 防护退出 | 4/4 第8轮负 rho 防护退出 | 4/4 rho 判停；默认290–296次，严格4429–4431次 |

各失败重放的最小 pAp 仍为正，防护原因来自 rho。double 逆将默认系统初始 rho 恢复为约+0.15074。

CPU 独立重建 A 后，double 逆在 rho=1e-14 下：

| 系统 | 最大真相对残差 | 两次重复解相对差 |
|---|---:|---:|
| 默认失败 A/b | 4.686e-9 | 2.941e-8 |
| 严格失败 A/b | 7.687e-9 | 1.430e-7 |

通过本项目1e-6固定系统真残差/重复解门槛。24次解的 CPU/GPU真残差最大绝对差1.32e-12。默认rho=1e-4的double组真残差仍约7e-4至3e-3，不能以 rho 判停代替欧氏精度通过。

原计划只有 wide 在有效PCG前提下达到上限才追加0.6N预算；实际wide均先出现非正rho，因此没有执行增预算分支。当前证据不支持将提高上限作为首要修复。

## 5. 下一步实现计划

1. **明确数值失效原因。** 在隔离新版本统一 host/Graph 对非零残差下 rho<=0、pAp<=0 和非有限值的诊断；零 RHS 特判正常完成。保存失败原因及快照，继续拒绝无效 trial。不得仅改成 `abs(rho_initial)` 来掩盖非正定预条件器。
2. **默认关闭地集成 double 逆。** 保留 v39 wide R/Z，并扩大局部逆计算/存储，更新缓冲区生命周期和 Graph 身份。对两个失败固定系统重复实际类/CPU核对，检查局部正定性；若仍失去正定性，再评估稳定分解或局部修正，先独立诊断，不能暗改全局 A 或物理参数。
3. **重新做整轨迹验证。** 先小场景控制，再兔子 host/Graph 各100帧、必要重复。只有完成后才扩大独立接受路径 CCD、翻转/物理质量和相同精度性能比较。double固定快照通过不保证新轨迹通过；旧失败帧后移也不是修复证据。

## 6. 复现与身份

```text
# v39_20261003，冻结模拟器整场景对照
python3 tools/benchmark_pcg_v40.py
# v40_20261003，独立固定系统工具
PATH=/usr/local/cuda/bin:$PATH cmake -S sources/mas_replay_v40 -B builds/replay -DCMAKE_BUILD_TYPE=Release
PATH=/usr/local/cuda/bin:$PATH cmake --build builds/replay -j2
python3 tools/benchmark_replay_v40.py
# 本地 CPU 验证
E:\Anaconda\envs\DL\python.exe tools/analyze_replay_v40.py
E:\Anaconda\envs\DL\python.exe tools/audit_negative_blocks_v40.py
E:\Anaconda\envs\DL\python.exe tools/verify_pcg_v40.py
```

整场景精确参数见 `PCG_V40_CONTROLS.json`；固定求解命令见 `downloads/replay_v40_20261003/reports/replay_matrix.json`。本地核验559个模拟器文件、118个诊断工具及依赖文件，v38原生内核逐位未变。远端结束复核v37/v39源码和二进制，GPU 0% / 24081 MiB free。

- 固定工具二进制 SHA256：`4dd400f9bf4123ecacbbc455fa23af383e9b7d955cb3c413a72464a09cef78b4`。
- 对照结果包196639194字节，SHA256：`398491e7704457a6ac648c3b8311da8a8a95e5460038bbcac028a87fd691dfe9`。
- 固定结果包31720052字节，SHA256：`92884d37c8fa5e4174339e269037dbf9d67b6a4356b9ec12fbfcc78340a12878`。
- 两个结果包下载完整校验后保留本地、删除远端重复包，远端原始结果保留。结束磁盘可用3974766592字节（约3.70 GiB）。

分析证据：`PCG_V40_RHO_SPLIT_V39.json`、`PCG_V40_RHO_SPLIT_NEW.json`、`PCG_V40_NEGATIVE_BLOCKS.json`、`PCG_V40_REPLAY_CPU.json`、`PCG_V40_VERIFICATION.json`。
