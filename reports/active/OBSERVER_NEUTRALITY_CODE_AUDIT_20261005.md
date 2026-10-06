# 固定兔子长窗观测中性：应用代码审查（2026-10-05）

本报告仅审查代码和既有报告，不启动 GPU、不修改构建输入、历史结果或质量范围。

**结论：速度导出没有直接改写求解状态的代码路径；每帧导出前已经完成设备同步。新增速度导出可能改变相邻帧的提交间隔与硬件执行条件，但当前证据尚不能把长窗分岔归因于它。应先用同一观测原版程序，仅切换速度导出，排除不同二进制带来的混杂。**

## 1. 差分与身份

直接对比两份完整 `gl_main.cu`，观测原版相对冻结原版只有一个差分块：在 `dump_frame` 中增加 9 行、受 `GIPC_TRACE_VELOCITY` 控制的速度 D2H 与二进制文件写入。初始化、帧循环、求解参数、位置导出和退出路径没有其他源码差异。

| 文件 | 本次读取的 SHA-256 |
|---|---|
| `sources/stiff_base/StiffGIPC/app/gl_main.cu` | `1273570708c222cdc188e9d95ced7f07d36efafabcd9610f2790e2117a4c4896` |
| `sources/stiff_base_observed/StiffGIPC/app/gl_main.cu` | `3e508c988f61c59d32d7a2375ed135ae6b2242eb00fedec3b07b428f41424cb2` |
| `sources/stiff_active/StiffGIPC/app/gl_main.cu` | `09bb8eead7e26201c80c6f859431033a733b06665b71b90fd0fcf7876fbf1e4f` |

这里的源码差分不证明两个已构建程序逐指令等价。两个独立编译的应用对象、主机代码布局和链接输入仍可能不同；其因果试验必须优先使用同一个程序。

## 2. 可直接确认的边界

| 检查项 | 具体源码位置 | 结论 |
|---|---|---|
| 原版的每帧同步与计时 | `sources/stiff_base/StiffGIPC/app/gl_main.cu:1851`，`IPC_Solver` 后 `cudaDeviceSynchronize`，随后 `stop` | 所有帧导出在本帧主求解计时之后；导出不是为了修补缺失的帧间完成屏障 |
| 观测原版的新增速度操作 | `sources/stiff_base_observed/StiffGIPC/app/gl_main.cu:1847`–1855 | 方向明确为 `cudaMemcpyDeviceToHost`，源为 `d_tetMesh.velocities`，目的为 lambda 内新建的 host vector；不存在 H2D 回写或求解器调用 |
| host 状态隔离 | 原版 `gl_main.cu:1840`，观测原版同一位置 | 局部 `state` 与 `tetMesh.vertexes` / `tetMesh.velocities` 不同；把局部 vector 从位置覆盖为速度不会改变下一帧输入 |
| active 的同类导出 | `sources/stiff_active/StiffGIPC/app/gl_main.cu:1862`–1884 | 同样只读设备速度、写局部 vector / 文件；计时 `stop` 位于1892，导出调用位于1906 |
| 速度读取时机 | 原版 `core/GIPC.cu:11352`–11356 | 求解器先 `updateVelocities`，再 `computeXTilta`，随后设备同步；应用读取的是完成更新的点速度数组 |
| 点速度来源 | 原版 `core/GIPC.cu:7854`–7875、11138–11149 | 自由点速度为端点差除 dt，固定点置零，并更新旧位置；这些写入属于原求解器而非导出 |
| ABD 原生速度 | `sources/stiff_base/StiffGIPC/abd_system/abd_system_function/update_velocity.cu:7`–46 | ABD 另有 affine `q_v` 更新；点速度导出不是原生12维 `q_v` 导出，不能把它称作全部 ABD 内部速度状态证据 |
| 导出频率 | 原版 `gl_main.cu:1838`–1839，active `gl_main.cu:1865`–1866 | 位置/速度共用 stride；初始帧和最后3个端点保留。切换 stride 会同时改变位置和速度频率，不能隔离新增速度观测 |
| 其他帧后诊断 | 原版 `gl_main.cu:1855`–1864，active `gl_main.cu:1893`–1902 | physics 能量评估和 stats 写盘在 solver stop 后；其调用与 scratch 操作应在观测中性试验中固定或关闭 |

应用批处理路径不是 `display()` 回调；后者在原版 `gl_main.cu:1559` 把设备位置写回 `tetMesh.vertexes`，但批处理帧循环没有调用该回调或 GLUT 主循环。不能把交互渲染路径的 host 写入当作本轮批处理观测的机制。

## 3. 哪些机制有证据，哪些尚待验证

**已验证：没有新增直接状态改写。**新增代码没有设备写入、CUDA kernel、计数器改变、求解选项改变或 host mesh 回写。文件写失败只会抛异常，使本次运行失败；不会默默改变求解分支。

**已验证：不缺帧间同步。**原版 `IPC_Solver` 结尾已经同步，应用每帧又显式 `cudaDeviceSynchronize`。在这些源代码路径中，新增速度 D2H 不是第一个等待本帧速度更新完成的操作。此审查没有发现“原版未完成上一帧，新增导出修好了竞争”的依据。

**支持但未证明：主机导出可改变后续执行条件。**速度 D2H、host 文件写入和文件流析构在下一次 `IPC_Solver` 启动前发生，会增加帧间间隔。它们可能改变共享桌面 GPU 负载、频率和执行调度；虽然不在本帧 solver timer 内，也不能保证下一帧的耗时或浮点执行顺序完全中性。此处是从控制流程得出的可能机制，不是本轮已测因果结果。

**支持但未证明：原子累加使微小差异可能进入求解状态。**冻结原版 `core/GIPC.cu:3235`–3246、3349等多处使用浮点 `atomicAdd` 累加接触梯度。碰撞对生成也使用计数器原子，例如1419、1498。浮点加法结合顺序改变可以产生微差；停止条件和接触分支又可能放大差异。代码存在这条路径，并不证明新增导出必然改变其顺序。尚未通过同状态重复 kernel 捕捉到直接因果链。

**待验证：长窗观测中性。**最近结果报告中3帧 PCG/方向/退出相同、位置微差约1e-11m，是短窗证据；固定兔子100帧原版重复及组合材料越界是长窗问题。二者不能互相替代，不能据此扩大旧质量界限或宣布导出是根因。

## 4. 唯一建议的小型因果试验

对固定兔子，用**同一个 `base_observed` 可执行文件**、相同输入哈希、`dt=.01`、legacy 默认停止、100帧，仅切换 `trace_velocity=false/true`。位置导出保持 stride=1、physics/substeps/残差观测关闭、其余配置与资源预算完全一致。

- 3组交错配对，共6次，次序交替 OFF→ON / ON→OFF / OFF→ON；GPU串行，单次120秒上限，不临时延长。
- 每次独立新运行目录；比较初始状态、每帧方向数/PCG/退出/alpha/beta/接触量、整段最大与p99拉伸，以及首个端点分岔帧。
- 比较 OFF 组内部重复差异、ON 组内部重复差异和跨组差异。两组自然范围高度重叠且跨组差异没有稳定方向时，只能说明没有检出速度导出的额外系统性作用，不能宣称逐位中性。
- 如果同一程序的OFF/ON出现可重复的工作量或材料变化，下一步再追踪首次分岔系统；不得由这6次直接认证候选质量或重设历史范围。

试验将速度导出与不同二进制、不同位置导出频率隔离。它不同时测试 sparse trace、无trace、不同程序或更严格停止条件，以免一个小试验承担多个原因。

## 5. 本次验证范围

执行：完整应用源码统一差分、相关 batch/velocity/solver 路径逐行阅读、应用源码SHA-256核对、原子累加位置检查。未构建、未启动GPU、未改代码或实验阈值。

既有实验引用：[IPC执行结果](IPC_EXECUTION_RESULTS_20261005.md)。所有长窗数值仍来自该报告；本审查不重新改判它们。
