# 重分析后的固定系统诊断结果（AutoDL，2026-09-30）

## 任务判断

执行计划为 `REPLAN_20260930.md`。当前先复核固定问题下的线性求解，随后核对状态恢复与 TOI/AL 完成性；尚未采用新的信赖域、阻尼或放松退出条件。正式质量匹配加速比仍为 N/A。

## v26 实验与结果

每帧第一个线性求解完成后，在本次调用内保持组装系统和预条件器不重新组装，保存原 x 和 b，从零初值重复同一 PCG 执行路径两次。Graph 另解 host 两次。记录解差和真残差，逐字节检查 b 未变、最终原 x 恢复。该方法不跨轨迹比较，也没有对 A 内容进行额外校验；重复预条件器是否稳定属于待查对象。

AutoDL RTX 4090、CUDA 12.8，路径 `/root/autodl-tmp/stiff_toi_cudagraph_20260929/v26`。源码摘要 `77d637094a98cdb21176a7ddee40d3efa8cabfdc9caf957cd00acf4702e3b223`，构建包 SHA-256 `ad047ffa17ac4c71bbcb6f5820e4f3a6ca4fd1e751f893076314faadca774446`，文件校验 674/674；二进制 SHA-256 `7ddcece9dfd1dcd59e830511e052f46ca32a61003ddb4b336a0f4a528c3bb0c7`。Release 编译及 6 个接触、17 个体积组件检查通过。

| v26 固定系统实验 | 抽样系统数 | 最大同执行路径相对解差 | 最大 Graph/host 相对解差 | 1e-12 重复解诊断门槛 |
|---|---:|---:|---:|---|
| 小布料，Graph，2 帧 | 2 | 6.02e-13 | 3.75e-13 | 通过 |
| 盒子布料，host + MAS，60 帧 | 11 | 2.57e-7 | — | 未过 |
| 盒子布料，Graph + MAS，60 帧 | 11 | 1.53e-7 | 1.35e-7 | 未过 |
| 盒子布料，host + 对角预条件器，60 帧 | 11 | 1.13e-17 | — | 通过 |

所有运行完成指定帧、状态有限、无重放迭代上限，b 未变且原 x 恢复；重复解迭代数均一致。Graph 对照中的两次 host 求解最大相对差 `1.19e-7`，Graph/host 差 `1.35e-7` 仍通过原 1e-6 同系统门槛。严格 1e-12 仅用于定位重放波动，不等同于物理误差门槛。MAS 普通 host 自身已有同量级波动，因此不能将此前全部轨迹分叉归给 Graph。

盒子实验共同参数：`--scene boxes1920_cloth_l --steps 60 --trace --audit-pcg --pcg-replay-from-frame 50 --platform autodl --suite 0`；`--arm base_toi`、`--arm base_toi_graph` 或 host 增加 `--preconditioner diag`。每组固定窗口为第 50–60 帧，每帧重复两次；全部关闭曲率、三角形、额外 TOI 能量采样。小布料使用 `--case 4 --steps 2 --arm base_toi_graph --pcg-replay-from-frame 1`。分析命令为 `python tools/analyze_pcg_replay.py <run>/output/stats.json --output <report>.json`。

对应本地 JSON：`v26_replay_smoke2.json`、`v26_boxes_host_replay60.json`、`v26_boxes_graph_replay60.json`、`v26_boxes_diag_replay60.json`。MAS 两组的 analyzer 退出码 1 是严格重复解门槛未过；仿真本身退出码均为 0。host + MAS 抽样系统的真相对残差中位数 `0.0149`，Graph + MAS `0.0154`；配置 rho 比阈值 `1e-4` 仍不应解释为欧氏残差 `1e-4`。

## v27 定位算子

v27 在重复 PCG 前，保持输入不变分别执行三次矩阵乘法、三次预条件器应用，以第一次为参考比较后两次输出。盒子仍为 60 帧、窗口为第 50–60 帧每帧第一个系统；各算子有 22 次重复比较。两组仿真完整完成、状态有限、重复求解迭代数相同，无迭代上限；b 未变，最终原解逐字节恢复。

| v27 实验 | 最大矩阵乘法相对输出差 | 最大预条件器相对输出差 | 最大 PCG 重复相对解差 | 严格重复解门槛 |
|---|---:|---:|---:|---|
| 小布料，Graph，2 帧 | 2.55e-14 | 0 | 6.66e-13 | 通过 |
| 盒子布料，host + MAS，60 帧 | 3.06e-16 | 1.36e-7 | 2.07e-7 | 未过 |
| 盒子布料，host + 对角预条件器，60 帧 | 6.44e-18 | 0 | 1.23e-17 | 通过 |

因此，在本轮抽样系统中，相同输入的 MAS 应用本身具有约 1e-7 的数值波动；双精度矩阵乘法差处于舍入量级。对角预条件器输出逐字节相同。MAS 路径的严格重复解门槛没有通过，不能将它写成全部正确性验收通过。小布料 Graph/host 最大相对解差为 5.75e-13。

AutoDL 路径 `/root/autodl-tmp/stiff_toi_cudagraph_20260929/v27`。源码摘要 `d3e678e674672fe170eb3bb7fc8ac022f1c0c54e19d4cbef062beb1f0084923c`，构建包 SHA-256 `c788a17b92142e29f0a2b299e639a8206461c46c6106a63be2d56452164bacb6`，文件校验 674/674；二进制 SHA-256 `8d17b34374bb996fb5e59e5cef5725063967f263104ca8cf341aed1f374fd1b6`。CUDA 12.8 Release 编译、6 个接触及 17 个体积组件检查通过。

运行命令共同部分与 v26 盒子参数相同。运行名为 `autodl_v27_boxes_mas_operator60`、`autodl_v27_boxes_diag_operator60` 和 `autodl_v27_operator_smoke2`；分析命令 `python tools/analyze_pcg_replay.py runs/autodl/<name>/output/stats.json --output reports/<name>.json`。MAS 分析器退出码为 1，其余为 0；三次仿真退出码均为 0。本地证据为 `v27_boxes_mas_operator60.json`、`v27_boxes_diag_operator60.json`、`v27_operator_smoke2.json`。完整原始 stats 和轨迹保留在上述 AutoDL 的 runs 目录。

## T2 的源码检查入口

已检查 `toi_solver.cu` 的方向差分/曲率/三角形采样及 `core/GIPC.cu::step_forward`、`abd_system_function/step_forward.cu`：当前采样结束调用 `step_forward(mesh,0,false)`。FEM 使用保存的顶点作为基点；ABD 则从 q_temp 恢复 q，再重新计算 J*q，而非直接复制保存的 ABD 顶点。因此，不能预先把该调用视为全部顶点逐字节恢复。

TOI 安全更新分别插值顶点和 q，浮点舍入可能使它们与重新映射后的 J*q 有微小差异。后续真正线搜索也从保存基点计算候选，所以仅凭源码不能断定采样会改变最终接受状态，更不能解释 v25 第 55 帧的首次分叉。T2 应先比较采样前后顶点、q、接触 slack/乘子及摩擦快照，再用同状态重复组装区分状态变化与数值累加差异；发现超出已观测波动的变化后再修复。

## 对后续任务的影响

对角预条件器分支用于定位数值波动；本轮没有证据表明它可作为 TOI 的速度修复。源码中的 MAS 多层残差使用单精度原子累加，与已确认的输出波动量级相容；累加顺序是合理解释，但本轮没有逐指令证明，也没有证明数据竞争。没有观察到重复解迭代数差异。该数值波动对非线性轨迹的放大仍需测量，不能由此直接解释数百次内层迭代、CCD 保守标记或物理偏差。

下一步按 T2 核对同状态恢复和重复组装，再按 T3 核对 TOI/AL 算法时序。本轮所有额外算子和求解时间均为诊断时间，不提供速度比，也没有调整退出容差、步长策略或默认预条件器。
