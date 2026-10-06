# 本机状态审计与小场景对比（2026-09-30）

## 本轮范围与环境

按用户最新指示，先清理 AutoDL，然后在本机完成状态恢复/重复组装审计，最后测量小场景。未修改其他对话目录。环境为 RTX 3070 Laptop 8 GiB、驱动 581.57、CUDA 13.0、SM86、MSVC Release。AutoDL CUDA 12.8 本轮仅做空间管理，没有运行新仿真。

## AutoDL 清理

删除了 20 个已用 SHA-256 核对本机备份的上传包，共 789,892,476 字节；删除 1,582 个可再生 `.o` 编译对象，共 330,281,048 字节。历史二进制、源码、配置、报告与失败日志保留。v26/v27 构建缓存保留。

v17/v18 的 runs 共 12,185 个文件、8,403,995,560 字节，归档到 `/root/autodl-tmp/stiff_toi_cudagraph_20260929/history_runs/v17|v18`。rsync 复制后按内容校验，确认无差异，再删除原副本并以符号链接保留原 runs 路径。系统盘空闲由 404,164,608 字节增至 9,888,501,760 字节（df 显示 9.3G）；数据盘剩 9,597,542,400 字节（df 显示 9.0G）。完整路径清单与校验记录为 `autodl_cleanup_result_20260930.json`。

上述空间数字为清理完成时实际 df/磁盘统计。结束本机实验后再次通过 SSH 复核时，连接返回 `banner exchange: Connection to UNKNOWN port -1: Connection refused`，未取得新的空间快照；没有由该错误推断实例状态。历史迁移的内容校验和路径校验已在清理成功执行时完成。

## 状态审计的改动与结果

增加默认关闭的 `--state-audit-from-frame N`。记录相同物理状态重新组装、转换后的稀疏 A/b 差，以及保存解是否变化；重组装不再次求解或分发新方向。另逐字节核对顶点、q/q_temp/dq、safe/trial、接触 slack/乘子、摩擦快照、方向和参数等受保护数据在能量采样前后的恢复。

能量诊断结束后改为直接复制保存的顶点和 q。旧 `step_forward(0)` 对 ABD 会重新计算 J*q；本轮同时量化该旧恢复方式的差异。修改只涉及诊断恢复路径，没有加入阻尼、调高迭代上限或改变默认停止条件。

| 审计 | 抽样系统 | 含活动接触 | 最大 A 相对差 | 最大 b 相对差 | 受保护状态逐字节恢复 |
|---|---:|---:|---:|---:|---|
| v28 小布料 Graph，2 帧 | 4 | 0 | 0 | 6.23e-16 | 是 |
| v28 盒子布料，60 帧，第 55–60 帧外层 0/1 | 12 | 0 | 0 | 6.21e-16 | 是 |
| v29 盒子布料，增加首次有接触外层 | 18 | 6 | 0 | 1.13e-16 | 是 |

所有指定帧完成，状态有限，严格 1e-12 重组装门槛通过。活动接触为地面接触；本轮没有覆盖 PT/EE 活动集或兔子复杂混合体。抽样中旧零步长恢复的顶点差为 0，没有证据表明它在这些状态中造成污染。直接复制明确了恢复契约，但不能把本改动称为已修复此前长循环或物理偏差。

早期 v28 增量构建初始化出现缓冲区尺寸溢出，完整重编译后消失；原失败保存在 `runs/local/local_v28_state_smoke2`，完整重编译日志为 `builds/v28_local_clean_build.log`。该现象符合旧对象缓存不一致的表现，不把失败运行计入计时。v28/v29 的 6 个接触、17 个体积组件检查通过。

审计命令：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/run_local.py --arm base_toi --scene boxes1920_cloth_l --steps 60 --trace --audit-pcg --state-audit-from-frame 55 --diagnostic-toi --curvature-from-frame 55 --triangle-audit --name local_v29_boxes_contact_state60 --timeout 900
& 'E:/Anaconda/envs/DL/python.exe' tools/analyze_state_audit.py runs/local/local_v29_boxes_contact_state60/output/stats.json --output reports/v29_local_boxes_contact_state60.json
```

## 严格 PCG 对照（v28）

`cloth_sphere7_l` 为 3,084 顶点、5,000 三角形的小场景，dt=0.01，100 帧对应 1 秒物理时间。四组共同使用 Newton 阈值 0.01、PCG rho 比阈值 1e-16，suite=0，只隔离 Graph；计时关闭审计、重放和能量探针。PCG rho 比阈值不等于欧氏真残差。

base 与 base+Graph 各三次完成 100 帧，求解窗口中位时间分别为 35.7773 秒、14.3769 秒，配对耗时比中位数 2.4885。Graph 首次捕获和重建成本计入，初始化/状态输出排除。第一对完整轨迹的归一化最大质量 RMS 为 2.0635e-4，高于预注册 1e-6 等价门槛；最大顶点差为 0.09858 米。因此这是原始耗时比，不能列作质量匹配收益。

TOI host 完成 51 帧后在第 52 帧达到 PCG 迭代上限；TOI+Graph 完成 44 帧后在第 45 帧达到 PCG 上限。失败臂没有继续重复，完整 100 帧耗时比为 N/A，不能把不同完成帧数的时间相除。报告与配置为 `v28_local_small_strict.json`、`v28_strict_base_graph_trajectory.json`、`configs/local_small_v28.json`。

## 默认配置小场景（v29）

配置为 `configs/local_small_v29_default.json`：共同 Newton 阈值 0.01、PCG rho 比阈值 1e-4，四组各 100 帧、三次轮换顺序；每帧输出轨迹用于独立比较。所有诊断选项关闭，默认预条件器与材料保持一致。12 次正式窗口均完整完成、状态有限，无 PCG/外层迭代上限。

| 方法 | 100 帧时间中位数（秒） | 三次时间范围（秒） | base 时间中位数 / 本方法时间中位数 | 配对耗时比中位数 |
|---|---:|---:|---:|---:|
| base | 13.1775 | 12.9308–13.4208 | 1.000 | 1.000 |
| base+Graph | 9.7703 | 9.2200–10.2102 | 1.349 | 1.349 |
| base+TOI | 47.0306 | 45.2633–48.4073 | 0.280 | 0.277 |
| base+TOI+Graph | 37.5479 | 33.5752–37.9554 | 0.351 | 0.351 |

Graph 在该配置中降低求解窗口原始耗时。TOI 中位时间仍为 base 的约 3.57 倍，TOI+Graph 仍为约 2.85 倍；按时间中位数，Graph 将 TOI 时间改善约 1.25 倍。上述比值均不是质量匹配加速结论。

以下按共同 base 帧掩码划分，列出分段时间中位数与配对比中位数。79 帧存在原生距离检查后的邻近候选，21 帧没有邻近候选；基于共同 `distance <= 1e-4 * 初始场景对角线` 判据的几何接触帧为 0，因此真正接触段比值是 N/A。邻近候选段不能改名为几何接触段。

| 方法 | 无邻近候选 21 帧：base/本方法时间（秒） | 原始耗时比 | 邻近候选 79 帧：base/本方法时间（秒） | 原始耗时比 |
|---|---:|---:|---:|---:|
| Graph | 0.6854 / 0.6116 | 1.117 | 12.4921 / 9.1586 | 1.364 |
| TOI | 0.6854 / 0.6474 | 1.012 | 12.4921 / 46.3867 | 0.266 |
| TOI+Graph | 0.6854 / 0.5940 | 1.152 | 12.4921 / 36.9530 | 0.338 |

### 工作量与质量

第一重复的实际统计如下，Graph 有真实捕获与命中记录，没有把请求开关当成实际执行证据。

| 方法 | 方向求解次数 | PCG 总迭代数 | TOI 外层数 | Graph 捕获数 / 命中数 |
|---|---:|---:|---:|---:|
| base | 558 | 23,911 | — | — |
| Graph | 580 | 26,635 | — | 447 / 133 |
| TOI | 1,888 | 83,931 | 1,226 | — |
| TOI+Graph | 1,953 | 74,839 | 1,210 | 884 / 1,069 |

TOI host 的方向工作量约为 base 的 3.38 倍，PCG 总迭代约为 3.51 倍，与约 3.6 倍的第一重复耗时相近。按整步时间/方向数粗略均摊，base/TOI 约为 24.1/25.6 ms；这不是单独 PCG 的事件计时，但显示本场景的主要差异是方向数量增加，尚没有证据说明靠一次线性算子的优化就能消除该差距。

所有重复的完整 101 个状态均作轨迹对照：

| 对照 | 归一化最大质量 RMS（三次） | 首次超过 1e-6 门槛的帧 | 轨迹门槛 |
|---|---|---|---|
| base / Graph | 1.987e-4、2.650e-4、2.614e-4 | 25、33、33 | 全部未过 |
| TOI / TOI+Graph | 1.875e-3、1.657e-3、2.284e-3 | 22、22、23 | 全部未过 |

补充同二进制 IPC host 控制（仅用于质量，时间标为诊断）显示，fused host 两次重复的归一化最大 RMS 为 1.583e-4；原始 base 两次重复为 1.996e-4；相同 fused 二进制 host/Graph 为 2.745e-4。首次坐标数值差均始于第 1 帧，超过门槛分别在第 30、31、25 帧。host 本身已有相近量级波动，不能把全部 Graph 对照轨迹差单独归因于 Graph。也不能据此放宽已设门槛。此处仍没有把首次差异定位到具体 GPU 内核；重复组装和保护状态检查只排除了本轮覆盖状态中的一类解释。

第一重复的同时间步末态物理对照：TOI/TOI+Graph 对 base 的自由顶点位置 RMS/场景尺度为 0.1948%/0.2175%，后差分速度相对误差为 243.4%/293.4%；最大布料边伸长为 1.0641/1.1331，base 为 1.0128。末态均有限、固定顶点无位移、无非正四面体 Jacobian，但这不足以证明物理质量。该 base 是同时间步对照，尚不是严格共同物理参考。

本轮没有对最新完整接受子步重新做独立连续 CCD，也未做新的 dt/2、dt/4 共同物理参考。位置/速度与同后端轨迹门槛尚未满足，因此正式质量匹配加速比保留 N/A，没有继续扩大场景矩阵。v17 的旧 CCD 结果不能替代本版本验收。

完整默认计时为 `v29_local_small_default.json/.csv`，首重复工作量为 `v29_small_work_counts.json`，轨迹为 `v29_default_base_graph_r01..03.json`、`v29_default_toi_graph_r01..03.json`，控制为 `v29_base_repeat.json`、`v29_fusedhost_repeat.json`、`v29_fusedhost_vs_graph.json`，物理量为 `v29_default_toi_vs_base_physics.json` 与 `v29_default_toi_graph_vs_base_physics.json`。

执行命令：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/run_matrix.py --config configs/local_small_v29_default.json
& 'E:/Anaconda/envs/DL/python.exe' tools/report_matrix.py runs/local/local_small_v29_default/matrix_status.json --output-prefix reports/v29_local_small_default
& 'E:/Anaconda/envs/DL/python.exe' tools/compare_runs_stdlib.py runs/local/local_small_v29_default_cloth_sphere7_l_base_measure_r01 runs/local/local_small_v29_default_cloth_sphere7_l_base_graph_measure_r01 --output reports/v29_default_base_graph_r01.json
```

## 下一阶段

本轮已完成受控状态恢复、接触窗口重复组装以及用户要求的小场景计时。PT/EE、复杂混合体和首次差异的 GPU 内核归因仍待完成。下一阶段应冻结长循环中的接触状态，核对 TOI/AL 的 safe/trial、slack、乘子、摩擦和退出时序；同时检查严格 PCG 失败系统的真残差与条件尺度。先解决方向/外层工作量及速度偏差，再考虑更大性能矩阵或新的阻尼策略。

## 可复现性

v28 源码摘要 `19e9de917b53801ed5afe2d3d49a1a1c912143e1de690772219ffa004846f072`，源码包 SHA-256 `0a5ade6644fcaa08dd639e8a5df768c803cf84a6dc7e0b62041954ff64892fac`，fused 二进制 SHA-256 `6d5561477e570f7dc987c5134936b125ec638ee7a71723bfa36186ed40e36baf`。

v29 源码摘要 `b72075846c78895020d7f724718ebe7bffc6d5514b6eceb14853400001e1ed62`，fused 二进制 SHA-256 `5b13fc3bffde9208d148bb360a4137ab295c140f167d3dbfce2794277ec1e907`；两轮 base 二进制 SHA-256 `1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205`。最新源码包 `bundles/stiff_toi_cudagraph_source.tar.gz` 含 678 个文件，SHA-256 `d2e45dbf5c1c490f5f81d3eb5f6fa6c9e2067a4432d801070303ce1dc76a4ea6`；已准备供后续传输，本轮未上传新仿真版本至 AutoDL。

各次 requested.json 保存实际源码、二进制和运行器哈希。各运行结果、stats、逐帧时长、轨迹与失败日志均保留；旧本机 fused 二进制保存到 builds/frozen。收尾核验 553 个源码/资源文件哈希与当前 manifest 一致，4 个新改 Python 工具可解析，12 个默认测量完整，6 个轨迹失败报告均保留。

实际构建命令为 `cmake --build builds/local-fused --config Release --target gipc --clean-first --parallel 2`（v28 完整重编译）和 `cmake --build builds/local-fused --config Release --target gipc --parallel 2`（v29 增量）；可执行文件使用 `D:/computer/cmake/bin/cmake.exe`。组件检查使用 `GIPC_VALIDATE_COMPONENTS` 指向 `reports/v29_local_components.json` 后运行 `builds/local-fused/Release/gipc.exe`，退出码为 0。
