# P2-A 本机实现、逐项审查与成本筛选

## 结论

完成两个默认关闭的工程原型：延迟 BVH host AABB 刷新，以及保持原 kernel/归约/host 加法顺序的 ABD 合并读回。完整 CUDA 构建、24 项 CTest、14 项 Python 契约/几何测试通过。79 次成功场景运行、384 帧；54 次同二进制检查点恢复均核验 40 个缓冲及初始位置/实际速度；77 组 ABD 两项能量比较位级一致，140 次普通 BVH refit/rebuild 验证通过。

**本轮没有证明可用的整段加速，也没有达到扩大 P2-B 的门槛。** 组合的同状态线搜索倍率为 1.035–1.040×，从零短段总 solver 倍率为 Boxes 1.003×、Mixed 0.991×、Bunny 1.002×。保留实验开关，默认关闭；不实现更复杂的 GPU trial 状态机。按既定顺序，下一项独立候选为 M2/P3 的 EE 融合遍历与分类成本调查、固定 F/W 原型。

这些是相对当前 B1 执行组合的工程筛选数据，不是相对原 StiffGIPC/B0 的 2× 认证。重复控制也有轨迹差异，未通过独立物理参考或全路径极端失败状态认证。

## 身份与配置

- 分支 `research/stiff-innovation-20261014`，起点 `a4b83c040161eeef6149c93ae6167644fdf19384`；修改尚未提交，未上传 GitHub。
- binary SHA256：`abd50f5a9ebf03df2b3737a52d88bc67dc73a8b94bfd8d758cb08af17b5bcf18`。
- 构建：`build/innovation_p2a_20261014`；37 个 native TU 的实际编译、对象和链接核验；完整可执行 native 源码另封存到 `build/innovation_p2a_native_snapshot_20261014/snapshot_identity.json`。
- 运行：`runs/innovation_p2a_verified_20261014`，最终分析 **`analysis_v2/`**。`screening_analysis/` 和 `analysis/` 是保留的中间分析；v2 补上控制组几何数据、按 parent scope 精确归属线搜索成本，并统一诊断状态标识，未修改原始数值。
- 本机 RTX 3070 Laptop 8 GB / CUDA 13.0 / SM86 / Windows WDDM。实际电源为**野兽模式**，GUID `52521609-efc9-4268-b9ba-67dea73f18b2`；每次前/中/后检查。桌面负载、GPU 时钟未锁定。
- `dt=.04`、`rho=1e-4`、legacy 累计阈值 `.01`、至少 6 更新、outer movement tol `.01`；原材料、摩擦、Full CCD、原融合 DCD 不变，forcing off。
- Control：当前 Graph K1 + swept refit + ordinary/swept 双存储及 ordinary refit(interval 8) + 已有 batch/reuse + direct FEM pullback + MAS pivot-only sync；P2 新开关全关。四臂仅改变两个 P2 开关；没有把已有 batch/reuse 计为本轮新增贡献。
- 从零前缀：Boxes 17、Mixed 15、Bunny 14 帧。捕获使用 movement-only 前缀，与 M0 事件选择一致；冷恢复比较统一使用 legacy。新程序重新生成检查点，未修改旧 binary_id。

`solver_ms` 是 batch 主循环中 `IPC_Solver` 加结束同步的 CPU 实际窗口；位置/实际速度导出、checkpoint 和统计写盘另在计时外记录。`line_search_ms` 是 native CUDA event 的完整线搜索包络，包含提交间隙。新 CostTrace/P2 audit 在配对计时关闭；已有轻量统计和双方相同导出保留。进程 wall 包含启动、装载、导出和监控等待，在 runs.csv 单列。父子相位不重复相加。

## 实现与逐行检查落点

详细消费者/stream/所有权检查见 [SOURCE_AUDIT.md](E:/university_class/ComputerGraphics/GIPC/stiff_innovation_20261014/reports/innovation_p2a_20261014/SOURCE_AUDIT.md)。关键改动如下：

1. `mlbvh.cu/.cuh`：device 根归约仍执行；host 缓存改为显式 accessor，并带“缓存当前”和“device 根已计算”两个有效性位。完整缓存随 ordinary/swept storage 交换，分配/释放及 swept 扩容使其无效。空/未构建根拒绝 host 读取。
2. `discrete_bvh.inl`：真实 refit 后标记 host 结果过期，默认仍 eager 刷新。`RefitFullCCD` 标记过期但不额外增加生产读回。`GIPC::init` 使用 accessor 后再计算 bbox、dHat、dTol。
3. `cal_abd_energy.cu/abd_system.h`：抽出原两项 enqueue；原标量 API 仍可使用。batch on 时 ABD 写槽 9、10，一次读取 11 个 double；scalar on 时读取相邻两个结果。逐项 host 累加顺序保留，零 ABD 输出清零。
4. `p2_options.h/toi_options.h`：严格 0/1，默认 off，不继承 `GIPC_ACCEL_SUITE`，限于 IPC；实际配置写入 `p2_execution` 并由运行工具核验。
5. 线搜索入口 `cudaDeviceSynchronize` 仅新增 CPU 观测，**保留同步**。原接受条件、回退循环、九次预算、错误路径、forcing retry 和方向/alpha 计算没有改动。

运行开关为 `GIPC_P2_LAZY_SCENE`、`GIPC_P2_MERGE_ABD_ENERGY`；`GIPC_P2_AUDIT` 是独立正确性诊断。工具配置对应 `p2_lazy_scene`、`p2_merge_abd_energy`、`p2_audit`。

## 同状态四臂，3 对

倍率为每对 `control/candidate` 的几何平均；不是把平均时间和倍率混算。三个检查点的 N 在四臂各轮中分别均为 19、7、9；P 有小幅波动。逐轮原始 N/P、时间、alpha、回退次数均在 `analysis_v2/runs.csv` 与原 stats 中。

| 场景/物理帧 | 单独 lazy scene：solver / LS | 单独 ABD：solver / LS | 组合：solver / LS |
|---|---:|---:|---:|
| Boxes 17 | 1.0223 / 1.0433 | 0.9871 / 1.0050 | 1.0088 / 1.0360 |
| Mixed 15 | 0.9909 / 0.9844 | 0.9860 / 0.9934 | 0.9997 / 1.0355 |
| Bunny 14 | 1.0064 / 1.0131 | 1.0045 / 1.0156 | 1.0039 / 1.0398 |

组合 LS 的三对 log-t 95% 筛选区间：Boxes `[0.9491,1.1309]`、Mixed `[0.9455,1.1341]`、Bunny `[0.8989,1.2027]`。全部跨 1；只有三个样本且有 WDDM 噪声，不声称正式统计把握，不追加样本追显著。

## 从零短段组合，3 对

| 场景/帧数 | 总 solver 倍率 | 完整 LS 倍率 | 总 solver 的筛选区间 |
|---|---:|---:|---:|
| Boxes / 17 | 1.0026 | 1.0197 | 0.7422–1.3544 |
| Mixed / 15 | 0.9909 | 1.0182 | 0.8955–1.0965 |
| Bunny / 14 | 1.0018 | 1.0381 | 0.9741–1.0303 |

例如 Boxes 三轮 Control N/P 分别为 `78/2410, 73/2352, 77/2777`，组合为 `76/2895, 83/2502, 74/2079`。该轨迹工作量波动使整段时间不能仅归因于读回变化。Mixed 组合第二轮 P=12665，而 Control=11825；没有据此宣称修改 PCG 数学后导致迭代增加，本轮 PCG 数学并未修改。

## 真正的成本在哪里

以下是成本诊断 Control，按 scope parent 链只统计**完整 LS 内**的候选读回窗口；GPU events 关闭。这些 CPU 窗口还包含不可省的 GPU 计算等待，不能全部当成 memcpy 或潜在收益。

| 状态 | 完整 LS CPU 包络 ms | AABB+两项 ABD 读回窗口 ms | 占包络 | 入口整设备同步 ms |
|---|---:|---:|---:|---:|
| Boxes 17 | 100.0467 | 4.3651 | 4.36% | 0.0530 |
| Mixed 15 | 100.5339 | 2.1285 | 2.12% | 0.0334 |
| Bunny 14 | 46.8306 | 2.2294 | 4.76% | 0.0279 |

所选完整物理帧中，Control 的 host AABB 读回分别为 40/16/20 次，lazy 臂为 0 次；初始建场的真实 CPU bbox 读取仍保留。ABD 两项原独立读回分别为 20/8/10 组，merge 臂归并进入原 batch 结果读回。没有减少 ABD 数学工作或 BVH device 归约。入口同步仅占该包络约 0.03–0.06%，保留它是有成本依据的选择。

同状态 LS 小幅改善与窗口账本一致。P2-A 的这两个传输优化当前不足以成为主加速机制。完整 GPU trial 编排的收益仍未验证；本轮没有给它扩大实现的证据。

## 正确性、轨迹差异与覆盖界限

- 77 组两项 ABD 位级比较全部通过，其中 32 组来自零 ABD Hang，45 组有 ABD。native standalone 另覆盖 1/3/513/0/2 body 的增长/归零、Free/Fixed/Motor、boundary offset，以及写入 11 项 batch 尾部不污染前 9 项。
- scene 生命周期 off/on：未构建、空 edge、singleton、ordinary/swept 交换、坐标增长后的 refit、swept refit、device scene utility。普通查询诊断包含真实容量增长/overflow retry，140 次 face/edge refit/rebuild 叶界与带类型/矩阵映射的接触多重集核对无失败。这是每次 query 内相同几何的核对，不能把独立轨迹的不同 contact 数量说成位级等价。
- 54 次 checkpoint 恢复每次 40 个缓冲字节验证，context/frame 及 trace 初始位置/实际速度一致。采用 cold replay，未称 hot 缓存恢复。
- 完整短段出现真实能量回退，共 20 次；所有初始/后置几何回退计数为 0。CPU 接受函数覆盖九次预算和 NaN/Inf、缩步边界；native 入口失败及报告写失败通过。**真实 postgeometry 缩步、trial energy 非有限、原生预算耗尽和 forcing retry 尚未新增同状态完整覆盖。M1.3 不标成全部完成。**

实际轨迹没有位级一致，且候选/Control 差异有时超过三次 Control 自身的重复范围：

| 场景 | 单检查点组合 cloth RMS，mm（三对） | Control 重复最大 RMS，mm | 从零末帧组合 cloth RMS，mm（三对） | Control 末帧重复最大 RMS，mm |
|---|---|---:|---|---:|
| Boxes | .06589 / .01861 / .02340 | .04467 | .77766 / 1.69503 / 1.03445 | 1.58982 |
| Mixed | .00556 / .11599 / .02097 | .02587 | 3.77996 / 7.91858 / 3.76991 | 5.40336 |
| Bunny | .07758 / .10640 / .07854 | .06080 | 7.29673 / 5.20705 / 6.79363 | 6.58020 |

这些是相对当前 Control 的轨迹差异，**不是对真实解的误差**。现有 MAS atomic 归约、较松线性/外层阈值和停止分支可传播重复差异；本轮未用额外对照唯一分离各个来源，不能笼统称为“全是噪声”、或宣称相同物理误差。

Mixed 非正 tet J 仍存在，Control 最小 `-0.952230`、组合 `-0.942011`，两者最多 2 个非正 FEM tet；沿用 M0 已识别问题，未改材料/注释的 injectivity 限步。逐帧 stretch/J/fixed drift 在 geometry.csv；actual velocity 与按 free/cloth/FEM/ABD/fixed 分类差异在 state_differences.json。

## 中止、工具修正与历史保留

1. 显存约 3.4 GB 时，两次 prefix 分别在 Boxes15、Bunny5 触发预算中止，所辖 Job 均清空，无 OOM；不计入 79 次成功或性能表。用户释放显存后，在新会话重做三个 prefix。
2. 一次 cost 恢复 0 帧失败：CLI 的 relative session 被作为 relative native load path。程序在独立输出目录中启动，正确拒绝不存在的 metadata。驱动修正为先 resolve 绝对 session；新增该入口的 CPU 契约测试。
3. 旧两份驱动存于本报告目录 `*_before_path_fix.py`；三类失败记录分别保留在 `runs/innovation_p2a_20261014`、`runs/innovation_p2a_bunny_20261014`、`runs/innovation_p2a_full_retry_20261014`。
4. 已完成的三个 prefix/checkpoint 没有重写；新会话只复制 committed checkpoint 文件并逐文件 SHA256 核对。`resume_inputs.json` 明确保存继承测量、原始路径和复制来源；元数据/binary_id 不改。

## 复核入口和有限决定

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/p2_contracts_test.py -q
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/m0_runtime_contracts_test.py -q
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/m0_geometry_test.py -q
& 'D:/computer/cmake/bin/ctest.exe' --test-dir build/innovation_p2a_20261014 -C Release --output-on-failure
```

本轮分别为 6/5/3 项 Python 通过、24/24 CTest 通过。构建、原始轮次、恢复、输入和每次 resolved 配置均已保存。最终 `DELIVERY_SEAL.json` / `analysis_v2/data_validation.json` 提供独立核验入口。

**决定：STOP 扩大 P2-B；两个 opt-in 保留供复核，全部默认 off。** 不基于几个百分比的 LS 改善认定主要创新或整场收益。下一次推进 M2 时，先观察 EE 遍历/分类成本，再实现独立固定 F/W 小原型；不同时调整停止规则、精度、PCG 或碰撞模型。未上传 GitHub，未输出新的全场性能/视觉 PDF。
