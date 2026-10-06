# 当前并行 MAS 候选：布料加速比实测

2026-10-04。本轮使用已有活动程序，仅通过 `mas_restrict=warp` 开启新限制算子，没有修改求解器、材料或收敛参数。

**三种布料相对同轮 StiffGIPC 的配对中位耗时比为 1.425×、1.053×、0.704×。** 相对旧串行 TOI 则为 1.206×、1.229×、1.462×。新限制算子有整场收益，但尚未解决固定兔子比 Stiff 更慢、轨迹与拉伸不满足相对基线范围的问题；不能认证同质量 2×。

## 1. 测试条件和组别

- 本机 RTX 3070 Laptop GPU；每个场景从零连续运行100帧、dt=.01，即1秒物理时间。
- 三种场景为悬挂 `cloth_hang_l`、落球 `cloth_sphere7_l`、固定兔子 `cloth_fixed_bunny_l`。本轮固定兔子不是混合兔子 `bunny_cloth_bunny_l`。
- 每种场景四组、每组三轮，逐轮旋转顺序，共36次正式诊断计时；另6次3帧预热/冒烟和6次100帧接受路径诊断。GPU实验串行，CPU重型分析在36次计时全部结束后运行。
- Stiff使用官方冻结程序原生MAS；其余三组使用当前稳定Cholesky MAS、conditional PCG Graph、refit、批量能量和能量复用。TOI保留现有choose-start与restart-guard。原串行TOI与新并行TOI仅限制算子配置不同。
- IPC Newton=.01、PCG rho=1e-4；TOI剩余推进量=.01、trial速度=.05。逐次核对实际 `scene.effective_run` 和活动程序 `resolved_config`；初态、拓扑、质量、材料一致。
- 正式计时关闭额外探针、Nsight、substep和速度导出。下表使用程序记录的求解时间，位置文件写出不在该计时内；不使用进程启动耗时作为求解时间。
- 单次预算120秒，不因慢或失败延长。共享桌面GPU负载不能独占控制，三轮结果是诊断测量，没有执行七对置信下界认证。
- 旧布料报告部分球/兔子使用对角预条件器、悬挂使用dt=.04，不能用它们的秒数替代本轮同配置基线。

## 2. 整场耗时与加速比

时间为三次运行的中位数，单位秒；加速比则先算同轮配对比，再取中位数，不是直接用两列中位时间相除。所有加速比大于1表示新并行TOI更快。

| 场景 | Stiff | 原串行TOI＋Graph | 新并行TOI＋Graph | 同组件并行IPC＋Graph | 相对Stiff | 相对原TOI |
|---|---:|---:|---:|---:|---:|---:|
| 悬挂 | 5.060 | 4.242 | 3.551 | 4.633 | **1.425×** | **1.206×** |
| 落球 | 10.503 | 12.386 | 9.566 | 9.571 | **1.053×** | **1.229×** |
| 固定兔子 | 14.766 | 30.360 | 20.916 | 18.902 | **0.704×** | **1.462×** |

| 场景 | Stiff / 新TOI：三轮范围 | 原TOI / 新TOI：三轮范围 | 同组件IPC / 新TOI：中位（范围） |
|---|---:|---:|---:|
| 悬挂 | 1.364–1.431× | 1.171–1.212× | 1.292×（1.287–1.317） |
| 落球 | 1.044–1.108× | 1.223–1.295× | 0.980×（0.978–1.010） |
| 固定兔子 | 0.695–0.729× | 1.452–1.607× | 0.916×（0.878–0.958） |

悬挂相对Stiff耗时下降约29.8%；落球约5.1%；固定兔子耗时反而增加约42.1%。TOI本身相对相同执行优化的IPC仅在悬挂明显有收益，落球接近持平，固定兔子仍慢约9.2%。这些是当前测量，不能从落球5%的差距推出受控负载下有稳定小幅加速。

## 3. 为什么固定兔子仍然慢

| 全100帧工作量，中位 | Stiff | 原串行TOI | 新并行TOI | 同组件并行IPC |
|---|---:|---:|---:|---:|
| 悬挂方向次数 | 442 | 340 | 341 | 442 |
| 悬挂PCG总迭代 | 14,021 | 12,455 | 12,497 | 14,009 |
| 落球方向次数 | 576 | 666 | 669 | 562 |
| 落球PCG总迭代 | 24,282 | 26,510 | 26,495 | 23,818 |
| 固定兔子方向次数 | 583 | 787 | 784 | 605 |
| 固定兔子PCG总迭代 | 37,186 | 41,966 | 41,596 | 37,425 |

固定兔子的并行TOI相对同轮Stiff多31.4–35.6%方向、多10.1–12.6%总PCG迭代。每方向PCG均值反而更低：TOI约53.1–54.1，Stiff约63.5–64.6。问题包括方向数量过多，不能只盯单次PCG迭代上限；36次计时都没有触顶或breakdown。

另一个独立证据来自仍有普通阶段计时的Stiff与并行IPC：固定兔子整线性阶段为8.31–8.61秒与12.46–13.65秒。同轮总耗时增加3.65–5.51秒，其中整线性阶段增加4.07–5.34秒，其余阶段合计基本持平或下降。稳定MAS组合路径仍有明显线性成本，但本轮不能进一步归因到某一个kernel。

**计时解释修正：** TOI普通 `phase_ms` 五项为未填充的零占位，不是耗时为0。分析器已将其改为 `null / all_zero_placeholder`。采用 `CLOTH_RESTRICT_QUALITY_TIMING_V2.json`；旧JSON保留为更正记录，其TOI阶段时间不可引用。整线性阶段包括矩阵转换、MAS准备、PCG和解分发，不等同于纯PCG循环。

## 4. 形变、轨迹与安全检查

三次运行的全程最大边长比（当前长度/初始长度）：

| 场景 | Stiff重复范围 | 原串行TOI重复范围 | 新并行TOI重复范围 |
|---|---:|---:|---:|
| 悬挂 | 1.128876204–1.128876222 | 1.131726081 | 1.131726081 |
| 落球 | 1.019301–1.021728 | 1.017889–1.018046 | 1.017779–1.018141 |
| 固定兔子 | 1.035521–1.035715 | 1.040099–1.040358 | 1.040312–1.040355 |

悬挂与固定兔子峰值超过Stiff重复范围。落球全程峰值更低，但三个新TOI重复各有14帧超过Stiff对应帧的最大拉伸范围。因此三者都不能仅凭峰值或无碰撞宣告相对基线不退化。

相对旧串行TOI，新限制算子的峰值接近；固定兔子的新峰值也落在旧TOI全程峰值范围内。然而逐帧拉伸和轨迹并不严格等价。悬挂、落球、兔子相对旧TOI逐帧最大拉伸上界的最大超量分别约7.80e-7、5.03e-4、1.94e-3；没有据此新增宽容差。

| 自由布料位置RMS的全程最大值，米 | Stiff自身重复上限 | 原TOI自身重复上限 | 新TOI对同轮Stiff | 新TOI对同轮原TOI |
|---|---:|---:|---:|---:|
| 悬挂 | 0.00000861 | 0.00036848 | 0.015567–0.015569 | 0.0000835–0.0005074 |
| 落球 | 0.025498 | 0.031726 | 0.056288–0.070722 | 0.023220–0.030694 |
| 固定兔子 | 0.019187 | 0.039542 | 0.094943–0.114303 | 0.024068–0.037135 |

位置RMS使用自由布料顶点质量加权。这是轨迹差异，不是相对物理真值的误差；全程最大值也不能替代逐帧重复范围。未重建或验证速度，未完成同质量认证。

36次均完成100帧，所有导出端点有限，三角形面积比为正，无PCG触顶或breakdown。悬挂2个固定点、落球483个固定ABD顶点、固定兔子19193个固定ABD顶点均纳入漂移检查；所有组最大漂移7.85e-17米，处于舍入量级。没有将空FEM统计当作FEM材料验证。

接受路径审计独立于计时：六组Stiff/新TOI诊断各100帧，逐帧核验safe编号连续及首末状态与导出端点字节一致，再用CPU BVH＋Tight-Inclusion检查。**六组覆盖与检查全部通过，共2969段、零保守碰撞标记**，其中2375段实际接受推进、594段跨帧静止桥接。汇总见 `CLOTH_RESTRICT_CCD.json`。这验证六次独立诊断轨迹，不能替代未导出substeps的全部计时轨迹审计。

| 场景 | Stiff路径数 | 新TOI路径数 | 两组保守碰撞标记 |
|---|---:|---:|---:|
| 悬挂 | 541 | 240 | 0 |
| 落球 | 670 | 401 | 0 |
| 固定兔子 | 690 | 427 | 0 |

三张实际导出网格图已按共同帧、视角、比例和色标检查：悬挂新旧TOI形态接近；落球和固定兔子有可见褶皱、下垂差异。Matplotlib遮挡排序不用于证明接触安全。

- [悬挂网格对比](figures/cloth_restrict_hang.png)
- [落球网格对比](figures/cloth_restrict_sphere.png)
- [固定兔子网格对比](figures/cloth_restrict_bunny.png)

## 5. 结论与下一步

| 结论 | 状态 |
|---|---|
| 并行限制相对旧串行TOI有约1.21–1.46×整场诊断收益 | 已有本轮三次配对支持 |
| 悬挂相对Stiff明显更快；落球接近；固定兔子更慢 | 已测量，限共享桌面条件 |
| 新TOI的物理效果相对Stiff不退化 | 未通过冻结的拉伸/轨迹范围，不能认证 |
| 并行限制与串行TOI严格轨迹等价 | 未证明，存在逐帧差异 |
| 稳定MAS已无优化空间，或慢完全来自某个kernel | 证据不足，不能如此归因 |
| 同质量、受控条件下整体2× | 未达到 |

本轮保持候选显式启用，不修改默认路径。后续性能工作应分别处理固定兔子稳定MAS组合的剩余线性成本，以及TOI额外方向/装配工作；同步核查与Stiff的求解充分性差异。不能通过放松阈值、移除稳定性修复或用局部kernel收益替代整场结果。

## 6. 复现入口和证据

预声明协议 `CLOTH_RESTRICT_PROTOCOL.md`；正式矩阵 `CLOTH_RESTRICT_TIMING_BATCH.json`；最终分析 `CLOTH_RESTRICT_QUALITY_TIMING_V2.json`；27次活动组配置/执行检查 `CLOTH_RESTRICT_CONFIG_CHECKS.json` 全通过。冻结Stiff没有resolved接口，以保存的manifest、scene/effective_run及初态核对。

活动EXE SHA256：`976dcf949c5c51d00c2eee84f49aa8bf3f15890f569093081bf4075de765152b`。

Stiff EXE SHA256：`1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205`。

以下为本轮实际执行入口，工作目录为 `stiff_toi_cudagraph_20260929`；重新实验需要新运行名称和输出路径，runner拒绝覆盖旧证据：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/prepare_cloth_benchmark.py
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/cloth_restrict_smoke.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/cloth_restrict_timing.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/analyze_cloth_restrict.py --output reports/active/CLOTH_RESTRICT_QUALITY_TIMING_V2.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/audit_cloth_restrict.py --prepare
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/cloth_restrict_audit.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/audit_cloth_restrict.py --output reports/active/CLOTH_RESTRICT_CCD.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/render_cloth_restrict.py
```

各运行的 `requested.json`、`resolved_config.json`（活动组）、`build_manifest.json`、`result.json`、`output/stats.json` 和完整位置轨迹保存在 `runs/active/cloth_restrict_*`，不删除失败或更正证据。
