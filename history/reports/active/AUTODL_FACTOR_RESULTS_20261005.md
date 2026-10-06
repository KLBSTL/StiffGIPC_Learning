# AutoDL 4090 因子逆候选复测（2026-10-05）

本轮完成干净构建、18 次启动检查及 54 次短窗口测试。FP64 因子逆局部作用的执行收益在 RTX 4090 上复现，但融合方案没有通过相对 Stiff 的物理质量门槛，混合兔子仍明显更慢。保持 `factor_inverse` 默认关闭、默认作用为 `triangular`；不启动 100/300 帧或七组正式性能认证。

本轮没有修改 CUDA/C++ 求解器。历史大清理另见[清理报告](AUTODL_CLEANUP_20261005.md)：净释放 68.93 GB，关键源码、六份历史系统、报告及当前实验保留。

## 实测速度与组件收益

三场景均从零连续运行，dt=.01，按原计划交错三轮。表中时间是三次求解耗时的中位数；比率是同轮配对比率的中位数，不能直接用表中两个时间相除替代。

| 场景／帧数 | Stiff (s) | IPC host (s) | IPC Graph (s) | TOI host (s) | TOI Graph 新作用 (s) | TOI Graph 旧三角作用 (s) |
|---|---:|---:|---:|---:|---:|---:|
| 悬挂／21 | 0.373046 | 0.416074 | 0.349865 | 0.232284 | 0.199359 | 0.264412 |
| 固定兔子／45 | 2.018979 | 2.419866 | 2.199466 | 2.369367 | 2.147524 | 2.599780 |
| 混合兔子／35 | 1.835300 | 2.542205 | 2.485132 | 5.058399 | 4.924148 | 6.475672 |

| 同轮配对耗时比（大于 1 表示后者更快） | 悬挂 | 固定兔子 | 混合兔子 |
|---|---:|---:|---:|
| IPC host / IPC Graph | 1.223× | 1.100× | 1.023× |
| TOI host / TOI Graph | 1.199× | 1.142× | 1.030× |
| 旧三角 TOI Graph / 新作用 TOI Graph | 1.323× | 1.260× | 1.314× |
| 同执行组件 IPC Graph / TOI Graph | 1.755× | 1.024× | 0.505× |
| Stiff / 新作用 TOI Graph | **1.887×** | **0.948×** | **0.374×** |

IPC/TOI host 与 Graph 使用相同的其余活动组件；旧三角参考只关闭因子逆作用。不能把这些收益相乘。混合兔子新作用比旧三角版约省 24% 耗时，但融合仍耗费 Stiff 约 2.67 倍时间，也没有胜过同执行组件的 IPC。

这些是**开启子步、物理量和速度导出的短窗口诊断结果**。Stiff 没有实际速度导出，诊断开销也不完全相同；本轮只有三次重复，未执行关闭重型诊断的七组配对与置信下界认证。采样未见外部 GPU 计算进程，仍不把这些比率称为正式同质量加速比。

数据依据：[完整 v2 分析](../../downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json)；[独立配对比率复算](AUTODL_WINDOW_PAIRED_AUDIT_20261005.json)；[独立工作量复核](AUTODL_WINDOW_PAIRED_AUDIT_20261005_v2.json)。

## 质量结论

| 场景 | Stiff 三次范围 | 旧 triangular TOI Graph | 新 factor_inverse TOI Graph | 判断 |
|---|---|---|---|---|
| 悬挂峰值布料边长比 | 1.128876190–1.128876229 | 约 1.13172608111538 | 约 1.13172608111538 | 新旧作用基本一致，但已有 TOI 相对 Stiff 拉伸越界 |
| 固定兔子峰值布料边长比 | 1.034899–1.035170 | 1.040030–1.040395 | 1.040026–1.040385 | 相对 Stiff 越界；峰值接近旧作用也不能代表逐帧通过 |
| 混合兔子最小 FEM J | 0.49786875–0.49786895 | −0.080438 至 −0.033469 | −0.144003 至 −0.005459 | 新旧 TOI 都超出 Stiff 范围，新作用最差一次又超出旧作用范围 |

固定兔子新作用三次分别有 7/11/11 帧的布料拉伸超出旧 TOI 逐帧重复范围，最大超出约 1.81e−5。第 3 次布料位置最大 RMS 差约 4.258 mm，旧作用自身重复约 1.544 mm；实际速度差约 .398 m/s，旧作用自身约 .0934 m/s。

混合兔子新作用第 3 次第 35 帧最小 J 为 −.144003，最大负体积为 4.3888e−8；旧作用最差分别为 −.080438 和 2.4515e−8。Stiff 此 35 帧窗内没有非正 FEM 单元，三次新作用均曾出现一个。这里使用预先声明的相对基线门槛，没有新增禁止所有 FEM 翻转的材料约束。

分对象轨迹差异已分别检查，不能以总体 RMS 抵消局部问题。位置分岔本身不等于对物理真值的误差；拉伸和负体积范围越界已经足以拒绝本候选的推广。Stiff 缺少真实速度，保留为不可用，不用位置差分冒充实际速度。

54 次均未出现 PCG 触顶或 breakdown；导出的物理帧端点状态有限，端点固定点最大漂移约 6.21e−17 m。独立 CPU Tight-Inclusion 已检查全部 54 组的 **7,707 条接受路径（含跨帧静止连接）**，碰撞标记为 **0**。CCD 通过不抵消上述 FEM 与布料质量失败。

## 工作量解释与下一步

| 混合兔子 35 帧，中位数 | Stiff | 同组件 IPC Graph | 旧三角 TOI Graph | 新作用 TOI Graph |
|---|---:|---:|---:|---:|
| 求解方向数 | 142 | 142 | 170 | 169 |
| PCG 总迭代数 | 3,655 | 3,646 | 21,099 | 20,722 |
| 每方向平均 PCG（上述中位数之比） | 25.7 | 25.7 | 124.1 | 122.6 |

新 TOI 总 PCG 约为 Stiff 的 5.67 倍，但方向数只多约 19%。第 24–35 帧占新 TOI 全窗 PCG 约 92.4–92.8%：18,215–19,393 次，而 Stiff 同区间仅 1,978–1,979 次。问题主要在接触发展后每个方向需要大量线性迭代；因子逆作用减少了执行成本，几乎没有改变这个工作量结构。

同样工作量的 IPC Graph，其完整线性阶段中位约 1.379 s，Stiff 约 .666 s，仍有稳定路径成本。但该阶段包含矩阵转换、准备、PCG 与解分发，不能归因某个 kernel。TOI 的普通 phase 字段是零占位，本报告将其记为不可用，不用来构造分项成本。

**2026-10-05追加原因复核后，修正下一步优先级。** 原先把无接触退出语义排在首位的判断依据不足：TOI的AL集为空，不等于IPC无屏障响应；两者第一次线性求解在退出判断之前就已不同，且混合第1帧TOI最后方向已经低于双方名义阈值。详见 [F1目标与时序复核](TOI_ITERATION_AND_F1_AUDIT_20261005.md) 及保留历史的 [退出假设修正](CONTACT_FREE_EXIT_HYPOTHESIS.md)。

优先验证**接触罚刚度的混合坐标尺度和困难方向的矩阵组成**。新只读重算发现：本机无接触冻结系统最大H对角3861.4073来自ABD仿射DOF，FEM＋布料最大仅2.08196；代码从前者计算µ=386.1407，再直接用于世界坐标接触。尺度来源已证实，是否足以解释5.67倍PCG仍需同状态对照，不能把约1855的对角比当作条件数或推荐的全局调参倍率。优先对F25/outer1的inner0→1（已有历史冻结）及当前F27/outer1的inner0→1，检查接触/非接触H、右端和预条件器；保留材料、停止阈值及完整CCD。见[逐方向增量](TOI_PCG_CAUSAL_AUDIT_20261005.md)、[冻结对角证据](LOCAL_FIXED_DIAGONAL_SCOPE_20261005.json)。本轮尚未修改求解器。

## 构建、运行与证据身份

- RTX 4090 / SM89；驱动 580.105.08；CUDA 12.8；CMake 3.22.1。GPU UUID `GPU-52acb437-5458-3bd2-0694-4c9fe6cf7b9a`。
- 原冻结包 `autodl_factor_diagnostic_20261004.tar.gz` SHA256：`659cef89983a818f0988973c1a96038e40c24ab81e9bcc2b9ad13149f20f4862`，647 个输入逐文件核验。
- Stiff 程序 SHA256：`2da5b1a2eecbd473901870bce71a63229cebb637a12e6d799d7682486a73948a`。
- active 程序 SHA256：`6ce6928c65892a5fc823b491d99ac21f65a815c50966ebe8c973d5a7c9fad4d9`。
- Linux 独立 CCD 程序 SHA256：`fe2f54e0375e5c82acffa0bcc48539b2f0aeab5ac2b0380419ed4e80dffed0f0`。
- 原有材料、停止参数、活动组件与单次 120 秒预算未变。GPU 测试串行，无运行失败、预算超时或自动重试。
- 远端实际 43 项组件、15 项守卫检查全部通过。这与之前本机六份历史数值系统验证是不同范围，不能互相替代。
- 本机分析入口使用 DL Python；远端分析使用已有 `/root/miniconda3/bin/python`、NumPy 2.3.2、Matplotlib 3.10.5。

首次分析器错误地要求 IPC 提供 TOI 专属的 `effective_preconditioner` / `mas.active` 字段，误报 18 组配置失败。旧分析及原 CCD 审计完整保留。v2 按 IPC 的实际导出格式、场景 MAS 选择、运行配置与执行日志重新核验，54 组全部通过；TOI 的有效 MAS 检查仍保留。修复只涉及分析工具，未重跑或改变 GPU 实验。

精确运行计划、构建与启动命令见 [AUTODL_FACTOR_EXECUTION.md](AUTODL_FACTOR_EXECUTION.md) 及回传的 jobs/配置记录。主要 CPU 检查命令（`ROOT=/root/autodl_factor_retest_20261004_659cef89`）为：

```bash
/root/miniconda3/bin/python "$ROOT/tools/active/autodl_analysis_job.py" --root "$ROOT"
/root/miniconda3/bin/python "$ROOT/tools/active/analyze_autodl_factor_v2.py" --root "$ROOT" --output "$ROOT/reports/active/AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json"
/root/miniconda3/bin/python "$ROOT/tools/active/reaggregate_autodl_audit.py" --root "$ROOT" --audit-dir "$ROOT/reports/active/AUTODL_FACTOR_CCD" --output "$ROOT/reports/active/AUTODL_FACTOR_CCD_REAGGREGATED.json"
```

证据保留状态、图像检查与最终文件校验见下方交付记录；完整 54 组原始轨迹继续保留在远端，紧凑回传包不能替代全量 CCD 重放输入。

## 图像检查

三张图均由实际保存网格生成，采用同场景一致相机/尺度及统一布料色标，每张图包含 Stiff、旧三角 TOI Graph、新作用 TOI Graph 各三次重复。已在本机逐张视检：场景、行列和帧号正确，几何与指标文字可见，无遮挡导致的空白面板或错误色标。图中标注是**所画最终帧**的指标，与本报告的全窗峰值范围不同。

- [悬挂第 21 帧](../../downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_FIGURES/autodl_factor_hang_final.png)：新旧 TOI 形态基本一致；不能据最终帧相似忽略此前峰值拉伸差异。
- [固定兔子第 45 帧](../../downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_FIGURES/autodl_factor_fixed_bunny_final.png)：布料覆盖固定兔子，Stiff 与 TOI 的褶皱存在差异；具体判断使用逐帧指标。
- [混合兔子第 35 帧](../../downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_FIGURES/autodl_factor_mixed_final.png)：整体外观不足以揭示单个 FEM 单元翻转，图上保留实际 J、非正单元和负体积数值。

## 最终交付与保留

紧凑证据已回传本机并逐成员核验：**1,289 个文件全部通过**，其中新增 1,042 个、已有且字节相同 247 个；没有覆盖冲突，没有遗漏超限日志。包内包含 72 次运行元数据、54 次原生 CCD JSON/日志/身份、修正前后分析、构建报告、执行工具和三张图的 27 个精确关键帧输入。

- [证据包](../../downloads/autodl_factor_compact_evidence_20261005.tar.gz)：62,814,091 字节，解包内容 152,804,751 字节。
- SHA256：`ae076e983bd8425e0ed5a2d5e339e0e14b37233da874fc6219d3ce083e3e61a0`。
- [本机逐文件校验收据](../../downloads/autodl_factor_20261004_4090/evidence_receipts/2b442ead3713411893c1d5c42ab839b7.json)。
- [修正后的 CCD 汇总](../../downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_CCD_REAGGREGATED.json)：54/54 接受路径审计通过，7,707 条路径、0 标记；原报告及原失败字段仍保留。
- [图像输入与数值清单](../../downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_FIGURES/render_manifest.json)。

下载后再次从本机原生 CCD 记录核对：54 份均有限、ABD 翻转为零，计数覆盖全窗口；接受子步固定点最大漂移也为 6.2063e−17 m。原分析总任务记录仍保留 `completed_with_findings`，对应旧分析器的配置误报；后续 v2 和重聚合明确记录更正，不篡改原状态。

全部仿真、CCD 和绘图进程已经结束。最后远端检查 GPU 无计算进程，系统盘约 21 GiB、数据盘约 48 GiB 可用。完整 54 组轨迹仍保留在 `/root/autodl_factor_retest_20261004_659cef89/runs/active`，便于下一次有界因果分析；本机紧凑包仅含选定关键帧，不能独立重放全量 CCD。本轮结论为：**测试执行完成；候选推广拒绝；2× 同质量目标未达成。**
