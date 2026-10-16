# P5 rank 8：免 QR 坐标作用验证

2026-10-16；分支 `research/stiff-innovation-20261014`，HEAD `07534faba7d377315ac8f2023e63eab73fef14db`。本轮只增加离线数学诊断及独立 CUDA 探针，生产源码、原生二进制、材料、FullCCD、外层停止和 PCG 阈值保持本轮开始时的字节；未上传 GitHub。

## 结果与决策

**省去大矩阵 QR，在所测三个固定目标系统保留了迭代收益，并降低 Mixed 的构造成本；计入参考作用成本后，条件模型仍未显示净收益。**

| 固定源 n1 → 目标 n2 | MAS 参考原次数 | 统计空间＋QR | 统计空间免 QR | 免 QR 粗矩阵条件数 |
|---|---:|---:|---:|---:|
| Mixed，第15帧 | 122 | 61 | **61** | 2.667 |
| Bunny，第14帧 | 184 | 154 | **154** | 31.400 |
| Boxes，第17帧 | 27 | 31 | **31** | 4.730 |

Boxes 的增量工作仍高于 MAS，保留负对照，不扩大该场景路线。两个方法每个目标都使用相同原 A/b、零初值及固定 B_ref 共同真残差。CPU 数字不是原生 GPU PCG 次数。

Mixed，同一次 GPU 进程、同输入、五对 AB/BA 的完整 setup 墙钟中位数：

| rank 8 | QR 强对照 | 免 QR 候选 |
|---|---:|---:|
| setup 中位 ms | 9.8098 | **7.7605** |
| setup 范围 ms | 9.5623–10.4003 | 6.8847–7.9820 |
| 参考 Graph 作用新增中位 µs/次 | 86.836 | 80.236 |
| 条件净差 ms | -3.218 | **-0.766** |

五对 setup 每对都更快，配对节省中位 `1.9838ms`，配对倍率中位 `1.2535×`。两个耗时中位数之差为 `2.0493ms`，与配对差中位不同。**这些是构造阶段/固定参考成本，不能当成整场或 native 提速。** 不用上批另一次进程的 `10.945ms` 与本批候选混配。

结论：保留 rank 8 免 QR 实现；暂缓 rank 16、在线 bank 和主求解器接入。下一关先补原生 FP32 MAS 生成的完整历史/统计校准，再依据实际预算选择投影优化。

## 为什么可以省 QR

设 `W=AZ`，`E=sym(ZᵀW)`，`Q=ZE⁻¹Zᵀ`。对可逆换基 `Z'=ZT`，有 `E'=TᵀET`，因此精确算术下 Q 不变；原 balanced 作用

`Q + (I−QAᵀ) B_ref (I−AQ)`

也不变。这是坐标变换恒等式，不是新谱理论或有限精度保证。实现继续使用 `Wᵀ`，没有用 `ZᵀA` 代替它；原 A 的微小非对称保持原样。

候选把源统计小 Gram 的最大8个特征向量系数除以其特征值平方根，形成 `Z=S·U·diag(λ⁻¹/²)`。缩放只改变坐标，不改变保留的空间。没有用目标解，没有重新选择场景或秩。

不要求 Z 欧氏正交；独立验证时才计算 Python QR 比较空间，这个 QR 没有参与候选构造计时。GPU Mixed 候选的 `||ZᵀZ−I||≈360.18` 是所选坐标的正常属性，不用它判定失败。相应目标 E 的条件数约2.667，比本轮 QR 坐标下约67.970更小；这只适用于当前系统，不外推为普遍改善。

为了保护有限精度，候选每次 setup 支付粗 E 的读回和小谱检查：有限、相对非对称小于 `1e-8`、`λ_min > 1e-12 λ_max >0`，再要求 Cholesky 成功。不裁负特征值、不加 jitter、不补秩。当前失败行为是明确拒绝，**尚未实现生产在线回退**。

预登记：[PLAN.md](PLAN.md)；逐项源码核对：[SOURCE_AUDIT.md](SOURCE_AUDIT.md)。

## 数值验证

- CPU 三场，免 QR/QR 作用最大相对差分别 `1.913e-13 / 4.941e-14 / 2.177e-13`。六份目标解均满足原 A 的显式相对 L2 和固定 B_ref 加权相对残差 ≤`.01`，并独立 Decimal50 复核。迭代数相同不表示解逐位相同，也不表示动态物理轨迹已经认证。
- 保存全部492条逐迭代记录；最终审计重新加载六份解，重新计算真残差、标量残差，并检查每种方法的迭代连续性及末行一致性。
- GPU Mixed：两种坐标的输出作用最大相对差 `1.210e-13`；免 QR 空间投影差 `3.406e-15`、AZ 相对差 `1.515e-14`、因子相对差 `9.923e-14`，相对独立 Python 作用最大差 `8.351e-14`。两种方法 Graph/host 的所测输出逐位一致。
- 小型正例两个方法通过；负例让实际历史全部共线、同时保留满秩正统计量，免 QR 必须以 `QR-free coarse rank/conditioning rejected` 拒绝，而不是因为其他错误退出。
- `90` 项 Python 测试通过，含新增6项换基、Wᵀ、能量缩放、缺秩、病态及非有限合同；小型完整路径 Compute Sanitizer **0 errors**。

机器审计：[AUDIT.json](AUDIT.json)。最终表：[CPU_final.csv](tables_final/CPU_final.csv)、[CPU_curves.csv](tables_final/CPU_curves.csv)、[GPU_checks.csv](tables_final/GPU_checks.csv)。CPU 表的 `coarse_condition` 是同一案例的免 QR 候选条件数，作为案例属性重复在两行；两个 GPU 坐标各自的条件数在 GPU_checks 中独立列出。

## 成本与边界

完整 setup 包括历史 D2D 回放、S 构造、统计 D2H、CPU 小 Gram/谱分解、系数缩放/H2D、GPU 提取、候选粗谱检查、目标 AZ/E、Cholesky 及必要同步。没有预先准备好系数再隐藏费用。

隔离阶段单次墙钟样本，不能相加替代完整 pipeline 中位数：

| 阶段 | QR ms | 免 QR ms |
|---|---:|---:|
| 历史 D2D | 1.3680 | 1.1764 |
| S | .3309 | .3214 |
| 统计读回＋CPU小谱＋上传＋提取 | 4.4426 | **4.6361** |
| 大 QR/秩检查 | 3.4027 | 省去 |
| 目标 AZ | 1.2442 | 1.0325 |
| E/检查/因子 | .8104 | .7627 |

候选统计阶段内部 CPU Gram约`.0249ms`、谱和系数约`1.7510ms`、统计D2H约`.0379ms`；余下费用包含上传、GPU提取及同步。事件记录的是 stream 跨度，CPU工作期间可包含GPU空闲，不解释为GPU忙碌时间。

保留旧 native 平均时标 `c=.194894639ms/iteration`，并假定 CPU 目标次数61能迁移，则最多可省 `(122−61)c=11.888573ms`。本批参考 Graph 增量约`80.236µs×61=4.894371ms`，加setup共约`12.654871ms`，条件净差约`−.766298ms`；若增量保持不变，setup需小于约`6.994202ms`才有正余量。这个阈值及净差只是工程筛选模型。

参考 MAS 仍是独立 FP64 固定系数实现，效率与原生优化 FP32 MAS不同。host 增量的中位数甚至由约104升至121µs，完整保留，不能只选有利窗口。Graph 的QR/免QR两组作用窗口在各自配置内与MAS配对，**两种坐标之间并非交错配对**，不能把两组增量之差单独认证为作用加速。

原始五对 setup：[setup_pairs.csv](tables_final/setup_pairs.csv)；所有作用窗口：[apply.csv](tables_final/apply.csv)；条件模型：[conditional_budget.csv](tables_final/conditional_budget.csv)。

## 原生适用性仍待验证

现有原生导出包含 A/b、MAS所有保存系数/校准作用和求解结果，但没有逐次 alpha/rho 和完整历史。CPU参考采用保存FP32系数的固定FP64线性作用；原生FP32转换、局部计算和归约舍入不满足严格线性/共轭假设。当前不能据此认定统计小 Gram 在原生上同样准确。

下一轮应先增加受保护、默认关闭的原生固定系统历史采集，对相同A/b记录alpha/rho/curvature、历史下标、x和最终残差，校验采集前后主方向/算子/Graph状态恢复以及未采集对照。通过后再检查实际统计Gram/直接Gram与目标作用；失败则先解决模型适用性，不能继续只缩减探针费用。暂不进行100帧、轨迹或视觉认证。

## 构建、资源与失败记录

新独立1TU构建，CUDA13.0.88、MSVC19.44、SM86、UTF-8 host编码，0警告/0错误；最终exe SHA256：`b55705ef2743dca01a8262e777f702c19821c83762a57aed6fe15785fa28e0b7`。未启用fastmath。公共参考运行时是上轮源码的文本一致前缀副本，生成时换行规范化为CRLF，不能称为原始字节一致；来源hash在 `tools/bench/qrfree_gpu/ORIGIN.json`，没有修改旧探针。

本机RTX3070 Laptop/WDDM，全部成功GPU Job均保持野兽模式GUID `52521609-efc9-4268-b9ba-67dea73f18b2`。Mixed启动前GPU利用率23%，共享桌面负载未受控；五对仅作成本筛选。显式设备分配255.24MiB，Mixed监测显存下降457MiB，未触发资源保护。见[launches.csv](tables_final/launches.csv)。

失败和重试完整保留：

1. 首个沙箱构建配置停滞，已中止；新目录沙箱外编译成功。未用未完成构建采数据。
2. 首次全体Python测试因系统沙箱TEMP的创建/删除权限失败；把该次测试进程TEMP/TMP指向本工作区专用目录后，90项全部通过，没有改测试或数值算法。
3. 首次memcheck在沙箱中未附加，最终180.172秒超时，owned Job清理为空；结果不计作内存通过。新名字沙箱外重跑12.265秒，通过且0错误。
4. 首次汇总审计在memcheck尚未完成时启动，因缺少launch.json失败；没有产生成功审计。旧执行版和日志保留，最终待有效memcheck完成后重新核验六解及全部GPU记录，输出tables_final/AUDIT。早期空tables目录保留。

实际5个GPU Job：正例、预期缺秩拒绝、Mixed、无效沙箱memcheck、有效memcheck；失败不是数值数据样本，不混入成本中位数。没有结束外部应用或改电源。

## 复现与保存

在本工作树用 `E:/Anaconda/envs/DL/python.exe -X utf8 -B`，主要入口：

```text
tools/bench/run_qrfree_cpu.py
tools/bench/prepare_qrfree_gpu.py
tools/local/build_qrfree_probe.py --label qrfree_gpu_v2_20261016
tools/local/run_qrfree_probe.py --name fixture_positive --input fixture_positive_inputs
tools/bench/validate_qrfree_gpu.py --name fixture_positive
tools/local/run_qrfree_probe.py --name fixture_negative --input fixture_negative_inputs --expected-rank-rejection
tools/local/run_qrfree_probe.py --name mixed_cost --input inputs
tools/bench/validate_qrfree_gpu.py --name mixed_cost
tools/local/run_qrfree_probe.py --name fixture_memcheck_v2 --input fixture_positive_inputs --memcheck
tools/bench/validate_qrfree_gpu.py --name fixture_memcheck_v2
tools/bench/qrfree_audit_export.py
```

入口拒绝覆盖已有批次/构建/结果；新轮需新名字/目录和身份。编译及memcheck使用沙箱外权限；测试命令为 `-m unittest discover -s tools/bench -p '*test.py' -v`，该测试进程设置工作区TEMP/TMP。日志在 `build/qrfree_*`，原始数据在 `runs/innovation_p5_qrfree_20261016`，每场保存解/曲线/条件数和每次GPU样本均保留。

本轮 `DELIVERY_SEAL.json` 封存源码、实际构建/输入/输出身份，并把父封存中的可变TODO映射到先行保存的字节一致副本；不纳入仍在写入的seal日志，封存后重新计算全部hash。旧源、旧结果、冻结baseline及历史报告保持原字节。
