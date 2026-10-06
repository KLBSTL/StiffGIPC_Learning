# 跨帧状态拆分实验：v31（2026-10-01）

## 结果

本轮完成 v31 实现、构建、固定输入检查、18 次有效仿真和八组完整路径检查。**跨帧保留 C 与 lambda、重置 gamma 与摩擦历史，是本次通过独立 CCD 的组中较好的工作量候选**：相对全部重置的 TOI，总方向求解 762 → 560（减少 26.5%），CG 内部迭代 31,229 → 23,560（减少 24.6%）。这是单次诊断结果，尚未证明可重复的性能收益。

只保留 C 的外层数约减半，但第 58 帧发生很长的内层求解，使总求解数不降、CG 工作量增加到约 4.25 倍。**不能用外层次数单独判断速度。**

“C + 独立摩擦历史”虽然方向数最低，但出现 9 个独立 CCD 保守标记，零标记门槛未过，应排除出加速候选。其余七组全部接受路径零标记、零四面体翻转。

所有组的布料物理质量仍未通过建议的 1% 门槛，共同严格参考也未充分收敛。开启诊断和子步导出的时间不用于正式加速比；**相对官方 base 的质量匹配加速比仍为 N/A**。默认配置保持原有 coupled 摩擦及每帧清空状态。

## 1. 实现内容和身份

工作目录：`E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929`。本轮未修改其他对话的目录，未使用 AutoDL。

- 版本：`v31-independent-warm-histories`。
- 553 个源文件/资源最终散列一致。
- `source_digest`：`cda285a54f71829ac45d00b2257f96479559e8e2f8b41cd34b8f9dee92053097`。
- 有效 v31 二进制 SHA256：`7ff7e39ce10b3efa3f229ac6c2af707bb0f11da12321100df69301f297dcd9eb`。
- v30 的两个变更前源码、runner 和 manifest 保存在 `research/v30_before_state_split`；结合未变文件，553 项 v30 源码可重建并通过散列复核。v30 二进制未变。

新增独立帧初始开关：C、AL lambda、AL gamma、摩擦历史。保留 lambda/gamma/摩擦历史需要同时保留接触身份 C。

### 摩擦为何需要独立状态

原实现每次 outer 通过 `lambda × gamma` 生成摩擦力代理；只清空一次缓存，会在下次 outer 被重新生成。v31 的可选 `independent` 分支增加 `friction_lambda/friction_gamma`，使用同一个当前 trial 约束值及罚参数，执行同样形式的 slack/乘子更新，但其帧初始继承独立控制。该代理不进入 AL 接触目标中的 normal lambda/gamma，进入现有摩擦计算。

这是用于分离影响的**实现消融**，不是论文摩擦算法复现声明。改变摩擦历史仍会改变 trial，随后改变 normal 求解和活动集，不能当作轨迹固定的因果实验。原 `coupled` 路径保留为默认；原 `--persist-contacts` 行为作为 legacy 对照。

每帧记录实际策略、继承接触数、lambda 总和/正值数、gamma 衰减数及摩擦代理总和。对全部重置和全部独立继承两组，逐帧初始及逐 outer 的独立代理/AL 标量差均为 **0**，证明在这两种边界配置中新增历史未自行偏离。它不证明跨运行长轨迹等价。

## 2. 构建与验证过程

初次增量构建在公共 `ToiState` 布局变更后，只重编了 `toi_solver.cu`。两次启动均在首帧前的设备拷贝触发 `invalid argument`。失败二进制、组件报告及日志保存于 `builds/v31_incremental_failed`，两次失败运行保存在原 `local_v31_warm_*_smoke2` 目录，未覆盖。

随后执行 Release `--clean-first`，完整重编 32 个 CUDA 单元；最后只增加只读标量一致性日志，再构建该源文件。新二进制的两次启动检查通过。这与旧目标文件混用公共布局的解释一致，初次失败不计为场景物理求解失败。

| 验证 | 结果 |
|---|---|
| 原 GPU 接触/AL 检查 | 6 项通过 |
| 固定 PT 输入：独立 warm 状态、AL 能量、实际 GPU 摩擦代理、乘子更新 | 7 项通过 |
| 体积尺度/极值检查 | 17 项通过 |
| 全重置 coupled / independent，2 帧启动对照 | 两次完成，最大顶点差 `8.14e-13 m` |
| 八种状态策略，30 帧 | 八次完成，实际开关/帧初始状态核对通过 |
| 八种状态策略，60 帧 | 八次完成，有限状态，无 PCG/外层上限 |
| 源码、二进制、runner、初态/拓扑/质量/边界、所有帧与 outer 导出 | 复核通过 |

有效运行前缀为 `local_v31r2_warm`。r2 是构建修复后的新运行目录，**不是第二次统计重复**。每种 30/60 帧策略各一次。仿真串行，独立 CPU CCD 最多并行两项；桌面图形程序存在，本轮无独占 GPU 性能资格。

## 3. 场景与八组策略

本机 RTX 3070 Laptop 8 GiB、CUDA 13、Release / SM86。场景 `cloth_sphere7_l`：3,084 顶点，可变形布料 2,601 顶点，固定 ABD 球 483 顶点。共同 `dt=0.01 s`、Newton/累计 TOI epsilon `0.01`、PCG tol `1e-4`、罚参数模式 diagonal、suite=0（host，无 Graph）。60 帧对应 0.60 s；接触诊断从第 22 帧开始。

“保留/重置”均指帧初始状态，帧内仍执行更新。

| 组 | C | lambda | gamma | 摩擦历史 | 模式 |
|---|---|---|---|---|---|
| cold | 重置 | 重置 | 重置 | 重置 | independent |
| legacy | 保留 | 保留 | 保留 | 随 AL | 原 coupled |
| C only | 保留 | 重置 | 重置 | 重置 | independent |
| C + lambda | 保留 | 保留 | 重置 | 重置 | independent |
| C + gamma | 保留 | 重置 | 保留 | 重置 | independent |
| C + lambda + gamma | 保留 | 保留 | 保留 | 重置 | independent |
| C + friction | 保留 | 重置 | 重置 | 保留 | independent |
| all independent | 保留 | 保留 | 保留 | 保留 | independent |

同一组内 gamma 的重置同时影响 AL 权重和当前删除规则。因此本轮拆开了帧初始标量，却没有把 gamma 的权重与接触删除角色分开。

## 4. 60 帧工作量

一次“方向求解”为一个全局线性系统/PCG 求解；CG 内部迭代另计。方向数 = outer 数 + 额外 inner 方向。

| 组 | outer | 总方向 | 额外 inner | CG 迭代 | 首五轮 alpha 中位数¹ | 诊断时间/s² |
|---|---:|---:|---:|---:|---:|---:|
| cold | 660 | 762 | 102 | 31,229 | 0.00845 | 20.067 |
| legacy | 600 | 899 | 299 | 40,285 | 0.25495 | 18.665 |
| C only | 341 | 769 | 428 | 132,643 | 0.41610 | 30.903 |
| C + lambda | 352 | 560 | 208 | 23,560 | 0.34878 | 11.272 |
| C + gamma | 556 | 829 | 273 | 35,542 | 0.22703 | 16.949 |
| C + lambda + gamma | 545 | 909 | 364 | 51,735 | 0.23445 | 20.162 |
| C + friction | 369 | 507 | 138 | 23,034 | 0.35953 | 12.039 |
| all independent | 527 | 920 | 393 | 45,862 | 0.26492 | 18.989 |

¹ 第 22–60 帧各帧前五次 outer 的完整 safe CCD alpha，非能量线搜索 r。² 包含 blocker 读回、日志和子步导出，只记录时间，不求正式收益比。

### C only 的反例

30 帧时 C only 的 outer 为 114（cold 为 156），总方向 132（cold 为 160），看起来有利。延长到 60 帧后，第 **58 帧、outer 1** 一次 inner 达 **248 次**；整帧 254 次方向、89,862 次 CG、978 次线搜索回溯。该 outer 最后安全 alpha=1，也不能抵消此前大量求解。

这使 C only 的总 CG 工作量为 cold 的约 4.25 倍。C + lambda 本次最大 inner 为 18，没有出现同量级长循环。不过各组 trial 轨迹不同，尚未固定输入证明 lambda 继承是唯一原因。

### gamma 和活动集成本

| 组 | 最大已发布接触数 | 加入次数，非唯一接触数 | 删除次数 |
|---|---:|---:|---:|
| cold | 4,653 | 128,618 | 0 |
| legacy | 2,491 | 10,462 | 9,853 |
| C only | 3,043 | 3,043 | 0 |
| C + lambda | 3,338 | 3,338 | 0 |
| C + gamma | 1,703 | 8,869 | 7,958 |
| C + lambda + gamma | 1,498 | 8,007 | 7,414 |
| C + friction | 2,965 | 2,965 | 0 |
| all independent | 1,584 | 7,260 | 6,792 |

保留 C、每帧重置 gamma 的三组均未删除接触，C 持续增长；保留 gamma 的组开始删除旧接触，却未在本次实验中给出更好的总工作量。根据源码，gamma 既控制约束权重，也控制淘汰阈值，需进一步分离，不能将重置 gamma 的短测结果直接作为长期默认策略。

![工作量、接触与布料质量，失败组已标记](figures/TOI_WARM_HISTORY_V31.png)

## 5. 阻挡接触和数值检查

每个受 CCD 限制的 outer 取一个实际全局最小接触分类；并列时只代表其中一个，不是所有接触的统计。

| 组 | trial 未包含，更新后仍缺失 | 求解后才加入 | 已包含但约束违反 | 已包含、线性满足但几何限步 |
|---|---:|---:|---:|---:|
| cold | 182 | 283 | 97 | 40 |
| legacy | 193 | 288 | 32 | 17 |
| C only | 86 | 83 | 23 | 25 |
| C + lambda | 79 | 81 | 30 | 25 |
| C + gamma | 176 | 273 | 32 | 4 |
| C + lambda + gamma | 191 | 233 | 36 | 16 |
| C + friction | 85 | 87 | 44 | 20 |
| all independent | 169 | 243 | 28 | 13 |

C + lambda 的未参加当轮求解次数从 cold 的 465 降到 160，但比例仍为 160/215 = **74.4%**；cold 为 465/602 = 77.2%。继承改善了 safe 推进和绝对工作量，尚未解决动态阻挡接触进入求解不及时的问题。

### 两个探针未通过项

分析最初在 `1e-8` 系数误差门槛处停止。核查发现只有两条超限记录，原门槛继续作为未通过项保留，未放宽：

- cold，第 59 帧 outer 13，PT `[1486,1515,1516,1517]`：safe unsigned 距离 `9.45e-10 m`，CPU 重建 vs GPU 平面系数最大差 `5.67e-8`。
- C + friction，第 54 帧 outer 11，PT `[2406,1863,1870,1871]`：距离 `5.55e-11 m`，系数差 `1.63e-6`。

它们的距离极小，单位法线归一化会放大端点运算差，和观测量级相符；这是数值敏感性的解释，尚未证明所有偏差来源。相关 active 接触的分类和 multiplier 核对使用**实际 GPU 平面**，不使用重建探针替代。

全部组的 lambda 更新误差最大约 `6.6e-17`（相对分母至少为 1），slack 最大差 `3.47e-18 m`，gamma 更新差为 0。其余六组的探针系数检查通过。不能因此宣称全局 KKT 或 PCG 真残差已验收。

## 6. 完整 CCD 和首个标记

独立 CPU BVH + Tight-Inclusion 使用原 `1e-9` 容差、Stable NH1 材料口径，检查全部帧初态及每个 outer safe 状态；相邻帧的重复初态段计入路径数量。

| 组 | 路径数 | 保守碰撞标记 | 四面体翻转 | 零标记门槛 |
|---|---:|---:|---:|---|
| cold | 719 | 0 | 0 | 通过 |
| legacy | 659 | 0 | 0 | 通过 |
| C only | 400 | 0 | 0 | 通过 |
| C + lambda | 411 | 0 | 0 | 通过 |
| C + gamma | 615 | 0 | 0 | 通过 |
| C + lambda + gamma | 604 | 0 | 0 | 通过 |
| C + friction | 428 | **9** | 0 | **未通过** |
| all independent | 586 | 0 | 0 | 通过 |

共检查 4,422 段。首个标记定位到 C + friction 的第 **54 帧**：

- `safe_0053_0009.bin → safe_0053_0010.bin`，path index 376。
- vertex–face `[2406,1871,1863,1870]`，全为布料顶点；表面范围已核对。
- 独立 CCD 返回 TOI `0.9999957085`，接近段末。
- 对实际 binary64 端点做 60 位 Decimal 计算：投影均在三角形内，端点 unsigned 平面距离 `1.3665e-9 → 2.7461e-10 m`；三角形两倍面积约 `9.85e-4 → 9.74e-4 m²`，不是面积接近零的三角形。

该 canonical PT 与两轮后的 CPU 探针超限是同一接触。端点距离已低于独立 CCD 容差，提示过于接近数值精度范围；端点同侧**不能证明整条变形路径无穿越**。9 个标记未被删除，未降低容差重测，没有把它们改记为通过。这里只能确认原安全门槛失败，尚不能把每个保守标记都解释为确定的实际穿透。

## 7. 布料物理诊断

使用布料单独质量 RMS；固定 ABD 球按场景锁定语义和拓扑排除，避免球的大质量稀释误差。共同时间 0–0.60 s，比较已有严格 base dt/4。

| 组 | 最大位置 RMS / 初始布料尺度 | 末态速度 RMS 差 / m·s⁻¹ | 全程最大局部边伸长比 |
|---|---:|---:|---:|
| cold | 5.301% | 0.632 | 1.401 |
| legacy | 3.488% | 0.480 | 1.152 |
| C only | 3.587% | 0.488 | 1.102 |
| C + lambda | 3.904% | 0.508 | 1.102 |
| C + gamma | 3.751% | 0.491 | 1.150 |
| C + lambda + gamma | 3.425% | 0.432 | 1.102 |
| C + friction | 3.536% | 0.495 | 1.102 |
| all independent | 3.601% | 0.471 | 1.102 |

全部未过建议 1% 位置门槛；参考 dt/2→dt/4 自身最大布料位置差 **2.333%**，末态速度差 **0.576 m/s**，不能作为精确真值。legacy 和 all-independent 的状态更新形式等价且独立标量自检为零，但总方向/outer 仍不同，延续已知同后端波动；当前一次对照不能确定每个效果的统计可靠性。

![九组实际导出几何，同一时间、视角和尺度](figures/TOI_WARM_HISTORY_V31_GEOMETRY.png)

## 8. 下一步

1. 以 **C + lambda** 为受控诊断候选，先固定输入复查并重复短窗/60 帧，确认工作量收益和质量偏差可重复；不设为默认。
2. 分离 gamma 的权重与接触存续年龄/删除角色，处理当前 C + lambda 没有接触淘汰的问题。优先保护 CCD 确认的 blocker，重放上一 trial 的候选筛选，避免保留大量无效约束。
3. 针对 C only 第 58 帧长 inner 固定状态重放，再评估与已批准 `velocity_tol=1.0` 的组合；不因减少 outer 就忽略 CG、回溯和布料质量。
4. 对近零 PT 建立受控安全案例，检查 ACCD 与独立 CCD 的裕量和法线稳定性；C + friction 分支保留为失败证据。
5. 共同参考与重复稳定性通过后，再关闭诊断做交错计时、组合 Graph，输出相对官方 base 的质量匹配收益。

## 9. 复现和文件

关键命令在本任务目录内执行，详细日志为 `builds/v31_*.log`。公共状态结构变动后须完整重编：

```powershell
& 'D:/computer/cmake/bin/cmake.exe' --build builds/local-fused --target gipc --config Release --clean-first --parallel 2
# 将同次构建的 gipc.exe、freeglut.dll、glew32.dll 放到 builds/local-fused-v31/Release
$env:GIPC_VALIDATE_COMPONENTS = Join-Path (Get-Location).Path 'builds/v31_toi_components.json'
& 'builds/local-fused-v31/Release/gipc.exe'
Remove-Item Env:GIPC_VALIDATE_COMPONENTS
& 'E:/Anaconda/envs/DL/python.exe' tools/run_warm_ablation.py --stage smoke
& 'E:/Anaconda/envs/DL/python.exe' tools/run_warm_ablation.py --stage pilot
& 'E:/Anaconda/envs/DL/python.exe' tools/report_warm_ablation.py --stage pilot
& 'E:/Anaconda/envs/DL/python.exe' tools/run_warm_ablation.py --stage diagnose
& 'E:/Anaconda/envs/DL/python.exe' tools/validate_warm_paths.py
& 'E:/Anaconda/envs/DL/python.exe' tools/report_warm_ablation.py --stage diagnose
& 'E:/Anaconda/envs/DL/python.exe' tools/verify_warm_ablation.py
& 'E:/Anaconda/envs/DL/python.exe' tools/plot_warm_ablation.py
```

运行器复用已完成结果，不覆盖历史。独立验证运行器禁止覆盖现有 CCD JSON；重新统计需新目录/协议。每次精确 CLI、退出码、二进制/runner 散列均在 `runs/local/local_warm_v31r2_20261001/*_status.json` 中。

最终复核 `execution_identity_coverage_verified=true`、18 次完成、553 源文件、组件 `[6,7,17]`、路径 4,422；同时明确 `independent_ccd_all_passed=false`，两个探针超限保留。执行与身份核对通过不等于全部质量门槛通过。

- 协议：`configs/local_warm_v31_20261001.json`。
- 30/60 帧汇总：`reports/TOI_WARM_HISTORY_V31_{PILOT,DIAGNOSE}_20261001.json`。
- 接触明细：`reports/TOI_WARM_HISTORY_V31_DIAGNOSE_20261001.csv`。
- 最终复核：`reports/v31_warm_verification.json`。
- 八组完整 CCD：`reports/v31_*_accepted_ccd.json`。
- 首个 CCD 标记及端点几何：`reports/v31_c_friction_first_ccd.json`、`reports/v31_c_friction_first_path_geometry.json`。
- 科学图件：`reports/figures/TOI_WARM_HISTORY_V31*.{png,pdf}`，已打开检查。
