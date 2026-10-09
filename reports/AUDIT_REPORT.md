# StiffGIPC / TOI / CUDA Graph 审查（2026-10-09，本轮结果）

本轮目标是相对原版 Stiff 的同质量整场速度至少 2×。全部场景目标尚未达到，不把
更换接触模型、更宽停止条件、旧 TOI 的局部改善或单个 kernel 的倍率写成达标。
下列数字来自本机新运行。v3/v4 的三场景完整 120 帧、每项三次配对均完成；
v4 通过回归、真实矩阵与同步验证。完整 TOI 及更严格内层停止实验也已完成。
最新执行组合配对中位倍率为悬挂 2.08432×、固定 1.85142×、混合 1.33158×。

## 选用源码与保留范围

- 从本地 `StiffGIPC_Learning/main` 的 `7f5f069` 建立隔离仓库
  `E:/university_class/ComputerGraphics/GIPC/stiff_speed_audit_20261009`，分支
  `perf/toi-pcg-audit-20261009`。
- 原仓库存在未提交的 PNCG、探针和实验改动，全部保留。没有切换、重置或修改
  原仓库。旁边多个目录是源码快照，不将快照目录名误当 Git 分支。
- `baseline/` 和原版 Stiff 可执行文件保持冻结。首轮还使用保存的历史程序，
  将它的原生文件逐一与隔离源码比较后才运行；随后建立了本轮全量干净编译。
- 每个 session 保留请求、实际配置、编译/源码/二进制身份、资源观测、日志、
  状态导出和失败记录。GPU 运行串行，使用原工作区共享锁和 Windows Job。

## 已核实的原因

### 1. TOI 的工作量和预条件器成本是两个独立问题

同一混合场景 `bunny_cloth_bunny_l`，35 帧，dt=.01、rho=1e-4：

| 配置 | solver 秒 | Newton 方向 | PCG 总量 | 每方向 PCG | FEM 最小 J |
|---|---:|---:|---:|---:|---:|
| 原 Stiff | 5.4601 | 142 | 3649 | 25.70 | .497869 |
| 活动 host | 5.6143 | 142 | 3658 | 25.76 | .497869 |
| Graph | 5.2118 | 142 | 3650 | 25.70 | .497869 |
| Graph + 执行组件 | 4.8228 | 142 | 3655 | 25.74 | .497869 |
| TOI + FP64 factor-inverse / serial restrict | 13.8039 | 192 | 6085 | 31.69 | -.042005 |

TOI 总 PCG 增加约 67%，既来自方向增加，也来自每个方向更难求解。不能只
比较总 PCG，或把不同预条件器的单次成本直接归因于 TOI 法向项。

独立短对照中，TOI + legacy MAS 为 5.6548 秒、185 方向、5943 PCG，最小
J=.069899。现有 warp 限制算子的另一次整场 TOI 对照为 8.6622 秒、191 方向、
6069 PCG，最小 J=.235540。它们仍偏离 Stiff 材料状态；没有采用这些 TOI 配置。
FP64 serial 的一次试验出现倒置，有限数值或完整 CCD 不能替代材料质量判断。

初始 TOI 系统的分离算子测试测得 legacy MAS 64.72 微秒，FP64
factor-inverse 1265.96 微秒；SpMV 分别 62.48/69.05 微秒。两次矩阵/RHS
的相对 L2 差异仅 6.48e-15/4.97e-15，但不是字节相同，所以不把这组跨运行
测量称为严格同矩阵因果对照。默认 rho 下二者均 6 次 PCG，真实未预条件相对
残差约 .001245。rho 阈值不是未预条件残差范数阈值。

更严格的现有保护性重放，在**同一个 A、b、Cholesky 因子和限制映射**上交错
比较 serial/warp restrict。二者均 6 次 PCG，解差约 1e-15，Graph solve 从
约 9.0 ms 降到约 3.4 ms。系统/RHS和主解恢复检查通过。该约 2.6× 是固定
PCG 子问题的诊断倍率，不是 Stiff 整场倍率。

### 2. Graph 捕获频繁，但已测捕获成本不足以解释整场差距

35 帧混合场景约 94 次捕获/实例化，CPU 总计约 9–10 ms，而整场数秒。
每次接触布局/缓冲区身份变化都可能合法地使 Graph 失效。不能用捕获次数
直接推断它是主瓶颈，也没有重复历史已失败的 K4 chunk 方案。

host/Graph 才能隔离同一活动程序的 Graph 收益；原 Stiff/Graph 比值还包括
活动实现差异。材料、完整 CCD、rho=1e-4、legacy 停止阈值 .01/min6 没有改动。

### 3. 线性阶段包含很重的数值准备

混合场景 Graph+组件在选定 6 帧（24–26、33–35）中，41 次线性构建的
CUDA event 包络约为：矩阵转换 131.18 ms、MAS 准备 189.06 ms、其中
legacy factor 109.02 ms、Graph 执行 155.02 ms。装配包络 351.64 ms。
这些区间是嵌套包含时间，带提交间隙，**不相加为 kernel 活跃时间**；没有
将缺失的 TOI phase 计时解释成零工作。

历史结构探针的 ordered raw 全相邻命中率约 8%，且 Converter 的数值
gather/reduce 占所采单帧 GPU 转换工作约 72–74%。因此没有凭接触总量或
canonical keys 相同复用旧数值 Hessian、RHS、MAS 因子，或直接重启旧符号缓存。

## 本轮实际改动

### FEM 直接稀疏回拉（已编译、短跑验证）

三角形和四面体的变形梯度 Jacobian 只有同分量的 2/3 个系数非零。
直接计算上三角 3×3 块，替代稠密 Jacobian、中间积和完整 9×9/12×12
位置 Hessian。材料梯度、材料 Hessian 投影、dt² 与面积/体积系数保持原公式。
保留旧路径，用 `GIPC_FEM_DIRECT_PULLBACK=1` 单独选择；默认关闭。

独立测试用有限差分重建 Ds*Dm^-1 的 Jacobian，并比较真实材料核的梯度、
投影 Hessian、块索引/转置和偏移。覆盖 1/127/128/129/257 个元素，刚度
1e2/1e5/1e8、倒置四面体及重排顶点。30 个 GPU 对照通过，最大缩放误差
2.74646e-12。原有 17 项回归测试在干净基线上全部通过。

35 帧、每项三次，原 Stiff/旧组合/直接回拉的 solver 秒中位数：

| 场景 | Stiff | 旧组合 | 直接回拉组合 |
|---|---:|---:|---:|
| cloth_hang_l | 1.57850 | .92002 | .88759 |
| cloth_fixed_bunny_l | 2.96099 | 2.24013 | 2.13980 |
| bunny_cloth_bunny_l | 5.15237 | 4.67180 | 4.23999 |

混合场景装配从约 1.24 秒降到 .85 秒，整场组合约再快 10%。方向数保持
142，PCG 3650–3656，最小 J 约 .497869。**不把装配局部倍率称为整场 2×。**
这批 v1 运行期间新增控制器选项，Python 进程仍使用启动时载入的控制代码；
请求和原生配置有完整记录，但控制器磁盘身份没有全程冻结。因此它只作探索
结果，最终采用当前静态控制器与 v3 程序重新配对，不用于性能认证。

### legacy MAS 主元同步优化（v3 数值/同步与整场计时已完成）

`__inverse6_P96x96` 的名字与当前实际尺寸不同：BANKSIZE=16，所以每个矩阵
是 48×48，生产 block 有 96 线程，处理两个矩阵。其消去循环的行更新只读取
固定的主元行和列，且每个线程写自己列的不同非主元行；下一主元的块同步
已经连接全部写入。删除行内额外同步，每矩阵少 48*47=2256 次 barrier。
保留其余主元同步、FP64 中间算术与 FP32 存储、数据布局和投影。

原核仍保留，`GIPC_MAS_PIVOT_ONLY_SYNC=1` 单独选择。新夹具比较完全相同
输入的两个生产核实例，覆盖多个矩阵、奇数尾块、SPD、零对角 padding 与
宽尺度；将验证逐字节输出、输入不变和并发安全。CUDA 对跨线程内存依赖的
同步要求见 [NVIDIA CUDA 13.0 指南](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)。
上述删同步安全性是针对本核依赖关系的分析。v3 夹具的 201,960 个 FP32
存储项与原核逐位一致，输入逐位不变；两个核的生产布局和奇数尾块通过
racecheck（0 hazards、0 errors、0 warnings）与 synccheck（0 errors）。
这两个检查仅覆盖该核的固定夹具，不声称整个模拟器通过并发审计。

夹具 1/3/32/129 矩阵批次、每项 3 次交替、每次 8 个 replay 的 kernel
时间中位数（原核/候选，毫秒）分别为 .320736/.238848、.349956/.256768、
.343680/.204160、.398904/.273024。局部约 1.3–1.7×，不作为整场倍率。

首次 v2 全量编译发现旧 Cholesky 夹具仍调用原接口；为模板保留默认原路径后
重新建立 v3 全量编译，失败目录与日志保留，没有把失败 build 作为程序身份。
初次 sanitizer 调用误用了此安装版本不支持的 `summary` report 选项，工具
在运行 kernel 之前退出；记录保留于 `runs/audit_pivot_v3_racecheck`。
改用其帮助文档列出的 `analysis`，新会话完成上述 racecheck，没有覆盖失败记录。

## 没有认定为缺陷的实现与覆盖边界

继续逐行审查还确认了一个接口缺陷：Stable Neo-Hookean 1 的投影函数通过
`H` 输出 Hessian，却声明返回 Matrix9x9 并从末尾落出（NVCC #940）。改为
void，与唯一调用方和另一材料分支的写法一致。此修正晚于 v3 冻结，属于
下一次编译；不把 v3 计时当作已包含该修正，也不预先声称它有速度收益。

- 接触发布之前已经按类型/key 排序，自接触/地面数组的顺序没有混排缺陷。
- 默认固定 slack 子问题保留正 slack、零乘子的曲率，属于当前方法语义。
  历史 slack 消元方案已在 27/34 帧超时，本轮没有无证据重试。
- generalized ABD 与 world-space mu 的混用问题在主分支已经修正为完整 ABD
  compliance 回拉。没有把未修正旧 TOI 作为当前基线。
- Robust 的 Kmin=1 并非简单的漏写 min6；作者的方法允许这个取值，见
  [原论文 §4.4](https://arxiv.org/html/2512.12151v3)。但它不保证与 Stiff 的
  离散结果同质量，因此 TOI 状态仍需独立比较。
- PCG 的 previous-rho 停止时序和异常 guard 保持原实现；没有为漂亮倍率
  放宽精度，或改回更严格 rho 人为增加 Stiff 的工作。

已沿调用链检查 TOI contact/slack/dual、world penalty、trial 与 safe 状态、
线搜索、完整 CCD、累计停止、PCG host/Graph、矩阵转换、FEM/ABD 分区、MAS
准备与 action，以及修改涉及的 FEM 回拉公式。暂未声称扫描并验证了所有
renderer、网格导入、第三方库和无关诊断的每一行。源码审查、组件测试、短跑、
完整场景和独立物理验证是不同证据，不能相互替代。

## 静态控制器的完整 120 帧结果（v3）

用户明确选择继续使用当前安静电源模式；没有切换电源计划。前后观测保存在
`AUDIT_POWER_SCHEME_BEFORE_120.txt` 和 `AUDIT_POWER_SCHEME_AFTER_120.txt`。
本轮每个场景三个 repeat，原 Stiff、旧执行组合、两项新优化组合交错运行，
27 个任务全部完成，无资源中止、PCG breakdown、线性或非线性迭代上限。

| 场景 | Stiff 秒中位数 | 旧组合秒中位数 | 新组合秒中位数 | 配对 Stiff/新组合中位数 |
|---|---:|---:|---:|---:|
| cloth_hang_l | 8.80043 | 4.69901 | 4.35281 | 2.02178× |
| cloth_fixed_bunny_l | 28.31722 | 16.09759 | 15.48115 | 1.78932× |
| bunny_cloth_bunny_l | 45.31133 | 38.69062 | 34.08328 | 1.32943× |

倍率是每个 repeat 内比值的中位数，并非两个时间中位数的比值。悬挂场景
三个比值为 2.12959、1.95501、2.02178；因此连悬挂场景也不能说每次都超过
2×。原始账本与逐帧状态比较：`runs/audit_v3_paired_120/ledger.json`、
`comparisons.json`。短轮拆分两个组件的 45 个任务另存
`runs/audit_v3_components_35`，没有用短轮数字代替完整 120 帧结论。

执行优化没有稳定地减少 PCG 工作，符合其设计。混合场景 Stiff 为
24130/23782/24172 次 PCG，新组合为 24071/23830/24129；固定场景也只是
随轨迹分支波动。这与 TOI 增加方向和每方向 PCG 的问题有区别。

质量尚未认证：初始拓扑/位置/质量/边界输入匹配，所有导出有限，但原版没有
实际速度导出；没有用位置差分伪造速度。原版自己的 120 帧重复也有明显接触
轨迹差异。固定场景新旧布料最大逐帧 RMS 差 2.30–3.89 cm；混合布料
1.21–1.84 cm、FEM 2.90–3.95 mm；悬挂布料为 .93–3.58 微米。
执行优化保持数学公式，并不意味着浮点重排之后轨迹逐位相同。

提前记录的原版观察包络在新的配对中被原版本身突破：固定布料最大 stretch
达到 1.036523，而原校准上界约 1.035756；混合原版最大 stretch 达
1.293069，而上界约 1.286483。混合新组合最小 J 的一次结果为 -1.11605，
也越过原校准下界 -1.03405。原版本来已有负 J，不能用它为候选任意放宽
质量门槛。保留这些失败，不看过候选后扩大包络宣布通过。

混合场景 repeat 1 的阶段秒数：

| 阶段 | Stiff | 旧组合 | 新组合 |
|---|---:|---:|---:|
| 装配 | 9.3132 | 9.2594 | 6.5445 |
| PCG（包含线性准备） | 18.4471 | 13.5901 | 12.5046 |
| CCD | 4.7971 | 4.2648 | 4.1399 |
| 线搜索 | 11.4781 | 9.9230 | 9.6572 |

这解释了为什么只追求 PCG 次数下降不足以达标：新组合非 PCG 工作约
21.58 秒，原版 2× 的预算只有 22.66 秒，即使其余成本完全不变，PCG 也
必须从 12.50 秒压至约 1.08 秒。必须同时减少线搜索/装配成本或总工作量。
阶段统计属于完整运行的包络；没有当成独占 kernel 活跃时间。

## 对后续方向的实际筛选

从固定布料 frame 57、混合 IPC/TOI frame 34，各捕获三个相邻真实 A、b、
legacy MAS 状态，存于 `runs/audit_v3_adjacent_systems`。诊断重放检查主解和
系统恢复。CPU 筛选只使用之前系统的方向，不使用当前系统的已知解作为起点；
FP32 MAS 重构的 reduction 顺序不同，结果仅作机制筛选，没有 GPU 速度声明。

- 前解的 rank 1/2 Galerkin 起点：固定布料第二/三系统 86→85、66→67 次；
  混合 IPC 19→10、21→7 次，但真实相对残差分别 .000866→.001459、
  .000713→.001628；混合 TOI 74→74、62→50。不能凭次数降低就采用。
- 前系统 rank 2 Ritz 平衡粗空间：固定布料 86→76、66→60；每次还增加
  四个粗空间内积、投影，以及构建 AU 和保存 Lanczos 基的成本。混合 TOI
  74→73、62→52，第三系统真实残差 .012446→.016231。没有稳定质量/成本
  优势，未植入生产代码。独立 SPD 和粗算子对称正定测试通过，仅说明 CPU
  原型实现符合所测公式。结果见 `AUDIT_RECYCLING_REFERENCE.json`、
  `AUDIT_RITZ_REFERENCE.json`，不冒称精确原生回放。
- 进一步测试 rank 4/8/16 正定增量 `M_new=M_old+U(UᵀAU)⁻¹Uᵀ`，使用
  原 `bᵀM_old b` 参考，原加权停止标准在精确算术中不会放宽。固定布料第二
  系统 86→74/75/74，第三系统 66→51/49/49；TOI 首个受约束系统
  74→73/73/74，下一系统 62→50/50/50。收益没有随 rank 增长，且需要
  额外粗空间操作与 setup。混合 IPC 第三系统三项真实残差仍比零起点大。
  `AUDIT_ADDITIVE_REFERENCE.json` 保留独立 SPD/增量半正定测试和所有结果。
  没有把这些 CPU 次数转换成未经测量的 GPU 倍率。
- 新 120 帧成本表揭示不同于历史固定布料短窗的混合场景线搜索负担，因而
  检查现有 opt-in 接触池。`runs/audit_v3_mixed_pool_check` 完整验证：640 次
  复用/候选集检查全部通过、对比 6,151,935 对接触、同状态能量与 Armijo 分支
  一致；109 次走原查询回退、没有观测到生产溢出，峰值池内存约 45.4 MiB。
  这次有验证插桩且与 CPU 编译/原型重叠，不用于性能声明。下一轮须关闭验证
  插桩，使用独立配对计时；若成本不改善则继续关闭，不覆盖历史负结果。

现有接触池的独立 v4 无验证插桩轮已完成，三次 120 帧，
`runs/audit_v4_mixed_pool_120`。新组合秒数为 42.9752/41.7729/42.3014，
加池为 39.7321/38.5713/38.2201；配对倍率 1.08162/1.08301/1.10679，
中位 1.08301。线搜索从约 11.8–12.2 秒降至 7.5–7.8 秒，但 CCD、
池准备与工作量变化抵消部分收益。三次运行合同均通过，保持完整查询回退，
不使用无验证时的 `production_overflow_queries=0` 声称已测总溢出为零。
一次最大 stretch 1.298515、最小 J -1.047314，超过提前记录的原版观察
包络；因此保留为有成本收益的 opt-in 实验，默认关闭，未认证物理质量。

## 当前验证与后续有限决策

干净程序：`build/audit_clean_active/windows_manifest.json`，37 个原生 TU。
FEM 程序：`build/audit_direct_v1/windows_manifest.json`，37 个原生 TU，
SHA256 `a2af6147bace0e32d5d16d2aa67064127963305449ea4977d9b443c13b99f930`。
旧源码/新源码的不可变副本保存在 `build/audit_clean_frozen`、
`build/audit_direct_frozen`。运行账本在 `runs/audit_*`。

v3 全量编译 37 个原生 TU，程序 SHA256
`11a1093f0f8ed9d6ef9cd92059282aa7df53ed6885bc0bf1a259697b950454cf`，
冻结于 `build/audit_pivot_frozen/snapshot_identity.json`。19 项 CTest、23 项
CPU 合同检查、MAS racecheck/synccheck 均通过；完整短轮/120 帧数据如上。

v4 全量编译完成，37 个原生 TU、82 条 CUDA 编译命令；程序 SHA256
`42ec2b7145c6e03afae4524871d2ba8a1aa0223546da3cce33faad051dd1ba47`，
不可变源码与程序身份在 `build/audit_corrected_frozen/snapshot_identity.json`。
19 项 CTest 全过；9 个真实系统的 9924 个 MAS 矩阵、12,146,976 个 FP32
存储项逐位一致，输入逐位不变；racecheck 0 hazards/0 errors/0 warnings，
synccheck 0 errors。记录为 `runs/audit_corrected_v4_{ctest,matrix,racecheck,synccheck}`。

编译后仅删除 FEM 文件中一处空行的尾随 CR；与冻结编译输入的全部词元相同，
运行身份仍精确指向上述不可变源码。`git diff --check` 通过。

## v4 最终配对与 TOI 完整结果

用户要求保持安静模式，本轮始终没有修改电源计划；末次观测在
`AUDIT_POWER_SCHEME_AFTER_ALL_RUNS.txt`。布料 12 个任务、混合 9 个任务全部
完成 120 帧，无资源失败、迭代上限或 breakdown，输入比较通过。

| 场景 / 配置 | Stiff 秒中位数 | 候选秒中位数 | 三个配对倍率 | 配对中位倍率 |
|---|---:|---:|---|---:|
| 悬挂 / 两项执行优化 | 11.29259 | 5.41788 | 2.10974 / 2.08432 / 1.96712 | 2.08432× |
| 固定 / 两项执行优化 | 35.93762 | 19.41079 | 1.78566 / 1.98140 / 1.85142 | 1.85142× |
| 混合 / 两项执行优化 | 47.31475 | 41.90208 | 1.33158 / 1.10745 / 1.83435 | 1.33158× |
| 混合 / 再加接触池 | 47.31475 | 36.71904 | 1.44185 / 1.29109 / 1.28856 | 1.29109× |

混合场景有较大计时波动，不能据三个样本认定接触池稳定优于两项执行优化。
尤其不能用候选时间中位数更小就替换配对倍率。采样的 SM 频率中位数均
1905 MHz，但个别运行仍明显快慢不同；当前证据不足以将波动单独归于电源
计划、频率或某一 kernel。共享 WDDM/CPU 负载尚未完全隔离，性能仅作本机
观测，未给出稳定 2× 保证或性能证书。质量观察包络仍有超出项；逐次违反项
见 `AUDIT_FINAL_STATUS.json`，未在看到候选后扩大门槛。

当前最快执行核用于一次**独立完整算法对照**：TOI + legacy MAS、稀疏回拉、
主元同步、Graph、现有复用/批量能量/CCD refit；不与旧昂贵 serial Cholesky
混为同一配置。`runs/audit_v4_toi_120`：

| 算法 | solver 秒 | 方向 | PCG | PCG/方向 | 外层 TOI | 最小 J | p99 stretch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 Stiff | 31.49137 | 750 | 24404 | 32.5387 | — | -1.120397 | 1.092520 |
| TOI 原内层阈值 .05 m/s | 64.07130 | 2064 | 116736 | 56.5581 | 775 | -.739353 | 1.123592 |
| TOI 严格阈值 .005 m/s（单独算法实验） | 72.29209 | 2522 | 135599 | 53.7665 | 762 | -.565797 | 1.121360 |

原设置 TOI 的方向是 Stiff 的 2.752×、每方向 PCG 1.738×，总量 4.783×，
整场慢约 2.035×。Graph 捕获仅约 60 ms，占总计时不足 .1%；不是这组工作
量增长的解释。TOI phase 字段缺少该后端的阶段计时，表中的 0 **不代表零工作**。

775 个内层退出中 650 个走 full_step，方向速度 .0504–4.4756 m/s，
高于小方向阈值 .05。代码的全步速度保护使用 100*阈值，并非没有速度检查；
Robust policy 也确实强制启用小方向退出。为检验更充分内层求解能否减少外层，
通过现有参数收紧到 .005：外层仅少 13 个，方向多 458 个，PCG 多 18863 个，
整场更慢；不采用。默认/严格的安全 alpha 中位数 .41029/.41222，alpha<.1
的次数 97/100，也没有显示全局推进显著改善。不能把全步退出直接当成一个
修改后必然变快的 bug，或将减少倒置数单独当成同质量胜利。

## 有限决定与复现

保留有独立数学/数值证据的 FEM 回拉、MAS 同步和 void 接口修正；前两项
仍为独立 opt-in，便于与原路径对照。保留已有接触池实验入口，默认关闭。
不采用 TOI、更严格内层停止、前解/Ritz/增量粗空间作为默认加速路径。
TOI 的总工作量与预条件条件仍未解决到足以满足目标，不能声称任务的全场景
2× 已完成。后续需要能同时降低 AL 方向数与求解成本的方案，并建立独立的
质量验证；本轮否定了仅删同步、复用少数方向、收紧内层停止就能解决的猜想。

所有运行身份、请求/实际配置、资源样本、完整状态与失败日志保存在对应
`runs/`；`AUDIT_FINAL_STATUS.json` 提供机器可读结论。原 Stiff 无实际速度导出，
没有用位置差分冒充速度；独立 accepted-path CCD 证书及渲染图像复核未完成，
因此未认证物理质量。审查覆盖边界如前，不声称全部第三方/renderer 源码验证。

在本隔离目录使用 PowerShell 7 与 DL Python。以下是本轮实际命令；重复运行
应换全新的 session/build 名称，保留已有记录：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/build_windows.py --kind active --label audit_corrected_v4 --jobs 2
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/audit_validation.py ctest --identity build/audit_corrected_frozen/snapshot_identity.json --session audit_corrected_v4_ctest
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/audit_validation.py matrix --identity build/audit_corrected_frozen/snapshot_identity.json --session audit_corrected_v4_matrix --matrix-input runs/audit_v3_adjacent_systems/real_mas_inputs.bin
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/toi_pcg_audit.py run --session audit_v4_cloth_paired_120 --frozen-identity build/audit_corrected_frozen/snapshot_identity.json --scenes cloth_hang_l cloth_fixed_bunny_l --arms base best --steps 120 --repeats 3 --timeout 180
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/toi_pcg_audit.py run --session audit_v4_mixed_paired_120 --frozen-identity build/audit_corrected_frozen/snapshot_identity.json --scenes bunny_cloth_bunny_l --arms base best pool --steps 120 --repeats 3 --timeout 180
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/toi_pcg_audit.py run --session audit_v4_toi_120 --frozen-identity build/audit_corrected_frozen/snapshot_identity.json --scenes bunny_cloth_bunny_l --arms base toi --steps 120 --repeats 1 --timeout 180 --direct-pullback --pivot-sync --toi-mas legacy
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/local/audit_toi_tolerance.py --identity build/audit_corrected_frozen/snapshot_identity.json --session audit_v4_toi_tol005_120 --tolerance .005
```
