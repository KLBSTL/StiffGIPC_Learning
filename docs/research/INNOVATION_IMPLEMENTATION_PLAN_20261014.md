# StiffGIPC 创新实现计划：先证据与独立原型，再选择主线

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；benchmark-regression
- Origin Mode: plan
- Origin Date: 2026-10-14
- Verification Status: UNVERIFIED（设计计划；没有运行本轮算法实验）
- Version Label: stiff_innovation_plan_v1
- 用户授权：按照所附研究方案另开分支，当前阶段先做计划。

## 1. 分支、起点与本轮边界

研究分支：`research/stiff-innovation-20261014`。

独立工作目录：`E:/university_class/ComputerGraphics/GIPC/stiff_innovation_20261014`。

固定起点：`a4b83c040161eeef6149c93ae6167644fdf19384`，即封存的 AutoDL 观察程序快照。按照用户方案选取此提交，而不是将审计分支尚未提交的原型混入研究基线。

方案原件：[StiffGIPC_a4b83c0_加速与创新性提升_研究方向及验证方案.md](D:/C_files/StiffGIPC_a4b83c0_加速与创新性提升_研究方向及验证方案.md)。SHA256 与源码树身份见 [INNOVATION_BASE_MANIFEST_20261014.json](INNOVATION_BASE_MANIFEST_20261014.json)。原件只作为研究输入，文中的设计建议不视为已实现或已验证结果。

旧审计工作目录和历史工作目录均保留；不迁移、清理或覆盖已有实验。`baseline/` 保持冻结。本轮没有上传 GitHub。此计划只创建文档及任务清单，尚未修改 native、测试脚本、物理参数或默认开关。

下一阶段遵循两个原则：每次只改变一个因果因素；每个原型先独立关闭/开启对照，通过才组合。P1–P5 是候选，不预设五项全部进入生产版本，也不预先承诺 2×。

## 2. 需要检验的假设和优先级

| 候选 | 要回答的问题 | 第一版范围 | 后续真正可能新增的机制 | 优先级 |
|---|---|---|---|---|
| P2 | 合并读回与减少提交间隙，是否在相同验收下节省完整线搜索时间？ | P2-A：BVH host scene 消费审计、ABD 能量结果合并、依赖核查 | 动态接触、容量和失败恢复下的 GPU trial 提交协议 | 第一工程原型 |
| P3 | 拆分 DCD EE 遍历与 FP64 分类后，队列成本是否小于资源压力收益？ | 原融合 F 对固定两阶段 W；先 EE，VF 保持原路径 | 有界队列、重尾任务粒度与在线 F/W 选择 | 与 P2 独立筛选 |
| P5 | 少量真实 FEM 慢模式是否能在相同线性精度下减少完整 solve 成本？ | 固定系统上的理想 rank 8/16/32 与邻系统迁移 | 接触演化下的模式筛选、数值更新和有成本依据的有效期 | 数值侧先验证潜力 |
| P1 | 线性误差、模型缺陷和 CCD 限步的分类，是否优于已有 EW？ | 影子观测与 eta=.01/.03/.1 的同状态反事实 | 由实际接受进展驱动的有限动作成本控制器 | 随 P5 诊断，后实现控制 |
| P4 | 保守包络能够复用多久，额外分类和证书成本是否抵得过查询？ | 同方向 trial 的旁路证书，原查询仍执行 | 候选完整性与路径安全分别管理的有效期 | 后置，需独立成本证据 |

完整 conditional PCG Graph K1、device 容差、现有 EW forcing、标量历史预算已在起点代码中存在，属于基线或强对照。固定两阶段、balanced 预条件公式、POD/Ritz、skin 缓存本身也不作为新增原创性主张。进入主线前，逐项核读方案列出的论文全文/作者代码及适用版本，记录已有机制与拟议差异；本计划没有新做穷尽性文献检索。

选择顺序是：**M0 证据底座 → M1 P2-A → M2 P3-F/W → M3 P5/P1 小规模数值诊断 → M4 决策 → 仅扩大胜出的模块 → 必要组合与正式验证。** 各里程碑按依赖推进，GPU 实验串行；不同时堆叠修改。

## 3. 基线口径与最新证据的影响

| 身份 | 精确定义 | 用途 |
|---|---|---|
| B0 | 当前提交的冻结 `baseline/` 原版，同机器独立构建；保留其原生停止规则并公开实际配置 | 最终“相对原 StiffGIPC”的加速声明 |
| B1-LEGACY | a4b83c0 当前最快执行组合 + 原累计 TOI；forcing off；新增候选全关 | 执行优化和数值增量的主要基线 |
| B1-MOV | 同一 B1，仅改为 movement-only | 停止规则消融，不能代称 B0 |
| B1-DIRECT | 同一 B1 + 累计 TOI + 当前直接残差 gate | 历史预算的强简单对照 |
| B1-COMP | 同一 B1 + 补偿累计 TOI + residual gate | 与 direct 的同前缀/同质量比较 |
| B1-EW | 同一 B1-LEGACY + 当前 adaptive forcing | P1 必须击败的已有实现 |
| R | 分别收紧内层和外层并检查自身重复/收敛性的数值参考 | 质量辨别，不冒称精确物理解 |

B0 已有累计退出，不能用“关闭累计的 Host 版本”代替未经修改的原版。B0 的 scene 导入、时间步和输出接口若不能直接与 B1 对齐，先写出适配边界和身份，不悄悄修改冻结求解器。适配仅作用于外围输入/观察，且单独审核；无法一致配置时，对应原版加速声明保持 pending。

B1 最快执行组合包括 Graph、FullCCD refit、energy batch/reuse、离散 BVH refit、direct FEM、legacy MAS、pivot-only sync；增量收益不能全部归为 Graph。

最新本机五场景报告：[REPORT.md](../../../stiff_speed_audit_20261009/reports/five_arm_retest_20261014/REPORT.md)。该路径从当前计划目录指向外部只读历史证据；不将旧测量重标成本分支的新测量。

| 最新复测观察 | 对实施计划的约束 |
|---|---|
| FAST/BASE-TOI：Boxes 1.069×、Mixed 1.265×、Bunny 1.973×、Table 1.373×、Hang 1.939× | 五场景尚无这一口径的 2×；最终仍需 B0 同身份、同质量实测 |
| 第100帧共同输入，Bunny/Mixed/Table/Hang 仅 1–3 次方向求解 | 不能只选第100帧检验 TOI/残差控制；补选真正激活累计退出的事件状态 |
| Boxes 参考自身轨迹 RMS 分歧约1603 mm，FAST 对参考约1762 mm | 全轨迹位置差不足以独立归因；先靠同起点单步、局部物理量和更可信参考 |
| Mixed 所有旧对照含较严参考均出现非正几何 J，第一轮首次第13帧 | M0 先核查材料定义、几何指标、合法变形和装配；不能只给 TOI 定责 |
| 旧参考仅将 rho 收紧到1e-6，外层 movement 仍为.01 | 不足以证明非线性求解精度；R 必须同时研究外层收敛 |

这些是既有报告事实，不是本轮重新测量。旧 AutoDL 三场景与本机五场景分别统计，硬件、二进制、记录方式、统计公式及电源来源不混用。

## 4. M0：先建立可比较的数据底座

### M0.1 构建、配置和观察身份

固定 B0/B1 输入 SHA256、材料参数、完整编译定义、编译器/驱动/CUDA、目标 SM、依赖、exe SHA256、完整环境及 resolved config。新代码或配置必须使用新 build seal 与 session，禁止覆盖旧结果。

本机当前 PATH 上的 nvcc 是 CUDA 13.0；原 AutoDL 快照记录 CUDA 12.8.93、SM89，本机旧构建为 VS2022、SM86。因此先冻结本机实际工具链，B0/B1/候选同工具链重建；不能把本机结果说成12.8/4090复现。复用依赖路径可行，复用旧 CMakeCache/旧 objects 不可行。检查当前显卡和空闲资源后设定正确 SM。

主场景固定以下 low 输入：

| Scene | scene key | 角色 |
|---|---|---|
| Boxes | boxes1920_cloth_l | 密集接触、线搜索/CCD/装配成本；质量解释须保留参考分歧 |
| Mixed | bunny_cloth_bunny_l | 混合 ABD/FEM 与接触变化；材料/J 检查前不得认证物理质量 |
| Bunny | cloth_fixed_bunny_l | PCG 长尾、P5/P1 的首选数值诊断 |
| Table | cloth_table_l | 未用于候选规则调参的完整场景验证 |
| Hang | cloth_hang_l | 小系统/低开销退化检测，作为留出验证 |

主配置延续 dt=.04、rho=1e-4、movement=.01、累计=.01、min updates=6、res=.03、Graph K1、legacy MAS、完整 CCD、原材料和摩擦。P2/P3 不改这些值或外层停止；movement-only 只做命名明确的消融。P1 精度变化是独立算法实验，初始必须是 legacy 外层。

### M0.2 离线统计，先不新跑长轨迹

流式读取当前提交的 `history/autodl_recording105_20261013/data/per_newton.csv.gz`，以及本机旧报告 `per_newton.csv`。后者包含逐行 `native_json`，已核实首行存在 energy/intersection backtracks、energy evaluations、alpha、beta、Kappa 和 active_pairs。不能由首行推定每行齐全。

统计：按版本/轮次/场景的首次 trial 接受率、能量/几何回退分布、接受 alpha 分位数、接触规模、PCG 长尾、退出类型、Kappa 变化和缺失字段比例。起始几何 alpha、每个 trial margin、容量增长和真正 GPU 内部阶段耗时若缺失，明确 unavailable，随后用最小 observer 补齐，不填零。

输出新的 `line_search_distribution`、`pcg_tail_distribution`、`field_coverage` 与 `representative_states`。记录开关造成的方向数变化与观察成本分开解释；不从两个不同轨迹的时间差直接扣除“日志成本”。

代表状态按事件规则在看候选速度前冻结：普通、首次接触、密集持续接触、释放/再接触、小 CCD alpha、真实能量回退、PCG 长尾、容量增长。三个开发场景每场最多8个自然状态；未出现的类别标缺失，失败/溢出用人工 fixture 覆盖。保存第100帧共同输入作为兼容性对照，但不强行用它替代困难事件。捕获名精确包含物理帧、方向序号、epoch、输入哈希。

### M0.3 参考和质量协议

先对共同起点做内/外层收紧与重复性研究，再登记误差阈值。线性采用显式 `r=b-Ax` 的 L2 相对残差及固定 `B_ref` 的加权相对残差；不能分别用候选自己的 `r^TBr` 作为不同终点。代码中 eta 映射为 rho tolerance=eta²，rho 比值不能直接当真实 L2 残差或位置误差。

非线性质量包括自由顶点与 cloth/FEM/ABD 分区的位置 RMS/max、真实速度 RMS/max、伸长、固定漂移、FEM 体积/J、适用材料合法性、独立路径相交检查、非有限状态/预算耗尽/迭代上限。固定顶点不能稀释动态部分 RMS。R 的重复分歧应明显小于所要辨别的质量差，否则该指标不作认证依据。

阈值在看候选结果前写入版本化质量协议；不得借用旧 dt/旧场景 stretch 阈值直接认证本轮。Mixed 先解释几何 J 与所用 SNK/应变限制模型的关系；若需修复真实物理/实现错误，单开修复提交并同时更新双方基线，不夹带进加速候选。

### M0.4 direct gate 与 compensated 的最小消融

同一完整 checkpoint、同一外层契约、同一个接受前缀，比较累计、direct residual gate、compensated gate。分别记录首次可退出序号、beta、当前残差、预算、最终误差及完整时间。三臂独立继续时前缀可能分叉，不能仍称同前缀。

确认历史预算的价值是否超过 direct；没有同质量净收益即降级为诊断。暂不解除现有 forcing 与 compensated 的禁组合保护。

M0 交付：身份 manifest、字段覆盖及事件清单、参考/质量协议、direct-compared 报告、下一阶段固定输入。没有可靠质量门槛时可继续同状态执行正确性与成本研究，但保持同质量加速认证 pending。

## 5. M1：P2-A 最小工程原型

### 源码落点与第一步

已核对 `StiffGIPC/core/GIPC.cu::lineSearch`、`collision/mlbvh.cu::calcMaxBV` 的 host AABB 读回、`abd_system/abd_system_function/cal_abd_energy.cu` 两类 ABD 归约，以及 `solver/line_search_acceptance.h` 的九次能量回退预算。

首先建立每次 D2H/同步的调用者、消费者与 stream 依赖表，测完整路径的频次和成本。通过后每项独立实现：

1. BVH 内部暂不读取 CPU `scene` 时延迟刷新；保留真正 host 消费者的有效性接口。
2. ABD 两项归约结果进入一次有稳定地址的结果读回；第一版保持原归约及 host 标量相加顺序。
3. 只有依赖证明成立时才替换整设备同步；异步 memcpy 后立即等待不记作重叠收益。

使用原融合 DCD，原有能量 batch/reuse 不重复作为新增优化。新增开关默认 off，只引入本模块需要的状态，避免先搭建泛化事务框架。

### 保持的语义与验收

原目标、下降方向符号、initial alpha、几何先验收、非增能量条件（当前 armijoParam=0）、两类回退计数、九次能量预算、后置缩步重算能量、非有限错误、forcing baseline retry 均保留。绝不对几何无效 trial 继续计算 barrier 来“隐藏失败”。

同状态比较分项能量、最终 alpha、接触 multiset/映射、接受/失败分支、错误代码、accepted 状态和缓存有效性；覆盖首次成功、空接触、真实能量拒绝、初始/后置几何回退、预算边界、NaN/Inf、容量变化。能量相对差很小不足以证明阈值附近分支一致。

M1 交付：消费/依赖清单、小补丁、同状态一致性和完整线搜索计时。阶段时间净减少约15%且一个主场景有整场至少5%减少空间，才扩大到 P2-B；否则保留有价值的小优化，停止复杂 Graph 改造。

## 6. M2：P3 固定 F/W 原型，与 P2 分开

在 `collision/mlbvh.cu` 的 DCD `_selfQuery_ee` 路径拆分遍历与原 FP64 helper 分类；初始不改 VF，不改 CCD。F=当前融合；W=固定两阶段。一次 query 的树、坐标、输入顺序、dHat 和过滤参数相同，观察输出使用私有缓冲，不污染生产数组。

每个 raw pair 保留 primitive 身份、方向、拓扑/body/边界排除。typed 输出核对多重集、重复贡献、类型计数和 MatIndex 下游含义，不能只比较数量或去重集合。排列改变后还需检查能量/矩阵归约和最终接受分支。

队列写入先检查容量；超一项、单 query 超容量、恰满、零输入、指针增长、重复 query 全部测试。不允许截断 pair。W 初版可采用明确有界的 query 批次和完整 F 回退，回退前恢复私有/生产计数；任何发现漏项立即拒绝。只有 W 已有净收益，才研究遍历 continuation 或在线选择器。

完整 query 计时包括遍历、队列写读、扫描/压缩、分类、额外 launch、计数读回和重试；另报寄存器、occupancy、local 流量及队列峰值。资源指标改善本身不等于加速。

M2 交付：F/W 同 query 回放、完整性测试、成本与显存表。W 完整 query 无净收益则停止 A；若只是部分负载获益，保留适用区间，不宣称全面替代。

## 7. M3：P5 潜力和 P1 因果诊断

### P5：先固定系统，后考虑真实 GPU

复用现有 `linear_system/solver/pcg_fixed_study.inl` 的捕获思想和 `tools/local/cpu_fixed_reference.py`/`audit_recycling_reference.py` 的独立验证；检查其读取的矩阵、零/固定自由度、MAS 系数和原解恢复契约后再扩展。旧诊断数据只作为线索，重新捕获 a4 基线代表系统。

同一 A,b 上比较原 MAS、旧 ABD rank-12、简单 FEM/接触局部模式、Ritz/POD rank 8/16/32。当前系统充分求解后提取的“理想空间”只能作为上界研究，不能放入在线加速表。

候选采用既有 SPD balanced 形式：

`E=Z^T A Z, W=A Z, Bnew=Z E^-1 Z^T+(I-Z E^-1 W^T) B0 (I-W E^-1 Z^T)`。

先检验对称性、正定作用、秩亏/近相关处理及共同真残差；不依赖精确算术理论掩盖 native MAS 有限精度。在一次普通 PCG 内冻结 B0/Z/W/E；跨系统可复用 Z，但当前 W/E/因子必须更新。补偿包装最终预条件作用，不能被后续 ABD 覆写。

第二关用 k 的已有信息预测 k+1，包含接触出生/消失/强度和分区变化；禁止偷看 k+1 的完整解。对照固定最近 r 向量、固定 POD/Ritz 与拟议选择策略。

第三关才做 GPU rank 固定原型：完整 solve 计入提取/正交化、当前 AZ/粗矩阵/因子、两次投影归约、apply、真残差和 fallback；只减少迭代不减少总时间即拒绝。Z/W 的 FP64 内存为16*n*r bytes，另计 staging 和构造工作区。P5 保留全空间自由度，但确实新增预条件内部低维空间，不描述为“没有粗空间”。

### P1：先分类，后控制

在同 objective epoch 捕获 `g_k, A_k, d_k, alpha_k, g_{k+1}`；源码方向 `d=-_moveDir`。定义 `ell=A*d+g`、`m=g_next-g-alpha*A*d`，检查恒等式 `g_next=(1-alpha)*g+alpha*ell+m`。先核查物理残差映射/质量/能量缩放，再使用速度量纲范数；它不是实际速度误差。

真残差若需额外 SpMV，观测/存储/读回成本单列；无同目标后继、Kappa/摩擦/边界目标变化时标 unavailable 并重置历史。范数不能直接相加变成误差上界。

从同一完整非线性起点分别以 eta=.01/.03/.1 求方向，每臂重新做原完整 CCD/线搜索并继续完成时间步。比较当前 solve 和后续 Newton/CCD/线搜索的总成本、accepted alpha、失败重求及最终位置/真实速度差，不沿用旧方向 alpha。

只有反事实显示稳定可预测的净收益，才设计有限动作控制器；对照 fixed 与现有 adaptive EW，保持同一外层 stopping。调参仅用开发场景/早期事件，Table/Hang 和其余留出事件冻结规则后验证。不能只因 alpha 小就放松 PCG，也不能因模型缺陷大直接提前退出。

M3 交付：同系统质量—成本表、理想与在线迁移分开的报告、接受步分解、反事实完整时间步报告。理想低秩无效停止P5；迁移失败停止recycling；控制器不能超过EW或固定策略则停止P1复杂化。

## 8. M4–M6：选主线、扩大与条件性 P4

| 已通过的证据 | 才允许推进的工作 |
|---|---|
| P2-A 有收益，当前首次 trial 成功率与回退成本支持净收益 | P2-B：origin冻结→trial→BVH→安全检查→接触/能量→统一提交或完整 legacy replay |
| P2-B 后真实回退仍占重要可节省份额 | P2-C：完整 conditional 状态机，逐分支映射原代码 |
| W 优于 F，额外队列成本受控 | P3-A：独立 VF/EE、批次预算、重尾 continuation 与在线F/W选择 |
| P5 小秩及迁移均有效，GPU完整solve节省时间 | 接触演化模式选择/更新的生产候选 |
| P1 反事实支持分类，且EW尚未利用收益 | 新 accepted-step 控制器，再单独评估与gate的兼容契约 |
| 原查询重复较多且候选扩张/证书维护便宜 | P4 旁路保守包络，之后才允许替代查询 |

P2-B/C 必须区分 objective、geometry、storage、query-certificate 有效性。矩阵数值变化与 Graph 拓扑变化不是同一个条件；指针稳定不保证接触/能量有效。容量不足在写之前转 CapacityOverflow，host 扩容后从 origin 重跑；禁止吞成 Accepted。快路径失败、恢复、重跑和 capture 成本全计入。首先复用现有 loose-forcing 的回滚清单，核对 q/q-v、positions/velocities、预测/边界/摩擦/Kappa/接触计数、BVH及必要缓存，不能只备份顶点。

P3-A 的预测仅使用分类前可得的规模/历史批次统计，计费采样/扫描；比较始终F、始终W、按场景固定选择、在线A和离线理想选择上界。理想上界不是在线算法。模块显存预算提前固定，不能无限吃掉余量。

P4 第一版只覆盖同一方向 alpha∈[0,alpha0]，原查询仍执行验证 typed multiset 覆盖。先核对 dHat 的实际平方/长度单位；按primitive位移界保守扩张/向外舍入。候选完整性与路径可行性分开维护，仍在当前trial重新分类、评估原能量并保留FullCCD。累计位移额度不得每trial清零，不得靠额外限步维持缓存。对照无复用、旧contact pool、uniform skin和新有效期；计算build/L+certificate+expanded classification+fallback，完整成本没有净收益就停止。

只选择一条主线：系统主线P2+P3（P4可选），或数值主线P5/P1其中一项为主。局部预松弛、PPN/稳定能量差、s-step、全FP64 MAS、K4等不在首轮实现清单；须相应症状和新因果证据才能另立里程碑。

## 9. 统一实验、计时和验收协议

### 实验阶梯与有限预算

| 阶段 | 预登记范围 | 决策 |
|---|---|---|
| CPU/offline | 字段/身份/状态契约；固定矩阵参考分析 | 先证明比较单位和质量指标可用 |
| GPU同状态 | 三开发场景每场最多8个事件；配套失败/容量fixtures；算子warm/cold分别记录 | 完整性或状态恢复失败直接停止，不作性能采用 |
| smoke | 通过候选与基线每场2帧；全五场景 | resolved开关、输出契约、无异常 |
| bounded screen | 最多少数候选，各开发场景固定3对窗口；AB/BA平衡顺序 | 看完整成本而非单kernel，淘汰无净收益 |
| 全场复测 | 每个入选候选与基线五场景105帧×3对，计前100帧 | 兼容此前表；3对仅筛选，不作稳定显著性宣称 |
| 正式确认 | 最终一个主候选，五场景固定7对；在候选测量前封存样本数、顺序及质量门槛 | 可认证的同质量净收益，或明确未证实 |

GPU 串行、进程由控制器持有、timeout/显存/磁盘门槛提前冻结，每个105帧初始硬上限600s，超时为失败而非收敛；依据M0基线如需调整限时，必须在候选计时前共同改协议。失败收据和原始输出保留，不覆盖或无界自动重试。

固定同一实际电源方案，不继承文本中的 quiet 标签；上一轮本机实际为野兽模式。每次运行前后核对 GUID、记录功耗/温度/时钟/显存与系统切换事件；途中变化整对排除并保存证据。WDDM桌面、OEM散热和未锁频是需披露的干扰，不能只凭GUID声称热状态一致。

### 必须输出的数据

- 身份：source tree + dirty diff、compile/toolchain/definitions、binary/input/scene SHA、requested/resolved config、开关是否实际激活。
- 每帧：Hess/assembly、LSolver、CCD、LineS、Misc、solver total、N、Ptot、失败/退出、位置/实际速度和质量指标；初始化、导出、写盘及应用wall另列。
- 每Newton：PCG次数/阈值/真残差/重求、initial/geometric/accepted alpha、beta、gate/comp预算、接触规模、能量/几何回退、Kappa与objective epoch。
- 定向PCG诊断：矩阵转换/排序归并、MAS setup、SpMV、MAS apply、dot/reduction、update、Graph capture/replay、host等待。完整 solve 包含所有准备/验收，不能用LSolver/P解释纯kernel。
- P2/P3：trial状态、分项能量/acceptance margin、版本/容量、fallback原因和成本、queue峰值/工作量；P5另报basis/当前setup/投影/验证成本。

普通正式计时用最轻观察模式；重型逐kernel事件、额外真残差、完整向量导出只在专门诊断运行开启。正式模式必需的控制/验证成本不扣除；保留相同轻量导出以供双方质量比较，wall中导出另列。先量记录成本和工作量变化，再决定优化观察接口，不盲目把诊断时间从总时间减掉。

相位为父级包络，DCD/PCG子项是解释项，不能重复相加。设备事件耗时、同步host窗口与整程序wall分别展示。同一操作只有一个账本归属。

### 指标与继续条件

每对 `s_i=T_baseline_i/T_candidate_i`，逐对公开时间/工作量；主统计为log(s)上的几何平均与预登记95%区间，兼容旧表另列三轮中位数及对应公式。3对不声称高统计把握；7对若区间跨门槛即未证实，不追加样本追显著。五场景总量必须注明工作负载权重；不能混用倍率均值和总时间比。

采用门槛：复杂模块目标阶段约15%净减少且整场至少5%净减少；5%时间减少对应S≥1.05263。每项在共同质量下对最强简单替代评估；无收益的小优化可作为实现整理保留，但不列强贡献。2×为B0同配置同质量时间至少减少50%，分场景报告；相对B1、相对MOV和相对B0分别列。

P2×P3 和 P1×P5 分别做开关2×2；单项未通过不组合，收益不简单相加/相乘。需拆Graph单项归因时另列执行组件消融，不能用整个bundle/Host倍率冒充Graph倍率。

## 10. 计划文件、源码落点与后续交付

| 位置 | 最小职责 | 当前状态 |
|---|---|---|
| docs/research/INNOVATION_BASE_MANIFEST_20261014.json | 起点、方案hash、旧工作区保全及外部证据身份 | 本轮生成 |
| docs/research/INNOVATION_TODO_20261014.md | 依赖、完成证据与停止决定 | 本轮生成 |
| tools/bench/（后续按模块增量添加） | portable schema、observer分析、固定状态/有限配对控制 | 待实现；复用既有身份和写新目录规则 |
| StiffGIPC/core/GIPC.cu / solver/line_search_acceptance.h | P2状态与接受协议；P1接入 | 待实现，原可信路径保留 |
| StiffGIPC/collision/mlbvh.cu/.cuh | P2 host AABB接口，P3 DCD，P4证书 | 待实现，每项独立开关 |
| StiffGIPC/abd_system/abd_system_function/cal_abd_energy.cu | P2 ABD结果读回 | 待实现 |
| StiffGIPC/solver/ipc_pcg_forcing.h / ipc_budget.h | 强基线与P1契约 | 待实现，先不解除禁组合 |
| StiffGIPC/linear_system/solver/pcg_fixed_study.inl / pcg_graph_impl.inl | 捕获/同状态replay及P5接入边界 | 待实现，Graph不混同数值缓存 |
| tests/（后续） | 状态/失败/容量/残差与数学性质等有意义的测试 | 待实现，不为计划文档新增镜像测试 |

每个里程碑交付：小补丁及实际开关、最小相关测试命令与结果、身份封存、正确性/质量报告、完整成本、明确continue/stop。独立提交划分为observer、P2-A、P3-W、P5诊断、P1诊断、胜出机制、必要组合，禁止一个提交同时改容差、PCG时序、接触模型与退出。

GPU build尚未执行。未来先用本机实测工具链配置新 `build/innovation_baseline` 和 `build/innovation_candidate`，保存configure/build日志；CPU契约可用 `E:/Anaconda/envs/DL/python.exe -X utf8 -B tools/bench/contracts_test.py -v`。native优先运行相关 CTest（例如line_search_acceptance、ipc_pcg_forcing_policy、ipc_budget_invariants、ipc_compensated_controller），之后才扩全场；这些命令在本轮只是计划，不表示已经通过。

最终通过的候选才更新之前格式的性能表、独立场景checkpoint、误差—时间图与真实关键帧PDF。视觉按真实状态、统一相机/尺度和完整覆盖输出，误差图与参考分歧同行展示。当前没有承诺出版级创新、新2×或物理正确性认证。

**实施的下一步：M0.1/M0.2，先封存基线配置并离线统计回溯、步长与PCG长尾，再决定具体同状态捕获；不是直接开始重写GPU回溯。**
