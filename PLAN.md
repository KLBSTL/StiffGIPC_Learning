# Stiff-GIPC 提前终止 base + TOI 解锁 + CUDA Graph 融合计划书

日期：2026-09-29。版本：v1，待用户审阅。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；benchmark-regression；PDF。
- Origin Mode：plan。
- Verification Status：已核对论文、官方提交、v3 配置及部分源码哈希；融合实现与新实验尚未执行。
- Version Label：stiff_toi_graph_plan_v1。

## 1. 目标与本轮边界

以用户指定的 [KemengHuang/Stiff-GIPC](https://github.com/KemengHuang/Stiff-GIPC) 为唯一主基线，保留它已加入的提前终止和当前材料、ABD/FEM、布料模型；在独立目录中加入论文的 TOI 推进/解锁方法，并迁移 `autodl_results_v3` 对应的 CUDA Graph 优化，形成可独立关闭的功能。

核心问题：在相同场景、物理参数、模拟时长和质量要求下，TOI 方法能否减少接触密集阶段的外层迭代，CUDA Graph 能否进一步降低每次线性求解的执行开销，两者合并后的总时间是否优于新 base？

本轮仅完成计划与取证。待用户审阅后，再建立源码副本、构建、改代码及运行实验。现有 `stiffGIPC`、`externals`、`paper_reproduction_20260928`、已有 AutoDL 结果目录均作为只读来源。

### 本计划对“TOI”的具体解释

采用论文完整的 TOI locking 解锁链条：**双状态 + 增广拉格朗日接触子问题 + CCD 活动集更新/过滤/衰减 + 累计 TOI 终止**。仅复制累计步长终止条件不能构成本次新增算法，因为新 base 已包含该类条件。

这会新增一个接触求解后端。论文允许中间状态穿透，因此不能在原对数 barrier 上直接继续求解穿透状态；需要在 TOI 后端替换法向 barrier 响应，同时复用 Stiff-GIPC 的材料、线性系统及 ABD/FEM 基础。IPC 后端仍作为可选择的 base。

## 2. 已确认的输入与关键事实

| 项目 | 当前证据 | 对计划的影响 |
| --- | --- | --- |
| 官方 base | GitHub `main` 当前为 `bb2849a7b292099581907937860d96ecfdf42588`；本地官方副本同提交且工作区无改动 | 实施时固定该提交，不从旧原型整仓开始 |
| 提前终止 | `StiffGIPC/core/GIPC.cu::solve_subIP` 中 `Kmin=6`、`semi_implicit=true`，接受步长后更新 `beta=(1-alpha)*beta`；另保留方向范数停止条件；Newton 上限 10000 | base 已有累计步长终止；必须记录真实接受步长、退出原因及上限命中 |
| 终止阈值 | 官方 `Assets/scene/parameterSetting.txt` 的 `Newton_solver_threshold=1e-2`，同时用于方向范数和 beta | 主比较保留官方默认；另做双方一致的 `1e-3` 敏感性比较，不能只给一种方法更松阈值 |
| 论文 | 用户 PDF 共 20 页；重点为第 5–7 页、算法 1–3、式 (9)–(14)，以及第 9–10 页与附录 A | 先以这些算法建立 CPU/小规模 GPU 数值参照，再迁移到求解器 |
| 论文公开实现 | 作者项目页指向 `wiso-enoji/libuipc` 的 `AL-release`；当前分支提交 `389f8a52a29606ccaf5640607dd1e38987500988` | 作为 AL 接触和活动集处理的实现参照；主工程继续是 Stiff-GIPC，初期不引入整套 libuipc 运行时 |
| CUDA Graph 来源 | v3 的 A 组运行配置与源码清单已读取；选出的 13 个核心优化文件与当前本地候选源码哈希一致 | 实施时先冻结并再次核对文件；哈希变动则重新定位原版本，不能从正在修改的目录盲目复制 |
| 本机 | RTX 3070 Laptop，8192 MiB，总空闲快照 6437 MiB，CC 8.6；驱动 581.57；`nvcc` 是 CUDA 13.0 | 本机先小规模验证，构建 SM86；CUDA 13.0 与远端 12.8 分别验证 |
| AutoDL | 用户指定 CUDA 12.8；既往机器为 RTX 4090 24 GiB | 执行阶段重新核对实例、空闲显存与 SSH；源码传输后重新编译 SM89，不复用 Windows 二进制 |

论文项目页与公开代码入口见 [作者项目页](https://simulation-intelligence.github.io/barrier-free/) 和 [公开实现说明](https://github.com/wiso-enoji/libuipc/tree/AL-release/apps/AL_examples)。公开实现说明列有移动边界及解析 PSD 投影方面的未完成项，不能将该代码视为全部论文特性的完整复现。

已确认存在正式勘误 DOI `10.1145/3839111`，本轮只取得元数据，尚未读到勘误正文；实施前核对其修改范围。任何有限步终止或精度结论都不能自动继承到本次迁移版本。

## 3. v3 优化应如何迁移

v3 中 A 是多项功能组合，不是纯 CUDA Graph 实验：

| 子功能 | v3 A 状态 | 迁移策略 |
| --- | --- | --- |
| conditional WHILE PCG + MAS | 开启 | Graph 主模块，优先迁移 |
| Graph 缓存 | 开启 | 增加与新动态内存机制一致的失效条件 |
| PCG dot/tail、continue 融合 | 开启 | 独立开关；验证停止语义与数值变化 |
| MAS 静态拓扑缓存 | 开启 | 与 Graph 分开消融；不能把动态接触拓扑误认为静态 |
| MAS 接触拓扑缓存 | 关闭 | 首版保持关闭 |
| 批量能量 | 开启 | 独立验证后纳入扩展优化组合 |
| 能量复用 | 开启 | 首版主比较不默认加入；先验证状态/目标函数版本是否一致 |
| CCD BVH refit | 开启 | 首版主比较不默认加入；与完整 rebuild 对照核查漏对 |
| B 的缺陷预算提前终止 | A 中关闭 | 本次不迁入；主基线已经有官方提前终止，避免叠加不同停止准则 |

旧 v3 报告的 base/A 原始时间比中位数为 1.136×，但未完整验证质量匹配，且盒子/兔子场景存在明显轨迹差异。这些结果仅用于选择迁移候选，不能作为新 base 上的预期加速承诺。详见 [v3 报告](E:/university_class/ComputerGraphics/GIPC/autodl_results_v3/final_autodl_full_four_arm_20260927/REPORT_FULL_FOUR_ARM.md)。

### Graph 适配重点

1. 从 PCG 内循环开始，包含 SpMV、MAS apply、归约、方向/解向量更新和设备端继续条件；外层 AL、CCD 和动态内存管理先留在主机调度。
2. 捕获前完成所需缓冲区扩容，捕获中不分配内存。矩阵、MAS、归约 scratch 和向量地址变动时同步后销毁旧图并重建。
3. 缓存键覆盖设备、DOF、全部捕获地址、逻辑长度、矩阵格式、MAS 层级结构/版本、容差、最大迭代数及后端。不能仅靠地址相同判断图仍有效。
4. 适配新 base 的 `DeviceBuffer` 动态扩容和 count/grow/rerun 机制，保留其容量保护；旧固定容量设计不随 Graph 一起迁入。
5. Graph 失败时仅在 CUDA 流状态安全且数据状态完整时回退普通 PCG；流已失效或结果不可恢复则明确报错，不吞掉错误。
6. 保持与 base 相同的 PCG 判断顺序、初值和实际停止语义，并额外记录真残差。官方 PCG 比较的是预条件残差内积 `abs(rz)` 与 `global_tol_rate*rz0`，其配置值不能直接等同于欧氏相对残差，也不能直接等同于论文 CG tolerance。先记录两者的对应关系，再设计严格参考。若 fused 操作改变数值，单独报告其影响。
7. CUDA 13.0/12.8 使用各自 API 签名编译分支；启动时验证 conditional Graph 功能，记录有效开关和 fallback。CUDA 12.8 的 WHILE 节点能力见 [NVIDIA 官方指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)。

## 4. TOI 后端的算法设计

### 4.1 状态和接触集合

- `x_safe`：最后一份已验证不相交的状态。
- `x_trial`：论文的中间状态，允许几何穿透；每个时间步从上一时刻初始化，在该时间步的外层迭代间持续保留。
- `x_tilde`：当前惯性目标，材料和时间积分参数沿用 base。
- `C`：按标准化 primitive key 去重的接触集合，保存 PT/EE/退化 PP/PE 类型、对象/顶点身份、线性化系数、乘子 `lambda`、衰减 `gamma`、状态及版本。
- ABD 状态：保留广义坐标 `q_safe/q_trial`；凸组合广义坐标后生成一致的世界顶点，不能只更新顶点而漏掉 ABD 内部状态。
- 摩擦历史：单独记录，明确法向力来源、切向基、历史状态及更新时机。

### 4.2 外层推进

每个时间步：初始化预测目标、safe/trial 状态、继承的活动集和 `beta=1`；随后按论文算法 1 的数据依赖顺序执行：

1. 在 `x_safe` 上定向并线性化当前接触约束。
2. 在 `x_trial` 上求解 AL 子问题；材料/惯性项、固定边界及 ABD/FEM 映射沿用 Stiff-GIPC。
3. 使用论文规定的 safe/trial 历史时刻执行活动集更新，收集 CCD 返回的 primitive pair 与各自 TOI，去重并过滤。对“上一 trial”与“新 trial”的具体时序建立逐行对照，不能在移植时自行交换。
4. 对 `x_safe -> 新 x_trial` 进行完整 CCD，并对需要的材料做 inversion 检查，得到保守可行推进量 `alpha`。
5. 以同一 `alpha` 更新 safe 顶点和 ABD 广义状态，保证输出轨迹不相交。
6. 达到 `Kmin` 后，按实际接受的 `alpha` 更新 `beta`。达到累计 TOI 终止门槛后输出 safe 状态并更新速度。

法向接触在 TOI 后端使用 AL；不能同时对同一对保留 IPC 法向 barrier 并计入 AL 力。第一版无摩擦单元验证通过后，再加入论文附录 A 对应的摩擦处理，最终性能比较保持双方相同的摩擦参数。

### 4.3 AL 子问题

在 safe 状态计算线性化约束：

`c_i(x_trial) = d_i(x_safe) + grad(d_i(x_safe))^T (x_trial - x_safe) - delta`。

维护 slack：`s_i = max(0, c_i - lambda_i / mu)`；接触梯度为 `mu * gamma_i * (c_i - s_i - lambda_i/mu) * grad(d_i)`，接触 Hessian 为 `mu * gamma_i * grad(d_i) * grad(d_i)^T`。

材料 Hessian 投影及稀疏系统装配复用 base，法向 AL 项通过同一全局系统接口装配。ABD 接触项需要正确变换到广义 DOF；同时核对交叉块、约束 DOF 消元与 MAS 分区。

内层使用减少 AL 目标的线搜索，记录其步长 `r_inner`。`r_inner` 与外层 CCD `alpha` 分开存储、分开计时，不能用内层步长更新外层 beta。内层完整步接受后按论文算法 2 更新乘子和衰减。

### 4.4 活动集、参数与停滞处理

- 保留至少是某个顶点最早新碰撞的 primitive pair；浮点 TOI 的并列情况使用明确、可复现的保守处理。
- 此过滤只影响 AL 活动集，**不允许用于删除保证 safe 推进的完整 CCD 对集合**。
- 非活动项乘子清零、`gamma *= Gamma`；活动项恢复 `gamma=1`；低于阈值时移除。
- 起始参照参数：`Gamma=0.9`、移除阈值 `0.01`、`mu_init=0.1*max(diag(H_E))`。接触 `delta` 按实际网格、厚度和碰撞语义校准，不能直接把 base 的平方 `dHat` 当作线性距离使用。
- `alpha < 1e-4` 连续 50 次的停滞策略，参照论文调整 `mu` 与 `delta`；记录每次调整，验证距离缩减不破坏模型厚度约束。
- 数值异常、不可行边界、CCD 不可靠、持续零 TOI、内外层上限命中分别退出并标记失败。
- 若提供回退 IPC，必须从完整时间步检查点恢复再运行；回退次数和额外成本单列，不混成“纯 TOI”结果。

### 4.5 首轮实现范围

首轮包含双状态、AL、活动集过滤/衰减、累计 TOI、固定地面、静态边界、布料自碰撞、FEM 和 ABD 耦合，以及相同参数下的摩擦。移动边界与复杂马达在后续扩展门槛中单独验收；未支持的场景明确列为不支持，不能静默改变运动条件。

论文第 5.1 节的解析 PSD/Hessian 装配优化另作候选，不加入首轮核心融合，以便定位 TOI 与 Graph 各自收益。

## 5. 独立目录与代码迁移位置

任务根目录：`E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929`。

```text
stiff_toi_cudagraph_20260929/
  PLAN.md / TODO.md               当前计划与审阅状态
  research/                      论文摘录、算法页图、来源哈希与核对记录
  sources/stiff_base/             固定官方提交的只读 base 副本，批准后创建
  sources/stiff_fused/            唯一改代码位置，批准后创建
  references/graph_v3/            经哈希核对的优化来源快照
  references/robust_al/           固定公开 AL 实现参照
  builds/local-base|local-fused/  新建构建缓存
  configs/ / assets/              冻结场景与本任务资产
  tools/                         构建、预检、运行、审计、报告脚本
  runs/local/ / runs/autodl/      按场景、方法、重复分组
  reports/                       比较报告、CSV、图、关键帧
  bundles/                       远端源码包与哈希
```

| 官方模块 | 计划修改内容 |
| --- | --- |
| `StiffGIPC/core/GIPC.cu/.cuh` | 后端分发、双状态推进、beta/退出原因、AL 接触装配接入；保留 IPC 原路 |
| 新 `StiffGIPC/toi/` 模块 | 活动集状态、线性约束、AL 内循环、乘子/衰减、参数与检查点 |
| `StiffGIPC/collision/ACCD.*`、`mlbvh.*` | 暴露逐对 TOI；保留全 CCD 可行性查询和动态容量增长 |
| `StiffGIPC/linear_system/solver/pcg_solver.*` | 设备标量与 conditional Graph、缓存、失败状态、真残差记录 |
| `StiffGIPC/linear_system/preconditioner/`、`solver/MASPreconditioner.*` | 捕获接口、静态拓扑缓存、结构版本、动态容量兼容 |
| `StiffGIPC/abd_system/` | safe/trial 广义状态与接触 Jacobian 映射、速度一致性 |
| `StiffGIPC/app/gl_main.cu` | 有限帧 benchmark 入口、配置、状态与关键帧导出 |
| `CMakeLists.txt` | 独立目标/功能、实际架构参数、CUDA 12.8/13.0 兼容与小规模数值测试 |

新 base 的目录布局和内存机制已不同于 v3，采用逐模块迁移。新增配置建议为 `contact_backend=ipc|toi_al`、`pcg_execution=host|conditional_graph`、`mas_static_cache`、`energy_batch`、`energy_reuse`、`ccd_bvh_refit`、`toi_eps`、`toi_kmin` 和迭代上限；名字待实现时与现有接口统一。输出同时记录 requested 与 effective 配置。

第三方源码快照保留其许可及来源记录。其他对话目录不做 reset、pull、重建或输出覆盖；不共享可写构建缓存。下载使用 `127.0.0.1:7892` 代理；远端外网需要时使用经过验证的隧道方式，不能把本机回环地址直接当作远端代理。

## 6. 对照组与消融设计

### 主四组

| 方法 ID | 接触后端 | 官方提前终止 | Graph | 意义 |
| --- | --- | --- | --- | --- |
| `base` | IPC | 开启，官方默认参数 | 关闭 | 本次主基线 |
| `base_graph` | IPC | 同 base | 开启 | Graph 单独收益 |
| `base_toi` | TOI/AL | 累计 TOI，匹配 eps/Kmin；记录后端特有退出 | 关闭 | TOI 后端收益 |
| `base_toi_graph` | TOI/AL | 同 base_toi | 开启 | 最终融合收益 |

Graph 主包为 conditional PCG/MAS + 图缓存；拓扑缓存、kernel fusion 分别消融后，在报告里明确组合内容。扩展 `graph_v3_full` 逐项加入批量能量、能量复用和 BVH refit；这些组用于复现 v3 组合效果，不改名为纯 Graph。

### 关键消融

1. 设备标量但无图 vs conditional Graph，区分减少 host 同步与图调度的收益。
2. 图缓存关闭/开启；静态拓扑缓存关闭/开启；dot/continue 融合关闭/开启。
3. TOI 不过滤/过滤、无衰减/衰减、罚函数/AL：仅在小受控场景上定位机制，失败也保留。
4. 同 eps 的 `1e-2` 与 `1e-3`；严格质量参考另使用更紧的停止和 PCG 条件。
5. MAS 与 block-Jacobi 作为诊断项：TOI 后端若暂不支持 MAS，使用双方都能工作的共同预条件器做桥接比较；此时不能标为完成 v3 MAS Graph 融合。

Newton/AL 上限不得出现旧实验的 256 vs 10000 不对称。不同后端的外层、内层计数单列，总工作量以实际线性求解次数和迭代数衡量；命中任何上限都不能按成功结果计速。

## 7. 本机与 AutoDL 场景安排

### 7.1 本机先行

先做无接触惯性运动、PT/EE 小接触、滑块摩擦、小布料自碰撞，以及官方小型兔子-布料混合场景。按 `1–2 帧 -> 20 帧 -> 100 帧` 递进。

主场景第一批用 L 档：悬挂布料、七球布料、桌面布料、固定兔子布料、十层布料；然后加入兔子-布料-兔子 L。M 档逐个预检后加入。本机 H 档和原始图四不预先承诺可运行。

原 v3 的 L/M/H 标记、顶点数和变换只作为输入参照；迁移到官方 importer 后重新导出初态、计数及拓扑集合并验算，不能仅凭场景名称认为一致。

### 7.2 AutoDL 扩展

本机过门槛后，先在 CUDA 12.8 上做同样的小场景和 Graph 功能测试；再扩展五个布料家族各 L/M/H、兔子-布料-兔子 L/M/H，共 18 个候选场景。资源允许时增加盒子布料三档和图四 O 档，形成最多 22 个候选场景。

每个主配对先 100 帧、3 次重复；需要覆盖更晚接触或验证持续稳定性的场景追加 200 帧，四方法同一窗口。盒子场景作为轨迹偏差专项，图四优先用保持物理含义的小规模变体调试；缩小版与原始 O 档分开命名。

### 7.3 资源预检

1. 记录初始空闲显存、模型规模、矩阵/活动集容量、图实例与 scratch；新 base 的动态内存不同，旧 v3 的内存估计不能直接套用。
2. 本机建议起始上限为 `min(0.75*空闲显存, 空闲显存-1536 MiB)`；按当前快照约 4.7 GiB。用于首轮资源保护，具体值在执行前固定并写入配置。
3. AutoDL 同样先 2 帧、20 帧预检并记录内存趋势；遇到显存高增长、CCD 对暴涨或持续小 alpha，暂缓该档完整测试。
4. 本任务进程记录 PID 和完整退出原因；停止仅针对本任务，不干预其他对话的进程。显存未释放到基线时等待并重查，不能立即启动下一组。
5. OOM、容量保护、超时、NaN、上限、CCD 不可靠分别统计。资源未完成组不填加速比；放宽资源上限属于显式另一次实验。

## 8. 正确性与质量验收

### 8.1 数值与模块门槛

- 小规模 CPU 参照核对线性接触能量、梯度、Hessian、slack 和乘子更新；有限差分只用于平滑分支，活动切换单独测试。
- PT/EE 退化、重复约束、TOI 并列、零接触、缓冲区扩容、流捕获失败都要有定向验证。
- 图版和普通 PCG 使用相同矩阵/RHS；核对实际真残差、迭代出口、解差和耗时，不能仅比较迭代计数。
- 内部 `x_trial` 的穿透属于允许状态；验收对象是 `x_safe` 输出和其真实更新路径。
- 核对所有缓冲区地址/版本变化都触发对应图失效；扩容压力场景不能复用旧地址图。

### 8.2 物理质量

所有方法：无 NaN/Inf、无错误计数、完成全部指定帧、未命中上限；输出路径通过独立连续 CCD 检查，固定边界和 ABD 顶点映射一致；需要非翻转的材料满足正 Jacobian。

Graph 属于执行优化，要求与同后端普通求解保持数值等价。建议预注册小确定性场景的质量加权位置 RMS / 场景尺度不超过 `1e-6`、线性解相对差不超过 `1e-6`，真残差同时满足声明门槛。接触敏感场景同时检查首次偏离帧、约束和视觉结果；不因某场景更快而事后放宽阈值。

TOI 会改变求解路径，不要求与松容差 base 逐顶点完全一致。使用更紧容差/更小时间步的共同物理参考，比较位置/速度、动量、能量、最大应变/拉伸、边界误差和接触间隙；通过 `dt, dt/2, dt/4` 在相同物理时间窗口检查精度趋势，并加入无接触运动衰减及摩擦滑动/静摩擦验证。

质量容差建议先由小场景参考误差与几何尺度标定，例如位移和速度误差相对严格参考均不超过 1%，应变与能量误差采用相同预注册表。具体容差在看到性能结果前固定。连续 CCD 失败直接判失败，不以 RMS 很小替代。

## 9. 计时、分段与结果记录

### 每帧 / 每迭代必须记录

- 整步 wall time、材料/接触装配、预条件器、PCG、BVH/CCD、内层线搜索、活动集处理、状态更新时间。
- 外层迭代、AL 内层迭代、线性求解次数、PCG 迭代和真残差。
- 原始碰撞 TOI、保守 CCD 步长、最终接受 `alpha`、内层 `r_inner`、beta、退出原因、零/小 TOI 连续计数。
- 宽相位候选、真实 CCD 交对、AL 活动集/过滤/移除数量、mu/delta/gamma 更新。
- 图 capture/instantiate/cache hit/invalidation/fallback 数量及时间；内存容量/峰值与扩容事件。
- 源码提交/文件哈希、场景与初态哈希、编译选项、GPU/驱动/工具链、requested/effective 配置。

### 比较口径

主指标为相同模拟窗口的累计整步时间，**包含第一次 Graph 捕获及本窗口内重建成本**，排除进程启动、资产载入和状态/制图导出；另报包含初始化的端到端时间，以及稳态帧指标。详细 CUDA event profiling 与轻量 wall timing 分开运行，避免大量同步改变性能结论。

本机四组顺序轮换，独立进程运行，禁止同时在同一 GPU 上跑性能测试；记录温度、频率、功耗和其他 GPU 负载。每场景 3 次配对，报告配对时间比、中位数、范围和单次值；该重复数主要反映可重复性，不作强统计显著性结论。若噪声接近收益，再对关键场景增加到 5 次。

主速度比：`base/base_graph`、`base/base_toi`、`base/base_toi_graph`、`base_toi/base_toi_graph`。不能把旧 TOI 与 Graph 加速比相乘当作融合实测值。

接触/非接触段使用同组 base 的统一帧号集合，但帧分类必须有实际窄相位距离/法向接触力证据，并定义一致几何距离与力阈值；有必要时加入固定场景时间窗口。仅 `native_contact_candidates>0` 时标成“候选段”，不能称为真实接触段。TOI 后端内部活动集数量不直接作为跨后端接触判据。

质量未通过时仅给诊断时间，正式质量匹配加速比为 N/A；失败、回退和未支持项留在主表。

## 10. 实施阶段、交付物与进入条件

| 阶段 | 工作 | 交付物 / 进入下一阶段条件 |
| --- | --- | --- |
| P0：审阅计划 | 用户确定本计划的 TOI 范围、四组和质量口径 | 本轮停止在此 |
| P1：冻结与 base | 创建源码独立副本；核对许可/勘误/版本；配置新构建；加入不改求解语义的必要记录 | `SOURCE_LOCK.json`、配置、base 小场景成功及初态/参数核验 |
| P2：Graph | 迁移 conditional PCG/MAS、图缓存；分开核对 kernel fusion/静态拓扑 | 同矩阵/RHS 数值核验通过，扩容失效测试通过，小场景和 100 帧 Graph 配对 |
| P3：TOI 核心 | CPU 接触参照、双状态、AL、逐对 TOI、过滤/衰减、safe 推进 | 无摩擦小场景及连续路径核查；停滞/失败状态可定位 |
| P4：完整融合 | 加入摩擦、ABD/FEM 耦合、边界；TOI 接入 Graph；测试后端和容量缓存键 | 四组 L 场景 100 帧通过；物理参考与模块验证通过 |
| P5：本机报告 | 扩展可运行的 M 档；质量评估、消融、重复计时 | `REPORT_LOCAL.md`、原始 CSV/状态、关键帧、已知失败表；冻结远端包 |
| P6：AutoDL | CUDA 12.8 Release 编译；小场景验证；资源预检；逐档完整矩阵 | `REPORT_AUTODL.md`、源码/二进制/配置哈希、重复统计、失败证据 |
| P7：对比报告 | 汇总机制与性能、Graph 单项/组合收益、局限 | `FINAL_COMPARISON.md`、导出 CSV、静态科研图和关键帧索引 |

P2 先取得可独立验收的 Graph 结果；P3–P4 是工作量和风险最高的部分。AutoDL 运行时长根据本机及远端小样本的每帧耗时、有效场景数和 3 次重复估计，运行前生成预算，不提前承诺固定加速倍数或完成时刻。

主 AutoDL 18 场景 × 4 方法 × 3 次 = 216 个候选运行；资源条件不满足的档位保留跳过原因，局部消融只在代表场景执行，避免把所有开关组合扩大成无必要的全场景矩阵。

## 11. 构建与运行入口草案

所有命令均在批准后执行；以下源码、构建路径和脚本尚未创建。

本机使用 PowerShell 7，CMake/Visual Studio 2022，CUDA 13.0，SM86；复用已存在的第三方 vcpkg 安装作为只读依赖来源，不复用其他任务的求解器或 CMakeCache。当前 vcpkg toolchain 已确认位于 `E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake`，MSVC 实例执行时由 CMake/vswhere 重新定位。

```powershell
cmake -S sources/stiff_fused -B builds/local-fused -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build builds/local-fused --config Release --target gipc
& 'E:/Anaconda/envs/DL/python.exe' tools/run_matrix.py --config configs/local_smoke.json
```

官方 CMake 的目标属性中还直接写有 `CUDA_ARCHITECTURES native`，实施时在新副本修正为尊重配置值，并对 base/fused 采用一致构建选项。旧 `stiffGIPC/build/CMakeCache.txt` 指向历史 `D:/code_2/stiffGIPC`，本任务新建全部构建缓存。

AutoDL 使用 CUDA 12.8、Linux Release、SM89；具体 generator、依赖路径和 OpenGL/Xvfb 条件实测后写入脚本。现有 batch 模式仍需要 OpenGL context，不把“隐藏窗口”误认为已完全去掉图形依赖。

## 12. 最终报告与验收

报告必须回答：

1. TOI 有没有增大有效 alpha、减少小/零 TOI 和外层/线性求解次数？
2. Graph 有没有减少相同求解工作量的耗时；捕获、重建和动态活动集使收益损失多少？
3. 融合后相对官方新 base 的总、接触、非接触时间和加速比分别是多少？
4. 每档分辨率的趋势、物理质量和显存成本如何变化？
5. 哪些场景失败或未支持，具体是资源、CCD、AL 停滞、Graph 失效还是质量偏差？
6. v3 的哪些组合优化能安全复现，哪些被撤回，是否存在变慢场景？

验收：用户指定的新 base 来源正确；全部工作在新目录；TOI 和 Graph 可分别关闭；本机先行验证后才上 AutoDL；所有正式速度结论有质量与完整运行证据；失败不隐藏；原目录和其他对话工作不被覆盖。

## 13. 来源索引

- [固定官方提交](https://github.com/KemengHuang/Stiff-GIPC/commit/bb2849a7b292099581907937860d96ecfdf42588)，本轮经代理 API 查询；记录：[upstream_commit.json](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/research/upstream_commit.json)。
- 用户论文 PDF：`D:/C_files/2026-Robust and Efficient Penetration-Free Elastodynamics without Barriers (1).pdf`；SHA-256 `0f18d16bd95a2e6abd6839a55e03d32420045057726336d6bf4d40446357a0ed`；记录：[paper_manifest.json](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/research/paper_manifest.json)。
- [论文 arXiv v3](https://arxiv.org/abs/2512.12151v3)、[作者项目页](https://simulation-intelligence.github.io/barrier-free/)、[公开 AL 实现](https://github.com/wiso-enoji/libuipc/tree/389f8a52a29606ccaf5640607dd1e38987500988/apps/AL_examples)。
- 勘误：[DOI 10.1145/3839111](https://doi.org/10.1145/3839111)，正文核查待实施前完成。
- v3 结果：[REPORT_FULL_FOUR_ARM.md](E:/university_class/ComputerGraphics/GIPC/autodl_results_v3/final_autodl_full_four_arm_20260927/REPORT_FULL_FOUR_ARM.md)。
- v3 优化运行配置：`E:/university_class/ComputerGraphics/GIPC/autodl_results_v3/benchmarks/stiff4-v1/runs/autodl-full-four-arm-20260927/cloth_hang_l/pcg_conditional_mas_cached_topology_energy_batch_prototype/r01/requested.json` 与同目录的 `source_files.json`；核心来源核对：[graph_source_provenance.json](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/research/graph_source_provenance.json)。
- CUDA Graph API：[CUDA 12.8 官方指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html)、[NVIDIA conditional nodes 说明](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)。
