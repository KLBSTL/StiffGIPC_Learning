# TOI 迭代增长与 mixed 第 1 帧：只读复核

2026-10-05。范围：已下载的 AutoDL 4090 三轮短窗口、冻结源码与原始参考；未运行新 GPU 实验，未修改算法。结论标记为“已证明 / 支持但未证明 / 待验证”。

## 1. TOI 的预期收益与当前实际工作量

**原方法确实同时追求较少 Newton 和较容易的线性系统；不能把它改述为只减少外层、不关心 PCG。** 它保留未截断 trial 来探索接触，以 AL 乘子避免依赖不断变大的罚刚度；但这不是对任意材料、坐标表示及预条件器都保证更少 PCG 的定理。论文也指出过多线性化约束会增加成本，过大的 mu 会使系统病态；mu 初始化公式是实用近似。参见[原论文 v3 §4.1–4.3、§5.2](https://arxiv.org/html/2512.12151v3)。

本项目称为 TOI 的路径实际上包含三层工作：物理帧中的 AL/CCD 外层；固定接触模型下的内层 Newton 方向；每个方向对应一次线性系统 PCG。总 PCG 是全部方向的迭代数之和，既受方向数量影响，也受每个线性系统及其预条件器影响。

- 当前源码 `sources/stiff_active/StiffGIPC/solver/toi_solver.cu:1157` 起是 outer，`:1213` 起是 inner，`:1249` 调用 `calculateMovingDirection`；`:1582` 更新 slack/乘子，`:1622` 起完成 CCD 与 safe 推进。
- 冻结公开参考 `references/robust_local/public/src/backends/cuda/engine/advance_al.cu:336` 使用单个 Newton 循环，但 `:419` 的 continue 延后乘子更新/CCD；`:427` 起才更新乘子和推进 safe。循环外观不同不能单独认定移植错误。参考身份见 `references/robust_local/SOURCE_MANIFEST.json`（public commit `389f8a52a29606ccaf5640607dd1e38987500988`）。
- `references/paper_al/apps/AL_examples/README.md:17` 的 Newton/PCG 数量对齐声明是 animal-well 前 300 帧与作者原实现的比较，不是针对当前混合 ABD/FEM/布料与 Stiff 的保证。
- **Stiff 基线已经有累计 TOI 停止**：`sources/stiff_base/StiffGIPC/core/GIPC.cu:10985` 设置 Kmin=6，`:11110` 起更新 beta 并按累计推进量退出。因此当前实验不是“无 TOI 的 Stiff”对“只新增 TOI 停止”。

已证明：35 帧 mixed 三轮中位数如下，来自下载的 `AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json`，并与各 run 的 `output/stats.json` 核对。

|实现|Newton/PCG 调用次数|PCG 迭代总数|总数/调用次数（中位数之比）|
|---|---:|---:|---:|
|Stiff|142|3655|25.74|
|同候选 IPC Graph|142|3646|25.68|
|TOI Graph（因子逆候选）|169|20722|122.62|
|TOI Graph（三角参考）|170|21099|124.11|

当前主要增长是每个方向的 PCG 成本：对 Stiff，方向数约 1.19 倍，PCG 总数约 5.67 倍，迭代总数/方向数约 4.76 倍。后一个比值是描述统计，不是同一 A/b 上测得的条件数。TOI Graph 第 24–35 帧各轮分别消耗 18215、19219、19393 次 PCG，占各轮总数约 92.4%–92.8%；Stiff 同窗为 1979、1978、1978。不能把此增长仅归咎于 Graph 发射或因子逆，三角参考也存在。

## 2. mixed 第 1 帧不是已证明的“相同无接触目标、仅退出不同”

原 `CONTACT_FREE_EXIT_HYPOTHESIS.md` 的直接日志证据是悬挂布料；不能未经核对把其因果解释移用于混合兔子。

以下是同轮 r1 的真实字段；其余两轮方向数和物理量现象一致。

|第 1 帧字段|Stiff|同候选 IPC host|TOI host|
|---|---:|---:|---:|
|方向次数 / PCG 总数|3 / 63|3 / 63|2 / 40|
|第一次 PCG 迭代数|12|12|6|
|第一次 rho_initial|未导出|1.7036039789298014e-4|1.5408440340210068e-4|
|FEM min J|0.8470838476367649|0.8470838464483121|0.9999806574518412|
|帧末 native narrow self pairs|2625|2625|2968|
|帧末最小自距离 m|0.0021503867909742295|0.002150386791031935|0.0014964303927938996|
|日志 geometric_contact|false|false|false|

TOI 的前帧 AL 接触数、warm lambda、warm friction 均为 0；唯一 outer=0，active_added=0，active_self=active_ground=0，完整 CCD alpha=1、beta=0，inner 为 `velocity_converged`、frame 为 `cumulative_toi`。**这证明第 1 帧的 TOI AL 子问题没有接触约束。**

但 `geometric_contact=false` 的分类距离是 `1e-4*bbox_diagonal`，不是 IPC 屏障半径：

- 实际相同 `output/scene.json`：bbox²=11.122765781907146，relative_dhat=.001，IPC 屏障半径 **0.0033350810757621988 m**；源码 `core/GIPC.cu:8487` 的 dHat 是该距离的平方。
- TOI delta 与日志分类距离为 **0.0003335081075762199 m**，恰为其十分之一。`solver/toi_solver.cu:966` 初始化 delta。
- `collision/mlbvh.cu:578` 等在平方距离 d<dHat 时写入 narrow pair；`core/GIPC.cu:10333` 按 use_toi 选择 AL 或原 barrier，`:10355` 同样替换地面项。`solver/toi_solver.cu:463` 的 publish_contacts 将当前 AL 集发布给接触缓冲；空集因此不装配 AL 接触项。
- 上表 IPC 帧末距离仍在屏障半径内，且存在大量 narrow pairs；“无几何接触”不等于“IPC 接触能量及其导数为零”。**表中 2625 是帧末数量，不是初始/第一次装配数量；当前紧凑日志没有逐项列出初始 barrier 活跃对，不能伪称已核验其初始精确列表。**

还有一个直接反证：第一次 PCG 的计数及 rho 在执行任何 inner 退出判断之前就已不同。因此“只差第二步后的退出条件”不是完整解释。rho 是预条件残差量，不是能量、真残差或条件数；单凭它也不能断言差值全部来自 barrier。接触目标不同是源码支持的首要混杂因素，完整 A/b 分项仍待验证。

TOI r1 最后原始方向轴速度为 **0.005339399881225453 m/s**，L2 最大顶点位移为 5.388436388497953e-5 m。此场景 Stiff 名义轴速度阈值是 `.01*sqrt(11.122765781907146)=0.03335081075762199 m/s`（位移 3.335081075762199e-4 m）。该 TOI 方向已低于 Stiff 名义阈值和 TOI 的 .05 阈值两者；不是落在两阈值之间的放宽区间。由于 Stiff 检查上一方向缓冲且接触目标不同，这不是同状态退出等价证明，但足以将“速度门槛偏松导致 mixed f1 差异”降为次要待验证假设。

## 3. 初始化、材料、质量与 ABD：已排除到哪一层

已证明：三臂实际 `scene.json` 完全相等。V2 分析与原 CCD identity 记录还核对了全场景各臂的 `state_0000.bin`、`topology.bin`、`masses.bin`、`boundary_types.bin`、`metadata.json`；结果 `same_initial_state_scene_materials_topology=true`。同候选 IPC host/Graph 的 f1 min J 与 Stiff 相符，而两种 TOI MAS 作用都接近 .999981，这不支持把初始分歧归因于 Graph 或因子逆候选。

源码支持但尚非完整状态转储证明：

- `sources/stiff_perf_v50/StiffGIPC/app/benchmark_scenes.inl:20` 从同一冻结清单载入几何、材料、dt 和边界；它没有给 TOI 单独设置初速度。`io/load_mesh.cpp:537` 等初始化零顶点速度，`abd_system/abd_system_parms.h:8` 默认零广义速度。
- active `app/gl_main.cu:1446` 在选择 IPC/TOI 前调用共同 `computeXTilta`。v50 `core/GIPC.cu:7974` 实现 x+v*dt+g*dt²；`:11296` 调用共同 ABD q_tilde，后者见 `abd_system/abd_system_function/cal_q_tilde.cu:19`。两后端在 `core/GIPC.cu:11401` 才分支。
- 共同 `computeGradientAndHessian` 在接触分支以外使用相同动能、ABD、FEM、布料和弯曲装配；FEM 质量对角见 `core/GIPC.cu:10265`，ABD 装配见 `:10371`。TOI 不在这些位置替换材料或质量模型。
- **仍未冻结证明**：首次求解时的 x_tilde、q/q_prev/q_tilde/q_v、ABD 完整广义质量矩阵、实际分项 A/b 的逐字节/数值等价。顶点质量一致不等于已经验过全部 ABD 广义质量块；基线缺少实际速度导出，也不能以位置差分补成速度证据。

## 4. 下一步应先做的有界审计

先在同程序、同进程受保护状态上冻结 mixed 第 1 次装配，分别输出 IPC 与 TOI 的非接触项及接触项 A/b、完整预测/ABD 状态与活动对，检验是否恰由接触项解释差值；不先改阈值，不撤掉任一安全 CCD。其次在从零推进的首次困难系统（24–26 帧）比较 AL 接触曲率相对于非接触 Hessian 的规模、约束重叠、预条件后谱/真残差与迭代量。必须区分已经不同的轨迹状态和同一系统内的预条件器影响。

若同状态审计表明初始非接触项一致、差异确来自 IPC 的有限距离屏障响应，应把它记录为目标函数/接触模型差异，再判断如何满足用户的“相对基线质量不退化”要求；不能用收紧停止条件假装恢复了同一目标函数。

证据入口：`downloads/autodl_factor_20261004_4090/reports/active/AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json`；`runs/active/autodl_window_mixed_{stiff,ipc_host,toi_host}_r1/{output/stats.json,output/scene.json,run.log}`（均位于同一下载根）；历史语义见 `reports/active/HISTORY_SEMANTICS.md`。本次只读复核支持以上定位，不构成修复完成、物理质量通过或性能认证。
