# Robust 源码直接适配 StiffGIPC：可行性、速度潜力与投入判断

日期：2026-10-01。对象：当前 v31 融合工程、公开 Robust/libuipc 快照、旧 Robust 场景复现程序。

## 1. 结论

**值得做一个范围有限的核心模块适配原型；现有证据不支持整套替换后端，也不支持承诺恢复旧表中的 2–20× 加速。**

本次重新读了实际代码，并在本机追加 30/60 帧、各四组、每组三次的交错诊断。公开实现的求解流程、活动集维护和帧级摩擦快照有复用价值；工程上的主要成本在状态与接口适配、行为核对和质量验证。数学公式并不需要重新研究一遍。当前 Stiff 也使用 ABD/FEM 混合自由度、3×3 块矩阵、Eigen 和 CUDA，这些共同结构有助于适配。

但公开代码也有明确的导数问题。本次固定输入检查确认，AL 边—边摩擦能量与其梯度采用了不同的顶点权重。直接复制会带入这个问题。旧 Robust 的宽松停止策略及与当前程序不同的接触间隙，进一步限制了旧速度结果的解释。

建议下一阶段做一个独立的 `Robust-derived` 路径：保持 Stiff 的材料和线性后端，适配当前 trial 活动集更新、持久 C/λ、独立接触年龄、真正的帧级摩擦快照，再评估 GPU 活动集维护。先用小场景判断收益，随后决定是否继续投入。

**本报告中的运行比值均为诊断耗时比；没有完成跨程序同等物理质量和连续 CCD 验收。**

## 2. “本地 Robust”具体是什么

| 来源 | 本次核对 | 用途 |
|---|---|---|
| `externals/history-Robust` | `wiso-enoji/libuipc`，提交 `389f8a52a29606ccaf5640607dd1e38987500988`；核心文件无工作树修改，动物井示例已有其他任务修改 | 公开实现的行为参考 |
| `references/robust_local/public` | 17 文件冻结清单中的公开核心文件与许可证 | 本任务内可复查的供体 |
| `stiffGIPC/barrier-free` | 旧场景复现源和已编译 DLL/EXE；本次 EXE SHA 与 9 月 27 日保存记录一致 | 独立程序的速度潜力探针 |
| `references/robust_local/prior_derived/advance_al_paper.cu` | 修改过外层计数和实际 alpha 的派生实现 | 另一个算法行为分支，不能混入公开列 |
| 当前 `sources/stiff_fused` | v31，base 提交 `bb2849a7b292099581907937860d96ecfdf42588` | 拟适配的目标工程 |

论文项目页的 Code 链接指向 libuipc 的 AL-release。其 README 明确区分 libuipc 实现与论文原始实现，并说明原始实现可联系作者获取。因此本次可直接适配的是公开 libuipc 实现；不能假定已经拥有论文原始 GPU 模拟器的全部优化。[项目页](https://simulation-intelligence.github.io/barrier-free/)、[固定提交 README](https://github.com/wiso-enoji/libuipc/tree/389f8a52a29606ccaf5640607dd1e38987500988/apps/AL_examples)。

本次供体目录只读。测试 EXE 和全部运行 DLL 复制到本任务的 `runs/local/robust_port_probe_20261001*/frozen_bin`；输出、工作目录和引擎 workspace 均在本任务内。原始资产通过原程序已配置路径只读加载。没有重编其他任务的源树或改动其结果。

## 3. 本次新测得的速度潜力

### 协议与限制

- RTX 3070 Laptop，8 GiB。开始时约 6.17 GiB 空闲，满足小场景预算；桌面图形程序仍运行，没有宣称 GPU 完全独占。
- 布落球 `cloth_sphere7_l`；布料 2,601 顶点，dt=0.01 s；分别测 30 帧和 60 帧，各组三次，GPU 串行。
- 交错顺序：base → Robust 0.05 → Graph → Robust 1.0，第二轮反序。
- 当前 base：Newton 参数 0.01，PCG rho 阈值 1e-4。Graph：v31、IPC 路径、suite=0。
- 旧 Robust 场景程序：PCG `tol_rate=1e-3`、累计 TOI threshold=0.001、decay=0.9、min_iter=6；仅覆盖 `velocity_tol` 为 0.05 或 1.0。
- Stiff 记录 solver 窗口；Robust 记录 `world.advance` 整步窗口，两者插桩边界不同。Stiff 导出逐帧状态，Robust 本轮只导出初态和逐帧计时。比值保留为诊断。
- 配对比值 = 同一重复编号的 base 时间 / 方法时间，再取三次比值中位数。**它不等于时间中位数之比。**

### 60 帧结果

| 程序 | 窗口时间中位数 / s | 配对诊断比值 | 三次比值范围 | 方向求解中位数 | CG 迭代中位数 |
|---|---:|---:|---:|---:|---:|
| 当前 base | 6.217 | 1.000× | 1.000–1.000 | 327 | 12,573 |
| 当前 base + Graph | 4.525 | 1.411× | 1.311–1.437 | 319 | 12,582 |
| 独立 Robust，velocity_tol=0.05 | 4.899 | 1.269× | 1.189–1.420 | 348 | 16,720 |
| 独立 Robust，velocity_tol=1.0 | 2.550 | 2.368× | 2.351–2.553 | 164 | 7,440 |

![60 帧实际计时与方向求解数，点为各次运行，柱为中位数](figures/ROBUST_PORT_POTENTIAL_20261001.png)

30 帧窗口中，base 1.566 s；Graph 1.294 s、配对比值 1.208×；Robust 0.05 为 1.298 s、1.205×；Robust 1.0 为 0.765 s、2.056×。30 帧较短，60 帧用于覆盖更深的接触演化及 v31 已出现问题的帧段。

两档合计 **24 次完成、1,080 个计时帧**，未观察到 Newton/PCG 上限。Robust 源码公式审计单独完成。运行完成和日志无上限不构成碰撞/质量通过。

### 这些结果说明什么

1. **独立 Robust 仍有速度潜力。** 在当前更快的 base 面前，1.0 阈值下仍出现约 2.37× 的原始耗时优势。这支持有限适配实验。
2. **“Robust 必然大幅减少求解数”取决于具体配置。** 60 帧下 0.05 阈值有 348 个方向，超过 base 的 327；1.0 阈值降到 164。两档同时改变 inner 停止与小方向下的线搜索接受行为，不能把收益全部归因于 CCD/活动集。
3. **0.05 档没有显示出超越现有 Graph 路径的时间优势。** 它的中位时间 4.899 s，Graph 为 4.525 s；两者都比 base 快。这降低了“为默认小阈值整体换引擎”的投入价值。
4. **单次执行成本也存在差异。** 0.05 档的方向与 CG 数较多，却仍比 base 的窗口短。矩阵、预条件器、kernel 融合、几何与插桩均不同，当前不能把这一现象归因于某一个模块。
5. **不能把 2.37× 与 Graph 的 1.41× 相乘。** 独立 Robust 已使用 fused PCG；移植后受益的成本占比与 Graph 缓存/迭代次数都会变化。

本次没有对旧 Robust 二进制修正导数问题。表中测的是现有程序的行为；修正后的速度和质量需要独立测量。

## 4. 能直接复用到什么程度

| 模块 | 可复用内容 | 必须适配的接口/语义 | 判断 |
|---|---|---|---|
| `advance_al.cu` | 预测、AL 线性化、Newton、CCD、累计 TOI 的控制次序 | `SimSystem::require`、scene config、time integrator、line search、ABD/FEM reporter 转为 Stiff 的状态/调用 | 高价值，适合先适配流程 |
| `global_active_set_manager.cu` | GPU per-vertex earliest TOI、排序/扫描、去重、接触保留及淘汰 | primitive 编号、canonical key、MUDA buffer/view 转 `cuda_tools`，容差及旧乘子继承规则 | 高价值；不是把文件加入 CMake 即可 |
| 帧级摩擦快照 | 帧开始冻结候选及法向乘子，使用上一帧几何确定切向基与坐标 | 必须独立于法向活动集增删，贯穿该帧所有 outer/inner；PT、EE、ground 分别处理 | 高价值，与当前代理历史有实质区别 |
| 法向 AL 公式 | 线性化间隙、slack、λ、低秩 PSD 接触 Hessian | 间隙/厚度、μ、符号、混合坐标映射 | 已部分具备；重点是同输入对照 |
| EE 摩擦函数 | 正确的 EE 运动映射和摩擦模型 | 先修正本报告确认的 Jacobian 调用 | 不能原样复用 |
| `linear_fused_pcg.cu` | device 标量、融合向量更新、SpMV+dot、周期 host 查询 | 当前 solver 接口、ABD/FEM 预条件器、rho 与停止顺序、Graph 缓存 | 作为单独线性实验；优先级低于活动集与控制 |
| BVH/CCD | simplex/half-plane 检测与 per-pair TOI 数据流 | surface manager、厚度、近退化、ACCD 参数含义、安全验证器 | 替换成本高，保留 Stiff 安全 CCD更利于首次适配归因 |
| 完整 libuipc 引擎 | 已有独立程序、scene/constitution/reporter 框架 | 引擎生命周期、材料、DOF 排布、动画边界、插件 DLL/so、构建与输出 | 属于后端迁移，当前不建议投入 |

公开实现使用 MUDA、系统注册、`GlobalVertexManager`、`GlobalTrajectoryFilter`、`GlobalContactManager` 等服务；目标工程使用自己的 `cuda_tools` 和 `gipc` 接口。公开 CUDA 后端要求 C++20，当前 Stiff 构建用 C++17。两者都有 3×3 block triplet 和混合 ABD/FEM 的概念，**转换有基础，但没有可直接链接的共同求解器 API**。

Windows 已编译 DLL 不能直接拿到 AutoDL Linux 运行。后续 CUDA 12.8 阶段需要从隔离源重新构建、测试；本次没有建立新的远程兼容性结论。

## 5. 最有价值的四个行为差异

### 5.1 当前 trial 的候选应及时进入下一轮求解

公开 `advance_al.cu:427–434` 更新 λ 后，对当前求出的 trial 做 CCD，并马上更新 C。当前 `toi_solver.cu:1221–1234` 用 previous trial 更新 C，再用新的 trial 做 safe CCD。

v31 的 C+λ 分支中，全局最小 TOI 的 blocker 有 160/215 次、约 74.4%，尚未进入生成本轮方向的活动集。它提示延迟探索可能有较高成本。它没有证明“切换 current trial 就必定消除所有 blocker”：候选筛选、其他顶点接触和运动路径会继续变化。这个差异值得优先做受控比较。

两套更新顺序应分别标记为公开行为和论文行为，不能合并后仍声称逐行复现原 Algorithm 1。

### 5.2 持久 C、λ 与接触年龄

公开实现持久保留 C 和 λ，以整数 `cnt` 表示释放历史；法向权重由 `pow(decay,cnt)` 得到，合并时阈值为 25。当前直接存 gamma，并用 `gamma<0.01` 淘汰；重置 gamma 同时重置了衰减与淘汰的依据。

两者在特定初值下可表示类似衰减过程，但淘汰阈值和继承规则不同。v31 C+λ 已把 60 帧方向从 cold 的 762 减到 560；C 只保留身份则曾出现 248 次 inner 的长循环。应引入独立接触年龄/释放状态，而不是继续把 gamma 当作多个角色的共同开关。

### 5.3 真正的帧级摩擦快照

公开 `advance_al.cu:300–304` 在预测前 snapshot；`global_active_set_manager.cu:761–776` 复制摩擦候选与 λ。摩擦函数用 `prev_positions` 确定坐标和切向基，在该帧求解中保持相应参考。

当前 `toiFrictionSets` 在外层重新从 AL 接触组装摩擦信息；v31 的独立 friction λ/γ 仍每轮按同样约束递推。**它只是历史消融，不能代表完整的公开帧级快照。** 上轮 C+friction 有 9 个独立 CCD 保守标记，也不能据其减少方向就选为生产默认。

这一适配需要同时固定候选、力、切向坐标、基底和刷新时间；只保存一个法向力标量不够。

### 5.4 GPU 活动集维护

当前 `swept_query:652–678` 下载 TOI、pair、surface 索引，在 CPU `map/set` 去重和筛选，再上传接触；每轮还下载接触状态。公开实现使用 GPU atomicMin、radix sort、exclusive scan 和 compaction。

还有筛选细节差异：公开 per-vertex earliest TOI 来自整批候选，使用接近 1 的排除阈值与 `1e-6` tie 容差；当前先排除已有接触，再对新集合求 double 最小值并用精确相等。迁移时不能只换容器，而忽略集合定义与继承次序。

GPU 维护适合较大活动集。小场景收益须测量；减少内层与外层循环通常比只缩短 CPU 排序更有潜力。既有 profile 来自不同版本/轨迹，不能用旧占比推算本次总加速。

## 6. 本次确认的源码问题及其他风险

### 6.1 EE 摩擦的导数映射错误：已有固定输入证据

本地冻结 `al_contact_function.h:179–186` 中，切向位移用 EE 运动，但 J 调用 `point_triangle_jacobi(basis,gamma,J)`。工具函数对四个顶点的权重分别为：

- EE：`[1−g0, g0, g1−1, −g1]`。
- PT：`[1, g0+g1−1, −g0, −g1]`。

二者一般不同。该调用也存在于本次浏览到的 [公开 AL-release 函数](https://raw.githubusercontent.com/wiso-enoji/libuipc/AL-release/src/backends/cuda/contact_system/al_contact_function.h)。本次局部证据以本地固定提交为准，不把远程缓存读取当作完整源码版本锁定。

CPU 检查选非平行、最近点参数均在边内部的普通输入 `g=(0.25,0.60)`；固定 prev 几何、basis、法向力，在平滑阈值外直接对该能量做中心差分：

| 差分步长 / m | 公开 PT 映射的梯度相对误差 | 正确 EE 映射的梯度相对误差 |
|---|---:|---:|
| 1e-5 | 0.4625733 | 4.37e-7 |
| 1e-6 | 0.4625730 | 4.37e-9 |
| 1e-7 | 0.4625730 | 4.45e-11 |
| 1e-8 | 0.4625730 | 6.88e-11 |

这是对已核对源码公式的 CPU 代数验证，未直接执行原 CUDA 函数。它确认该映射在普通输入上与能量不一致；尚未统计具体场景触发次数，也没有证明旧速度主要由此造成。修正 J 会同时改变梯度与 Hessian 的组装，速度可能提高，也可能降低，必须保存修正前后两列。

### 6.2 小方向绕过能量下降

公开 line search 使用 `E<=E0+1e-12 || newton_converged`。提高 velocity_tol 不只减少 inner 迭代，还扩大了可以越过能量下降判据的方向范围。这与当前“已接受且能量下降后允许小方向退出”的实现不同。

因此 `velocity_tol=1.0` 的 2.37× 不能自动移到保留当前严格接受规则的适配版，也不能直接解释为论文方法的纯算法收益。

### 6.3 未推进 safe 时仍累计 beta

公开代码仅在 alpha 超过下限时推进 safe，但下一句仍按 alpha 累计 beta。之前派生版对未推进状态改用 alpha=0。新适配应保留实际几何进度的修正，并明确标为派生行为。

### 6.4 距离退化与几何余量

公开 PT/EE 的线性化仍有 `GradD/(2*D)`，没有因为名字是 Robust 就自动消除 D 接近零的奇异性。旧 Robust 的 AL 约束减去 thickness 与 d_hat；本场景 d_hat≈0.002503 m，当前默认 AL delta≈0.0002503 m，约差十倍。

两个 CCD 的 eta/保守比参数语义也不同。不能为减少小 alpha 直接照抄 `0.001` 到 Stiff ACCD 的 `0.2` 参数位置。当前独立 CCD 门槛继续适用。

## 7. 为什么旧 2–20× 不能作为移植预期

旧多场景表比较的是较旧、较慢的 base，一部分数据在 RTX 3090 Ti/CUDA 12.6 上；当前 base 已有累计 TOI 提前终止。旧 Robust 运行又覆盖了 `velocity_tol=1.0`。硬件、分母、停止与接触语义均需分开。

同机历史数据收紧 Robust 阈值到 0.05 时，方向数明显增加，轨迹也改变。桌布和悬挂布的质量偏差曾达到毫米到厘米量级。旧记录提供“某些场景值得尝试”的证据，未提供“复制代码即可得到同质量 20×”的证据。

当前 v31 也未形成可交付质量结论：八组 60 帧布料位置偏差约 3.4%–5.3%；细步参考自身仍有 2.333% 的差异，不是收敛真值。继续适配的价值应由可完成性、迭代工作量、物理诊断与当前 base/Graph 的实际收益共同决定。

## 8. 投入量级

以下为熟悉当前工程的一名开发者的**初版工程量粗估**，不包含未知收敛问题的全部排查，也不是交付承诺。各项可共享工作，不能机械相加。

| 路线 | 初版量级 | 主要额外成本 | 判断 |
|---|---|---|---|
| 已有独立 Robust 二进制探针 | 小 | 解释物理和计时差异 | 本次完成 |
| 在隔离 Robust 副本修 EE 导数、重编并对照 | 约 1–2 人日 | 大型供体构建和修正前后质量 | 建立可信行为参考有价值 |
| Stiff 后端上的控制/持久接触/摩擦快照适配 | 约 3–7 人日 | PT/EE/ground 的固定输入核对、长帧验证 | 推荐的有限原型 |
| GPU 活动集移植 | 约 3–7 人日 | key/去重/继承、buffer 生命周期、可重复性 | 算法路径稳定后投入 |
| fused PCG 另做同 A/b 接口适配 | 约 2–4 人日 | 实际残差、混合预条件器、Graph 合并 | 依据 profile 再决定 |
| 整套 libuipc 后端替换 | 数周起 | 材料/动画/混合自由度/CCD/部署与全矩阵验证 | 当前投入收益不清楚 |

最大的不可预估成本是物理质量与复杂接触收敛。大量编译成功的代码不一定产生更快、同等质量的结果。已有框架相似处会缩短机械适配；源码问题及语义差异会增加验证工作。

## 9. 建议的下一步与停止条件

### 阶段 A：明确可信供体

在新隔离子目录冻结旧 Robust 和 v31，修正 EE Jacobian；对 PT/EE/ground 做能量导数、切向刚体平移、摩擦耗散检查。保存公开、修正派生两列。先测修正是否明显改变同一小场景的方向数或运动。

### 阶段 B：有限适配原型

保持 Stiff 的惯性、布料和 ABD 能量及当前安全 CCD，建立可切换的 `Robust-derived` 控制路径：

1. 当前 trial 的完整 CCD 候选进入下一轮活动集。
2. C 与 λ 跨帧保持，接触年龄独立存储，明确衰减/释放规则。
3. 帧开始建立独立摩擦快照，其 force、coordinates、basis 在该帧保持一致。
4. 保留未推进 safe 不累计进度的修正；明确严格接受规则与公开小方向放宽规则的差异。
5. 将 Kmin、累计 TOI、velocity_tol 和 PCG rho 分别记录，避免继续用相同参数名表示不同退出语义。

先比较与 v31 的**同输入状态/候选更新和固定 A/b**，再跑 30→60→100 帧。首次实现可保留现有 host 活动集，方便与 GPU 版本逐条核对。

### 阶段 C：决定是否继续

- 原型若在相同 Stiff 后端减少方向与 CG，且未增加异常回溯/近零间隙，才投入 GPU 活动集。
- 小场景通过已有安全和物理诊断后，关闭额外审计做三次交错计时；同时比较当前 base 和 base+Graph。以相对 Graph 有稳定增益为继续工程投入的证据，而不是仅胜过旧 base。
- 若只在 1.0 阈值下快、质量要求下回到 base/Graph 附近，应记录为“宽松停止配置的收益”，停止扩大整体适配。
- 若修正 EE 后速度或轨迹发生大变化，先处理可信供体，不把未修正旧程序当加速目标。
- 若有限原型持续比 Graph 慢，且主要成本来自原问题的 Newton/CCD 工作量，应调整算法范围或保留 Graph 成果；不继续扩大为整套框架迁移。
- 有稳定小场景收益后，再增加桌布或多层布料一个接触更密集的场景，验证收益是否随接触强度增加。AutoDL CUDA 12.8 放在这之后构建与测量。

第一轮目标是证明“值得适配”的最小收益路径。约 2× 的独立程序潜力可作为调查目标，不能作为新原型验收的预设结果。

## 10. 证据与复核

- `ROBUST_PORT_SOURCE_AUDIT_20261001.json`：具体几何、两套映射、四档有限差分、源码 SHA；Hessian sandwich 字段只比较相同解析二维 Hessian 下的两套 J，未执行公开投影 Hessian。
- `ROBUST_STANDALONE_POTENTIAL_20261001{,_60F}.json`：24 次原始运行、有效配置、二进制/DLL SHA、配对比值。
- `ROBUST_PORT_POTENTIAL_20261001.csv`：两档窗口的汇总。
- `ROBUST_PORT_PROBE_VERIFICATION_20261001.json`：24 次完成、1,080 帧、无观察到的迭代上限；明确 `quality_gate_checked=false`、`qualified_speedup=false`。
- `figures/ROBUST_PORT_POTENTIAL_20261001.{png,pdf}`：实测成本与方向数，已打开检查。
- 历史依据：`ROBUST_SPEED_GAP_ANALYSIS_20260930.md`、`ROBUST_SOURCE_COMPARISON.md`、`TOI_WARM_HISTORY_V31_20261001.md`。

在本任务目录执行：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/audit_robust_port_feasibility.py
& 'E:/Anaconda/envs/DL/python.exe' tools/probe_robust_standalone.py --steps 30
& 'E:/Anaconda/envs/DL/python.exe' tools/probe_robust_standalone.py --steps 60
& 'E:/Anaconda/envs/DL/python.exe' tools/report_robust_port_probe.py
```

本次三个检查/运行脚本均退出 0，报告验证脚本输出 `runs_verified=24`、`total_measured_frames=1080`。运行器复用已有完整结果，不覆盖运行目录。30 帧源 runner 首次执行后加入了 60 帧参数支持，原协议保留当次 runner SHA，报告没有声称两批使用逐字节相同的 runner。

最终逐文件散列复核确认清单中的 553 个源文件全部不变，官方 base 和 v31 二进制 SHA 仍与原锁定值一致。新写入仅为隔离探针、公式审计、统计/绘图工具及其结果。本阶段完成的是适配价值分析与潜力诊断。
