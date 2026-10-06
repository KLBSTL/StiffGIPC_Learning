# K4 Graph边界：完整窗口成本、状态保护与有限预算

2026-10-06，只读分析；没有修改原生代码/工具或运行GPU。

**判断：固定K4可作为一个有界测量假设，但尚未证明有生产收益。当前两布料完整窗口的线性占比约45%–46%，要净省整窗5%，必须在线性总阶段净省约11%；节点追踪里的30µs空隙不是已证明可回收成本。** 本批仅预声明一个实现假设H1：保持K1为参考、严格保护停止状态的K4展开。最多两个实现假设是上限，不要求凑满两个；没有依据时不增加IF图、K网格或退休融合。

## 1. 应使用哪些数据作预算

主依据是 `local_bvh_rounds_20261006` 两场景各三轮 **D1S1参考**：固定兔子从零59帧，悬挂布料从零51帧；池关，普通/swept refit均开，legacy .01/min6、rho1e-4、dt=.01，程序`b2ad…f812`。这些是完整已声明短窗口，不是100/300帧最终验收；共享WDDM负载也没有变成受控独占基准。

`phase_ms.pcg`在当前记录中是包含线性准备/转换/预条件器/迭代及分发的阶段，**不能将其全部视为conditional Graph循环**。完整窗口的阶段比例来自逐帧和，不能用单个f57的49.84%包络替代。

| 场景/轮次 | 全窗口T秒 | 线性L秒 | L/T | 净省整窗5%所需ΔL ms | 所需L净减幅 | 对应线性加速比 |
|---|---:|---:|---:|---:|---:|---:|
| fixed r1 /59f | 7.728402 | 3.499199 | 45.277% | 386.420 | 11.043% | 1.1241× |
| fixed r2 /59f | 7.350037 | 3.287296 | 44.725% | 367.502 | 11.179% | 1.1259× |
| fixed r3 /59f | 7.328312 | 3.292960 | 44.935% | 366.416 | 11.127% | 1.1252× |
| hang r1 /51f | 2.791654 | 1.289216 | 46.181% | 139.583 | 10.827% | 1.1214× |
| hang r2 /51f | 2.530805 | 1.169736 | 46.220% | 126.540 | 10.818% | 1.1213× |
| hang r3 /51f | 2.499712 | 1.141190 | 45.653% | 124.986 | 10.952% | 1.1230× |

公式：`f=L/T`，若其他阶段不变，整窗省q需要`ΔL=qT`、线性净减幅`q/f`、线性加速比`1/(1−q/f)`。整窗10%需要线性减约21.64%–22.36%。当前f<.5，即使线性阶段免费，也不足以相对当前参考实现2×；这是成本上界，不是优化承诺。

若候选仅改变PCG入口/循环，用`g=PCG可改变部分/L`，则所需该部分净减幅是`q/(fg)`；不能用q/f降低局部验收要求。f57 OFF中PCG entry包络65.286ms/linear80.610ms，g约.810，仅可作局部参考；若整窗g类似，局部PCG约需净减13.6%–13.8%才能贡献5%整窗。完整窗口的g尚未直接测定。

交叉核对：Step3完整pool-off fixed三轮的线性比例45.015%/45.573%/45.024%，hang首轮47.286%，与主预算量级一致，但绝对时间不同，不能混成新配对。Step3被显存保护中止的hang on2不纳入完整窗口分母。`EDGE_RESULTS`的两场景明确`full_scene_timing_usable=false`；6.347186s/2.238890s含局部probe，只用于确认路径完成，**不作为K4预算/速度分母**。

Step3的历史fixed Stiff/off配对中位约1.4748×也未认证；若只作算术，达到2×还需再省当前总时间约26.26%，约相当于省掉当前线性阶段58%。不能用本轮5%组件筛查把它包装成2×已可达。

## 2. K4可省回边与必付开销

[LINEAR_DIRECTION](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_rounds_20261006/LINEAR_DIRECTION.md)已核对f57 OFF：467迭代、8406=18×467个Graph节点，同solve的461个回边空隙共17.734125ms；ON：484迭代、8712节点，478个空隙15.308129ms。两组边界中位30.272µs，已扣所有捕获GPU活动，但仍可能包含Nsight、WDDM调度/抢占；不能认定都是CUDA WHILE本身。

假定某次solve实际迭代I>0且K4正确保留停止时刻：

- 原回边`I−1`；K4回边`ceil(I/4)−1`；减少`B=I−ceil(I/4)`。
- 尾部额外算子子步`Q=4ceil(I/4)−I`，每solve0–3次。它们仍运行SpMV/MAS/CUB，不算实际PCG迭代但必须计入耗时。
- 按当前13槽守卫设计，原p_continue拆为p更新和finalize，再每块一个handle更新。若原18节点组成保持，其名义节点为每块`4×19+1=77`，而原四步72。候选减少的是WHILE回边，**不是总GPU节点数**；最终必须以实际capture验证节点数。

从完整D1S1 stats的真实迭代数逐solve计算：

| 场景/轮次 | 方向/实际PCG | 减少回边B | 额外尾步Q | Q/I | 每减少回边至少需净省µs，才能省整窗5% |
|---|---:|---:|---:|---:|---:|
| fixed r1 | 334 /19854 | 14750 | 562 | 2.831% | 26.198 |
| fixed r2 | 319 /18869 | 14021 | 523 | 2.772% | 26.211 |
| fixed r3 | 335 /19760 | 14684 | 544 | 2.753% | 24.953 |
| hang r1 | 297 /9101 | 6715 | 443 | 4.868% | 20.787 |
| hang r2 | 297 /9105 | 6719 | 439 | 4.822% | 18.833 |
| hang r3 | 297 /9099 | 6713 | 445 | 4.891% | 18.618 |

末列为`0.05T/B`，**尚未扣尾算子、新增finalize/handle、门控、capture和显存成本**。按77节点/块估计，总节点数fixed反而多约9.9%–10.0%，hang多约12.1%–12.2%。挂布短solve更多，尾步更易抵消收益；不能只测固定兔子长系统。

可复算净预算模型：`净省 ≈ B×可消除的生产回边成本 − Q×尾算子成本 − 新增门控/节点/捕获成本`。f57 OFF把已捕获空隙直接乘.75仅得13.300594ms的乐观上限，未包含上述抵消项，且不能外推到无Nsight完整窗。K4必须在非Nsight同系统重放中证明净省；若生产回边成本明显小于末列，候选应立即停止。

## 3. 尾部重算：主状态、诊断与M工作区必须分开

详细顺序与跨block防竞争依 [GUARD_DESIGN](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_graph_rounds_20261006/GUARD_DESIGN.md)。本节强调观测/比较契约。

当前K1标量布局为：s0=current rho、s1=pAp、s2=new rho、s3=alpha、s4=beta、s5=converged、s6=count、s7=initial rho、s8=previous rho、s9=error。新s10=active、s11=pAp_work、s12=rho_next_work将scratch结果与原报告槽分开。

| 对象 | 停止后的要求 | 对照方式 |
|---|---|---|
| x/r/p、b | x/r/p保持最后一个**真实K1 Graph迭代**终态；b始终只读 | 真实终止点与后续0–3空步逐位比较；production主解在诊断退出后恢复 |
| s0..9及返回值/实际迭代数 | 首次停止后冻结；不得后续scratch改变rho/pAp/错误、停止类型或count | 与K1原始终止体比较；既检查结果也检查首次停止/失败子步 |
| s11/s12，z/Ap，CUB临时区 | 允许空步重算覆盖；不是可导出的最终逻辑rho/pAp | 单独标为scratch，故意poison并验证不传播至冻结槽/主状态 |
| MAS `d_multiLevelR/Z`及64位对应四个buffer | apply会重置/重写，空步允许改变 | 完整快照保留，operator身份仅排除这四个已声明工作向量；比较下一次apply/solve是否被残留污染 |
| A、b、MAS层次/映射/因子/模式、ABD逆块、维度/地址/布局 | 必须保持持久算子身份，不因尾重算改变 | 完整snapshot/hash与graph signature；不能把未知变化一概归为scratch |

**pAp首个失败值尤其容易被破坏。** 当前`fail_pcg(code,rho,curvature)`写入breakdown_rho/curvature；若尾部dot继续直接写s1，即使x/r完全不动，也会把首个非法曲率换成后续搜索方向的值。必须让CUB只写s11，alpha在active时才提交到s1；s2同理。错误rho、NaN/Inf类别与首次发生时刻必须保留；JSON把非有限值写为null不能单独作为一致性证明，fixture应另核原始位模式或明确非有限类别/符号。

比较对象必须是K1 Graph，不能误用host终态：host在x/r后判previous-rho并跳过末次M/dot/p；当前K1仍做末次M/dot及p更新，错误情况下也可能写最后一次p。K4只能冻结**这次真实K1终止体之后**的空子步，不可顺便改变已有最后一次p或s2行为。s1–s4原初始化不全部赋值；比较前应给两路相同初始字节，不能拿未初始化历史值伪造不一致。

预条件器内部状态审查：当前GlobalLinearSystem先r→z，再ABD局部覆盖，MAS清R上层/清Z后限制、局部作用、延拓。ABD apply读逆块、写z，不修改逆块；MAS apply的四个R/Z向量是既有明确scratch排除项。其余因子、连接表、restriction表、索引、地址不能修改。下一次调用须保留原清零和覆盖顺序；若不同尾长度影响下一次输出，说明scratch并非独立，停止候选而非扩大排除清单。

K4捕获签名需要K/标量布局及地址、buffer容量、n、max_iter、容差和M身份，K1↔K4切换必须失效旧图。现有无active/error守卫的fused_diag_update应显式拒绝与K4同开；能够替换apply的探针/实验回调需禁用或单独验证，不能静默回退。多block p更新与finalize分节点，不能用同kernel线程0改active影响其他block，也不能用`__syncthreads()`替代全kernel边界。

## 4. 预声明三阶段计划与停止门槛

这是供主任务执行的有限计划，不是本报告已执行的测试。只做H1固定K4；K1作为同一新程序中的原路径。任何修复都仅纠正H1的合约实现；不更换K、不顺手引入融合、不以“最多两个”作为必须展开第二条分支的理由。

### 阶段A：把停止语义做成可验证的具体实现

一次独立构建、现有回归及一次新真实Graph守卫fixture套件。必须调用生产K4门控；N至少513，覆盖三block、尾元素、块内第1/2/3/4步停止、fixed-iteration1–5、max_iter2/4/5/6、零RHS、rho=0但r非零、负/非有限rho、非正/非有限pAp、alpha/beta溢出。用负值/NaN空步scratch验证首次失败证据不被替换；核对count、iteration_limit、返回值及handle。验证模式切换、维度/地址增长和cache失效。

**停止：**任一主状态/首失败标量改变、持久A/M修改、跨block遗漏写、非法模式静默回退、现有材料/停止默认变化，均不得进入速度测试。允许修复明确实现错误；不能通过放松比较标准或替换K1终止语义使fixture“通过”。

### 阶段B：非Nsight、同进程冻结系统证伪成本假设

执行顺序以已冻结的 [LOCAL_GRAPH_OPTIMIZATION_PLAN_20261006.md](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/docs/LOCAL_GRAPH_OPTIMIZATION_PLAN_20261006.md) 为准：先fixed f2/n1，再fixed f57/n1，必要时hang f41/n1，最多三个从零推进批次；每批后作继续/停止决定，不依赖未验证等价的checkpoint。每系统两次预热、七对交错K1/K4。计完整PCG入口（初始M/rho、capture/cache情况、replay和最终读回分别列出），另外提供CUDA-event区间；不同计时口径不可相加。保存K1自身重复范围、解与CPU重算真残差，rho门槛保持1e-4，不能冒充真残差门槛。

记录真实I、Q、chunk数和capture成本；K4净收益必须含尾部重算/新节点。诊断结束恢复主解，核对输入和持久M身份，检查下一次K1应用不受工作区残留影响。短系统不准以迭代数少而删除；完整候选weighted收益也不能仅取最好的长系统。

**进入C的必要条件：**正确性与主状态保护通过；固定系统完整线性重放稳定净省至少15%，短系统无稳定退速；结合已测可改变阶段比例能解释至少5%完整窗口预算。完整求解重放与包括装配/准备的全场`phase_ms.pcg`必须分别命名，不得将前者局部收益当作后者收益。15%是有限筛查条件，不是整窗保证。若真实可省回边成本低于第2节需求、tail/新节点吃掉收益、K1重复范围外出现未解释差异或资源预算触发，停止H1；最多一次有明确证据的修订，不追加K2/K8/IF图等网格来延长分支。

### 阶段C：两布料完整短窗口，各三组配对

仅B通过后运行：fixed59帧、hang51帧，从零；同一新程序K1/K4、无重探针/Nsight，各三对交错（K1→K4、K4→K1、K1→K4）。最多12次场景运行；B最多三个从零推进批次。原单次120秒、显存保护、GPU串行和完整安全CCD不变，资源失败保留并停止，不现场加时重试。

逐轮审查初态/配置/程序身份、旧材料界限、布料真实位置/速度与ABD/FEM分组、方向/PCG、真实线性阶段净收益、尾部计数及退出。冻结计划要求整窗净省至少5%且无明显质量回退才保留；阶段目标为两布料相对当前Graph的几何平均至少1.10×，任一场景不回退超过3%。分别公布两场景三对结果和更快的对数，不用均值掩盖场景退速。原参考或候选超旧材料界限、轨迹超预声明参考自身范围均保留待确认，不自动放宽，不宣布同质量。

**最终停止/交付：**任何必要门禁未通过即保持K1默认，交付失败证据与明确“本候选未达标”；完成阶段C也只代表当前短窗口组件结果，不自动认证2×、100/300帧或独立CCD。若需正式推广，另按原全场景质量与七组配对/置信界要求验收。本批不以继续小kernel、增加迭代预算或参数扫描替代未达标结论。

## 5. 可复算性与限制

- 主成本：`runs/local_bvh_rounds_20261006/{fixed,hang}_r{1,2,3}/analysis.json`中D1S1的`timing.prefix.solver_ms/phase_ms.pcg`；各run `output/stats.json`中每个newton.pcg.iterations用于I/Q/B。六份身份见[BVH_RESULTS](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_rounds_20261006/BVH_RESULTS.json)。
- 完整阶段比例交叉核对：[Step3 ROUND_REVIEW.json](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step3/ROUND_REVIEW.json)仅completed、pool-off；[EDGE_RESULTS](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_rounds_20261006/EDGE_RESULTS.json)明确排除其重探针全窗计时。
- 节点/空隙沿用已有只读SQLite核对及LINEAR_DIRECTION；CPU等待/NVTX包络从不与GPU活动相加。固定兔子节点成本不能按同百分比直接套给悬挂场景。
- 内部状态基于审查时K1 `pcg_graph_impl.inl`、`pcg_solver.cu::snapshot_operator_identity/fail_pcg`、`MASPreconditioner::preconditioning/graph_signature`、`ABDPreconditioner::apply`及本批GUARD_DESIGN。正在实现的K4还需要阶段A实际证据，本文件不是实现通过声明。
