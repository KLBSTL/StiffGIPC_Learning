# 当前线性执行路径：一个有条件候选及停损依据

2026-10-06。只读当前 `StiffGIPC/` 与 Step4 已完成的 fixed f57 off/on 节点追踪。没有编译、GPU运行、新捕获或原生修改；没有重新开放退休 MAS-dot、SpMV+pAp 或 ordered restriction。

**结论：没有发现可直接证明足以省整帧 5% 的零贡献预条件工作。唯一保留的候选是固定 K=4 的 conditional-WHILE 分块执行，但必须先在同一冻结 A/b/M、无 Nsight 的重放中证伪/验证迭代边界成本。现有 30µs 边界空隙是真实捕获现象，不是已证明可消除的生产开销；若普通重放不支持收益，立即停止，不开始别的小 kernel。**

## 1. 成本口径

两次捕获均为 `cloth_fixed_bunny_l` 第57物理帧，程序 SHA `b2ad19884fc01b81827d560f9f1984346b5488851995c2fd495f7427eac0f812`，IPC legacy `.01/min6`、MAS legacy、rho `1e-4`、conditional Graph。ON/OFF仅请求池开关不同，但它们不是相同状态：六次 Newton 的 PCG 合计467/484，接触与alpha亦不同。以下 ON 只作交叉核对，当前默认池关为主依据。

| f57诊断成本 | OFF | ON |
|---|---:|---:|
| CPU物理帧NVTX包络 ms | 161.738546 | 148.197846 |
| CPU linear.total包络 ms | 80.610369 | 84.910731 |
| linear / 物理帧 | 49.840% | 57.296% |
| 全GPU活动union ms | 112.626585 | 98.475947 |
| Graph实际kernel节点sum ms | 34.625436 | 39.575132 |
| SpMV类GPU sum（含输出清零）ms | 12.294395 | 13.427814 |
| MAS局部作用GPU sum ms | 4.929819 | 3.594397 |
| MAS限制GPU sum ms | 2.531676 | 3.259865 |
| MAS延拓GPU sum ms | 1.128766 | 2.283739 |

CPU父子scope不可相加；GPU时间与包含它的CPU等待不可相加。上表只是单个完成物理帧，不等于59/100帧总时间。若该帧要省5%，OFF须省8.086927ms，即线性总包络的10.03%；ON须省7.409892ms，即8.73%。完整轨迹收益仍需原质量门禁下的匹配配对验证。

## 2. 已证无须做或可能零贡献的工作，为何不是本轮主候选

### 2.1 纯悬挂布料不存在“空ABD kernel”大项

`cloth_hang_l`场景只有FEM布料；ABD数量为0时，[LaunchCudaKernal_default](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/cuda_tools/cuda_tools.h:82)在`total<1`直接返回。[ABDPreconditioner::apply](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/preconditioner/abd_preconditioner.cu:44)按body数量派发。纯布料没有这项可消除GPU成本；不能把固定兔子的ABD时间迁移到悬挂目标。

### 2.2 固定ABD的数值贡献为零，但不能随意留脏输出

固定兔子的ABD物体在输入中`fixed_mode=all`。当前固定体梯度严格置零（[setup_abd_system_gradient_and_hessian.cu:184](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/abd_system/abd_system_function/setup_abd_system_gradient_and_hessian.cu:184)）；ABD-ABD固定耦合被屏蔽（同文件:119）、barrier gradient不累加固定体（:316、:343），ABD-FEM交叉块遇Fixed也清零（:394）。因此有限数据和一致mask下，零初值PCG的固定ABD残差/搜索方向子空间保持零，其12×12逆作用对最终方向为零。

但当前[ABD apply kernel](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/preconditioner/abd_preconditioner.cu:10)仍负责**覆盖写出z**；`z`此前还被点积作为partial scratch使用。不能仅跳过kernel后让这12维保留旧partial。允许的重整必须明确维持z=0、固定/动态混合mask、类型变更、Graph签名和完整非有限诊断，不可将所有ABD一概删除。固定体Hessian为质量矩阵，不是恒等矩阵；求逆kernel确实工作（[同文件族:435](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/abd_system/abd_system_function/setup_abd_system_gradient_and_hessian.cu:435)），不能简单把矩阵值声称为I。

实际ABD apply为OFF473次/2.619770ms、ON490次/3.947737ms；ABD求逆各6次/1.941373、1.955802ms；收集对角块另.010240/.010144ms。即使这些都免费，总额仅OFF约2.83%、ON约3.99%帧包络。零作用特例可作为将来的小整理，但不够本轮≥5%主线依据。

### 2.3 全局r→z初始化确实被当前MAS局部覆盖，仍是小项

[gipc_system.cu:52](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/gipc_system.cu:52)创建ABD/FEM两个分区；[global_linear_system.cu:38](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/linear_system/global_linear_system.cu:38)为其连续排布偏移。MAS配置创建ABD和FEM MAS局部预条件器，不创建全局Diag（[gipc_system.cu:72](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/gipc_system.cu:72)）。当前[apply_preconditioner:348](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/linear_system/global_linear_system.cu:348)仍先全量复制r到z，再局部覆盖。ABD为赋值；MAS最终[collect:929](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/MASPreconditioner.cu:929)逐FEM节点赋值、不读旧z。因此在当前完整分区/完整覆盖合同下，这次seed copy的值没有最终贡献。

不能只凭`m_local_preconditioners`非空泛化删除：Diag/global、部分局部覆盖、新子系统或总节点/子视图不一致均需要保留fallback。若同时跳过固定ABD局部覆盖，必须补固定z写零，两种省略不能各自独立假定对方仍覆盖。

Graph内`memcpy32_post`每迭代1次，OFF467次/1.411710ms、ON484次/.624896ms；这是GPU活动，不是API等待。即使把本项与上述ABD项全部理想消除，也仅约OFF3.70%/ON4.41%；不把多个小项拼成已过门槛的候选。

MAS内部两次memset则**不能删除**：限制在粗层有atomicAdd（[MASPreconditioner.cu:888](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/MASPreconditioner.cu:888)），局部作用对mZ同样atomicAdd（:1102）；下一次迭代必须清旧累加。SpMV的输出清零也不是普通覆盖式输出，不能因每迭代都执行便认定冗余。

## 3. 唯一值得条件性验证的包络：WHILE迭代边界

现有Graph每个WHILE body恰好一个PCG迭代，18个实际GPU节点。节点顺序：SpMV清零/乘法、pAp两阶段归约、alpha、x/r更新、seed copy、ABD作用、MAS两次清零/限制/局部作用/延拓、rho两阶段归约、beta、零rho核查、p_continue。两份SQLite所有Graph节点总数分别8406=467×18、8712=484×18，实际CUB与memcpy/memset kernel均在序列内。

本核查按每个重复的Graph节点集合与对应stream分出六个solve，以`graph_p_continue`结束→同solve下一次`fill_y_zero_kernel`开始定义边界；减去**全部已捕获KERNEL、MEMCPY、MEMSET的区间union**。并非只扣本Graph活动。每个边界内其他已捕获GPU活动覆盖均为0；不跨越两个不同solve。

| solve序号 | OFF迭代 / 边界数 | OFF空隙ms | ON迭代 / 边界数 | ON空隙ms |
|---|---:|---:|---:|---:|
| 1 | 97 / 96 | 3.862527 | 99 / 98 | 3.221466 |
| 2 | 93 / 92 | 2.804224 | 89 / 88 | 2.685273 |
| 3 | 81 / 80 | 3.039446 | 64 / 63 | 1.931900 |
| 4 | 67 / 66 | 2.324416 | 73 / 72 | 2.517339 |
| 5 | 70 / 69 | 2.903353 | 76 / 75 | 2.486268 |
| 6 | 59 / 58 | 2.800159 | 83 / 82 | 2.465883 |
| 合计 | **467 / 461=467−6** | **17.734125** | **484 / 478=484−6** | **15.308129** |

边界中位两组均30.272µs；OFF范围29.088–846.015µs，ON29.471–359.968µs。空隙分别占物理帧10.965%/10.330%，占linear包络22.000%/18.028%。这些空隙不含跨Newton的碰撞或装配，已排除“漏统计同帧CUB/Graph节点”这一简单解释；但**仍包含Nsight节点采集、设备调度、可能的桌面抢占**。不能据此归因CUDA WHILE固有代价，更不能当作可全部省掉的生产时间。

### 固定K=4分块的理由与限度

只保留一个K值，不开2/4/8参数网格。若边界开销能在普通运行中复现，四个严格串行迭代共用一次WHILE回边，可尝试减少约3/4回边。直接按捕获空隙乘.75的乐观预算为OFF13.300594ms/8.224%帧、ON11.481097ms/7.747%帧；尚未扣新增门控、较大图、末块多执行的最多3次算子，也未证明这些空隙可省。两步方案的乐观预算仅约5.48%/5.16%，几乎没有维护余量，因此不建议从多轮小步调参开始。

这不是“把capture体for循环四遍”即可：当前[p_continue:38](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/solver/pcg_graph_impl.inl:38)无停止保护地更新p，alpha又覆盖previous-rho，CUB写s1/s2。直接展开会在已经收敛、breakdown或上限后继续改变状态，改变原有算法。

任何实现必须有**黏性有效迭代状态**：真实迭代才更新x/r/p、previous/current rho、迭代数及失败信息；收敛/error/limit一旦发生，后续子步不得解除它。最后仍在原来的x/r更新之后按previous-rho判停（[host:532](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/solver/pcg_solver.cu:532)），不提前/延迟接受一步，不松rho、不改max_iter−1规则。若为了少改重算尾部SpMV/MAS，应把它们明确计为额外scratch工作，保护错误曲率/状态、主x/r/b及持久A/M；不能声称“已跳过完整迭代”。若选择IF子图跳过尾部，则需重新计入它自身调度成本，不能预设更快。

## 4. 为什么不选初始化或CPU读回

Graph每次capture/instantiate CPU总计OFF1.445916ms、ON1.283097ms，连全部去掉也不足1%帧。不能从六次capture直接推出大收益。

`graph.initial_readback` CPU包络7.049857/8.541226ms；`graph.final_readback`54.039479/56.927947ms。初始化读回等待先前装配/求逆/初始M作用；最终读回等待刚发出的整个Graph。全帧真正GPU memcpy总时长仅.632928/.768319ms。不能将两种readback的CPU包络与Graph GPU时间相加，也不能将它们解释成搬80字节耗时几十毫秒。本轮不再提D2D Async或Graph初始化读回候选。

Graph最后一次迭代在previous-rho终止时仍额外做M与dot，host会在此前break；这与边界策略有关，但每solve至多一次，仅六次。没有成本支持为这项尾部工作另写复杂条件图。

## 5. 有意义的有限验证与退出条件

本报告只制定验证，没有启动以下实验。

1. **先验证生产包络。**由主任务在既定资源/120秒规则内，用相同冻结A/b/M、同x0、同rho、同最大迭代，交错原WHILE与单个K=4候选，CUDA events测完整solve；关闭Nsight/cost events，不改系统或重复复制数据来制造收益。包含高接触fixed系统，配一个短系统防止尾部成本反转。先检查原WHILE正常重放是否仍有足够可省总成本；无≥5%帧预算就停止。固定系统加速不能直接当整场加速。
2. **边界正确性用生产逻辑。**零RHS、非零r但rho=0、负/非有限曲率、alpha/beta非有限、max_iter<=1 fallback、max_iter−1触顶；收敛/错误恰在块内第1/2/3/4步，以及fixed-iteration诊断1–5步。核对实际迭代数、previous-rho、错误代码/原始标量、x/r/b、完整矩阵与持久M前后身份；分别记录允许改变的scratch。不得只看x接近。
3. **算子和生命周期。**保持MAS/SpMV/CUB原顺序、mask、材料及接触；覆盖纯cloth、固定ABD、动态ABD/FEM混合，Graph缓存命中/失效、维度与buffer增长、容差改变。不能把固定体零作用特例扩展成动态体例外。
4. **收益门槛。**OFF主证据需要约8.09ms/该帧、约10.03%线性总包络的净减少；上述capture包络不能充当通过值。短系统额外尾部/门控抵消收益、任一数值守卫失败、资源止损或无≥5%整窗净收益，即封存该分支；不追加K网格，不恢复退休融合。通过固定系统只允许进入原已声明的布料短窗配对及质量/独立CCD核查，不能直接推广默认。

**风险：中高。**收益依据来自受节点追踪影响的空隙，而实现容易破坏上一rho判停或错误/迭代计数；比“删一个copy”复杂。若主任务要求候选在实施前已经证明生产≥5%，目前证据不足，应明确本轮没有合格线性实现候选，保留此项为测量假设，而不承诺更快或2×。

## 证据与可复算性

- [PROFILE_OFF.json](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/PROFILE_OFF.json)、[PROFILE_ON.json](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/PROFILE_ON.json)；归类为GPU symbol统计，Graph内部NVTX仍未唯一归属。
- [fixed_off.sqlite](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/local_step4_profile_exports_20261006/fixed_off.sqlite)，SHA `56b7ab952ae30d0391327bf6602ed956ef5247ea9586fe8202d86c19cb4d6d74`；[fixed_on.sqlite](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/local_step4_profile_exports_20261006/fixed_on.sqlite)，SHA `27809629d60a7f72ee0d3fb5ef774ae60be29cf8df20abc8755bc15161237611`。
- SQLite通过`mode=ro`与`PRAGMA query_only=ON`打开；按`CUPTI_ACTIVITY_KIND_KERNEL.graphNodeId!=0`取实际节点，`demangledName`连接`StringIds.id`。每个同capture节点集合/stream独立匹配fill与p_continue；验证18×iteration计数与原stats一致。对每对边界减去三类GPU活动全局union，最终求和；不使用runtime API等待替代GPU活动。
- [现有f57查询成本复核](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/PROFILE_REVIEW.md)说明两条轨迹分岔和计时边界。本报告只新建当前文件，没有修改其分析或数据。
