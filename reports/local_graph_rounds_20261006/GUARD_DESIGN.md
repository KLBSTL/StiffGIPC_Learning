# K=4 Graph 子步守卫设计审查

2026-10-06。只读核对当前 Graph、host PCG、guard fixture，以及主任务提出的 13 标量最小方案。本文没有修改原生代码、工具或运行 GPU。遵循仓库 AGENTS 与 benchmark-regression：此处是语义设计，不能作为正确性或加速验收。

## 结论与比较对象

只采用固定 K=4；保留 K1 原路径，不增加参数网格。主任务提出的 `s[10]=active`、`s[11]=pAp_work`、`s[12]=rho_next_work`，结合独立向量 p 更新、单线程 finalize、每四步一次 handle 更新，可以保护停止后的状态，而不用改 SpMV/MAS/CUB 的算式。

必须明确“严格等价”的对象是**当前 K1 Graph**。host 在 x/r 更新后按 previous-rho 判停，并在停止时跳过 M、下一 rho 和 p 更新；当前 K1 仍执行这些工作，beta 在 previous-rho 已达标时早退，但 p_continue 仍无条件改 p。K4 为了不混入另一个行为修复，可保留最后一个真实迭代的这次 p 写入，仅冻结此后空子步。不要据此声称终止内部 p 与 host 逐位一致。

依据：[host PCG](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/solver/pcg_solver.cu:495)、[Graph 标量/向量核](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/solver/pcg_graph_impl.inl:17)、[原 capture 序列](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/solver/pcg_graph_impl.inl:162)。

## 最小状态与子步顺序

`s[0..9]` 仍保留 K1 含义；`active` 只允许 1→0，整次 solve 初始化时才能重新置 1。它在一个子步中对所有向量线程只读，直到该子步最后的独立 finalize 节点。

1. SpMV、pAp 的两阶段归约继续执行；CUB 只写 `s[11]`，不直接覆盖 `s[1]`。
2. alpha 核**先检查 active，再做任何写入**；仅真实子步复制 `s[11]→s[1]`，随后原样更新 previous-rho `s[8]`、alpha `s[3]` 和错误 `s[9]`。即使 previous-rho 已足够小，仍须先检查曲率并做本轮 x/r 更新，不能提前接受。
3. x/r 核要求 `active && s[9]==0`；算式、元素顺序和维度不变。将 alpha 置零不能替代门控：`0*NaN` 仍可能污染状态。
4. 继续 M(r) 和 rho 归约；CUB 只写 `s[12]`。beta 核先检查 active，再复制 `s[12]→s[2]`，**然后**执行原 error/previous-rho 早退及 beta 逻辑。若先按 error 返回再复制 s2，错误真实迭代的标量终态将与 K1 不同。
5. 零 rho 检查在 CAS 前检查 active；保留 `r[i]!=0` 的原判据，包括 NaN 不等于零。不能让空子步把旧错误替换为后续 scratch 的错误。
6. 独立多 block 的 p 更新仅检查 active，保留当前 K1 对最后真实迭代的 p 更新；它不能同时写 active/stop/count。
7. 独立 `<<<1,1>>>` finalize：inactive 立即返回，不能再写 0..9；active 时把 s6 加一，原样计算 `converged=(fixed>0 ? count>=fixed : abs(previous_rho)<=tol*initial_rho) || (rho==0 && error==0)`，写 s5，并令 `active=!converged && error==0 && count<max_iter-1`。每四个子步后另一个单线程节点仅把 active 写入 WHILE handle。

仍然最多执行 `max_iter-1` 个真实迭代。非收敛触顶时返回 max_iter，统计实际执行数为 max_iter−1；收敛/错误优先级、fixed-iteration 诊断与 exact-zero 提前退出均按原公式保留。`max_iter<=1` 继续原 host fallback，不在本轮重新解释它。

## 多 block 竞争的准确边界

现有 p_continue 中其它线程只读 s4 并写自己的 p[i]；线程 0 写 s5/s6/handle，其它线程不读这些控制字段。因此当前源中没有“其它 block 读 stop 被线程 0 提前关闭”的竞争。**若新增 active 门控却仍由该 kernel 的线程 0 改 active，才会引入跨 block 竞争**，造成部分 p 分量少一次更新。`__syncthreads()` 不解决跨 block 问题。分离 p 与 finalize，依赖同流 capture 的节点先后顺序，才有完整 kernel 边界。

## 允许的额外 scratch 与禁止项

空子步仍会写 z、Ap、CUB 临时区和 MAS 的多层 R/Z 工作区；最多增加三个完整 SpMV/MAS/归约子步。它们不能算“已跳过迭代”，全部必须计入候选完整 solve 时间。允许变动的 scratch 清单须与 x/r/p/b、0..9、A/M 持久输入分开。

当前 [apply_preconditioner](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/linear_system/global_linear_system.cu:348) 写 z；[MAS action](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/MASPreconditioner.cu:2525) 重置/重写 R/Z 工作区并读取因子及层次映射。保持原 memset、限制、局部作用和延拓顺序，不能顺便删清零。原 dot 借用 z/Ap 作 partial 的行为也仍属 scratch；它们不应成为“主状态未变”的比较对象。

K4 必须拒绝与现有 fused_diag_update 同开，或者另行实现并验证守卫；本轮最小范围应直接拒绝，不静默改请求。其 [fused kernel](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/preconditioner/diag_fused_update.cuh:8) 直接写 x/r，既没有 active 也没有 error 门控，不能放进空子步。M 阶段研究/算子诊断等会替换 apply 行为的模式也须明确排除或验证，不把静态默认 MAS 结论推广到所有可选回调。

K 值、标量布局/地址及新增 scratch 地址必须进入 capture signature；K1/K4 切换、buffer 增长、维度/容差/max_iter 改变均应重新 capture。K1 的 capture body 不得意外使用 K4 work 输出槽。

## 现有 fixture 的缺口与有限验收

[pcg_guard_fixture.inl](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/solver/pcg_guard_fixture.inl:1) 现有 15 case 仅调用 init/alpha/beta/zero-rho，残差一元素、全部 `<<<1,1>>>`。它没有运行 x/r 更新、p_continue、真实 conditional handle、四子步尾部、max_iter 或跨 block 门控；保留通过记录，但不能用它证明 K4。

新增验证应调用生产 K4 守卫和实际 conditional Graph，而不是另写同公式的测试实现。最低覆盖：

1. 在块内第 1/2/3/4 步分别首次停止；比较该点与剩余 3/2/1/0 个空子步后的 x/r/p、s[0..9] **逐位不变**。至少 N=513，覆盖三个 block 和尾元素。故意让 s11/s12 的空步结果为负数/NaN，确认不会改错误和主标量。
2. previous-rho 达标但 x/r 尚须更新一次；以及同一步曲率非法时错误不能被容差停止吞掉。覆盖 exact-zero r、rho=0 但 r 非零、负/非有限 rho、非正/非有限 pAp、alpha/beta 溢出，并核对首次错误的 rho/curvature 与 K1 相同。
3. `max_iter` 为 2、4、5、6 等跨块边界；fixed-iteration 1–5，且 exact-zero/错误仍可先终止。验证实际计数、返回值、s5/iteration_limit 和 handle，不只看 x 接近。
4. 对同冻结 A/b/M、同 x0 比较 K1/K4 完整 solve，恢复原生产主解并核对 b、矩阵和 M 持久输入前后身份；MAS R/Z 排除于持久 M 比较。K1 与 host 的数值/退出对照另列，不能用 host 终止 p 去否定刻意保留的 K1 行为。
5. 验证 capture 命中/失效与 K1↔K4 切换，并保留零 RHS 在 Graph launch 前退出的原行为。错误 case 要有失败证据，不能把异常后缺记录当通过。

若要求比较 s[0..9] 全部位模式，应先给两路相同的初始标量字节；当前 graph_init 不初始化 s1–s4，不能拿不同分配中的未定义历史字节当差异证据。已停止状态后的冻结检查则必须包含这些槽。

## 成本与停止条件

沿用 [LINEAR_DIRECTION.md](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_rounds_20261006/LINEAR_DIRECTION.md) 的单帧 Nsight 包络作为假设：它不能证明生产 WHILE 的可省成本。先进行未开 profiler 的同系统完整 solve 交错 CUDA-event 比较；新增 finalize/handle 节点、尾部算子、capture 与显存均计入相应报告。没有净收益或任一守卫失败即停止，不新增 K 网格、IF 图或旧融合分支。本设计不承诺加速，更不证明整场达到 2×。
