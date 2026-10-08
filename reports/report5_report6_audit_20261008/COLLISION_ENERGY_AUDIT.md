# 报告 5 / 6 对照：碰撞安全、BVH、能量与线搜索

2026-10-08。活动仓库 `E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning`；审查起点 HEAD `9192ad1b8637757290440ff5ae16e01d301465e2`。本分工完整读取两份外部报告及根 `AGENTS.md`，最初只读 native；确认线搜索接受门问题后，由主 agent 明确分配三项文件进行修复。未改冻结 `baseline/`，未运行 GPU，未提交 Git。

报告 5：184 行，SHA256 `F80F8DFDF27B862671FAD44DDA93347FE695CCB9723BF10DD1EDD5E9D32F6528`。报告 6：401 行，SHA256 `005C277C53F83A840DEFE94A527C1D0333CF3971EF0BCEA81C6B1F60695A658B`。下文 `R5:Lx` / `R6:Lx` 指上述原文件行号；native 链接指本轮工作区，函数名是行号移动后的稳定检索键。

## 1. 当前证据及达到 2× 所需成本

`reports/autodl_seven_20261008/ANALYSIS.md`、`TIMINGS.csv` 是当前七场景证据，21 次均连续完成 120 帧，每臂每场景只有一次。完整组合的单次 solver 比值为 **1.075–1.414×**，描述性几何平均 **1.263×**。R5:L3 的 1.6–1.95× 来自更早两场景跨阶段描述比值，不是本轮七场景质量认证；R6:L3–5 的“现代 GPU 达到 2× 可行”没有本项目可消除成本证明。两个报告的收益百分数不可相加、相乘或写成实测结果。

`all` 实际包括 Graph K1、FullCCD BVH refit、batch energy、accepted energy reuse、普通 BVH refit（interval=8）。**不包括** segmented energy、row gather SpMV、AL-TOI、残差停止、K4、退休融合、接触池；本文没有把它们重分类为已实现。

以下数值只由本轮 CSV 计算。`need = 1 - base/(2*all)`；最后一列是假设把 all 的 CCD+线搜索整个阶段耗时减半的算术反事实，不是候选预测。该阶段还含 BVH、接触生成、碰撞保守推进、能量、CPU 读回，不能全部归于单个 kernel。

| 场景 | base/all | 达到 2× 还需削减 all | all CCD+LS/solver | 若 CCD+LS 全减半，base/new |
|---|---:|---:|---:|---:|
| hang_l | 1.4139 | 29.31% | 39.35% | 1.7602 |
| hang_m | 1.3230 | 33.85% | 30.73% | 1.5632 |
| sphere7_l | 1.2166 | 39.17% | 47.45% | 1.5950 |
| sphere7_m | 1.2553 | 37.24% | 41.74% | 1.5863 |
| fixed_bunny_l | 1.3455 | 32.72% | 32.46% | 1.6062 |
| fixed_bunny_m | 1.2397 | 38.01% | 24.73% | 1.4146 |
| mixed_bunny_l | 1.0753 | 46.24% | 40.88% | 1.3515 |

因此碰撞和线搜索优化不能独立保证 2×，必须与经过独立成本验证的线性/装配改进共同形成最终组合。总体 solver、core event 和进程 wall 三种口径必须分别报告。

## 2. 报告建议逐项映射

| 报告原点 | 当前状态 | 实现证据与纠正 |
|---|---|---|
| R5:L25，R5:L96–112：引入 BVH / 中心网格替代 O(n²) | BVH **已实现**；中心哈希示例 **错误** | [mlbvh.cu](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1104) 为 triangle/edge 叶盒；`ConstructRebuild` Morton radix sort、内部节点和 AABB；`_selfQuery_vf` / `_selfQuery_ee` 已遍历 LBVH。不是从全对检测起步。大 triangle/edge 或 swept path 跨多格，中心仅入一桶及相邻桶无保守覆盖保证。若未来独立评估网格，必须插入所有覆盖格、处理巨大 primitive、跨格去重和容量重试，包含全准备成本。 |
| R5:L25：动态 BVH refit | **已实现** | [GIPC::buildBVH_FULLCCD](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:9197)、`lbvh_f/e::RefitFullCCD`；普通 refit 在 [discrete_refit](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/discrete_bvh.inl:146)、interval policy 在 L165。刷新全部叶盒和内部 union；不是复用陈旧几何。普通/swept 两组缓存独立。 |
| R5:L27，L114–124：小速度跳过 / 有限次数 approximate TOI | 当前全 ACCD **已实现**；伪代码 **错误/不适用** | [edge_edge_ccd](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/ACCD.cu:373)、[point_triangle_ccd](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/ACCD.cu:456) 是保守推进，不是报告假定的二分二次方程求根。只在去共同平移后的 `max_disp_mag==0` 返回完整步，不能替换为任意 small 阈值。现有 50000 次保护返回当前保守 toc，不能改成接受任意近似 TOI。 |
| R5:L27，L115：分层包围体筛选 | **已实现部分** | [swept leaf AABB](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1129) 合并全部 primitive 顶点的开始/末端；`_selfQuery_vf_ccd` / `_selfQuery_ee_ccd` 使用 `sqrt(dHat)` gap。这已经是线性运动全区间的保守 broad phase；新 sphere 筛选只有覆盖整个 motion、具严格分离证明才可跳精确 CCD。 |
| R5:L33：上一帧接触缓存、位移小就不完整检查 | **不适用**；拓扑复用已实现 | 普通与 swept refit 复用树的 topology/mapping，仍执行全部当前几何查询。接触池曾受限于单 Newton direction，现在在 all 中关闭；退役方案不能因为“时序一致性”重新启用。上一帧接触集不保证包含新接触。 |
| R5:L9、L19、L77；R6:L35–65、L246–277：整主循环一次 Graph | **部分实现**，报告捕获边界不成立 | 当前 Graph 是实际 PCG/预条件作用路径；碰撞候选数、BVH 重建/容量增长、线搜索拒绝次数和能量 CPU 决策是动态的。不能将固定 `numIters/numSteps` 的静态捕获伪代码直接包住现有 solver。图生命周期要绑定真实指针、尺寸与 workspace，不用 function-static 一份图跨对象复用。全覆盖还需测量能省的独占调度成本。 |
| R5:L15、L45；R6:L25：warp 归约代替单地址 atomic | 能量 block 归约 **已实现**；碰撞 append 仍存在 | [GIPC_CUB_BLOCK_SUM_AND_STORE](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:158) 与所有 production energy kernels 用 CUB block sum/valid_items；之后 CUB DeviceReduce。能量并不是报告示例“每个元素 atomicAdd 一个 scalar”。碰撞 `_cpNum` atomic 是输出槽分配，不是可不保留索引契约的简单浮点和。优化必须保存 pair 数、tuple、MatIndex 和溢出重试。 |
| R5:L47：复用不变结果；减少归约/readback | **部分实现** | [computeEnergy](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10877) 已把九项 FEM block partials 放独立区间，九次 CUB sum 后一次 FEM readback；这叫 batch，不叫 segmented。ABD kinetic/shape 仍各有 device reduction 和 host scalar conversion；[DeviceVar::operator T](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/cuda_tools/cuda_buffer_view.h:656) 为同步 D2H。 |
| R5:L47：初始能量缓存 | **已实现**，有效域有界 | `solve_subIP` 中 energy_valid 初值 false，lineSearch 导出最终 accepted energy；`postLineSearch` 改 Kappa 则缓存失效，见 `energy_valid=reuse_energy && Kappa==old_kappa`。动画比例/摩擦基准在外层续进变更，每次重进 solve_subIP 会重置缓存；不能把本缓存推广为跨帧/跨 objective 的 static 缓存。后安全回退能量门缺口本轮已修复，见第 4 节。 |
| R5:L17、L41；R6:L67–84、L279–298：碰撞/窄相/状态更新多流 | 完整并行 **缺失**；报告任务独立性 **错误** | 更新 geometry → 当前 BVH → 查询 → pair count → CCD/energy → alpha 决策存在直接依赖。位置上传后同流 build，再主流 wait 的示例保序但不产生两份独立工作；pageable host scalar 不会自动获得重叠。PCG 下一轮预条件亦依赖本轮 residual。改多流需显式列清每个写者、读者、事件和私有 workspace。 |
| R5:L13；R6:L117–139、L323–342：单体 AoS 改 SoA | 属性分离 **已实现**；坐标 SoA **未实现/待剖析** | [device_TetraData](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/fem/device_fem_data.cuh:21) 已有独立 vertex/velocity/mass/constraint arrays，坐标用 `double3`，并非报告 Particle 一个大 struct。geometry primitive 间接访问是 3D vector gather，拆 x/y/z 不自动改善 locality；布局重构应先限定热点 kernel 与实际访存指标。 |
| R5:L39、L126；R6:L86–115、L300–321：融合位移/速度 | 执行层融合 **部分实现**；示例 **改变数学** | R6 原顺序 `pos += dt*oldVel; vel += dt*a`，改后 `vel += dt*a; pos += dt*newVel` 差 `dt²*a`。实际 [stepForward](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:7857) 是 `x_temp-alpha*moveDir`，而 [updateVelocities](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:7876) 在完成物理帧后计算接受位移/dt。不能把 Newton trial 更新与物理帧速度更新合并。 |
| R5:L11、L43；R6:L24：最高 occupancy / volatile / 固定寄存器限制 | **待验证**；因果表述过强 | BVH 查询每线程 DFS stack、材料矩阵局部状态确有资源压力可能，但本轮没有 Nsight Compute spill/occupancy 证据。不能以 occupancy 高等于性能高，也不能盲加 volatile 或 -maxrregcount；寄存器限制可引入 local memory。 |
| R6:L163–187、L344–364：删同步、移循环内 malloc | 缓冲复用 **已实现部分**；盲删 **不适用** | native 使用 DeviceBuffer/PCG reduction scratch；closeConstraint clear 只改 logical size，不能把它当每轮 cudaFree。pair 数超容量会明确增长并重新完整查询。同步 D2H 后 CPU 立即决定容量/alpha/energy，不能只删同步。Newton 中事件 create/destroy 仍可另做生命周期优化，但当前没有独占成本足以列主要候选。 |
| R5:L31、L149；R6:L189–207、L382–396：能量/碰撞 float/half | **不适用本轮** | 保持 FP64、materials、rho=1e-4、legacy .01/min6/fullCCD。energy 对接受门的错误不应靠放宽精度预算遮蔽。`cuda-memcheck --precision` 不是自动度量物理舍入误差的合同；需要数值参考与轨迹质量测量。 |
| R5:L35：近似 strain/阻尼代替 Newton | **不适用本轮** | 修改材料/目标函数/求解模型超出执行层优化；不能给该变化记同精度 speedup。 |
| R5:L51、L57、L66；R6:L207、L240：容差与安全测试 | 验证基础 **部分实现**；论文例阈值 **错误** | 不能把单个 1e-5 displacement/energy 阈值用于所有尺度，也不能用同 contact count 证明无漏检。球球/盒盒不是当前 PT/EE triangle mesh 的代表。需要整段 material/真实 velocity/位置、逐 accepted segment 独立 CCD；七场景独立 CCD 仍待验证。 |

## 3. 碰撞不变量、筛选反例与缓存边界

### 3.1 实际完整路径

`solve_subIP` 保留现有 ground feasible step、当前接触 self CCD；之后 `buildBVH_FULLCCD(temp_alpha)` → `buildFullCP(temp_alpha)`，完整候选在满足既有分支条件时交给 `self_largestFeasibleStepSize`。后者 [GIPC.cu](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:8932) 调 `_cub_reduct_self_step`，使用 PT/EE ACCD 并按 reciprocal max 得全局保守步。最终 `lineSearch` 每个试探 state 仍经过普通 BVH/接触，前/后 `isIntersected` 的 ground 与 edge-triangle 安全检查保持。

原 `min(temp_alpha, swept_step)` / `max(alpha,alpha_CFL)` 及 `temp_alpha>2*alpha_CFL` 路径保留，属于现有 Stiff 执行合同。本审查没有凭报告近似伪代码替换它；独立接受路径 CCD 尚未完成，不能把“代码仍调用 FullCCD”写成完整物理质量认证。

`_selfQuery_ee_ccd` 从 leaf 的 `element_idx` 取 **原 edge ID**；`obj_idx<self_eid` 防重复。VF signed tuple `-vid-1` 与各 body/boundary eligibility 保留。输出容量不足时 count 仍覆盖所有发现，主机扩容并重跑，不静默截断；`GIPC_CCD_PAIR_LIMIT` 是运行资源失败门，不能砍 pair 换速度。

### 3.2 为什么报告速度阈值不能用

距离 `ε>0`、法向相对总位移 `2ε` 的点/面，在约半个 interval 接触。对任意固定“小速度/位移”阈值，都可取更小 ε，使位移低于阈值而仍撞击。R5:L119 又要求 `distance<threshold` 才 skip，恰好放大近接触风险。近零共同刚性平移可以抵消，但必须像现有 ACCD 那样对 **所有 primitive 顶点** 去共同平移后证明没有相对运动，不能只看 center velocity。

可接受的保守筛选范式：整个区间的 swept AABB 分离，或已证距离下界 `L` 严格大于该区间的相对运动上界 `M` 与厚度、安全舍入余量之和。比如 primitive A/B 线性运动的最大顶点位移和可给 Hausdorff 运动上界，但距离估计必须是 **下界**，有误差时不得乐观放大。NaN、Inf、退化 geometry、边界相等或证明不确定一律回到完整当前 ACCD。该范式是安全要求，不是本轮已实现的新优化。

### 3.3 Refitting 已经保守，但诊断覆盖有限

`RefitFullCCD` L2467/L2584 重算全 swept leaves、重排 fresh bounds、清内部 flags 后刷新 union；普通 refit 在 `discrete_bvh.inl:146` 重算当前叶盒，间隔满/存储签名失配时强制 full rebuild。`discrete_signature` 记录 primitive 数、geometry/topology/body pointers 和八组 owner buffer 的 pointer/size/capacity。它依赖 topology 在 init 与显式 invalidation 之间不原位修改；任意 remeshing/原 ID 重新编号必须 invalidate，不能以指针相同推断 topology 相同。

ordinary/swept storage selection 不保留外部 raw pointer，当前场景只有一个 GIPC host thread拥有缓存。R6 多流建议如果让普通与 swept build 同时交换这一组 public buffers，会破坏此拥有关系；必须先改所有者与借用 API，不能直接加 stream。

普通 refit 诊断比较完整 typed pair **multiset**、五个计数和 MatIndex bijection；FullCCD `GIPC_AUDIT_REFIT` 当前仅每帧首个调用比较排序去重后的 **set**。后者可隐藏重复数差异，且不覆盖该帧所有 Newton directions。下一次验证应明确 exact tuple/multiset/count、首/末/复杂接触方向与 storage lifecycle；诊断重建会改变顺序并增加成本，验证关时才计时。

## 4. 确认并修复的线搜索接受门

### 4.1 修改前的确定问题

1. 原条件 `energy>baseline && backtracks<=8` 可执行九次 half step；若第九次仍升能，退出后只 printf，`lineSearch` 仍 return false，caller 继续 `postLineSearch`。所以“达回退警告阈值”被错误当作“接受”。
2. IEEE NaN 的 `>` 为 false，非有限 energy 可绕过同一条件；nonfinite baseline/threshold/alpha 没有独立接受检查。
3. energy 回退后的第二轮 intersection half steps 若改变 state，只有 `accepted_energy!=nullptr` 才重算 energy；即使重算也未重查接受条件。reuse 关闭时更会用旧 state 的能量默许新 state。
4. safety half loop 没有检测 alpha 下溢到零/无法严格减小，可能重复无进展的 state。

这是控制流与接受判据问题；未声称七场景旧运行已经触发上述分支，也未用它解释现有轨迹差异。

### 4.2 本轮修复与行为边界

修改文件只有 [GIPC.cu](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:11101)、[line_search_acceptance.h](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/line_search_acceptance.h:8)、[line_search_acceptance_test.cpp](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/tests/line_search_acceptance_test.cpp:23)。CMake 接入交主 agent。

- 纯 helper 检查 baseline、trial、slope、threshold、alpha 有限；alpha 必须正。当前 `armijoParam=0` 保持，接受判据为有限 **非增能量**，相等可接受。
- `report_line_search_threshold=8` 保持，以 static_assert 绑定原最多 **九次** energy backtrack；第九次 finite 合格 trial 仍可接受，仍升能或非法 trial 则受控失败。没有自动延长预算、静默回退或新 solver 路线。
- 普通合法路径原来的 geometry/BVH/CP/safety/energy 时序及装配数保持；只在后置 safety 改 state 的路径重算最终 state energy（reuse 关闭也计算），再检验。如还有剩余 energy budget则按相同 half-step 路径消耗；预算耗尽失败。
- half-step 必须得到有限正且严格变小的 alpha；正无穷 CFL 表示无额外限制，可以保留。零/NaN/负 CFL 或 alpha 下溢/无进展失败。仍执行完整 `isIntersected`，没有按 small-velocity shortcut 接受。
- 失败先写 `newton.line_search_failure`（原因、baseline/trial/threshold、有限标记、energy 是否属于当前 state、alpha、CFL、两种 backtrack 数、eval 数、预算），结束 contact pool，然后 throw。现有 [app controlled failure](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/app/gl_main.cu:1983) 保存 stats 并 `quick_exit(4)`。

严格门可能使以前静默接受的运行明确失败；这属于纠错而非新增速度收益。修复后必须使用新 source/build identity 重新做完整GPU回归，不继承旧21次运行的通过状态。

### 4.3 当前验证

PowerShell 7 下执行：

```powershell
& 'D:\computer\mingw64\bin\g++.exe' -std=c++17 -O2 -Wall -Wextra -Werror -pedantic `
  'tests\line_search_acceptance_test.cpp' `
  -o 'reports\report5_report6_audit_20261008\line_search_acceptance_test.exe'
& '.\reports\report5_report6_audit_20261008\line_search_acceptance_test.exe'
git diff --check
```

结果：**2304 checks passed**，diff 无 whitespace 错误。覆盖 finite/NaN/±Inf、positive alpha/非法 alpha、乘法或加法阈值溢出、zero-slope 非增/相等与 nextafter 边界、九次预算首/末、final safety state 不能继承旧 trial 接受、CFL clamp、subnormal 下溢和严格缩步终点。本分工 **未编译 native CUDA、未执行 GPU**；CPU 合同不等于 native 运行验证。

## 5. 成本判断：分段归约不是当前主线

旧 `reports/report_execution_20261007/MAIN_ANALYSIS.md:L79–96` 保存了真实且范围受限的成本：

- 两个代表捕获帧的 batch energy 最终 CUB 归约约占捕获 GPU 工作 **0.106% / 0.068%**；全程 CPU 归约提交不到 0.2%。因此不可把整个 line-search 阶段占比挪给九段 final reduction。
- sphere 第49帧 `_selfQuery_vf+_selfQuery_ee` **102.190ms**，安全 `_edgeTriIntersectionQuery` **40.010ms**，合约 **54.5%** 捕获 GPU 工作。只是一帧旧程序/旧轨迹的显著证据，不是本轮120帧占比或严格上界。
- 旧全程 `COST_sphere.json` energy evaluation inclusive CPU 10.36%，production submission 0.249%，batch reduction submission 0.193%，batch readback inclusive 3.43%。不能把等待已有排队 GPU 工作归于 copy 本身或把嵌套 CPU envelopes 相加为可省 GPU 时间。

过去候选 A 未达到“独占整场至少10%，理想减半至少5%”门槛并明确跳过，本轮不补一个空 segmented 开关。当前候选最多以下两个，均由新的同身份热点证据触发；没有估计百分数冒充实测。

## 6. 两个可执行候选与停止条件

### C1：保持完整集合的 stackless BVH 遍历

**原因与范围。** 当前 VF/EE 查询每线程 `uint32_t stack[65]`，安全 face→edge 查询每线程 stack[64]；实际是否 spill/访存瓶颈需 Nsight Compute。已有普通 BVH refit节省build，却不消除遍历。若当前完整窗口证明这几项查询独占成本达到至少10%、显著 local loads/stores 或长 tail，再在 `mlbvh.cu` 和 `GIPC.cu::_edgeTriIntersectionQuery` 实现 escape-index 的 stackless DFS，先选最贵一个 query，避免同时换全部路径。

**实现合同。** 新 escape-index 是只依赖已有 topology 的视图；按旧 DFS **右优先**的访问顺序生成 next/escape（原代码 push left 再 push right），每个 internal/leaf overlap 与 predicate沿用原函数。对不重叠节点跳到 escape，对重叠 internal进入其 right child，leaf处理完进入escape。不得添加旧退役 eligibility subtree/pool/TOI approximate 分支，不修改 gap、原 element ID、body/boundary、signed tuple、capacity/count 或 narrow phase。ordinary/swept两缓存各拥有逃逸视图，full build、storage/count/topology失效时重建；refit保持topology可复用。单 leaf / 空 edge明确走现有边界处理。

**准备成本。** 约一项 uint32 per node，即约 `4*(2n-1)` bytes/树/缓存，另计生成kernel与准备时间；不能只给遍历计时。原 child/parent节点仍是保守 oracle。验证排序typed **multiset**、CCDtuple、五count及MatIndex双射、boolean/逐face安全结果。先固定几何离线fixture，再同系统/同state query compare；不靠总contact count。

**判定。** 局部含视图准备的完整query路径至少15%改进，三对从零100/120帧整场至少5%、另一主场景不退化超过3%，完整质量门通过后保留。最多一次有明确 profiler依据的修订。若 stack 不spill、逃逸多访存抵消、完整query不到10%独占，停止本候选而非猜寄存器参数。

### C2：合并完整 FP64 energy 的生产/结果读回边界

**原因与范围。** 当前 batched FEM 除九归约外仍调用 ABD kinetic/shape各自同步 scalar readback；混合/刚体场景的每次 energy评估至少三个 host decision读回。候选先保留全部九生产kernel和逐项CUB归约，新增 ABD device-only输出接口，让两种ABD结果进入同一11 scalar buffer，再一次copy全部到host，按当前CPU加法/系数次序累加。不是把 energy 改float，不是删ABD贡献，也不把segmented当主成本。

**源与生命周期。** `computeEnergy`、`Energy_Add_Reduction_Algorithm`；[ABD kinetic/shape](/E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/abd_system/abd_system_function/cal_abd_energy.cu:120)、`abd_system.h`。保留旧scalar API为关闭路径。空FEM/ABD/ground/friction段必须每次归零，保留dt²、Kappa、friction系数，TOI objective独立明确，不暗改all backend。workspace所有者只属本对象/host线程，resize在任何producer前完成；不能跨流共享一个CUB temp storage。kernel fusion或Graph capture仅在这个有界数据流内另有独占提交成本证据时讨论，不能静态捕获全Newton次数。

**启动门。** 新诊断拆 energy producerGPU（kinetic、FEM、triFEM、bend、constraint、barrier、ground、friction、ABD）、final reducers、CPU submit、readback阻塞与GPU union，且取冷启动、初始、接触、线搜索峰值窗口。只有可消除的独占production/readback/调度成本≥10%整场、理想减半潜力≥5%才实现；若仅九个final sum小，明确跳过。不能把旧10.36%inclusive energy直接当启动证据。

**正确性与验收。** 同状态每项FP64结果相对误差≤既有1e-10、最终能量/接受分支一致；near-equality拒绝/接受边界必须单列。保持CPU最终累加顺序减少无必要浮点变化；固定、运动ABD、无接触/有接触、friction开关、Kappa变化、动画objective重置和缓冲增长覆盖。满足完整energy路径15%、整场5%与3%退化界、整段质量后才计入最终组合，最多一次修订；否则保留cost结论，关闭候选。

## 7. 顺序与完整质量门

1. 先将本轮接受门CPU测试接入CMake，独立clean native构建并保存37单元及编译/链接/程序/source/input身份。短native测试覆盖常规接受、九回退边界、非有限失败和后置安全state；失败是失败，不重跑替换或自动改预算。
2. 新程序在受控、GPU串行条件下，以all参考对C1/C2启动窗口做有界诊断。所有诊断run与正式timing分开；如果可省成本门未达到，停止那个分支。
3. 每个候选先同state完整operator oracle，再从零paired三个100/120帧，交替A/B顺序，保持materials/FP64/rho1e-4/fullCCD/legacy .01/min6/dt.01。报告Newtons、PCG、energy evaluations、两种backtracks、bvh rebuild/refit、pair work、prepare bytes/cost及所有阶段。
4. 独立CPU接受路径CCD需要 **每个真实accepted segment的前/后geometry和boundary更新段**。现有 `GIPC_TRACE_SUBSTEPS` 主要导出accepted后位置；没有实际独立CCD审计工具时不能把这个trace标记视为通过。参考PT/EE距离/TOI计算必须独立于native broad phase，覆盖与薄壳/同FEM自接触相关全部primitive eligibility，报告漏检、负distance/翻转、不确定区间；不只比较contact counts。
5. 整段材料max/p99 stretch、FEM最小J和非正单元、ABD翻转/固定漂移、分体位置/真实velocity按预声明范围检查。当前mixed base/all均有非正FEM，不能以数量下降认证。原版重复范围只描述repeat variation，不自动放宽认证容差。
6. 最终与主线线性/装配候选组成通过单项门的组合，至少七场景三对同轮Stiff/all/new确认，保留host/Graph因果臂。严格 >2×必须定义逐场景或预声明整体统计及置信区间，同时通过质量/独立CCD/长测试；达不到则报告具体剩余成本，不用报告的5–30%预期填补差额。

本文属于源码纠错与可执行实验交付，GPU成本/安全/2×目标均不宣称本轮已验证。
