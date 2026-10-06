# 安全 edge-triangle 查询：下一轮只读方向审查

2026-10-06。**唯一支持进入有界验证的执行候选：按现有普通 face BVH 的叶子排列分配 face 查询线程。** 不增删任何 triangle/edge，不改变 overlap、相邻顶点排除、边界条件或 segTriIntersect，不重建第二棵树，也不恢复已停止的 BVH eligibility、bounded CCD、接触池分支。当前证据支持“可能改善查询局部性”，没有证明足以改善整帧5%，更不支持承诺2×。

本次只读取源码和 Step4 已有输出，用 CPU 计算几何局部性代理量。未运行 GPU、未修改原生或历史报告。

## 1. 调用链与实际判定契约

调用链为 [GIPC::lineSearch](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10936) → step_forward/set_trial → [buildBVH](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:9037) → [isIntersected](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10920)。isIntersected 先检查 ground；若未命中，再调用 checkEdgeTriIntersectionIfAny → edgeTriIntersectionQuery → [_edgeTriIntersectionQuery](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:8163)。

初次试探位置必须通过相交检查，否则 alpha 减半、重新 step_forward/buildBVH 再检查。能量线搜索改变 alpha 后，末尾还保留对应的相交检查及必要的接触重建。初始推进另有 isIntersected 调用。不能因为 full CCD 已运行而删除这一层，也不能用任意一次历史“无相交”结果跨试探位置复用。

原查询按原始 `faces[idx]`，每线程负责一个 face，遍历普通 edge BVH，任何一个符合条件的 segment-triangle 命中便将共享标记写为 −1，主机读回负值后返回 true。不是接触对集合输出，也不是求最近距离：

- 查询 face 的包围盒由三个当前顶点构造；`gapl=0`，dHat 参数没有用于扩大该查询盒。
- `_overlap` 使用严格区间重叠：当某轴差值 `>=0` 即拒绝。因此零宽/恰好接触的盒语义必须原样保留；不能把该代码的结果表述为对所有数学退化情形的完备相交判定。
- 排除 face 与 edge 任一端点共享同一顶点编号的组合。
- 排除三个 face 顶点与两个 edge 顶点全部 `_btype>=2` 的组合。
- **没有 bodyId 排除**。不能借普通VF/EE查询的body过滤来“优化”此安全路径。
- [segTriIntersect](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:860)先排除端点严格在面同侧，再以 `abs(det)<1e-20` 排除近奇异情形；其余使用原行列式和 inclusive 重心/线段参数界限。候选必须保留这些运算及阈值，不引入新容差，不替换谓词。
- 当前线程仅在自身命中时提前返回，不轮询其他线程的共享标记。此次不提出全局提前退出候选，避免在同一轮混合新的并发读写语义。

覆盖结论：在合法普通 BVH、合法节点深度和当前数值谓词的前提下，每个原始 face 都运行相同遍历；任意现存命中通过最终布尔 OR 表达。重排查询线程只改变执行顺序，不改变这套覆盖或单查询的浮点运算顺序。

## 2. BVH 数据与映射

`buildBVH()`先构造/重选普通 face 树，再构造/重选普通 edge 树。`LBVHStorage`保存普通与 swept 的独立完整 buffer；公开 `_nodes/_bvs/_indices` 会随选择交换所有权，**不能跨 Construct、选择或增长保存借用指针**。

[sort_morton_codes](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:2356)先填充原始ID，再对 Morton64 key/ID 做 SortPairs。key的低32位包含原始ID，打破相同空间码的平局。[calcLeafNodes](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1185)定义：

`leaf = N - 1 + query_index; nodes[leaf].element_idx = indices[query_index]`。

叶子是 `[N−1,2N−2]`，内部节点是 `[0,N−2]`；`element_idx`是原始面/边编号，不是重排后的数组编号。原始 `_faces/_edges`未按 Morton 顺序重写。`sortBvs`把原始元素盒映射到对应叶位；refit沿原来的ID排列更新盒，不会破坏排列的双射性，但空间排序可能已过时。本轮默认普通树每8次按策略重建，故“采用已有叶序”不等于“每次重新按当前坐标排序”。

唯一候选在 `buildBVH()`完成后借用**当前普通 face 节点**，把原 `face = faces[idx]` 改为语义上的：

`face_id = face_nodes[face_count - 1 + idx].element_idx; face = faces[face_id]`。

其余 edge 树遍历、两个 child 的处理顺序、谓词与最终命中标记保持原样。无新排序、无新workspace、无拓扑/材料改变，且该查询位于PCG Graph之外，不需要修改PCG Graph的buffer签名。实现仍需明确模式开关、requested/resolved输出及普通树所有权；默认原路径保留，禁止静默改变所有调用方。

## 3. Step4 实际拓扑否定了一个错误方向

两次 capture 的 topology.bin 完全相同：24969个顶点，32082个表面三角形，79935个体四面体。由表面三角形三条边去重得48273个无向表面边；这是用于核对规模的CPU重构，未把它冒充运行时edge数组的方向和排列。

**两份 boundary_types.bin 的24969个值全部为0。** 因此该安全查询里“五个顶点全部 `_btype>=2`”的排除在此场景没有可利用的工作，不能根据场景名“固定兔子”推断这一过滤可省掉大量遍历。ABD的固定状态由另一路模型状态表达，不能擅自映射为新的edgeTri排除规则。

`trace/topology.bin`由 [gl_main.cu](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/app/gl_main.cu:1841)直接导出当前 `_faces`，故其面顺序确实对应查询的原始face序。boundary_types由原生BoundaryType导出。表面边从 surface 三角形生成并去重，正常非退化三角网格的非空面至少有三条边。

## 4. 空树、单叶与其他边界

| 输入 | 当前edgeTri行为/前提 | 对下一轮的要求 |
|---|---|---|
| 0 face | wrapper直接false，不启动kernel | 重排路径同样短路，不能计算无符号`N−1` |
| 1 face、edge数≥2 | face查询可运行；face叶位0映射有效 | 保证单face映射读取合法 |
| 0 edge且face数>0 | wrapper没有edge数参数，kernel仍读edge root | 现有一般接口边界未受保护；正常有效表面网格不会形成这一组合，不宣称现实现已支持 |
| 1 edge且face数>0 | edge root本身是叶子，left/right为UINT_MAX；当前kernel默认root为内部节点 | 现有潜在越界边界，不能拿原实现未定义行为当数值参考；需要显式host短路/leaf参考或明确拒绝的独立合约 |
| 重复Morton码 | key含原始ID，叶映射仍为排列 | 合成例验证没有遗漏/重复face |
| 普通/swept切换、增长、refit/rebuild | public buffer及次序可能改变 | 调用现场借用有效普通节点，验证mode及映射身份 |
| 深树 | 原kernel固定stack[64]，没有动态栈边界报告 | 本轮不改变遍历深度/栈策略；合成深树需在明确支持范围内，越界不能计作通过 |

上述0/1 edge问题是原路径边界审查结果，不是本次32082面场景的性能原因。若进入实现，应把边界合约和执行候选分清：0 edge可直接false；1 edge对各face调用相同overlap、排除和segTriIntersect，无需访问不存在的child。没有授权把本轮扩成整个碰撞模块重写。

## 5. 查询顺序证据：空间代理量，不是GPU收益

CPU只读两份 `state_0057.bin`，按每个face的AABB中心计算局部性，场景对角长度记D。原序与“在该保存状态重新计算Morton序”的对照如下：

| 代理量，均除以D | OFF原序 | OFF新算Morton | ON原序 | ON新算Morton |
|---|---:|---:|---:|---:|
| 每32个连续face中心盒对角的均值 | .5389024 | .0798382 | .5399017 | .0804849 |
| 上述32组跨度的95分位 | .7755097 | .1609687 | .7770968 | .1641981 |
| 相邻face中心距离均值 | .1550554 | .0100499 | .1553649 | .0100652 |
| 相邻中心距离95分位 | .4557289 | .0275173 | .4566377 | .0276552 |

复算方法：从topology头三个uint32取得N/Nf/Nt及后续uint3 faces；positions为little-endian FP64 xyz。`c=(min(positions[face])+max(positions[face]))/2`；D为全体face盒的场景对角。用源码的10bit/axis Morton量化（归一化×1024，截断到0..1023，expand_bits后x/y/z权重4/2/1），以原始face ID作为平局键排序。32组跨度只用完整32元素组，邻接距离覆盖全部相邻face。已验证所有索引在范围内且排序是完整排列。

这支持“原始线程组的查询位置分散，已有空间叶序可能提高同warp访问相似节点的机会”。**不能把约6.7倍的中心跨度改善当作kernel加速倍数。** 保存的f57是完成帧状态，既非六次query试探状态，也非实际保留refit叶序；两份capture没有导出每次face/edge节点、访问数、分歧或缓存命中。因此不能从这个代理量给出性能预测，也没有证明任何可裁掉的查询。

## 6. 可用成本与预声明的有限验证

[PROFILE_REVIEW](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/PROFILE_REVIEW.md)已核实：OFF/ON各6次edgeTri实际GPU kernel合计9.784111/9.508461 ms；ON完整swept另8.475375 ms，不能由本候选删除。ON CPU帧包络148.197846 ms。即使edgeTri完全消失，串行相减的乐观上界也只有约6.4%帧耗时；要单靠它净省5%帧时间，须省约7.41 ms，即约78% edgeTri kernel耗时，现无证据保证达到。

**只允许一个候选、一次有限对照，不扫描排序算法/线程参数网格。** 若父任务选择执行，建议先声明：

1. 正确性先行：在受保护同状态查询输入上对原序/叶序分别输出私有结果，比较总命中布尔、逐face命中以及节点/叶访问与predicate调用计数。重排应使逐face工作量相同；这是执行顺序变化，不是裁剪。不得覆盖主解或production输出。对任意重排均成立的face排列合约先用CPU验证。
2. 合成边界覆盖：无face、单face、明确的0/1 edge合约、两叶树、相同Morton码、共享顶点、全固定/混合边界标志、不同bodyID、严格分离、明显穿过、恰好touch、共面/近det阈值、无命中和多命中。退化情形以原谓词为契约，不能悄悄“修好”其中一个分支。
3. 同状态配对时必须预热、交错原/新查询，用已有固定兔子初始/接触输入，包含借用映射和主机读回成本；池保持关闭，不改变材料、CCD、停止规则、PCG。只保留一组固定预算；输入无法同状态冻结时不将轨迹差解释成kernel收益。
4. 性能只看总edgeTri调用成本而非代理量。局部改善不足以解释至少5%的目标帧成本，或数据未显示稳定收益，就停止该方向，不为凑“多轮”延长参数探索。只有局部门槛有证据支撑才进入既有质量边界与完整配对帧窗；单帧仪器化结果仍不认证整场/2×。

这次只读审查没有发现可证明应删除的edgeTri安全查询，也没有找到可以直接回收的CPU“等待成本”。候选依靠线程查询排列提高执行效率，价值必须由后续有界同状态实验决定。

## 7. 身份

- 两份 topology.bin SHA256：`33640f21acba865d8932bd58897fbddd2f2375cb88289b1258b1988fe2f16e4f`。
- OFF state_0057.bin SHA256：`bb96b9a1c83f22393ad6533d1170ec17dec07ca162c5726572c70cf2ee0902d6`。
- ON state_0057.bin SHA256：`8bcaba6229e80d47edb4c541f6241eb43af16d888c6910071fd08c4e5aa1a7ee`。
- 当前程序、源码、profile的完整身份沿用Step4的requested/manifest及PROFILE_REVIEW；本报告新增期间未修改这些输入。
