# IPC 下一步成本定位结果

## 1. 结论与范围

**已验证：**当前普通/FullCCD双缓存组合完成六次有限运行、四份单帧 Nsight 节点追踪；真实热点已细化到碰撞查询与 PCG 算子。**本轮没有修改求解器，没有新的加速比或默认推广。**材料、legacy 停止 .01/min6、PCG rho 1e-4、原子 MAS 和完整安全 CCD 均保持。

**本轮排除：**图外初始化合并首次读回暂不实施。三个追踪帧中相关 GPU 空闲包络均低于本轮 5% 启动门槛，还包含不可直接归因的捕获、调度与 profiler 开销。这只排除本轮候选，不永久否定其他场景、硬件或实现。

**支持但未证明：**下一次唯一值得核查的候选是碰撞查询的保守子树资格摘要，将原来到叶节点才执行的同体、全固定过滤提前到内部节点。必须先证明同状态候选集合完全一致，再判断节省是否足以抵消摘要维护成本；目前不能承诺其整场收益。

**待验证：**正式质量、受控性能、原版实际速度补齐、独立接受路径 CPU CCD、300 帧与 AutoDL；本轮不继承历史零 CCD 标记，不改变质量范围。

## 2. 可复现运行

程序 SHA256：`fb3b46246a710febecd8badb9e5b1fcf82125fc5deebefa0b9b2c19cd6017d1d`。复用上轮37编译单元的已验证程序，本轮不重建、不重跑原八项GPU fixture。各运行保留 requested/resolved 配置、构建清单、实际位置与速度、原始统计、成本记录和 profiler 身份。

全部从零、dt=.01、单次120秒上限、公共 runner 串行锁及原资源保留线。生产配置为 `preset=combined, discrete_bvh_refit=true, interval=8, mas_fused_dot=false, spmv_fused_quadratic=false`。host 仅用于算子辨认，不是冻结同状态因果对照。

| 运行 | 帧数 | 方向总数 | PCG总量 | 结果 |
|---|---:|---:|---:|---|
| 固定兔子，观测关闭 | 39 | 190 | 8589 | 完成 |
| 固定兔子，仅第39帧NVTX/CPU观测 | 39 | 190 | 8605 | 完成 |
| 固定兔子，Graph，第39帧节点追踪 | 39 | 189 | 8553 | 完成 |
| 固定兔子，host，第39帧节点追踪 | 39 | 190 | 8577 | 完成 |
| 固定兔子，Graph，第57帧节点追踪 | 57 | 311 | 18245 | 完成 |
| 悬挂，Graph，第41帧节点追踪 | 41 | 224 | 6466 | 完成 |

全部状态有限、PCG无触顶或breakdown、固定点漂移满足原冻结比较阈值。固定ABD最大漂移3.10e-17～5.72e-17m，悬挂0；并非数学上逐位为零。数据仅是39/41/57帧前缀，不能替代100帧材料验收。

## 3. 成本分解（毫秒，诊断轨迹）

下表 CPU 帧 wall 与 GPU 活动 union 分开。其余为实际 GPU kernel/活动时间总和；不同列有包含关系，**不能横向求和**。Graph 内部节点缺少 runtime correlation 时保留未归因，不伪装成具备阶段归因；kernel 名称仍可识别 SpMV 等算子。

| 目标帧 | 固定39 Graph | 固定39 host | 固定57 Graph | 悬挂41 Graph |
|---|---:|---:|---:|---:|
| CPU帧wall | 210.748 | 301.482 | 231.709 | 287.619 |
| GPU活动union | 132.234 | 121.748 | 158.422 | 107.683 |
| 无runtime归因Graph节点 | 46.394 | 0 | 52.588 | 22.607 |
| SpMV kernel | 16.467 | 9.557 | 18.085 | 4.862 |
| 普通EE查询 | 14.932 | 15.326 | 17.860 | 15.364 |
| 普通VF查询 | 4.611 | 7.331 | 11.108 | 9.314 |
| swept EE查询 | 9.159 | 7.977 | 7.671 | 3.488 |
| 边三角交检测 | 11.793 | 9.880 | 12.721 | 6.386 |
| 矩阵转换，exclusive runtime归因 | 5.562 | 4.559 | 10.440 | 4.374 |
| memcpy GPU记录 | 0.537 | 1.951 | 0.647 | 0.778 |

host与Graph的工作量、接触集合、kernel时长和运行负载不同，不能从表中推出Graph让SpMV变慢或host数值更好。Profiler改变时长，表中wall亦不是正式性能分母。

固定39/57普通EE+VF为19.543/28.968ms，占各帧9.273%/12.502%。仅优化这组查询，要净省整帧5%，包括新增维护成本至少需减少53.92%/39.99%。若只有EE受益，需要减少70.57%/64.87%。加入swept EE/VF的四个查询合计30.150/39.745ms，占14.306%/17.153%；理论要求降34.95%/29.15%，**实际可剪访问比例尚未测量**。

CPU `cudaMemcpy` 等待并非复制成本。固定两个Graph帧的GPU memcpy只占wall .255%/.279%，不能把其中包含PCG执行等待的百毫秒API时间当作可省传输。`graph.final_readback` CPU等待75.591/79.642ms主要包含Graph执行，也不能另加到GPU时间。Graph捕获/实例化CPU阶段1.268/2.800ms约占 .60%/1.21%，没有重写缓存的优先依据。

完整kernel、runtime、阶段、相关覆盖、嵌套语义与活动边界均保存在分析JSON；矩阵转换、MAS准备的CPU/GPU时序亦分开。CostScope CUDA事件关闭，生产计时器仍有事件；不宣称全程序事件数量为零。

## 4. 图外初始化候选为何停止

现有顺序为 `r=b → M作用 → 初始rho → p=z → 初始化与zero-rho检查 → 首次80字节D2H → 循环Graph → 最终D2H`。候选是把相同初始化纳入Graph前置阶段，设备决定是否进入while，并只在末尾统一读回。

本轮直接从SQLite取每次initial_readback开始到首次Graph kernel开始的窗口，扣除窗口内全部kernel/memcpy/memset的区间union。剩余只是**GPU不活动包络上界**，仍含捕获、操作系统、驱动与profiler影响，既不是保证可省量，也不是整场上界。

| 目标帧 | 初始化包络GPU不活动ms | 占诊断帧wall |
|---|---:|---:|
| 固定39 | 6.185 | 2.935% |
| 固定57 | 5.450 | 2.352% |
| 悬挂41 | 13.570 | 4.718% |

即使极乐观地全消除，也未过本轮5%门槛，因此不改条件图入口/签名，不引入零RHS、初始错误与首次捕获的新风险。正常设备初始化与MAS/归约工作不因合并读回而消失。

## 5. 观测与质量限制

固定39无观测/CPU观测的退出分布相同（22 movement、17累计），方向数同190，PCG差16次（约0.186%）。全前缀布料最大单点位置差5.301mm、实际速度差.366m/s，frame RMS最大.269mm/.014925m/s。两者最大位置/速度分歧已发生在第34/35帧，早于第39帧观测窗；目标39帧差2.805mm/.166m/s。

因此这些差异不能归因成第39帧观测导致，也不能反过来宣布观测已统计中性。原子累加及运行负载下的重复差异需要单独对照；本轮只确认读取配置、观测样本、流程和实际数据覆盖，不扩展确定性任务，也不扩大质量容差。位置与速度是实际导出，包含初态0共40个样本。

独立CPU审计、原版速度与旧固定兔子基线材料协议不可认证的缺口仍保留。本轮没有新组件，不能把上述差距解释成一个新算法的必然质量代价。

## 6. 下一次唯一候选与停止边界

当前 `_selfQuery_ee/vf` 与swept对应路径到重叠叶节点后才判断bodyID与fixed-fixed；被排除的子树此前已经经历遍历。保守摘要可以仅在**整棵子树所有叶都必被原条件排除**时跳过，不能复用旧接触集或缩减安全CCD。

先核查摘要维护与过滤的可行性，不同时调整block size、树布局或重建周期：

1. 摘要使用原谓词的实际leaf key，不能假设三角形/边各顶点bodyID一致；只有统一且非负body key才用于同体排除，`-1`自碰撞保留。fixed规则严格沿用`>=2`，不把边界`!=0`等同固定。
2. 普通/swept两个缓存各自拥有摘要。树重建、叶映射、地址/数量/拓扑、bodyID、BoundaryType变化时失效；仅有几何refit不证明资格不变。首版可每次重算摘要避免隐藏状态假设，并计入成本。
3. 默认关闭。冻结同状态旧/新typed pair多重集、MatIndex及CCD tuple；非空、无接触、固定ABD、混合ABD/FEM、自碰撞、缓存切换、容量重试、动态映射和边界变化都需覆盖。独立接受路径CCD不省略。
4. 未测到足够可剪访问、净查询收益不能支持整场至少5%或集合不一致，即停止该候选。不能仅凭查询占比宣称收益，也不进入参数网格。

## 7. 工具、证据和验证命令

本轮新增公共有限计划导出器和只读分析/交付核验；已有求解、runner、历史结果不改。导出支持任意冻结计划，不再复制硬编码旧批次；输出拒绝覆盖，初始化包络与嵌套计时限制写入分析。首次离线核验因错误要求固定ABD漂移逐位为零失败，修正为既有冻结比较阈值后完成；未重跑GPU、未修改数值或原始结果。

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/batch.py --plan configs/active/ipc_next_cost_20261006.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/export_cost_plan.py --plan configs/active/ipc_next_cost_20261006.json
& 'E:/Anaconda/envs/DL/python.exe' tools/active/analyze_next_cost.py
& 'E:/Anaconda/envs/DL/python.exe' tools/active/verify_next_cost.py
```

计划：[ipc_next_cost_20261006.json](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/configs/active/ipc_next_cost_20261006.json)。数据：[分析](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_next_cost_20261006_analysis.json)、[批次](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_next_cost_20261006_batch.json)、[核验](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/ipc_next_cost_20261006_verification.json)。六运行与本轮候选结论追加唯一索引，561项历史条目及旧决定保留。
