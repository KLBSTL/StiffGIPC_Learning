# 线性结构复用分支：完整成本门槛与 Stage-1 决策

本轮决定：暂不进入生产符号缓存实现。完整 120 帧探针已得到真实相邻结构命中，然而仍缺少同一完整工作负载中可消除符号步骤的关键路径成本，以及未来最小正确性 guard 的净成本。现有证据不足以证明至少 5% whole solver 净成本减少；也不足以将该分支列为达到原 Stiff 2× 的主路线。这里的“暂不进入”是本轮有限决策，不是依据单帧宣告任何结构复用永远无效。

本报告只新增分析文档。原生代码、测试、工具、baseline 和已封口旧报告均保持冻结。所有收益推算明确标为条件模型；没有实现旧 Hessian、RHS、MAS 数值块或因子的缓存。

## 1. 证据范围与可比口径

| 证据 | 完整性与用途 | 不能支持的结论 |
|---|---|---|
| [REFERENCE_COST_ANALYSIS](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/REFERENCE_COST_ANALYSIS.md) / JSON | 两场景、四臂、3 轮交错，24/24 次完整 120 帧；提供 whole solver 诊断分母与阶段预算 | 共享 WDDM 的计时不是独占性能认证；阶段 event 包含提交间隙与空闲 |
| [PROFILE_COST_ANALYSIS](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/PROFILE_COST_ANALYSIS.md) / JSON | sphere f49 node trace、fixed f40 whole Graph；完整单帧 NVTX 窗口内唯一 owner 的 GPU kernel 并集 | 单帧不代表整场；fixed Graph 内部没有算子分解；CPU 包络不是纯 GPU |
| [STRUCTURE_QUERY_ANALYSIS](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/report56_stage1_20261008/STRUCTURE_QUERY_ANALYSIS.md) / JSON | `--require-complete` 分析成功；两场景 validated_completed；所有 Newton 的生产 Converter 观测及私有 exact snapshots | canonical 相等不证明 MAS 拓扑可复用；诊断同步、输出和 BVH replay 不是未来最小 guard 成本 |
| [PROBE_LEDGER](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/report56_stage1_20261008/PROBE_LEDGER.json) | observer_off/probe × 两场景 4/4 次 completed，120 帧；程序 SHA256 `d4da6bd5ec8544b87d0f7b77f67c6bb02a95c2afa19a24a8a6013f69d89f3902` | 单次 observer_off/probe 差值不能隔离线性观察器成本或归因轨迹变化 |

固定条件仍为全局 FP64、原材料与 legacy MAS、rho=1e-4、dt=.01、累计停止阈值 .01/min6、完整 CCD。reference 的 `all` 臂包含 Graph K1、FullCCD refit、批量能量、能量复用和普通 BVH refit（间隔 8）。两份 Nsight capture 为 Graph K1，但这四类执行组件均关闭。probe 使用 `all` 加 cost/structure/BVH 诊断。因此不能把单帧 Graph-only 的符号占比直接乘以完整 `all` 分母，也不能把不同运行的系统数和时间拼成净收益实测。

原 Stiff 的速度、部分工作量和完整 PCG breakdown 缺失仍然缺失。probe 的逐系统 Hessian/RHS/M 前后 checksum 没有导出，数值缓冲区 bitwise unchanged 字段为 null。组件合同测试通过与私有比较通过不替代物理质量或独立 CCD 验收。最终结构分析核对了相同 exe/source 与初始静态输入：observer_off 与 probe 的位置/实际速度导出只在 frame0 字节相等，frame1–120 均不同，因此没有通过全程 bitwise neutrality。该事实不能凭这一次配对归因于观察器修改了生产数值；其 per-body RMS/max 差异与旧程序重复差异参考已单列于结构分析，旧程序重复差异不是本程序中性的验收阈值。

## 2. 允许复用的符号与必须更新的数值

| 路径与当前源码 | 可申请删除或替换的工作 | 命中后仍必须执行的工作 / 额外 guard |
|---|---|---|
| [Converter::_radix_sort_indices_and_blocks](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/utils/converter.cu:235) | ordered raw keys、owner/device/dims/count/offsets 全相等时，编码/index 生成与 CUB SortPairs 可用独立保存的 permutation 替代 | `set_dst_val_kernel` 读取本次新 blocks 的数值 gather 必须保留；恢复 permutation 的 copy/所有权管理也有成本 |
| [Converter::_make_unique_block_warp_reduction](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/utils/converter.cu:296) | partition flags、exclusive scan、unique count、compact row/col 生成可在完整 raw guard 后使用独立符号快照 | compact keys 在生产输入中会被装配覆盖，不能假设旧行列仍有效；恢复或保持独立 storage 有成本。当前 blocks 的 zero/init 和 `FastSegmentalReduce` 不能跳过 |
| [LocalPreconditioner::calculate_subsystem_bcoo_indices](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/linear_system/i_preconditioner.cu:54) | 选中的 subsystem BCOO indices 是符号数据，可在 compact pairs、范围/offsets、subsystem partition/拓扑 epoch 一致时复用 | canonical equality 只证明输入 pair 列表。FEM/ABD 范围、owner 与 partition 身份没有导出完整 guard；不能仅凭 pair_equal 删除 DeviceSelect |
| [MASPreconditioner::ReorderRealtime](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/MASPreconditioner.cu:1876) | hierarchy 的邻接、levels、clusters、映射可能成为符号缓存对象 | 需精确验证 base topology、当前 collisionPairs 及其 consumer 依赖、subsystem partition、owner/device/epoch 和布局。Converter pairs 或 permutation 相等都不自动给出这个合同 |
| [MASPreconditioner::setPreconditioner_bcoo](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/MASPreconditioner.cu:2404) | 仅在以上 guard 成立且不重复计算既有 `reuse_static` 收益时，申请省 hierarchy | `reuse_static` 已针对启用 `GIPC_MAS_STATIC_TOPOLOGY`、cpNum==0、owner 和有效静态缓存处理；不能把已省的工作计为新收益。numeric buffers 的尺寸/初始化与 PrepareHessian 始终在 hierarchy 之后执行 |
| [MASPreconditioner::PrepareHessian_bcoo](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/solver/MASPreconditioner.cu:2217) | 本轮符号缓存没有可删除的数值步骤 | 本次 Hessian scatter/fill、numeric aggregate、legacy factor/inverse 都必须更新。保留既有精度、算术路径和 preconditioner，不复用旧数值因子 |

`raw_equal` 是排序前的完整 ordered raw row/col 一一相等，保护旧 permutation/partition 的使用；本轮 compare 是 GPU 逐项 exact compare，hash 不替代相等。`pair_equal` 是转换完成后 compact canonical pairs 一一相等，具有独立 owner/device/dims/unique_count 形状，不受 raw count/offsets 变化限制。两者服务不同的符号对象。尤其 pair guard 发生在转换之后，不能用它倒推本次转换之前已具备跳过 sort 的条件。

`converted_joint` 在本报告中专指 **mapping AND partition** 相等，不是全部 MAS guard 的联合。private snapshots 保留历史符号；没有保留历史矩阵数值。

## 3. 单帧 profile 的合法可消除成本

下面重新使用现有 `analyze_profile_costs.py::load_capture` 以只读 SQLite、唯一 outside-Graph runtime correlation 与最内层 native NVTX owner 过滤 `linear.matrix_convert`。再按实际函数 token 将这一个 owner 分为六组；两份均无未分类 launch，分组并集精确回加到 4.807225/4.447097ms。不能用全局符号类的 gather/reduce 相减：同名核函数也出现在其他装配或 ABD Converter 中。

| `linear.matrix_convert` kernel GPU union ms | sphere f49 / launches | fixed f40 / launches | 复用含义 |
|---|---:|---:|---|
| encode/index | 0.061760 / 14 | 0.073824 / 7 | raw 命中时可删除 |
| owner 下 CUB radix-sort/scan | 1.069470 / 168 | 1.067965 / 84 | raw 命中时可删除；不是所有 CUB 的成本 |
| partition flags | 0.063744 / 14 | 0.052992 / 7 | raw 命中时可删除 |
| emit compact row/col | 0.038240 / 14 | 0.033664 / 7 | 可用 owned layout 恢复替换，恢复非免费 |
| 当前 block numeric gather | 2.764988 / 14 | 2.531261 / 7 | 保留 |
| 当前 block numeric segmented reduce | 0.809023 / 14 | 0.687391 / 7 | 保留 |
| 合计 | **4.807225 / 238** | **4.447097 / 119** | — |

仅 encode/sort/scan/partition 的可删除 kernel 为 1.194974/1.194781ms。即便先乐观把 compact emit 也全计入，Converter 符号预算只有 1.233214/1.228445ms；numeric gather+reduce 为 3.574011/3.218652ms，占 74.35%/72.38%。通过另外的数据流重构减少 numeric gather 属于新的优化方案，不能写成结构命中后数值 gather 已可删除。

| MAS / 相邻符号 owner 的 kernel GPU union ms | sphere f49 | fixed f40 |
|---|---:|---:|
| mas.hierarchy | 0.715998 | 0.441791 |
| mas.numeric_fill | 1.448669 | 2.183711 |
| mas.numeric_aggregate | 0.314847 | 0.316192 |
| mas.legacy_factor | 7.630674 | 6.885586 |
| mas.prepare kernel 合计 | 10.110188 | 9.827280 |
| 必须更新的 MAS numeric 三项 | **9.394190（92.92%）** | **9.385489（95.50%）** |
| local_preconditioner.prepare kernel 集合（乐观全部计为符号） | 0.085056 | 0.047360 |

乐观覆盖 Converter 符号+compact emit、MAS hierarchy 和 subsystem selection 的 kernel 集合仅 2.034268/1.717596ms，约占该帧 CPU wall 的 0.757%/1.040%，或 GPU work union 的 1.206%/1.485%。这只是这两帧已观测 kernel 的符号预算；它既不是整场收益，也不是把 host 提交/同步省略后的严格整体上界。

CPU `linear.matrix_convert` 为 11.127744/6.897861ms，CPU `mas.hierarchy` 为 6.789256/6.736245ms；`mas.prepare` CPU 为 7.391366/7.120698ms，反而小于对应 GPU numeric/prepare union。异步发射、前序工作等待与嵌套范围使这些包络不能相加或相减得到“纯 CPU 可省”。删掉一个同步点后，等待可能移到后续同步点。只有经过 timeline/对照证实会从 whole 关键路径消失的提交、copy、readback 或等待才能计入 credit。

同样，Converter kernel 首次到末次跨度 252.283683/140.138544ms 跨越多个 Newton，间隙 247.476458/135.691447ms 不属于 Converter 可删除 GPU 工作。owner 下 D2D/其他 memcpy 单列，不与 CPU 包络重复相加，也不默认可免费删除。

## 4. 完整 120 帧的相邻结构命中

两场景分别 676/691 次 production linear system observations，1 个 owner。observation_index 连续，gap_before 与 pair_gap_before 均为 0；没有 empty eligible comparison。真实相邻机会分别为 N−1=675/690，首次观测不作为相邻机会。本表的“全相邻命中”以 N−1 为分母；分析 JSON 的 `hit_fraction_all_observations` 另以 N 为分母。

| 场景 / 结构 | 相邻可比数 / N−1 | 相邻可比率 | exact 命中 / 可比 | 可比内命中率 | 全相邻命中率 | 全 observation 命中率 |
|---|---:|---:|---:|---:|---:|---:|
| sphere / raw | 72 / 675 | 10.667% | 54 / 72 | 75.000% | **8.000%** | 7.988% |
| sphere / mapping AND partition | 70 / 675 | 10.370% | 54 / 70 | 77.143% | **8.000%** | 7.988% |
| sphere / canonical pairs | 189 / 675 | 28.000% | 176 / 189 | 93.122% | **26.074%** | 26.036% |
| fixed / raw | 70 / 690 | 10.145% | 58 / 70 | 82.857% | **8.406%** | 8.394% |
| fixed / mapping AND partition | 70 / 690 | 10.145% | 58 / 70 | 82.857% | **8.406%** | 8.394% |
| fixed / canonical pairs | 204 / 690 | 29.565% | 185 / 204 | 90.686% | **26.812%** | 26.773% |

raw 不可比：首次 1，各有 603/620 次 shape_changed；另外可比的 raw 内容改变为 18/12 次。converted joint 不可比为首次 1、shape_changed 605/620；canonical 不可比为首次 1、shape_changed 各 486，内容改变 13/19 次。这是尺寸或身份判定后立即 miss，不是随机抽样。此处 shape 改变不能仅解释为某一个 count 改变，metadata 的全部合同仍需保留。

canonical hit 但 mapping/partition joint 未命中有 122/127 次。这说明 canonical 单独可比范围更广，也直接说明不能把其 176/185 个命中都送给旧 raw permutation 或认定旧 MAS 分区可用。partition 单独 exact 命中 68/65 次，也不是 joint 或 MAS 命中。

reference 选出的重窗口 sphere 41–43、48–50，fixed 39–41、57–59 内 raw/joint 都是 0 可比。sphere 两窗口以及 fixed 57–59 的 canonical 也是 0 可比；fixed 39–41 的 canonical 只有 3 次可比、1 次命中。这些结果削弱了“重帧天然更适合结构复用”的假设，但窗口仍不是整场成本分布的替代。计数命中率尚未按可删除符号时间加权，不能直接乘以 GPU/CPU phase 预算。

## 5. 真实比较与快照成本：测到的是诊断器

下表为完整 probe 运行的六个阶段。CPU 包含对应同步及 readback 子区间；GPU 是有效 stage event 区间。`n` 是实际有效 event 次数，不以全部系统数冲淡单次 compare 成本。CPU 与 GPU 不能相加。

| 场景 / stage | n | CPU inclusive ms | GPU event ms | readback CPU 子区间 ms |
|---|---:|---:|---:|---:|
| sphere / raw_compare | 72 | 30.1519 | 0.672864 | 28.8293 |
| sphere / raw_snapshot | 622 | 189.4397 | 4.424064 | 0 |
| sphere / converted_compare | 70 | 7.8494 | 0.801856 | 6.5368 |
| sphere / converted_snapshot | 622 | 45.5159 | 6.898432 | 0 |
| sphere / pair_compare | 189 | 13.3675 | 4.936896 | 9.1447 |
| sphere / pair_snapshot | 500 | 13.3304 | 7.846464 | 0 |
| fixed / raw_compare | 70 | 29.3491 | 1.245184 | 27.9859 |
| fixed / raw_snapshot | 633 | 199.1829 | 7.901696 | 0 |
| fixed / converted_compare | 70 | 12.1840 | 1.233920 | 10.8900 |
| fixed / converted_snapshot | 633 | 82.4654 | 10.358496 | 0 |
| fixed / pair_compare | 204 | 14.8476 | 4.888704 | 9.9031 |
| fixed / pair_snapshot | 506 | 14.3819 | 5.259136 | 0 |

linear probe 总 CPU inclusive 为 300.8015/353.5747ms，均值 444.973/511.686μs 每 observation。其中 event setup 为 0.0011/0.0015ms，workspace 为 0.2113/0.3031ms，已经包含，不能再加六阶段或父级 CostScope。以实际 compare 次数计算，raw compare 平均 CPU 为 418.776/419.273μs，GPU 为 9.345/17.788μs；pair compare 为 CPU 70.728/72.782μs、GPU 26.121/23.964μs。CPU 与 GPU 差值不能全当作可省同步。

raw/converted snapshot 次数分别正好 N−raw/joint hits；pair snapshot 为 N−pair hits。低全系统命中意味着完整观察器经常准备新快照，不能只报告少数成功 compare 的成本。最大 raw 数为 139975/298552，unique 为 20718/42266；三个快照池峰值总容量为 2769984/6247296 字节（2.642/5.958MiB）。实际优化缓存还可能需要其他 owned symbolic layouts、hierarchy 和 Graph 缓冲区，不能把这个 probe 峰值当最终缓存峰值。

这六阶段是在 events=true 下逐阶段同步等待的诊断实现；此外同一 probe 臂还运行完整 cost 输出和 BVH 私有 replay。raw_snapshot 的 CPU 区间在转换前可等待前序装配，converted/pair 阶段也可能等待生产转换。未来最小 guard 可以改变提交和同步组织，但必须保留完整精确比较、shape/owner/epoch 合同、可靠 miss 路径和缓存 ownership。**本报告不将 300.8015/353.5747ms、observer_on−off 差值、JSON 输出或 replay 成本当作未来最小 guard 的必需开销、下界、上界或预期加速。** 未来 net overhead 和时间加权命中率仍 pending。

## 6. 至少 5% whole 净成本的必要条件

门槛定义为 `T_candidate <= 0.95 * T_reference`，即 paired speedup 至少 `1/0.95 = 1.05263158×`；仅 1.05× 相当于 4.7619% 成本减少，不够严格 5%。诊断参考的完整 `all` 中位数给出以下计划预算，最终仍必须使用新实现的同程序同臂完整配对对照：

| 场景 | 当前 all 120 帧中位 solver s | 5% 净节省至少 ms | candidate 上限 s | 当前 base/all 配对比率中位数 |
|---|---:|---:|---:|---:|
| sphere | 8.1070887 | **405.354435** | 7.701734265 | 1.4618× |
| fixed | 11.9955370 | **599.776850** | 11.395760150 | 1.4417× |

如果仅在 Graph-only 臂评估，需分别对 9.2028584/13.0889275s 取得至少 460.142920/654.446375ms；该臂的结果不能直接晋级到 all 臂。当前 all 相对本轮 Stiff 观测 2× 目标还需进一步减少约 26.91%/27.92%。即便新分支恰好达到 5%，也不表示已达到 2×。

对同一完整执行的第 i 个系统，设 C_i 为完整 ordered-raw guard 后真正可消除的 Converter 符号关键路径，S_i 为额外 subsystem guard 成立时可省的 indices 准备，H_i 为完整 MAS 拓扑 guard 成立时可省的 hierarchy。其 credit 必须排除当前 numeric gather/reduce/fill/aggregate/factor，并排除已启用的 static cache 已省部分。令 R_i/P_i/M_i 为各自充分 guard 成立时的 0/1 指示，O 为本次完整执行额外引入的所有关键路径成本，则必要条件为：

```text
Net = sum_i(R_i*C_i + P_i*S_i + M_i*H_i) - O
Net >= 0.05 * T_reference
```

O 包含所有 hit/miss guard、miss 时快照准备、layout 恢复、额外数值移动、容量增长/冷启动、缓存管理和 Graph 失效/重新捕获的真实净成本；重叠时间按关键路径计一次。不得把 kernel union、CPU inclusive、event phase 与 readback 相加。分别统计三个 predicate，不能用 canonical 的较高命中率给 raw 或 MAS 发 credit。所有系统连续观测；暂停、owner/device/dims/epoch 变化和未观察的系统建立 gap，不能跨 gap 计相邻 hit。

若为一个简化分支定义 **时间加权** hit p、whole 中可消除符号关键路径占比 F、whole 新增净成本比例 c，则：

```text
p*F - c >= 0.05
p >= (0.05+c)/F
F >= 0.05+c is necessary even if p=1
```

| 条件示例（非实测） | c=0 时最低 p | c=1% 时最低 p |
|---|---:|---:|
| F=4% | 不可能 | 不可能 |
| F=8% | 62.5% | 75.0% |
| F=10% | 50.0% | 60.0% |

本轮 raw 全相邻计数命中只有 8.00%/8.41%，canonical 26.07%/26.81%。**只有额外假定每 observation 可省符号成本相同**，计数才可代替 whole 模型的 p；此时必须把首次 miss 也计入 whole 的 N 分母，使用 raw 7.988%/8.394%、canonical 26.036%/26.773%。在该假定、c=0 下，raw-only 要求 F 至少 62.59%/59.57%，canonical-only 要求 F 至少 19.20%/18.68%。这只是指出均匀工作量模型的严格要求，不是实测 F、whole 上界或缓存不可行证明。重窗口低命中提示不能默认命中集中于高成本系统。完整时间加权 credit 仍待量化，且 canonical-only 中 MAS predicate 未有充分 guard。

## 7. 本轮有限决策与下一次进入条件

| 分支 | 本轮判断 | 必须补齐后才有意义的下一步 |
|---|---|---|
| A：ordered raw 符号快照，复用 Converter sort/partition/compact layout | 暂不实现生产缓存。全相邻命中约 8%，Converter 单帧大部分 GPU 工作是必须更新的数值；没有证据支持 whole ≥5% | 在完整同臂执行中量化 raw hit 上的 C_i，证明 guard+owned restore+miss/cold+Graph 代价后仍可能达到 5%；保持本次数值 gather/reduce 的计算顺序与结果合同 |
| B：canonical/subsystem/MAS hierarchy 符号复用 | 暂不实现生产缓存。canonical 相邻命中约 26% 扩大机会，但约 93–96% 单帧 MAS prepare kernel 成本属于不可缓存 numeric；MAS 充分 guard 仍缺失 | 给出独立 topology/partition/contact consumer 的完整 guard 及 owner/epoch 合同，量化 S_i/H_i 和新增成本；先保留全部 numeric prepare，不用 pair_equal 代替其合同 |

继续门槛测量最多一个完整两场景回合：在冻结数值边界下，用同一执行配置、连续全部系统的可省符号组件时序与 owner 身份建立时间加权模型，同时单列未来最小 guard/恢复/冷启动成本。若即使乐观合法 credit 扣除已确认必需成本也不足 5%，停止该分支；不能重复挑更有利单帧。若存在 ≥5% 的合法预算，再进入一个有范围上限的符号缓存原型，并做完整 fallback、owned layout、空/尾/owner/shape/gap/Graph 合同验证。

原型最终是否晋级取决于完整未重度 instrument 的同程序配对计时与既有物理质量验收，两场景都需满足净 ≥5%，保持完整 CCD、材料和数值停止标准。若实现改变算术顺序、PCG 工作量或轨迹，记录为实质变化并按既有质量门控处理，不能将少算物理步骤包装为符号加速。退役 fusion/K4/contact pool 不在本报告路线内。

当前组件验证证据来自 root 运行的第三 clean build：17 CTest 全通过（20.50s），其中 CPU probe contract 为 98 checks，GPU fixture 为 41 cases/71250 checks。它们验证 probe 合同和生产 Converter 接口，不是缓存原型测试。该 agent 只进行了 CPU 只读证据分析，没有启动 GPU、构建或提交 Git。

## 8. 可复核性

- 已完成分析复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_reference_costs.py`；`analyze_profile_costs.py`；`analyze_structure_queries.py --require-complete`（本报告引用已经生成的结果，不重新运行 GPU）。
- profile 分组方法：`load_capture(folder)` 只读 SQLite/query_only，检查前后文件哈希；仅取 `not graphNodeId` 且 `outside_graph_runtime_owner == 'linear.matrix_convert'`，以 compute_hash_and_index、CUB DeviceRadixSort/DeviceScan、compute_sorted_partition、set_row_col_from_partition、set_dst_val、fast_segmental_reduce 的互斥 token 分组并校验总和。其结果已在本报告第 3 节逐项给出。
- 结构计数：读取 `linear.per_system`，检查 observation_index 连续与 gap；读取 `linear.reuse.{raw,converted_joint,canonical_pairs}`，排除 first/shape/owner/gap/empty，分母分别保留 N−1、eligible 和 N；阶段 cost 读取 `linear.costs.six_stages`，CPU/GPU/readback 不重复加和。
- 输入身份：REFERENCE_COST_ANALYSIS.json SHA256 `53df828f7e5916d5f87c33b32ab6c580077e798442c96cb418493ee7556ecd36`；PROFILE_COST_ANALYSIS.json `24528591ad49a421d3970c627a341349cc7aad27f5e6a35aad057ba95946cb0c`；含质量补充的 STRUCTURE_QUERY_ANALYSIS.json `286af167e17639a1351a0626c429f227092e4d90c6bf825260a3fed8faa857a1`。结构 JSON 的各 requested/resolved/output/cost 与双侧封口身份由其 `inputs`、`observer_off.closed_inventory` 和 PROBE_LEDGER 保存，不改原始 probe 文件。

状态：真实计数命中与诊断 compare/snapshot 成本已完成；全场可消除符号关键路径 F、时间加权充分 guard 命中 p、未来最小 guard 的 O/c 和最终净 ≥5% 仍 pending。以上 pending 项没有用 single-frame、canonical equality 或诊断输出成本替代。
