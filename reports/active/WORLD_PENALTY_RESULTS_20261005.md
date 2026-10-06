# 接触罚刚度坐标修正：实现、因果对照与本机复测

本轮完成了 `world_block` 修正候选及有限本机验证。**已证明该尺度规则会显著影响接触后的总求解工作量；尚未达到相对 Stiff 同质量或 2× 的目标。** 候选保持默认关闭，未启动 AutoDL、100/300 帧或最终七对验收。

## 1. 实际修改

活动 overlay 新增 `solver/toi_penalty_scale.h`，在 `toi_solver.cu` 增加世界坐标刚度估计；配置、runner、resolved 校验同步支持 `mu_coordinates`。

旧规则是 `mu=0.1*max(diag(H_generalized))`。混合场景中的最大值来自 ABD 仿射形变坐标，而 AL 约束是以米为单位的空间距离。

新规则为：

`mu = 0.1 * max(可动 FEM 对角项, diag((J H_ABD^-1 J^T)^-1))`。

这里 `H_ABD` 是完整的无接触 12×12 体刚度，包含惯性、投影后的形状项及 motor 项；`J` 使用实际生产顶点 Jacobian。没有丢掉 ABD，也没有把全局 mu 除以此前的 1855 倍对角比。固定节点排除，异常 pivot、无效 body ID 或顶点 compliance 直接失败，不静默换算法。

这仍是一个**全局对角启发式**，并非精确接触 Schur 补。它消除了 ABD 广义参数单位变化造成的尺度依赖，没有解决所有接触异质性和材料非线性。FEM 对角惯例及世界坐标轴保持原来定义。

该规则从每帧无接触状态计算；AL 能量、梯度、Hessian、slack/乘子更新共同使用新 mu，随后按正常路径重建 MAS。没有只改 A 沿用不一致的 b。材料、dt、停止阈值、完整 CCD、重启守卫和预算都保持不变。一次 PCG 内的预条件器不变。

默认 `generalized` 保留旧算法分支。七项配置测试通过，候选仅允许 TOI diagonal/movable 模式，冻结程序不接受此开关。历史配置缺少新字段时仅可解释为generalized；历史分析器比较保留原配置及哈希，不允许旧记录声称执行world候选。不能把保留旧分支等同于长轨迹逐位可重复；旧路径仍有明显接触后数值敏感性。

## 2. 构建及算子检查

- Release 程序 SHA256：`ae49936ffe29a39ce3b8ea5f8ee2511e5ea1e3d80cfc2df9081d961c20fbcd29`。
- 构建记录：`builds/active/provenance/20261005T022308_495197Z_world_penalty_20261005_compile/manifest.json`，37 个翻译单元身份及编译/链接输入有记录；最终构建 0 错误。历史源码/程序身份检查通过。
- 现有 43 项组件检查、15 项 PCG 守卫及新增 3 项世界尺度检查通过。ABD 坐标单位重参数化的 compliance 相对差异 `1.67e-16`；GPU 与 CPU 世界刚度结果误差 0。无效 pivot/映射均正确拒绝。
- 无 ABD 的悬挂布料、固定 ABD 兔子布料各 2 帧启动检查完成；这些不是布料加速比或 100 帧回归。
- 6 次正式短窗 TOI、1 次成本诊断、2 次边界启动，共 9 份实际 resolved/运行配置检查全部通过，无 PCG 上限或 breakdown。

最初构建受沙箱 Windows SDK 读取限制阻断，获准读取 SDK 后继续；一次候选临时缓冲成员命名错误已修复。最终验证使用上述成功构建，没有使用失败构建的程序。

## 3. 同状态矩阵对照：不能只归因于瞬时接触曲率

先复用 f25 n7/n8/n9 冻结 A/b/M。正常 AL 项完整重建，改变其曲率；b 和原 MAS 固定。全部接触为 FEM 接触，无摩擦、无 reduced-slack。世界尺度在该受保护内层状态重建，mu 约从 386.1407 变为 4.82356。

|系统|原 A PCG|缩放接触曲率 A PCG|原/新真相对残差|
|---|---:|---:|---|
|f25 n7|29|28|0.001428 / 0.002872|
|f25 n8|221|225|0.032888 / 0.020560|
|f25 n9|337|303|0.005510 / 0.005729|

所有案例在预声明的 1000 次/90 秒预算内达到原 rho 阈值。**这些真残差没有达到 CPU 准确参考的 1e-8；rho 收敛不能当作同精度证明。** 此对照固定旧 MAS，不代表一致重建预条件器后的求解成本，更不是物理验收。

结果说明，减少某个已形成困难状态中的 AL 曲率，不能立即消除 PCG 增长。此前 AL 方向曲率仅占很小比例的反证仍成立。完整非线性状态如何形成，必须一起考虑。

## 4. 三轮交错连续测试

场景 `bunny_cloth_bunny_l`，dt=.01，从零连续 35 帧，每次 120 秒预算。三次 Stiff、三次 generalized、三次 world；warp restriction/triangular action 和其他执行开关在两个 TOI 组完全一致。正式短窗关闭重型诊断；另一次成本诊断单列，不混入配对计时。

|指标（三轮中位）|Stiff|旧 generalized TOI|world TOI|
|---|---:|---:|---:|
|求解耗时 s|5.237|19.481|9.397|
|总 PCG|3654|21826|6025|
|方向/PCG 调用|142|172|191|
|每调用平均 PCG|25.73|126.90|31.54|
|TOI outer|—|54|90|

配对旧 TOI/world 耗时比为 **1.809、2.476、2.059×**，中位 **2.059×**。配对 Stiff/world 为 **0.557、0.567、0.559×**，中位 **0.559×**：新候选仍比 Stiff 慢约 **1.79 倍**。共享桌面负载未受控，只有 35 帧，没有正式性能认证。

总 PCG 分别下降 **68.93%、77.38%、72.49%**。第1–23帧两种 TOI 都恰好1503次 PCG，位置最大坐标差仅约 `0.7–1.3e-14 m`；超过 `1e-8 m` 的首次差异均在第24帧。接触后 PCG 从17910–25130降到4502–4529。候选 live mu 始终约4.82356，来源为世界 ABD 最大刚度48.23564，FEM最大值另有记录；不是把 ABD 刚度排除后仅取2.08。

因此，**单一尺度规则改变了第24帧之后的非线性过程，大幅降低每个方向的线性难度**。这是一条有连续运行支持的因果证据。它并不单独证明改善全部来自当前 A 的条件数，也不能将72%下降精确分配给材料、预条件器或接触项。固定状态矩阵对照的收益有限，恰好说明状态演化/M重建这条中间链不能省略。

## 5. 质量和剩余工作

|指标|Stiff 三轮|旧 TOI 三轮|world 三轮|
|---|---|---|---|
|最小 FEM J|约0.497869|−0.104464、−0.053977、−0.064445|0.235540、0.074551、0.073696|
|最多非正 FEM 单元|0|1|0|

候选消除了这三次短窗的翻转，但仍有明显过度压缩，且最小 J 对重复运行仍敏感。三次都不满足相对 Stiff 的质量门禁；微小布料拉伸/轨迹差异也没有被自动放宽。没有将“无翻转”作为“同质量”的替代。

world 的 outer 固定90次，单方向 outer 增为47–48次，而旧版只有7–9次；接受 alpha<.5 的 outer 为24–25次，旧版7–14次。PCG 已从每调用约127降到约32，但方向数及接触更新/装配次数增加，仍高于 Stiff 的142方向。不能再主要靠降低 PCG 发射开销解决总成本。

成本诊断选择24–26、33–35帧，2040条事件、63线性系统，嵌套一致性检查通过。所选帧求解3.766秒，线性区间累计2.534秒；其中 Graph replay约1.804秒、矩阵转换0.199秒、MAS准备0.411秒。区间包含提交间隙/追踪开销，仅用于定位。

Graph最终读回的主机等待约1.807秒与前面的Graph重放重叠，**不能把它再加成1.807秒新增传输开销**。节点级算子未在Graph内部拆分，63次普通MAS作用只覆盖初始化代表调用，不能把Graph全部时间再次归成三角kernel。

本轮未新增接受路径CCD验证，未达到其前置质量门禁；原AutoDL的零CCD标记不能继承到新轨迹。候选导出36份位置和速度，Stiff冻结程序仍未实际导出速度，完整同质量认证还有该证据缺口。

## 6. 当前结论及下一项有界工作

- **已证明：**世界罚刚度估计可实现、配置确认、坐标参数化一致、该混合短窗PCG显著下降。
- **支持但未证明：**旧尺度导致困难状态/材料条件恶化，继而增加每方向PCG；中间机制还需同状态完整A/b/M和能量项核对。
- **已排除：**仅靠缩放已形成困难系统的AL曲率即可解释全部收益；本轮已取得相对Stiff同质量2×。
- **待验证：**TOI外层/接触状态更新能否同时减少90次outer和过度压缩。不能因这次相对旧TOI 2.06×就继续长测。

下一项应聚焦24–26和33–35帧首次异常接触簇：记录完整目标分项、互补/可行性残差、safe–trial形变及每次接受推进量；对照首次同状态的接触模型和更新前后状态，解释为何大量outer只做一个方向仍需重复安全推进。只有证据支持时再改局部接触权重/更新机制。继续保留材料及CCD，不收紧/放松PCG停止、不全局半步、不增加上限，不做mu缩放网格。

## 7. 复现及证据

工具使用本机 `E:/Anaconda/envs/DL/python.exe`：

```text
tools/active/build.py build --label <fresh-label> --jobs 2
tools/active/test_contracts.py
tools/active/penalty_matrix_probe.py runs/active/frozen_contact_export/state_window/f25_n7 runs/active/frozen_contact_export/state_window/f25_n8 runs/active/frozen_contact_export/state_window/f25_n9 --output <new-output>
tools/active/batch.py --plan configs/active/world_penalty_pair_plan_20261005.json
tools/active/batch.py --plan configs/active/world_penalty_repeat_plan_20261005.json
tools/active/analyze_world_penalty.py --output <new-output>
tools/active/cost.py runs/active/world_penalty_mixed_cost_20261005/cost.jsonl --output <new-output>
```

命令从任务根目录调用时给脚本正确相对路径；现有运行名/输出保留，runner拒绝覆盖。重复执行需新运行名和输出。两份batch计划、边界启动计划、成本配置均在configs/active中。全部结果及输入哈希在 `WORLD_PENALTY_MATRIX_PROBE_20261005.json`、`WORLD_PENALTY_RESULTS_V2_20261005.json`、`WORLD_PENALTY_ALL_CONFIG_GATES_20261005.json`、`WORLD_PENALTY_COST_20261005.json` 和相应runs/active目录。
