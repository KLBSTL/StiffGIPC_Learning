# CUDA Graph＋TOI 全局实施与阶段结论

日期：2026-10-04。主场景 `bunny_cloth_bunny_l`，dt=.01。本轮全部使用本机 RTX 3070 Laptop；桌面共享负载，因此全部时间为诊断，**尚未获得同质量2×加速，也没有完成最终验收**。

## 1. 已落地的实现

- 一个活动覆盖层 `sources/stiff_active`，继承冻结v50公共实现及v54 TOI；历史源码、失败记录和旧可执行文件保留。
- 一个公共入口 `tools/active/{build,config,run,batch,analyze,index}.py`。每次实验独立目录、有限帧数/时限、串行GPU锁；清除环境中的隐式GIPC参数；记录请求、实际配置、源码/程序身份及运行环境。
- 独立对象目录、源路径对应的对象名称、实际编译与链接证据、有效工程头文件依赖闭包、每次构建的清单和小型源码/二进制归档。
- `ToiOptions` 和 `ToiObserver` 提取配置及观察逻辑，四种停止阈值独立表达，原默认值保持：IPC .01、TOI剩余 .01、trial速度 .05、PCG rho 1e-4。没有重写材料、ABD/FEM和完整Newton状态流程。
- 统一帧/outer/inner/接触模型/线性系统身份，选择窗口成本事件，固定A/b/M检查、CPU准确参考、接触状态离线分析以及RHS交叉实验。
- 默认关闭的FP64固定归约树限制算子候选。它仅改变求和顺序；保留原Cholesky因子及延拓。Graph签名含执行模式；受保护对照只在两次完整PCG之间切换，运行中不静默回退。

代码整理尚不是整个计划完成：迁移轨迹门禁和物理质量门禁仍须满足，部分完整目标/互补与速度验收尚未闭合。

## 2. 本轮发现并修复的构建与诊断问题

**已证明：头文件变化曾漏触发CUDA重编译。** 仅隔离对象目录不能解决这个问题。本轮发现MAS对象保留旧版`CostScope`布局，而PCG对象使用新版布局，固定算子探针在第二次SpMV前崩溃。新增工程include闭包检查后，准确作废了陈旧的MAS对象，3帧探针完成且主解/右端项/算子输入保持。失败日志未删除。

干净构建编译37个TU；单个PCG源文件内容修改及恢复各只更新该TU和device-link对象；最终no-op更新0个对象且EXE哈希不变。证据：[增量验证](../../builds/active/provenance/20261004T130851_529554Z_phase_a_incremental_verified/verification.json)。这验证的是本轮构建路径，不保证所有外部SDK依赖都由自定义闭包管理。

首次Nsight运行模拟正常但未产生报告：使用普通NVTX字符串而未启用匹配。公共runner已设置`NSYS_NVTX_PROFILER_REGISTER_ONLY=0`并检查真实`.nsys-rep`文件；原两次记为追踪失败。Graph和节点报告分别成功生成。依据[NVIDIA官方说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-command-lines)。

`phase_ms.pcg`在当前TOI输出中是零占位，不能作为零成本。新分析器将此类未测值写为null；早期活动报告中TOI的零值不参与成本归因。

## 3. 正确性与迁移状态

- 43项组件、15项PCG守卫、六份历史MAS GPU系统通过；包含Graph重放和缓冲增长。
- 六历史CPU参考真残差最大约4.69e-11；第2、25、34帧三个完整新系统CPU准确参考最大约1.33e-14，均低于1e-8。
- 保存默认rho对应的真实残差和解误差。默认rho=1e-4不是欧氏真残差1e-4，更不是1e-8；没有据此宣称默认PCG达到高精度参考标准。
- 前3帧活动版和冻结v54差异约1e-16m；35帧测试的24–26窗口差异小于1微米，但超出已冻结两次v54重复范围。**迁移验收待确认，未扩大范围。** 33–35窗口在原冻结范围内。
- 相对Stiff的质量已经失败：同轮Stiff35帧最小FEM J约0.498、零非正单元；当前TOI最小J约-0.00567、最多1个非正单元。另有分物体轨迹差异。采用相对基线标准，不新增禁止所有FEM翻转的约束。
- 全轨迹独立接受路径CCD、全部速度来源验证和100帧质量范围尚未完成。因此完整质量认证始终为false，未用检查点替代连续从零推进。

证据：[迁移门禁](MIGRATION_TOI_WINDOW.json)、[实际配置](RESOLVED_CONFIG_GATES.json)、[CPU新系统](CPU_NEW_WINDOW_SYSTEMS.json)、[初值质量与工作量](TOI_INITIAL_GUESS_GATE.json)。

## 4. 原因已经缩小到哪里

### 4.1 工作量而非单一发射开销

|35帧从零运行|总秒数|方向次数|PCG次数|outer|最多非正FEM单元|
|---|---:|---:|---:|---:|---:|
|同轮Stiff三次|6.388 / 6.488 / 6.413|142|3654–3657|不适用|0|
|冻结v54两次|47.494 / 44.122|171 / 165|21716 / 20326|54 / 52|1|
|活动版默认|41.801|169|20413|54|1|
|只继承trial候选|46.457|198|21962|103|121|

上表不是配对最终加速比。活动版与冻结版存在轨迹分岔，不能把41.8对44–47秒解释为整理代码带来速度收益。

### 4.2 首次迭代膨胀的因果对照

第25帧outer1从第7到第8方向，接触身份、lambda和gamma没有改变；1129个接触全部正slack、lambda=0。高罚权、trial压缩在正常低迭代系统中也存在，因此二者都不是充分解释。

**固定A/M、交叉RHS的CPU诊断：**

|固定算子|右端项b7|右端项b8|
|---|---:|---:|
|A7/M7|29|247|
|A8/M8|28|221|

停止规则同为rho相对1e-4。CPU和GPU归约不同，因此不要求每次迭代数相等。**支持但未证明：b8显著激发了现有预条件器难处理的方向；不需要发生接触增删或突然的矩阵装配错误，就能重现迭代膨胀。** 这不排除接触罚权与材料状态长期影响谱，只排除“看见罚权大即可直接缩小”的简单归因。

离线局部顺应度归一化的影子权重会使中位罚权缩小约两万倍，尚无质量证据支持。该分支停在离线检查，不进入全局倍率网格，也不把影子结果当作已测试的完整算法候选。

证据：[交叉RHS](CROSS_RHS_F25.json)、[异常簇](CONTACT_SHADOW_F25_R2.json)、[正常对照](CONTACT_SHADOW_NORMAL.json)、[历史语义](HISTORY_SEMANTICS.md)。

### 4.3 真实执行热点

|观测范围|限制|局部三角作用|解释|
|---|---:|---:|---|
|三个窗口512次固定输入MAS应用|652.021 ms|224.439 ms|MAS总应用885.863 ms；含发射间隙，不是生产迭代计时|
|Nsight第25帧首系统28次应用|34.868 ms|13.339 ms|分别占该系统GPU kernel时间60.1%、23.0%|

最长限制CSR行19193项。原每标量一个线程按顺序累加，在GPU上形成长尾；此处比先改三角作用更值得优化。

TOI三个窗口64个系统：生产Graph重放17.356秒，线性准备0.689秒，其中矩阵转换0.219秒、MAS准备0.440秒；固定探针额外审计约10.460秒必须剔除。不能把包含探针的28.670秒根区间当正式线性时间。

IPC对应50个系统：Graph重放1.989秒、准备0.549秒。两者系统和轨迹不同，表格用于解释成本结构，不是同A的速度对照。

Nsight中`cudaMemcpy`的CPU耗时包含等待Graph完成。高readback占比不能直接解释成PCIe带宽瓶颈。图捕获管理仍只占很小部分，本轮没有重写全TOI Graph或GPU活动集。

证据：[TOI成本表](TOI_COST_SERIAL.json)、[IPC成本表](IPC_COST_SERIAL.json)、[候选预声明门禁](RESTRICT_CANDIDATE_PROTOCOL.md)。

## 5. 优化候选与剩余门禁

固定归约树候选已完成本轮有限验证，默认serial保持。六历史系统和三个新窗口的CPU参考、R/P结构、正二次型、对称性通过；六历史GPU作用对CPU最大相对差约1.25e-9，低于预设1e-8门槛。新增第25帧第7/8方向的困难系统也通过准确参考和算子完整性检查，后者CPU真残差约4.98e-13。

同一A/b/M的三个代表系统，3组交错配对：host PCG中位加速2.12–2.23×，Graph PCG中位加速2.22–2.28×；迭代数6/27/25保持一致，解相对差约1e-15至1.13e-14。首次困难系统第25帧n8仍得到约2.12× host、2.21× Graph。这里**不包括共同的准备成本**，不是整场加速。

困难系统默认PCG具有重复波动：serial为221–223次，warp为221–223次；warp到第一份serial解的差异在本轮serial自身重复范围内。默认解对准确CPU解误差仍约6.4–6.6%，欧氏真残差约.02–.05；这属于原停止规则和困难系统的局限，不能被2.2×局部收益掩盖。本轮没有全局收紧rho。

|同程序35帧交错顺序|秒数|方向次数|PCG次数|最小FEM J|
|---|---:|---:|---:|---:|
|serial r1|43.142|170|21173|-.04205|
|warp r1|21.814|169|20698|-.05042|
|warp r2|23.445|177|22664|-.03550|
|serial r2|37.630|168|20407|-.04695|

两个配对耗时比为1.978×、1.605×，描述性中位1.791×。没有七组配对或置信下界认证。轨迹已分岔，不能宣称严格同质量1.79×。

选择窗口剔除探针审计后，完整线性区间从18.210秒降到10.205秒（44.0%下降，1.784×）。两次诊断工作量不同：64/91个系统、9687/13026次PCG；按PCG数归一化为2.40×，仅作解释，不能冒充同工作量测量。同系统PCG配对、共同准备仍保留及这份线性成本证据支持执行收益超过15%，**不构成轨迹推广条件通过**。

Nsight同一帧首系统28次应用：限制kernel从34.868毫秒降到5.382毫秒；新路径中局部三角作用占GPU kernel时间44.5%。该单次追踪不能独立认证6.48×kernel收益；它与固定系统配对共同确认长尾已明显减少。本轮没有据此再开启新的三角kernel分支。

|同等稳定MAS与refit/能量组件的35帧诊断|host|Graph|
|---|---:|---:|
|IPC|7.399秒|7.282秒|
|TOI|21.094秒|21.814 / 23.445秒|

TOI没有胜过同等执行优化的IPC。不同TOI轨迹有不同迭代数，这个小表也不能用来断言Graph本身变慢。Stiff同期中位6.413秒，TOI warp中位22.630秒，相对Stiff的诊断加速比仅约0.283×（耗时约3.53倍）；这不是最终100帧加速比。

![35帧执行成本与PCG工作量，阴影为实测重复范围](figures/window_cost_and_work.png)

四组短窗口均未取得完整相对基线质量认证：TOI仍有1个非正单元、显著FEM状态差异；IPC两组虽零非正单元、最小J约.498，但严格逐帧冻结范围也有小幅越界，未自动放宽。候选继续默认关闭。**按质量门禁停止100/300帧和布料长测，未声称全计划通过。**

证据：[同系统配对](RESTRICT_PROTECTED_PAIRS.json)、[困难系统配对](RESTRICT_DIFFICULT_PAIRS.json)、[CPU困难系统](RESTRICT_CPU_DIFFICULT.json)、[完整线性区间](RESTRICT_LINEAR_COST_COMPARISON.json)、[35帧质量](RESTRICT_WINDOW_QUALITY.json)、[四组窗口](FOUR_WAY_WINDOW_QUALITY.json)。

整体阶段状态：A代码/构建/配置已实施、轨迹迁移待确认；B成本与历史核心证据已建立；C初值候选拒绝、权重候选无支持而不推广；D限制热点实现及局部验证完成、正式推广未通过；E未进入。C的权重分支只完成离线因果筛查，不能称作第二个完整算法候选已运行失败。

即使D通过，当前TOI的物理质量与总工作量仍不足以支撑2×目标。下一次算法修改应针对已经冻结的困难RHS及对应非线性轨迹，先在同A/b/M检查预条件器与难方向，再决定是否改变接触模型。不得再次放宽收敛、增加上限或恢复低精度逆。

## 6. 可复现入口

在任务根目录，使用 `E:/Anaconda/envs/DL/python.exe`：

```text
tools/active/build.py build --label <unique-name> --jobs 2
tools/active/test_contracts.py
tools/active/fixtures.py --name <unique-name> --mas-restrict serial
tools/active/run.py --config configs/active/toi_window.json --name <unique-name>
tools/active/batch.py --plan configs/active/restrict_evaluation.json
tools/active/index.py
```

运行名称不可复用。已执行batch中的名称也不可复用；复现实验先复制JSON并改运行名称和报告路径。`requested.json`、`resolved_config.json`、`build_manifest.json`和结果保存在每个运行目录。统一入口：[实验索引](EXPERIMENT_INDEX.json)。

本轮最后实际执行的验证命令（以上述Python解释器为前缀）：

```text
tools/active/test_contracts.py
tools/active/fixtures.py --name restrict_final_fixtures --mas-restrict warp
tools/active/verify_fixture_identity.py runs/active/restrict_warp_fixtures runs/active/restrict_final_fixtures --output reports/active/FINAL_FIXTURE_IDENTITY.json
tools/active/verify_systems.py --prefix runs/active/restrict_difficult_systems/fixed/f25_n7 --prefix runs/active/restrict_difficult_systems/fixed/f25_n8 --output reports/active/RESTRICT_CPU_DIFFICULT.json
tools/active/validate_run.py runs/active/restrict_ipc_host_window runs/active/restrict_ipc_graph_window runs/active/restrict_toi_host_window --output reports/active/FOUR_WAY_RESOLVED.json
```

结果：4项配置合同测试、43组件、15守卫、6历史GPU系统通过；最终构建与CPU审计所用构建的66份因子/作用输出逐字节相同；两个困难系统CPU参考通过；四组配置与实际执行核对通过。最终EXE SHA256：`976dcf949c5c51d00c2eee84f49aa8bf3f15890f569093081bf4075de765152b`。正式质量与性能认证仍未通过。

100帧七组配对、单侧95%下界、300帧及三种布料回归只在前置门禁通过后执行。本轮没有启动这些最终验收，也不把35帧诊断当作它们的替代品。
