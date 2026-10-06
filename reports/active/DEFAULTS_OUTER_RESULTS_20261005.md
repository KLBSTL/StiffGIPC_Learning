# 两项默认开启与下一步外层诊断

按用户要求，活动源码和公共 runner 现在默认开启 **因子逆作用 `Bᵀ(Br)`** 和 **世界坐标 TOI 罚刚度估计**。显式 `mas_factor_action=triangular`、`mu_coordinates=generalized` 仍能关闭；原材料、ABD/FEM、完整 CCD、四种容差和预算未改。历史报告中的默认关闭是当时决定，本次用户选择已覆盖它。

**默认值已验证，质量问题尚未修复，2×目标尚未达成。** 本次另外完成了外层能量/状态诊断及第34帧单个子问题的退出充分性对照，没有将退出候选推广为默认。

## 1. 默认值与证据身份

|适用路径|当前默认|保留的关闭方法|
|---|---|---|
|稳定 Cholesky MAS|`factor_inverse`，FP64 B=L⁻¹ 后作用 Bᵀ(Br)|`GIPC_MAS_FACTOR_ACTION=triangular`|
|TOI diagonal/movable|`world_block`，以完整 ABD compliance 转换到世界刚度并保留 FEM 尺度|`GIPC_TOI_MU_COORDINATES=generalized`|
|MAS restriction|仍为 `serial`|本次短窗另显式测试 `warp`；没有顺带改第三项默认|

公共 `preset=toi` 将两个 `auto` 解析成上述设置。legacy MAS 不激活因子逆；历史 mass/full-scope 路径保留兼容的尺度估计。源码、runner和实际 `resolved_config.json` 一致。

三个最终2帧原生默认smoke覆盖混合、无ABD悬挂和固定ABD兔子。runner精确移除了这两个环境变量，程序仍确认 `factor_inverse/world_block`，restriction为serial。关闭两项的2帧smoke确认 `triangular/generalized`；与旧 generalized 参考位置最大坐标差仅 `1.11e-15 m`。这只验证启动和关闭兼容，不能替代碰撞阶段质量验收。

两次Release构建分别用于组合/观察诊断及单点退出探针；最终程序 SHA256 为 `57782cfeabdc35eb3151187a0f466911ef7b2ecf831b13ecc60ef221385b75f6`。构建清单登记37个编译单元，本次最后改变3个对象，编译/链接与源码身份记录在 `builds/active/provenance/20261005T034848_056273Z_selected_outer_exit_probe_20261005/manifest.json`。组合三轮使用此前同日程序 `e50310ed…ce0e`；最终探针关闭时不修改物理计算，新增字段是观测/诊断字段，不宣称两个程序逐位轨迹等价。

8项配置合同、46项组件（43旧＋3世界尺度）、15项PCG守卫、六份历史系统检查通过。13份活动运行的实际配置/执行检查通过，无PCG触顶/非有限值；这些检查不证明物理质量通过。

## 2. 本机35帧组合测试

RTX 3070 Laptop，共享桌面负载，`bunny_cloth_bunny_l`、dt=.01、从零连续35帧。关闭重型探针的计时组，与单独的outer探针分开。**没有在本次重测4090；用户引用的54组4090数据仍是历史因子逆/旧罚尺度结果。**

|配置|求解时间 s|总 PCG|方向数|outer|最小 FEM J|
|---|---:|---:|---:|---:|---:|
|同日当前 Stiff，一次|5.536|3654|142|—|0.497869|
|两项默认＋显式warp，三次中位|8.822|6109|192|91|三次分别0.085215 / 0.235540 / 0.235540|
|全部当前默认，serial，一次|14.415|6163|193|88|−0.049119，1个非正单元|

对同日这一次 Stiff 的描述性比值为 **0.627×（显式warp）**、**0.384×（全部默认serial）**。不是七对交错，不是受控负载，也不是同质量加速。旧世界尺度＋三角作用三轮中位9.397秒、6025PCG；本次8.822秒约1.065×的跨批比值不足以证明稳定的额外整场收益。

三次组合组都不满足相对Stiff的FEM压缩门禁；完全默认组更出现了非正单元。仍存在重复轨迹敏感性。用户要求的两个默认值保留，但不把“程序能运行”改写成“问题已修复”。所有图中J均为接受的物理帧端点，不混用trial形变。

![35帧接受端点FEM J](figures/defaults_fem_j_20261005.png)

证据：[组合几何及工作量](DEFAULTS_COMBINED_GEOMETRY_20261005.json)、[全部默认几何](DEFAULTS_NATIVE_WINDOW_GEOMETRY_20261005.json)、[组合batch](DEFAULTS_COMBINED_BATCH_20261005.json)。

## 3. 外层追踪的确切发现

新增 `outer_probe` 仅在指定帧启用，记录帧/outer/接触模型/线性系统编号，覆盖模型建立、内层结束、乘子更新、安全接受四个时刻。完整能量分项包括惯性、ABD形状、FEM、布料、弯曲、约束、AL接触及摩擦；能量探针前后保护的物理状态逐字节一致。

第1–3、24–26、33–35帧共32个模型、128条事件，能量分项和完整目标最大差 `1.73e-18`。19个outer只做一次方向并以full_step退出，零个单方向outer以velocity_converged退出。被记录的可行性/互补量属于已求解的仿射接触模型，不能称完整物理KKT证书。

首次观察运行的第34帧：

|outer|内层次数|末方向速度 m/s|接受alpha|trial最小J|safe接受最小J|
|---|---:|---:|---:|---:|---:|
|1|2|0.476|0.230|0.475|0.473|
|2|1|0.496|0.520|0.095|0.191|
|3|1|0.221|0.536|0.004|0.040|
|6|1|0.204|0.676|0.080|0.073|

outer2→3的已有接触，在同一个warm trial上重线性化的最大仿射gap变化达13.3δ。outer2完整目标从0.010636降到0.009066，**能量下降仍伴随局部FEM压缩**；它没有证明“总能量不降”或“线搜索没有执行”。新接触模型、slack/乘子和非凸材料状态之间的作用需进一步分项归因。

对照本地冻结的公开Robust `externals/history-Robust/src/backends/cuda/engine/advance_al.cu:334,419`：其Newton循环和min_iter计数本来横跨安全推进。当前帧级六次计数与这一历史语义相符；仅因新outer只做一次，不能认定移植漏了“每模型六次”。当前safe重启守卫保留。没有执行全局velocity_only或将所有模型统一重置六次。

这份诊断说明大量outer的退出并不等于原.05 m/s小方向收敛；它提供具体检验对象，**尚未证明其造成所有重复推进或所有质量偏差**。证据：[外层诊断](DEFAULTS_OUTER_DIAGNOSIS_20261005.json)。

## 4. 下一步已经执行：单个outer退出充分性对照

先声明预算，再从零运行一个native诊断、两个选定诊断。仅第34帧outer2抑制full_step启发式，保留原.05 m/s小方向阈值，最多8次内层迭代（比生产1024更小），每次35帧/90秒。所有其他模型采用原规则；此探针默认关闭，实际目标在resolved配置和Newton记录中确认。

|诊断组|指定子问题N|指定末速度|第34帧outer / PCG|整场PCG|整场最小J|
|---|---:|---:|---|---:|---:|
|native|1|0.3067|7 / 434|6129|0.235540|
|选定r1|6|0.0422|4 / 797|6219|−0.075168|
|选定r2|3|0.0453|8 / 627|6042|0.085758|

所选子问题均在八次上限内达到原速度阈值，但没有一致减少后续outer或整场工作，质量也未通过；r1第35帧出现非正单元。探针组计时含诊断，不用于速度推广。按预声明条件结束这条局部退出候选，不扩大次数/阈值网格。

**因果限制：**第33帧（干预前）native到r1/r2的FEM位置RMS已为 `5.10e-5 / 3.01e-6 m`，ABD位置RMS为 `1.68e-4 / 6.98e-6 m`。不能把r1翻转或所有工作量差异全归因于选定退出改动，也不能把r1少三个outer包装成同状态算法收益。这个有限对照不支持直接默认推广；要证明局部机制，必须有受保护的同状态A/b/M和接触状态探针，而不能依赖未验证续算等价的检查点。

证据：[预声明协议](DEFAULTS_OUTER_PROTOCOL_20261005.md)、[退出batch](OUTER_EXIT_PROBE_BATCH_20261005.json)、[退出诊断](OUTER_EXIT_DIAGNOSIS_20261005.json)、[退出几何与干预前差异](OUTER_EXIT_GEOMETRY_20261005.json)。

## 5. 结论与下一项工作

- **已证明：**两项源程序/runner默认值及关闭开关有效；本轮配置、组件、守卫和历史系统检查通过；接触后仍存在大量full_step单方向outer和显著FEM压缩。
- **支持但未证明：**重线性化、slack/乘子更新与材料非凸状态共同影响困难方向及重复推进；完整目标下降不足以排除局部压缩。
- **本候选不采用：**第34帧单模型增加内层求解没有稳定工作量/质量收益。它不是全局退出规则的最终证伪，也不是新的默认算法。
- **待验证：**接触类型/身体对应的权重及更新机制在同状态下的贡献；完整相对Stiff质量和2×目标。

接下来集中在首次异常接触簇，冻结同进程的接触模型、slack/λ/γ和完整A/b/M；分开统计ground–FEM、ABD、布料及自碰撞的残差与方向曲率，并核对linearize和dual更新前后。先确定究竟是哪一类接触在接受安全位置前把FEM带入压缩状态，再做一个有证据的局部权重/更新对照。保持世界坐标估计、材料、PCG阈值和完整CCD；不再使用统一增加内层次数作为修复。

没有启动100/300帧、七对性能认证或新的AutoDL长测；相对质量门禁仍失败，本轮没有补独立接受路径CPU CCD。旧54组4090的CCD证据不能继承给新轨迹。最终Stiff冻结程序仍未实际导出速度，完整同质量认证还有该证据缺口。

## 6. 复现与索引

在工作区 `E:/university_class/ComputerGraphics/GIPC`，解释器 `E:/Anaconda/envs/DL/python.exe`：

```text
-m unittest discover -s stiff_toi_cudagraph_20260929/tools/active -p test_contracts.py -q
stiff_toi_cudagraph_20260929/tools/active/build.py build --label <fresh> --jobs 2
stiff_toi_cudagraph_20260929/tools/active/fixtures.py --name <fresh> --mas-restrict warp
stiff_toi_cudagraph_20260929/tools/active/run.py --config stiff_toi_cudagraph_20260929/configs/active/defaults_native_window_20261005.json --name <fresh> --program-defaults
stiff_toi_cudagraph_20260929/tools/active/batch.py --plan configs/active/defaults_outer_plan_20261005.json
stiff_toi_cudagraph_20260929/tools/active/batch.py --plan configs/active/outer_exit_probe_plan_20261005.json
stiff_toi_cudagraph_20260929/tools/active/analyze_outer_probes.py runs/active/<fresh> --output reports/active/<fresh>.json
stiff_toi_cudagraph_20260929/tools/active/plot_defaults_quality.py
```

已有batch运行名/输出不可覆盖；复现须先复制plan并改名。实际配置检查详见 `DEFAULTS_CONFIG_GATES_20261005.json`、`DEFAULTS_FINAL_CONFIG_GATES_20261005.json`、`DEFAULTS_NATIVE_WINDOW_GATES_20261005.json`。16项新增运行/fixture索引已追加，原174项记录原样保留（含手工汇总与失败决定），共190项且证据路径/ID核验通过。修正公共index重扫丢失手工汇总的缺陷，不改历史默认或质量判定。
