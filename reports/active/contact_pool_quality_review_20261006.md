# Swept barrier 候选池：本轮最小质量审阅

日期：2026-10-06。只读审阅 `NEXT_CONTACT_POOL_PLAN_20261006.md`、冻结协议及现有分析工具；未构建、运行 GPU、修改 native 或旧协议。本文件是待执行检查单，不是候选已通过的证据。

冻结协议：`ipc_revision_20261005_quality_protocol.json`，SHA256 `1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f`。每轮分析开始和结束都核对此 SHA；保留历史 fixed_bunny 基线失败，不因最新单次通过重建门槛。

## 1. 最小运行与覆盖

| 层次 | 从零运行与必要覆盖 | 判定边界 |
|---|---|---|
| 小型 GPU fixture | 零/单/尾块、重复 primitive、PT/EE 全部 typed 类、近平行 mollification、fixed/free、原始方向/ID、容量不足后完整重试、epoch/映射/属性变化、无效 alpha、ABD 机器包含反例 | 在首个场景前通过；失败保存证据并停止候选，不在运行中延长预算 |
| 同状态场景 guard | hang 51f、fixed bunny 59f、mixed **35f**，每次 validate 开启，120s 上限及既有显存预算不变 | 都应检查非空比较及真实 pool 使用；全回退可以说明 fallback 可用，不能说明 pool 主路径正确或有收益 |
| mixed 最小轨迹对照 | 若无同一二进制的 35f off 参考，补一次 off 35f，与上述 on guard 比较 | 一对仅作 ABD/FEM/布料兼容诊断，不能建立重复范围；3f 启动成功不替代 35f |
| 短窗性能筛选 | 3 轮交错 off/on：hang 51f、fixed bunny 59f；同一观察开关，关闭 validate/profile/cost 重诊断 | 全前缀和冻结接触窗口同时列出；不从守卫运行计算加速比 |
| 有条件完整验证 | 仅在两场景质量证据与原计划净收益门槛均成立后，3 轮 100f Stiff/off/on | 100f 仍为120s；本轮不因失败自行扩成参数网格或增加长运行 |

冻结统计窗口：hang 全前缀1–51及接触41–50；fixed 全前缀1–59及接触22–59、密集45–59；mixed 全前缀1–35及1–3、24–26、33–35分窗。完整100f另列hang41–50和fixed22–100/45–59。不得按某臂更有利的接触起止点移动配对窗口。

每个窗口另列实际 `native_narrow_self_pairs`、ground pairs、Newton `active_pairs` 的非零帧与峰值。`geometric_contact=false` 不等于 IPC 没有接触力：已核对 hang 的 barrier 半径为0.001m、分类阈值0.000141421m；fixed分别为0.002538208m和0.000253821m。mixed35必须在实际统计中显示活跃接触及成功 pool 查询，尤其包括后两个窗口；缺覆盖明确标待验。

## 2. 同状态保护：比较什么，不能省略什么

- 旧 query 与 pool 窄相读取同一实际位置、body/boundary/mapping、dHat、origin/direction/alpha；epoch不能仅依赖未改变的设备地址。验证必须早于诊断引起的任何生产状态变化。
- 比较完整、带重数的 typed DCD 与必要 CCD 多重集；保留原始 vertex/face/edge ID、PT/EE方向、负编码及近平行 EE 的 `-obj_idx-2` 身份。不得只把四顶点排序去重后宣布等价。类型计数与总数相符，MatIndex在每种类型内是正确置换并对应输出类别。
- 容量、真实 written count、epoch与poison分别验证；零容量增长后必须从完整源集合重试。旧/新比较结果均完整，不能把共同截断视为一致。
- 包含守卫必须覆盖VF的query vertex与target face、EE的两条edge，以实际trial位置检查相应swept bounds；ABD `J(q-alpha*dq)`的机器舍入不能由实数推导或经验padding替代。超区间、NaN/Inf、generation/映射/属性/半径失效均记录明确回退原因。
- 保留FullCCD及地面检测；普通BVH维护、接受步后`isIntersected`等安全调用继续执行。报告这些入口的实际调用覆盖，不能只看到pool counter就推断安全检查未受影响。
- 诊断私有输出不能污染生产pair/count/MatIndex、当前坐标、方向、步长、Kappa或缓存身份。恢复后继续同一生产路径；抽查分项能量、方向、接受alpha及其有限性。集合相等不足以证明不同累加顺序产生的目标值和接受步也等价。
- 每个实际 pool-use 查询必须有对应验证结果；拒绝 `all([])` 式零记录通过。记录queries、eligible、pool-used、validated、各回退原因和未覆盖路径；无需强迫正常场景自然触发每个失效分支，失效分支由fixture覆盖。

## 3. 原模型的质量检查

所有正式轨迹从零开始。先核对初态、拓扑、质量、boundary/body IDs、scene材料及effective_run停止参数；active还核对requested/resolved、程序身份和逐帧开关。PCG rho停止值不能当作真残差阈值。

每个全前缀与冻结窗口输出：位置有限性；PCG触顶/breakdown；方向、PCG总量及每方向分布；退出原因；逐帧最大/p99布边伸长、固定漂移；FEM最小J、非正单元数量及负体积；ABD最小J。混合场景保持相对基线要求，不新增“任何FEM非正都失败”的物理规则。纯布料空tet是合法输入，fixed bunny的19,193个固定ABD顶点必须纳入漂移检查，不能只看boundary标记。

| 既有100f冻结界限 | hang | fixed bunny |
|---|---:|---:|
| 最大伸长 | 1.1288772279548063 | 1.0361727662908526 |
| 最大逐帧p99伸长 | 1.032410340416372 | 1.0135636480371302 |
| 固定漂移/m | 1e-10 | 1.0000005721958499e-10 |

短窗越界立即保留失败；短窗不越界不是100f已通过。hang全局材料峰值在接触前f18，因此41–50逐帧指标必须另列；fixed材料峰通常在45–50，不能用前22f自由落体结果替代。

布料/FEM/ABD的位置与**实际导出的速度**分别比较，给off自身重复、on自身重复、同轮off/on差异；不能只报全体顶点总体RMS。候选自身波动变大不能用来放宽off的原范围，超出时保留待确认。Frozen Stiff未实现真实速度导出：明确unavailable，不从位置差分伪造；请求`trace_velocity=true`本身不证明导出存在。current每份速度还需独立检查完整字节数和finite，不能因Stiff缺速度跳过。

## 4. 独立安全与验收口径

私有旧query多重集验证不是独立accepted-path CCD。若候选进入完整质量检查，每个布料场景至少一次连续100f开启substeps；必要时直接复用已完整导出的guard路径做短窗审计。用既有独立CPU Tight-Inclusion validator逐段检查所有accepted segments和跨帧bridge（包括静止bridge），验证路径数、端点身份、fixed/ABD/finite、原始退出码与实际flags。记录validator/source/input/output SHA及stdout，数值失败与执行失败分开保存。缺路径或预算中断就是覆盖不足，不能默认为零碰撞。

计时与重诊断分开。正式配对报告整段solver time、冻结接触窗time、普通query节省以及pool准备/包含守卫/分类压缩/回退的成本；包含线性阶段计时的`pcg`字段不是独立PCG kernel时间。全前缀收益与维护净收益均满足原计划后才前进。共享桌面不认证2×；没有独立CCD或Stiff真实速度对照时，不认证同质量。原计划的最多一次候选修订与失败停损保持。

## 5. 可复用工具及现存边界

- `ipc_benchmark.metrics()`（70–116行）：可复用材料、固定ABD、PCG与阶段统计；有限性只覆盖位置，velocity目前只计文件数，调用者必须独立验证速度。混合fixed/body映射仍须与场景声明核对。
- `component_tuning.state_comparison()`（27–59行）：已有分体位置/实际速度比较及缺失标记；调用者先核对初态/质量/scene并强制期望帧数，不能仅依赖其拓扑相等检查。
- `contact_optimization.window()`（62–76行）：可复用`trace/frames.csv`的真实`solver_ms`与冻结窗口工作量；调用者先断言完整帧数。不要从stage总计反推接触窗时间。
- **不可直接照搬旧门禁：**`contact_optimization.py:17,35`仍是mixed1–3/3f；94行`all(a['passed'] for a in audits)`在空列表上为真；125行性能门禁未合并102行材料判定。新pool runner必须显式修正这些契约，不改旧失败报告。
- `audit_autodl_factor.inspect_trace()`可复用端点/bridge清单；其高层CLI固定54次AutoDL布局，不能直接当本轮批审计入口。`audit_factor_windows.py`同样固定旧factor场景配置，不直接调用其prepare。现有本机validator为`builds/validator/Release/diagnose_first_path.exe`，参数模式为`<trace> <new_report> substeps --stable-nh1`；先确认exe存在及实际输入契约，不启动本审阅中的GPU或CPU CCD。
- 已生成的`contact_baseline_quality_20261006_agent.json`可作本次原始Stiff/current100f的质量背景：四次材料指标通过，但每臂只有一次、固定兔子布料位置RMS差达27.448mm，未认证轨迹等价，且没有Stiff真实速度对照。它不能为新pool候选背书。

## 审阅结论

计划的单变量、default-off和机器包含回退边界合理。最小有效推进需要**非空同状态typed比较、mixed35活跃接触、恢复生产状态、原材料门槛与实际速度覆盖、独立accepted-path CCD分离**同时落地。当前没有新pool运行结果，所有候选正确性、质量、净收益及2×结论均待验证。
