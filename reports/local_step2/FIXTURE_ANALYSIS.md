# 本机 Step 2 夹具证据复核（2026-10-06）

本报告只读分析 `runs/local_step2_fixtures_20261006` 的 13 个 `result.json`、9 个原生 `fixture.json`、3 个完成的 `f2_n1_study.json`，并核对 requested/build manifest、输入身份和对应夹具源码。没有重新运行 GPU、编译、修改原始结果或改变验收门槛。

**结论：12 项完成；混合 Cholesky 固定系统测试被预先设定的显存预算中止，仍待验证。完成的固定系统测试证明受保护诊断成功运行及主状态恢复，不构成统一的 `1e-8` 真残差通过。** 两个布料固定系统在严格 rho 阈值下仍只有约 `2.7e-7` 至 `4.9e-7` 真残差；混合 legacy 的严格诊断约 `1.05e-9` 至 `1.09e-9`。本批没有性能或完整轨迹质量认证。

## 1. 身份和运行范围

- 全部 13 项 requested 和 build manifest 指向同一程序：`build/local_step2_v3_20261006/Release/gipc.exe`，10,840,064 字节，SHA-256 `57fc289968126db380e57ade8d1ce9b63ab73d97009365c6aad07a2f018769b2`。
- 全部 requested 的 source digest 为 `b87263b06344883067813a13079bc2730b26445d67d773ed590b809e1d3e234c`，`identity_unchanged=true`。每项 manifest 保存源码、相邻 DLL 和输入哈希；该 manifest 自身不冒充完整编译证明。独立构建证据由 `build/local_step2_v3_20261006/local_diagnostic_seal.json` 另行保存。
- GPU 为 RTX 3070 Laptop GPU，8 GiB，WDDM，驱动 581.57。全部运行明确标记 `load_controlled=false`、`diagnostic_under_desktop_load=true`、`performance_certified=false`、`physical_quality_certified=false`。
- 每项 120 秒上限；没有为失败项延长预算。WDDM 的进程显存为 N/A，不能据此排除其他计算或将全卡显存变化精确归属到本程序。

| 项目 | 结果 | 能支持的结论 |
|---|---|---|
| component | completed，46 个原生检查均 passed | AL 导数/更新、历史和归一化体积界限的已有夹具合同 |
| pcg_guard | completed，15/15 | host 分类与实际 Graph 标量守卫 kernel 的合同 |
| contact_pool | completed，53/53 | 合成同状态的 typed 多重集、回退、容量重试和生命周期合同 |
| 六个 `mas_*` | 6/6 completed | 六份历史输入的 MAS 局部作用与保存探针覆盖；不是六份完整 A/b/M 的生产 PCG 重放 |
| fixed_cloth_legacy | completed，8/8 子运行无 error/limit | 当前布料冻结系统的 host/Graph、两档 rho 和主状态恢复 |
| fixed_cloth_cholesky | completed，8/8 子运行无 error/limit | 同上，稳定 MAS serial + factor_inverse；不等于 PCG 全程逐位确定 |
| fixed_mixed_legacy | completed，8/8 子运行无 error/limit | 当前混合 ABD/FEM 冻结系统的相同诊断 |
| fixed_mixed_cholesky | memory_budget，无 study JSON | 资源门禁中止；数值结果缺失，不能判定通过或算法失败 |

## 2. 固定系统：真实数值结果

两个 cloth 项的场景都是 `cloth_fixed_bunny_l`，17340 自由度、39536 个存储块，含 1 个 ABD 预条件块和偏移 4 的 MAS。mixed 为 `bunny_cloth_bunny_l`，60858 自由度、136033 个存储块，也含 ABD 和 MAS。均从零推进到第 2 帧第 1 个方向，在当前进程冻结 A/b/M，诊断解从零初值开始。正式配置仍为 IPC、dt=.01、legacy 停止、累计容差 .01、至少 6 次更新；`1e-16` 只属于受保护诊断臂。

**原生 study JSON 没有根级 `passed`。** runner 的 `fixture.native_passed` 只检查 8 个规定模式/容差/重复齐全、无异常/上限、残差有限及系统/主状态恢复；并明确输出 `numerical_acceptance_certified=false`（`tools/local/fixture_windows.py:225–235`）。不能将它解释为任何精度门槛通过。

`true_relative_residual` 是重新执行现有 GPU SpMV 后，在 host 汇总 `||b-Ax||/||b||` 的直接残差，区别于 PCG 递推 rho；它不是独立 CPU 高精度矩阵求解或 CPU LU 参考（`StiffGIPC/linear_system/solver/pcg_fixed_study.inl:73–81`）。rho 阈值不能直接等同真残差阈值。以下每条记录均为 `error=""`、`limit=false`、残差有限。

`Δhost` 表示同一 rho 下相对第一个 host 解的 L2 差；`Δrepeat` 表示相对同模式首次解的 L2 差。重复 1 的 `Δrepeat=0` 是定义，并非额外重复一致性证据。

| 系统/预条件器 | rho 阈值 | 模式 | 重复 | PCG 次数 | 真相对残差 | Δhost | Δrepeat |
|---|---:|---|---:|---:|---:|---:|---:|
| cloth legacy | 1e-4 | host | 1 | 6 | 4.198429558140e-2 | 0 | 0 |
| cloth legacy | 1e-4 | host | 2 | 6 | 4.198429761925e-2 | 1.230652e-8 | 1.230652e-8 |
| cloth legacy | 1e-4 | graph | 1 | 6 | 4.198429712377e-2 | 1.397552e-8 | 0 |
| cloth legacy | 1e-4 | graph | 2 | 6 | 4.198429740782e-2 | 1.275033e-8 | 1.071505e-8 |
| cloth legacy | 1e-16 | host | 1 | 72 | 4.162395051047e-7 | 0 | 0 |
| cloth legacy | 1e-16 | host | 2 | 72 | 2.994175446126e-7 | 7.025343e-10 | 7.025343e-10 |
| cloth legacy | 1e-16 | graph | 1 | 72 | 3.481388427819e-7 | 2.802881e-10 | 0 |
| cloth legacy | 1e-16 | graph | 2 | 72 | 3.100069461666e-7 | 7.245553e-10 | 5.343483e-10 |
| cloth Cholesky | 1e-4 | host | 1 | 6 | 4.198429445957e-2 | 0 | 0 |
| cloth Cholesky | 1e-4 | host | 2 | 6 | 4.198429445957e-2 | 4.067440e-16 | 4.067440e-16 |
| cloth Cholesky | 1e-4 | graph | 1 | 6 | 4.198429445957e-2 | 4.003905e-16 | 0 |
| cloth Cholesky | 1e-4 | graph | 2 | 6 | 4.198429445957e-2 | 4.006868e-16 | 4.085348e-16 |
| cloth Cholesky | 1e-16 | host | 1 | 70 | 2.738346179226e-7 | 0 | 0 |
| cloth Cholesky | 1e-16 | host | 2 | 69 | 4.948027731104e-7 | 3.353603e-9 | 3.353603e-9 |
| cloth Cholesky | 1e-16 | graph | 1 | 70 | 2.733603144206e-7 | 6.998251e-12 | 0 |
| cloth Cholesky | 1e-16 | graph | 2 | 71 | 2.786636506803e-7 | 5.763250e-9 | 5.764558e-9 |
| mixed legacy | 1e-4 | host | 1 | 6 | 1.439583563636e-3 | 0 | 0 |
| mixed legacy | 1e-4 | host | 2 | 6 | 1.439587683678e-3 | 4.207944e-8 | 4.207944e-8 |
| mixed legacy | 1e-4 | graph | 1 | 6 | 1.439583119929e-3 | 3.672205e-8 | 0 |
| mixed legacy | 1e-4 | graph | 2 | 6 | 1.439582265177e-3 | 4.446535e-8 | 3.418592e-8 |
| mixed legacy | 1e-16 | host | 1 | 130 | 1.086152492539e-9 | 0 | 0 |
| mixed legacy | 1e-16 | host | 2 | 130 | 1.092580488646e-9 | 5.154667e-10 | 5.154667e-10 |
| mixed legacy | 1e-16 | graph | 1 | 130 | 1.080405242465e-9 | 1.249848e-10 | 0 |
| mixed legacy | 1e-16 | graph | 2 | 130 | 1.046478470311e-9 | 1.471313e-9 | 1.406377e-9 |

布料 Cholesky 的单独 M 作用三次重复逐位相同，但独立 SpMV 重复仍有约 `1.15e-16` 相对差，完整严格 PCG 为 69–71 次，不能称完整 PCG 逐位确定。legacy 布料 M 重复最大相对差为 `1.06935e-7`，mixed 为 `2.73930e-7`。这些是该冻结系统的观测，不证明所有 host/Graph 差异只由一个算子导致。

三个完成 study 均 `system_unchanged=true`、`primary_restored_bitwise=true`。`full_snapshot_unchanged_including_scratch=false` 也必须保留：逐项核对发现，legacy 只变化 MAS `d_multiLevelR/d_multiLevelZ` 哈希，Cholesky 只变化 `d_multiLevelR64/d_multiLevelZ64` 哈希；这些是明示排除的工作 scratch。A、b、ABD 逆块、MAS 算子/映射等身份未变化。不能宣称“全部 scratch 也逐位恢复”。

cloth legacy 与 Cholesky 来自各自从零运行，冻结 A 的 values 哈希并不相同；因此跨这两项的差异不应解释为严格同 A/b 的预条件器因果比较。每项内部 host/Graph 共享本项冻结系统，才是这里直接支持的比较。

## 3. 六份历史 M：实际覆盖及缺口

六个 requested 指向六个不同历史 prefix，manifest 分别记录输入文件哈希；meta 和 `d_inverseMatMas` 哈希也各不相同。每项实际输出 `p0_v0..9.bin`、`p1_v0..9.bin`、`p2_v0..9.bin` 共 30 个结果，因此不是六次读取同一个空报告。

| 本批目录 | 历史来源目录（prefix） | MAS 节点 | offset（3-vector 单位） | clusters | meta SHA-256 前 16 位 |
|---|---|---:|---:|---:|---|
| mas_smoke | v37 / autodl_perf_v37_mas_smoke / mas_audit_initial | 1939 | 0 | 2256 | 3fbb833f95c59c65 |
| mas_default | v39 / autodl_perf_v39_wide_bunny / mas_audit_failure | 20282 | 4 | 23152 | 17f00b34168a67cad |
| mas_strict | v39 / autodl_perf_v39_wide_bunny_strict / mas_audit_failure | 20282 | 4 | 23168 | e7d703ac264fcbf8 |
| mas_v41_graph | v41 / autodl_perf_v41_double_bunny / mas_audit_failure | 20282 | 4 | 23136 | 296079bb3489c0cb |
| mas_v41_host | v41 / autodl_perf_v41_double_bunny_host / mas_audit_failure | 20282 | 4 | 23136 | 5cbf945cc05cc8f9 |
| mas_v41_wide | v41 / autodl_perf_v41_wide_guard / mas_audit_failure | 20282 | 4 | 23152 | b98a67be1caf77f9 |

对应源码 `StiffGIPC/solver/mas_cholesky_fixture.inl:6–57,62–163` 的覆盖为：

1. 读取保存的 MAS 局部矩阵及映射，从每个历史探针中截取 MAS 区间；依次 legacy → Cholesky → legacy，10 个探针/阶段，每个组件图重放 3 次。`graph_bitwise=true` 的主循环判断限于 Cholesky 阶段，不代表 legacy 原子路径也逐位相同。
2. 本次配置都是 stable restriction `serial`、`factor_action=factor_inverse`，块宽 48；没有测试 warp restriction 或 triangular 模式组合。
3. Cholesky/R64/Z64/CSR 扩容后的签名和 M 输出一致；factor-inverse 独立扩容、重新捕获和三次 Graph 重放一致。off 签名恢复；六项这些标志均 true。
4. 零输入 M 在 direct 和 Graph 下输出零；私有异常 pivot 被检测（status=1），非有限 inverse 被检测（status=2257），fault probe 主缓冲摘要未变化。这里的 `zero_rhs_host/graph` 测试的是 M(0)，不是完整零 RHS PCG。

六份 `mas/fixture.json` 的 SHA 相同：`65e0ef87092a101ea05984d0978f3b23b365842ae4334bd6200f75b621fa3b92`。原因是该 JSON 只记录相同结构的布尔结果/状态，不含每份输入身份；不同输入的身份由相邻 manifest 和 requested 提供。

**边界：** fixture 没有求解保存的全局 A/b，没有应用历史 ABD 区间，也没有独立 CPU 准确求解或报告全局真残差。六份输入中虽然含 A、b、ABD 等文件，manifest 将其封存并不等于实际执行。不能将六份 M 作用测试改写为“六个历史失败全系统 PCG 均修复”。

## 4. 其余原生夹具

`component` 的六组共 46 个检查全部 passed。其中距离/完整 Hessian/slack 更新对照最大误差 0，能量梯度有限差分最大 `2.85829e-10`；reduced-slack 的 gradient 最大 `9.99999e-7`、curvature 最大 `1.40994e-5`，按其既有夹具合同通过，不能统一称所有误差小于 `1e-8`。warm history 更新误差最大 `2.22045e-16`；world penalty GPU/CPU 对照误差 0。

`pcg_guard` 的 15 项包括零 rho 与非零残差区分、负/非有限初始或更新 rho、零/负/非有限曲率、alpha/beta overflow、前一 rho 已收敛等。host 分类与实际 Graph 标量守卫全部匹配。它是标量守卫测试，不能替代完整零 RHS、所有异常矩阵的生产 PCG 回归。

`contact_pool` 的 53 项检查全部 pass。计数为 attempts=42、reuse=16、old=26；同状态私有验证 16 次全部通过，其中 10 次非空，共比较 875 个 pair；pool 记录 VF=394、EE=10，包含 signed tuples/重复保留与 type-local MatIndex 双射。诊断扩容重试 20 次，生产容量溢出测试 2 次；属性/方向/映射/区间改变均触发对应回退。fixture 包含实际 ABD 舍入反例的 out-of-bounds 回退、空输入、尾块、零容量及完整重试。

pool 的 `requested=false/validate=false` 是夹具作用域恢复后的全局配置摘要，不能据此抹掉上述显式 fixture 内执行计数。它仍明确 `production_scene_validation_required=true`：本合成夹具不证明所有 PT/EE Voronoi 子类全覆盖，也不替代真实接触帧的同状态 energy/Armijo、完整 CCD、质量或性能验收。

## 5. mixed Cholesky 的资源中止

原资源公式为 `min(.75*free, free-1536)` MiB，且运行时 free 不低于 768 MiB。本项启动前 free=3850 MiB，因而预算固定为 2314 MiB。四次采样 free 为 3850、3560、3476、1492 MiB；最后一次在约 1.657 秒，相对启动前减少 2358 MiB，超预算 44 MiB，触发 `memory_budget`。不是 120 秒超时，也不是 free<768 分支。

`run.log` 显示第 1 帧完成，随后进入下一次 solve；没有完成的 `fixed/f2_n1_study.json`，没有每档容差残差。runner 中止返回 exit_code=1 不能解释为 PCG pivot、非有限、收敛上限或 CUDA OOM。由于全卡共享 WDDM 遥测，不能断言额外 2358 MiB 全由当前程序占用；同样不能据此认定算法已通过。该项准确状态是**在原资源约束下未完成，数值待验证**。

## 6. 可用于本轮决策的证据

- 支持：清理后的程序能执行已有局部合同；六份历史 M 的 stable 作用/组件 Graph/扩容/fault probes 通过；三份当前冻结 A/b/M 诊断完成，host/Graph 输出差异和实际残差可查，主状态恢复。
- 未支持：统一 `1e-8` 残差门槛、独立 CPU 准确参考通过、六份历史完整全系统重放、混合 Cholesky 固定系统通过、全轨迹质量不退化、加速比或 2× 认证。
- 本报告不放宽旧门槛，不把资源中止改写为成功，也不因布料严格残差未到 `1e-8` 就在此改变生产 PCG 停止规则。需要准确参考验收时，应在独立预声明诊断中核查矩阵/残差与参考解；本批结果不能代替该步骤。

原始证据入口：`runs/local_step2_fixtures_20261006/<case>/result.json`、`requested.json`、`build_manifest.json`；固定系统为 `<case>/fixed/f2_n1_study.json`。本报告未使用任何墙钟或 operator_mean_us 数值宣称加速。
