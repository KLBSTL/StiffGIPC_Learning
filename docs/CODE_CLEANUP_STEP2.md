# 第二批原生清理实绩与待执行验证

2026-10-06。实施前身份：`edaf2388c3e8b201ed5b56cd843341aebaa45b54`。
清理前完整归档：`archive/pre-cleanup-20261006`。

**Source ready；尚未编译或运行 GPU。** 本机的这批修改不代表远端正在执行的
质量诊断程序；本机没有上传或改变远端二进制。旧 guard 的失败/待确认结果、
`allow_next_round=false` 及已冻结质量协议不受此清理影响。

## 已实施

移除了 MAS final-dot、SRBK SpMV+pAp、legacy ordered restriction 的 12 个专属
实现/研究文件，以及公共类中的方法、候选 partial/CUB 工作区与调度分支。
conditional Graph 删除两候选共 10 个签名字段，MAS Graph 签名单独删除 ordered
mode；没有删除普通向量、reduction scalar、因子或映射的地址签名。

host 初始 rho、后续 rho、SpMV/曲率点积，Graph 循环，MAS 的 legacy/wide/Cholesky
三条作用路径都直接使用清理前候选关闭时的表达式和顺序。previous-rho 判断、
zero-RHS、迭代上限和 breakdown 规则保持。普通 SpMV 仍按每个已存块做原来的
对称扩展；global/ABD/local 预条件器完整写入顺序保持。

稳定 MAS 仍拥有 `d_restriction_starts/nodes`，其分配、释放、快照和 Graph 地址
签名保留。`prepare_restriction_map` 只删除 ordered 研究专属的测时参数、同步和
成员字段；稳定路径原先就使用默认不测时分支。CSR 构造、排序、读回、上传及
公共 CostScope 未改。`mas_restrict=serial|warp`、`factor_inverse`、
`triangular` 和通用 MAS stage/fixed-system 诊断仍在。

专属历史说明保留并加退休标识。删除清单和全部修改文件 SHA256 位于
`docs/CODE_CLEANUP_STEP2_STATIC_CHECKS.json`，实施设计位于
`docs/CODE_CLEANUP_STEP2_PLAN.md`。本次未改 `tools/bench`、`tests`、CMake、
README、`baseline/`、原实验目录或 remote。

## 显式拒绝旧请求

`main` 既有首行 `reject_retired_components()` 现在额外检查：

- `GIPC_MAS_FUSED_DOT` / `GIPC_FIXED_MAS_DOT_STUDY`
- `GIPC_SPMV_FUSED_QUADRATIC` / `GIPC_FIXED_SPMV_QUADRATIC_STUDY`
- `GIPC_FIXED_LEGACY_RESTRICT_STUDY`

上述只接受未设置或严格 `0`；`1` 报已退休，其他值报非法布尔。
`GIPC_LEGACY_RESTRICT` 只接受未设置或 `atomic`；`ordered` 报已退休，
其他值（包括空串）报非法枚举。检查在 fixture dispatch、场景与 GPU 初始化前执行。
原六系统、PCG guard、组件与 contact-pool fixture 入口保留。

resolved 仍提供旧 false/atomic 字段并增加 availability=false。每个生产 PCG
提供两候选 requested/effective/available=false、partial_count=0 和
fallback_reason=`retired_component`，匹配当前 validator 的关闭状态要求。
不生成旧专属研究文件或假计时数据。公共 `expand` 仍可读历史 true/ordered
配置；执行边界能力拒绝由主任务的 runner 修改另行完成。

## 已完成的 CPU 静态核对

使用 `E:/Anaconda/envs/DL/python.exe -X utf8 -` 运行只读核对，以
`git show HEAD:<path>` 取清理前源码，按明确关闭分支化简后比较去注释/空白文本。
37 项检查全部通过，详细逐项结果见 JSON：

- 整个 host PCG 和整个 conditional Graph（除专属签名）与原关闭路径一致。
- 普通 reduction/update/guard/operator-audit 函数不变。
- MAS 三条作用、原子限制、GROUP/非 GROUP collect 与原关闭路径一致。
- 稳定 Cholesky 准备、CSR 非测时路径及完整 owned-buffer 快照列表保持。
- 普通 SpMV 全部 kernels/方法、全局 apply、全局 Graph 签名保持。
- 通用 fixed study 只删除三条退休 dispatch。
- 13 个公共数值/fixture/碰撞文件与实施前逐字节一致，包括 core GIPC、
  contact pool、mlbvh、ACCD、稳定 restriction/Cholesky/factor、六系统 fixture。
- 12 个删除路径无残留 native include，实际编译单元仍为 37。

`git diff --check -- StiffGIPC docs/CODE_CLEANUP_STEP2.md` 作为交付前格式检查。
静态相同不替代编译、GPU 运行、Graph 生命周期或数值正确性证明。

## 待主任务串行执行的具体验证

以下是需要交给公共有限预算 runner 的子进程环境/命令，不是绕过 GPU 锁、
负载/显存/磁盘门槛的独立运行许可。每次从干净 GIPC 环境开始，即复制继承环境时
删除所有 `GIPC_*` 再仅设置本例参数。每子进程上限 120 秒，超时仅停止该 runner
拥有的进程组；不延长预算，不复用已有输出目录。先冻结新 build/source/exe 身份。

### 六份历史 M 作用 fixture

保留下面六份输入的所有相关 A/b/M、meta 和 probe 文件，迁移时记录逐文件 SHA256。
路径相对原实验目录，具体远端根由主任务的资产清单确定：

| case | prefix |
|---|---|
| smoke | `downloads/autodl_perf_v37_20261003/runs/autodl/autodl_perf_v37_mas_smoke/mas_audit_initial` |
| default | `downloads/autodl_perf_v39_20261003/runs/autodl/autodl_perf_v39_wide_bunny/mas_audit_failure` |
| strict | `downloads/autodl_perf_v39_20261003/runs/autodl/autodl_perf_v39_wide_bunny_strict/mas_audit_failure` |
| v41_graph | `downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_double_bunny/mas_audit_failure` |
| v41_host | `downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_double_bunny_host/mas_audit_failure` |
| v41_wide | `downloads/autodl_perf_v41_20261003/runs/autodl/autodl_perf_v41_wide_guard/mas_audit_failure` |

对每个 prefix 调用新程序 `build/active/gipc`，使用：

```text
GIPC_MAS_CHOLESKY_FIXTURE=<完整prefix>
GIPC_MAS_FIXTURE_OUTPUT=<全新输出目录>/<case>
GIPC_MAS_RESTRICT_MODE=serial
GIPC_MAS_FACTOR_ACTION=factor_inverse
```

先六例默认模式，确认每例 exit=0、fixture.json passed=true，zero/pivot/Graph
增长项均真实存在并通过。warp 和 triangular 的保留模式用既有代表系统回归，
不要把某一模式通过外推到未运行模式；若共享映射存在疑点则扩为该模式六例。
可另外调用保留的 `GIPC_MAS_REPLAY_FIXTURE=<prefix>`（同 OUTPUT）核查
legacy/wide/inverse64 原有相位；其 JSON 与 Cholesky fixture 模式不同。

这些入口重放 **M 作用及保存 probe**，不是历史完整 PCGSolver 的 A/b/M 求解。
未实现或声称六份历史全局 A/b/M 的生产 PCG 载入与重放。

### 当前进程的完整固定 A/b/M host/Graph 对照

每个场景从零运行，冻结最后帧方向 1。建议两个小输入为
`cloth_fixed_bunny_l`、`bunny_cloth_bunny_l`，均 2 帧；这两个 scene 键已经与
`Assets/benchmark_scenes/` 实际文件名核对。
每个输入分别以 legacy 和 Cholesky 默认配置运行，共最多四个 fixture 进程。
主环境沿当前 runner 的公共物理设置，再设置：

```text
GIPC_SCENE=cloth_fixed_bunny_l        # mixed 对照改为 bunny_cloth_bunny_l
GIPC_CONTACT_BACKEND=ipc
GIPC_STEPS=2
GIPC_DT=0.01
GIPC_NEWTON_TOL=0.01
GIPC_PCG_TOL=0.0001
GIPC_IPC_TERMINATION=legacy
GIPC_IPC_CUMULATIVE_TOL=0.01
GIPC_IPC_MIN_UPDATES=6
GIPC_PCG_EXECUTION=conditional_graph
GIPC_PCG_PRECONDITIONER=mas
GIPC_MAS_CHOLESKY=0                   # Cholesky 对照改为 1
GIPC_MAS_WIDE_APPLY=0
GIPC_MAS_INVERSE64=0
GIPC_MAS_RESTRICT_MODE=serial
GIPC_MAS_FACTOR_ACTION=factor_inverse # legacy 可写 triangular，与当前 runner 对齐
GIPC_LEGACY_RESTRICT=atomic
GIPC_FIXED_STUDY_DIR=<全新输出目录>/fixed
GIPC_FIXED_STUDY_FRAMES=2
GIPC_FIXED_STUDY_DIRECTIONS=1
GIPC_FIXED_STUDY_COMPACT=1
GIPC_MAS_SNAPSHOT=1
GIPC_RESOLVED_CONFIG=<全新输出目录>/resolved_config.json
GIPC_DUMP_STATE=<全新输出目录>/final.bin
GIPC_OUTPUT_PATH=<全新输出目录>/output
```

所有 `GIPC_FIXED_*_STUDY` 专项开关和两退休 fusion 开关均不设置或为 `0`。
创建 output 与 fixed 目录后，公共 runner 启动 `build/active/gipc`。通用
`fixed_system_study` 会自行做 host/Graph 对照，不能仅检查进程成功：检查
`f2_n1_study.json` 的系统身份、生产 x 恢复、误差/真残差和实际运行模式/次数。
保留 snapshot 的全局/局部预条件器数据，mixed 确认含 ABD 与 FEM；不能用总 RMS
掩盖缺失某一部分。默认 rho 与欧氏真残差保持区分。

注意原有 compact study 自带默认 `1e-4` 与严格 `1e-16` 两档诊断 rho，每档
host/Graph 两次，并有原有算子观察循环；本次没有修改这些旧诊断。这不改变
生产 rho，但仍可能耗时或触发严格档失败，必须记录每条 runs 的 error/limit，
遵守同一 120 秒预算，不能将仅 JSON 已写入视为全档通过。

还需用原入口运行组件/PCG guard（`GIPC_VALIDATE_COMPONENTS`、
`GIPC_PCG_GUARD_FIXTURE`）及保留 contact-pool fixture。通用 stage-only probe
与完整 fixed study 的覆盖不同，不能互相替代。全部结果属于清理回归，不宣称
新的加速比或质量门槛通过。
