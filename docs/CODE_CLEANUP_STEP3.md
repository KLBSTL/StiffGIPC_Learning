# 清理 Step 3：两个已封存 AL 退出入口

2026-10-06。基于已推送的 `2befec75b217fdcc424d922719453959f3b212cc`，只修改 `solver/toi_options.h`、`solver/toi_solver.cu` 和 `core/retired_components.h`。旧源码、二进制、构建身份及本机结果不被覆盖；本次新源码必须重新构建并生成新身份，旧 executable `57fc2899…` 的结果不能作为此修改已经运行通过的证据。

## 变更

- 退休 `GIPC_TOI_INNER_EXIT=velocity_only`：从接受的模式中删除，并消除它对原 full-step 出口的抑制分支。`native` 和 `full_step_only` 保留，原速度阈值、六次 full-step 历史/重启条件、完整 CCD、材料和接触历史均保留。
- 退休 `GIPC_TOI_FULL_STEP_EXIT_PROBE`：删除 frame/outer 状态、解析及选中子问题的八次内层预算和专属异常分支。普通内层上限仍为 1024，未调高生产预算。
- 原生启动在 fixture 分发、场景加载和 GPU 初始化之前明确拒绝上述请求。旧 probe 接受 `frame:outer`，因此设置为空或 `0` 也报退休错误，不能静默按关闭处理。报错给出历史 commit。
- 为旧结果读取器保留 JSON `full_step_exit_probe=null` 和 `full_step_exit_probe_active=false`。`requested_environment` 键也保留；旧历史 JSON 的解析不受影响。没有修改公共 runner，也没有把历史配置自动改写为新算法。

历史依据见 `history/reports/DIRECTION_REVIEW_V46_20261004.md` 与 `history/reports/active/DEFAULTS_OUTER_RESULTS_20261005.md`。这是清理已停止的分支，不是新提速候选。

## 静态核查（已执行，未编译/GPU）

用 `E:/Anaconda/envs/DL/python.exe -X utf8 -` 读取 `git show HEAD:<path>` 和当前文件，执行以下 CPU 断言；所有断言通过：

1. 旧 `toi_solver.cu` 仅把已退休且关闭的 `exit_probe`、不再可选的 `velocity_only` 作常量化后，整个文件的非注释 token 序列与新文件相同。没有改写默认求解公式、状态更新或调用顺序。
2. `ToiOptions` 仅删除退休字段/解析，缩小枚举接受集合并把旧默认 null 日志字段常量化；其余 token 序列相同。
3. 对存活的 `native/full_step_only` 两种模式，以及 velocity/full-step/robust/restart/history 条件的 256 个 Boolean 组合，退出判定和 `restart_full_step_blocked` 与旧默认关闭路径一致。
4. 新原生退休 guard 之外，原 `reject_retired_components()` 内容 token 相同；除此三文件外，所有 tracked native 源文件均未变。
5. `git diff --check` 通过。差异为 `+20/-27` 行：retired header `+11/-0`，options `+3/-19`，solver `+6/-8`。

这些检查证明的是枚举关闭路径的源码等价和有限布尔逻辑等价，不替代编译、设备执行或轨迹回归。未增加新算子，不需要以此为理由重新宣称六历史完整 A/b/M 重放；正常构建及代表默认路径/错误入口回归仍由主任务执行。

| 文件 | 修改后 SHA-256 |
|---|---|
| StiffGIPC/solver/toi_options.h | `6005654e505a3392667d51a91509f5812e34759ef8571d4ed6afd1a29e2a2bdc` |
| StiffGIPC/solver/toi_solver.cu | `08b6651be67db1ed23ac5cc8c3b2bc1d932874751f194a7d0f9af181a3506e0a` |
| StiffGIPC/core/retired_components.h | `9486429d7cf621a35e3be2277ab9e4b97aa89be92ab14999c3dd8a9bfaa7324b` |

## 未改变的结论

剩余 stable MAS、conditional Graph、接触池及 TOI/AL 代码仍有各自测试范围；保留或默认开启某功能不代表已达到同质量 2× 目标。本机 Step 2 夹具结论见 `reports/local_step2/FIXTURE_ANALYSIS.md`：布料严格 rho 诊断并未统一达到 `1e-8` 真残差，mixed Cholesky 被显存门禁中止。后续主场景的资源中止也不能归因成新 kernel 的算法失败，不能计入完整性能认证。

## 后续实际构建与入口回归

上述静态核查之后，主任务已完成新的 Windows Release 构建：

- 路径 `build/local_step3_20261006/Release/gipc.exe`；SHA-256
  `b2ad19884fc01b81827d560f9f1984346b5488851995c2fd495f7427eac0f812`。
- 独立封存 37 个源文件、37 个唯一原生对象及实际链接输入；包含构建前输入
  哈希、实际 NVCC/CL 命令、编译器和应用 DLL。摘要见
  `../reports/local_step3/VERIFICATION.json`。没有用旧程序的通过状态替代新程序。
- `ctest --test-dir build/local_step3_20261006 -C Release -R 'ipc_budget|ipc_compensated' --output-on-failure --timeout 120`：2/2 通过。
- 组件导数/更新 46 项、实际 PCG 标量守卫 15 项、接触池合成查询 53 项检查通过；
  三个实际 GPU 进程均 completed、身份未变化。它们不替代接受路径 CCD 或完整场景。
- 20 个有界负面启动用例全部返回非零且明确提示退休，组件 fixture 输出均未生成。
  包括 `velocity_only`、probe 的 `1:0`/`0`/空值、旧执行开关/fixture 和非法旧布尔值。
  guard 的源码顺序和未生成 fixture 支持其在分发前拒绝；没有以此声称 GPU 时间线覆盖。
- 本机 runner、资源监控及 CPU 矩阵读取共 26 项 CPU 合约检查通过。

完整 Step3 轨迹暂未运行；共享桌面空闲显存约 2.9–3.0 GiB，而 Step2 两次完整窗口
之后已触发原显存保护。保持失败收据、原质量界限和预算。此处不继承 Step2 加速比，
不推广接触池，也不将本次删除分支表述成已测到新的性能提升。
