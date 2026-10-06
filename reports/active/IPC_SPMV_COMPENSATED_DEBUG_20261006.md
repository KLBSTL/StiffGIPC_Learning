# 新增组件与补偿 TOI 调试入口

本轮新执行组件是 `spmv_fused_quadratic`；另一项是把报告已有的 `compensated` 整理成独立入口和控制模块，并修正时序边界。两者没有绑定开启。先前的 MAS 点积融合和普通 BVH refit 仍是独立开关，默认关闭。

## 1. 文件与职责

| 模块 | 活动 StiffGIPC 内的位置 | 检错重点 |
| --- | --- | --- |
| SpMV 旧路径与候选入口 | `linear_system/utils/spmv.cu/.h` | 原 kernel 保留；分开清零输出；按每个存储块的原对称扩展计算 |
| 二次型候选 kernel | `linear_system/utils/spmv_quadratic_kernel.cuh` | 不读未完成 Ap；无效尾线程贡献零并参加所有 collective；每块一个 partial writer |
| CPU 参考和界限 | `linear_system/utils/spmv_quadratic_reference.h` | 独立逐系数展开，包含 lower 块和转置贡献，误差界在运行前固定 |
| 配置与 PCG 接入 | `solver/spmv_quadratic_options.h`、`linear_system/solver/pcg_spmv_quadratic.inl` | requested/effective/fallback；host 与 Graph 使用独立 scratch，地址/大小入图签名 |
| 受保护 GPU study | `linear_system/solver/pcg_spmv_quadratic_study.inl` | NaN poison、三输入、首次/重放、私有空矩阵；A/b/M/主向量/Graph 恢复 |
| 补偿参数 | `solver/ipc_options.h` | 独立累计/残差阈值、严格整数/布尔解析、resolved 输出 |
| 预算公式 | `solver/ipc_budget.h` | 复用原 beta/u/z 递推；激活前后非法输入均永久禁用本次 solve 的附加出口 |
| 纯 CPU 控制器 | `solver/ipc_residual_controller.h` | 参考冻结、接受步 pending、下一次装配审计、动画/次数门控、movement 优先 |
| 主流程接入 | `core/GIPC.cu` | 原物理计算保留；提交最终 alpha；normal/terminal 记录不覆盖；旧 beta 和真实预算分列 |
| 控制专项测试 | `sources/stiff_active/tests/ipc_residual_controller_test.cpp`（项目根目录相对路径） | 直接编译生产控制器和选择器，800 步预算性质与控制/解析边界 |
| 公共实验入口 | `tools/active/spmv_compensated.py` | 冻结配置、串行运行、资源失败停止、真实日志独立标量重放 |

详见 [SpMV 模块说明](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/linear_system/utils/SPMV_FUSED_QUADRATIC.md) 与 [补偿 TOI 模块说明](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/sources/stiff_active/StiffGIPC/solver/IPC_COMPENSATED_TOI.md)。

## 2. 入口和开关

```json
{
  "scene": "cloth_fixed_bunny_l",
  "preset": "toi_compensated",
  "execution": "conditional_graph",
  "steps": 3,
  "dt": 0.01,
  "timeout_seconds": 120,
  "spmv_fused_quadratic": true
}
```

此配置只组合本轮两项。`toi_compensated` 展开为 **backend=ipc、compensated/.001/.03/min6**；`toi` 仍是 **backend=toi_al**。前者改变 IPC 附加停止规则，后者改变接触求解算法，报告和统计中不得混称。

改 `execution=host` 做同组件 CPU 控制对照；设 `spmv_fused_quadratic=false` 关闭融合。用 `preset=graph` 或 `host` 回到原 legacy/.01 停止，不用 movement-only 当速度分母。所有 preset 默认不启用新 SpMV、旧 MAS 融合或普通 BVH 组件。

`GIPC_SPMV_FUSED_QUADRATIC=0|1` 与 `GIPC_FIXED_SPMV_QUADRATIC_STUDY=0|1` 为严格原生开关。公共 fixed study 要求仅 fixed 诊断、末帧 direction1、无其他 fixed study。通常由 runner 映射，不手工继承环境污染。

## 3. 逐层检查

SpMV 先比 `requested.json.expanded_config` 与 `resolved_config.json.report_components`，再比每个 PCG 的 `spmv_fused_quadratic_requested/effective/fallback_reason/partial_count`。启用请求不能替代实际生效。unsupported 系统必须有原因；一次 PCG 内不能静默切换。

受保护系统文件为 `fixed/f2_n1_spmv_quadratic_study.json`。先确认 passed 和恢复布尔值；再看 actual/zero/signed 三输入的 host、Graphfirst、replay 误差与界限。`Ap_bitwise_equal` 是记录，不是原子累加算子的唯一验收依据。算子 FP64 界限不是材料形变容差。检查空矩阵输出有限为零，且 `full_PCG_zero_rhs_reexecuted=false`、`production_graph_growth_invalidations_tested=false` 等覆盖边界不被省略。

补偿 TOI 看 `ipc_residual_observations`：首次 active 应在 accepted_updates=5，audits=0；第六次最终 alpha 提交后，下一次正常装配才出现 audited=true/audits=1。reference 不随 Kappa/objective_epoch 重置。`gated_ready/compensated_ready` 是数学条件，`*_exit_allowed` 还包括 animation>0.99、min updates、valid/pending 等状态。

退出行若是 compensated，应该有 normal residual 和 `ipc_exit_assembly_ms`，不应有该行 PCG。movement 仍先退出，不能声称它保证残差门槛。legacy shadow 的额外终端审计保存在 `ipc_terminal_residual`，不能覆盖本轮 normal 观察。帧末 `final_beta` 为旧累计量，`budget_beta/budget/unit_budget/budget_reference/budget_pending/budget_audits` 才是补偿状态。

## 4. 本轮实际命令

在项目目录、PowerShell 7：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/test_contracts.py
& 'E:/Anaconda/envs/DL/python.exe' sources/stiff_active/StiffGIPC/linear_system/utils/spmv_quadratic_cpu_check.py
& 'D:/computer/cmake/bin/cmake.exe' --build builds/active --config Release --target ipc_budget_test ipc_residual_controller_test --parallel 2
& './builds/active/Release/ipc_budget_test.exe'
& './builds/active/Release/ipc_residual_controller_test.exe'
& 'E:/Anaconda/envs/DL/python.exe' tools/active/build.py build --label ipc_spmv_compensated_20261005 --jobs 2
& 'E:/Anaconda/envs/DL/python.exe' tools/active/build.py build --label ipc_spmv_compensated_20261005_tail_guard --jobs 2
& 'E:/Anaconda/envs/DL/python.exe' tools/active/spmv_compensated.py prepare
& 'E:/Anaconda/envs/DL/python.exe' tools/active/fixtures.py --name ipc_spmv_compensated_20261005_fixtures
& 'E:/Anaconda/envs/DL/python.exe' tools/active/spmv_compensated.py run --stage guards
& 'E:/Anaconda/envs/DL/python.exe' tools/active/spmv_compensated.py run --stage smoke
& 'E:/Anaconda/envs/DL/python.exe' tools/active/spmv_compensated.py run --stage activation
& 'E:/Anaconda/envs/DL/python.exe' tools/active/spmv_compensated.py analyze
& 'E:/Anaconda/envs/DL/python.exe' tools/active/verify_spmv_compensated_delivery.py
```

第一次 GPU 构建后、GPU 运行前，静态审查发现 tail collective 参与问题，第二次仅重编依赖对象；所有本轮 GPU 数据使用修正后的程序。首个 CPU build 在默认沙箱因 SDK 元数据读权限失败，获自动批准后重试成功，日志保留。

TAG 是任务在10月5日开始时冻结的身份，跨日继续后不改写；结果书日期为10月6日。所有输出名不可覆盖，不能在原证据上直接重跑命令。新的实验先另取身份并冻结有限预算。

本轮 startup 三帧数据只能作局部状态/计时诊断，不能认证材料质量或完整加速比。23帧窗口从零推进，不降低min_updates来伪造激活；资源失败后停止后续同场景诊断，保留失败。完整100帧/300帧/AutoDL和实际Graph扩容仍按后续条件分别验收。
