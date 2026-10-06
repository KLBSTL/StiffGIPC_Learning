# 当前组件与保留边界

2026-10-06；基于第三批清理后的本机源码。此表是代码能力清单，不表示完整
质量或性能已认证。最新程序已全新构建、封存源/对象/链接身份，CPU 停止测试、
三项 GPU 夹具和退休入口检查通过，随后完成9次短窗；100/300帧、独立CCD与
完整质量协议仍待验证。结果见
`../reports/STATUS_20261006.md`，第二、三批变更见对应清理文档。

## 活动功能

| 功能 | 当前入口/默认 | 依据与状态 |
|---|---|---|
| IPC、ABD/FEM/布料、完整碰撞安全 | 当前执行实验使用 IPC；材料与完整 CCD 保留 | `core/gipc_system.cu:10`、`core/GIPC.cu`、`collision/ACCD.cu`；本次清理未改数值公式 |
| 原停止与另行研究的停止规则 | `GIPC_IPC_TERMINATION=legacy`；另有 movement_only/gated/compensated | `solver/ipc_options.h:9–16`；legacy `.01` / min6，PCG rho `1e-4`，研究模式不属于同停止条件执行优化 |
| AL/Robust TOI | `GIPC_CONTACT_BACKEND=toi_al` 独立启用；policy paper/robust | `solver/toi_solver.cu`、`solver/toi_options.h:61–132`；跨帧接触/λ/γ已存在，不作为新待实现组件 |
| host / conditional PCG | `GIPC_PCG_EXECUTION=host|conditional_graph`；原生未设置为 host | `core/gipc_system.cu:61`、`linear_system/solver/pcg_graph_impl.inl`；previous-rho、零 RHS、breakdown/上限守卫保留 |
| legacy / 稳定 MAS | legacy 默认；Cholesky、wide/inverse64 对照路径保留 | `solver/MASPreconditioner.cu`；稳定 CSR、factor 和六历史 M fixture 仍依赖这些共享结构 |
| 稳定限制与局部作用 | `serial|warp`；`triangular|factor_inverse` | `solver/mas_restrict_options.h`、`mas_factor_action_options.h`；原生默认 serial、factor_inverse；不可与退休 ordered 混淆 |
| 用户指定世界刚度默认 | diagonal/movable 条件下 `world_block` | `solver/toi_options.h:119–124`；用户默认选择记录于 `history/reports/active/DEFAULTS_OUTER_RESULTS_20261005.md:3` |
| 既有执行组合 | FullCCD refit、batched energy、energy reuse、无接触 MAS topology reuse | `core/accel_features.h`、`core/GIPC.cu:9056,10766,10943`、`solver/MASPreconditioner.cu:2439`；原生开关默认 false，runner combined 明确开启所选项 |
| 普通/swept BVH 独立缓存 | `GIPC_DISCRETE_BVH_REFIT`，默认 false，周期8 | `collision/discrete_bvh.h:17–22`；有历史收益但未取得2×同质量认证；见组件实测报告 |
| swept barrier contact pool | `GIPC_CONTACT_POOL`，默认 false；validate 独立 | `collision/ipc_contact_pool.h:13–27`；保守机器 bounds 检查及不满足回退保留。Step1 四guard的typed/energy与质量结论分开，旧 `allow_next_round=false` 保留；见 `reports/autodl_step1/ROUND0_AND_GUARDS.md` |
| diagonal update fusion | `GIPC_PCG_FUSED_DIAG_UPDATE`，默认关闭，MAS 不适用 | `pcg_graph_impl.inl`、`tests/diag_fused_update_tests.cu`；只支持相应全局块对角路径，不是本次删除的 MAS-dot |

`tools/bench/plans.py:10–19` 的当前接触池实验使用 IPC＋legacy MAS＋Graph＋
combined＋普通 BVH 缓存；不能把这里的显式配置说成所有程序默认都开启。

## 保留的诊断入口

- main 早退出：`GIPC_VALIDATE_COMPONENTS`、`GIPC_PCG_GUARD_FIXTURE`、
  `GIPC_CONTACT_POOL_FIXTURE`、`GIPC_MAS_REPLAY_FIXTURE`、
  `GIPC_MAS_CHOLESKY_FIXTURE`（`app/gl_main.cu:1785–1794`）。六历史 M fixture
  不等于六份历史全局 A/b/M 完整 PCG 重放。
- 同进程固定系统：`GIPC_FIXED_STUDY_DIR/FRAMES/DIRECTIONS/COMPACT`；
  `GIPC_FIXED_MAS_STAGE_STUDY`、`GIPC_FIXED_RESTRICT_STUDY`、
  `GIPC_FIXED_FACTOR_STUDY`，见 `linear_system/solver/pcg_fixed_study.inl`。
- 算子/失败观察：`GIPC_AUDIT_PCG`、`GIPC_AUDIT_GRAPH`、`GIPC_MAS_AUDIT`、
  `GIPC_FAILURE_SYSTEM`、`GIPC_STATE_WINDOW_DIR`、`GIPC_COST_TRACE`。
  它们保留诊断成本和恢复边界，正式计时不能混开。
- 碰撞/能量同状态核对：`GIPC_AUDIT_REFIT`、`GIPC_AUDIT_ENERGY`、
  `GIPC_DISCRETE_BVH_VALIDATE`、`GIPC_CONTACT_POOL_VALIDATE`。
- 帧状态/真实速度/物理能量与 checkpoint 观察仍在
  `app/gl_main.cu:1818–1932`。checkpoint 只是一种入口，未通过续算等价的
  混合状态不能作为算法因果对照，正式轨迹继续从零推进。

## 已退休及保留边界

已从活动实现删除：bounded-request CCD、BVH eligibility、MAS final-dot、
SpMV+pAp、legacy ordered restriction，以及下面两个 AL 退出入口。`core/retired_components.h` 明确拒绝
启用，历史说明/报告和 Git 恢复入口保留。v55/v56 三角优化没有进入当前标准
`cholesky_action` 实现，无需在活动源码中重复删除它们。

以下两个有明确负面证据的入口已在 Step 3 退休；新程序已独立构建，旧已测程序仍保留。静态等价、程序身份和已执行检查见 `CODE_CLEANUP_STEP3.md`：

| 入口 | 证据 | 已实施边界 |
|---|---|---|
| 全局 `GIPC_TOI_INNER_EXIT=velocity_only` | `history/reports/DIRECTION_REVIEW_V46_20261004.md:19`：660次方向/101969 PCG仅推进一次outer；改变原停止语义且未形成可接受结果 | 删除这一枚举及对应抑制条件；保留 native、full_step_only、原速度阈值、完整CCD；原生启动明确报退休错误 |
| 选定 outer 的 `GIPC_TOI_FULL_STEP_EXIT_PROBE` | `history/reports/active/DEFAULTS_OUTER_RESULTS_20261005.md:64–72`：有限8次内层探针无一致工作量改善，质量仍未通过，按预声明停止 | 删除解析/状态/专属预算分支；保留常规 full-step 出口及通用 outer observer，旧日志字段固定 null/false；原生启动明确报退休错误 |

其余路径不能仅因“仍是实验”或代码多而定为失败：wide/inverse64 服务数值回归，
triangular 是稳定参考；factor_inverse/world_block 是用户已选默认；
gated/compensated、choose_start/restart guard 与完整 AL 后端涉及另行研究的数学
行为。本轮没有足够的新证据授权把它们合并删除。

保留功能或当前默认开关不代表取得同质量 2× 加速；精度、完整轨迹和受控负载计时仍须分别验收。
