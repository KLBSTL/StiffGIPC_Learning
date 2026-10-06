# TOI 耗时诊断（AutoDL，2026-09-30）

范围：`boxes1920_cloth_l`，100 帧，CUDA 12.8，v17。同一批非 profile 单次运行的 solver 时间：base `5.542887082 s`，base+TOI host `72.820049671 s`，即 TOI 耗时为 base 的 `13.14×`（按 base/TOI 表示加速比为 `0.076×`）。质量匹配门槛未过，这不是正式性能结论。

## 按 base 轨迹的几何接触时间分段

| 帧 | base 时间 | TOI 时间 | TOI 额外时间 | TOI 外层轮数 | TOI Newton 方向求解 |
|---|---:|---:|---:|---:|---:|
| 1–65 | 3.252 s | 3.888 s | 0.636 s | 319 | 323 |
| 66–79（base 几何接触） | 1.414 s | 46.532 s | 45.118 s | 520 | 2,437 |
| 80–100 | 0.877 s | 22.401 s | 21.524 s | 935 | 986 |

约 67% 的额外时间集中在第 66–79 帧，约 32% 在第 80–100 帧。**分类只依据 base 轨迹**；TOI 自身在第 55–100 帧共 46 帧检测到几何接触，尤其第 80–100 帧全部有接触，不能把该段解释为 TOI 的真正无接触开销。两条轨迹已分离，物理质量仍未匹配。

## 迭代与候选工作量

| 100 帧累计 | base | TOI host |
|---|---:|---:|
| Newton/PCG 方向求解次数 | 548 | 3,746 |
| PCG 内部迭代总数 | 28,938 | 185,058 |
| TOI 外层轮数 | 0 | 1,774 |
| TOI 活动集更新的 swept CCD 宽相位候选次数累计 | — | 516,282,103 |
| TOI safe step 的 swept CCD 宽相位候选次数累计 | — | 544,129,507 |

后两项是各轮重复处理的候选次数之和，**不是不同几何对的数量**。`swept_query` 每次重建 full CCD BVH/候选，并对候选执行 `pair_ccd`。活动集更新还将时间、碰撞对和表面顶点下载到 CPU，再用 `std::map` / `std::set` 筛选；safe step 之后又做 BVH/候选更新与 `isIntersected` 检查。实现位置：`sources/stiff_fused/StiffGIPC/solver/toi_solver.cu` 的 `swept_query` 与 `solve_subTOI`。

第 70、74、77、78 帧分别耗时 `5.982、7.922、7.378、5.399 s`，合计 `26.681 s`，占 TOI 总 solver 时间约 36.6%。这四帧的方向求解次数为 `514、490、414、297`。第 70 帧一个外层轮次持续 485 次内层求解，其中 473 次线搜索仅接受 `r=0.25`；第 74 帧一个轮次持续 452 次，其中 410 次 `r=0.25`。当前内层退出条件要求一次完整 `r=1` 步，因此反复小步的 Newton/PCG 求解构成主要尖峰；不能仅归咎于 CCD。

## AutoDL 单独 profile 复跑

命令：`/root/miniconda3/bin/python tools/run_local.py --platform autodl --arm base_toi --scene boxes1920_cloth_l --steps 100 --name autodl_v17_boxes_toi_profile100 --profile --trace --timeout 900`，目录 `/root/stiff_toi_cudagraph_20260929/v17/runs/autodl/autodl_v17_boxes_toi_profile100`。完成 100 帧，solver `67.318 s`；该复跑仅有 2,925 次方向求解，而普通计时为 3,746 次，说明不能把分阶段比例直接当作普通计时的精确占比。`GIPC_PROFILE=1` 在时间戳前同步 GPU，也改变了执行时序。

| profile 阶段，累计 | 时间 | 占 profile 外层总时间约 |
|---|---:|---:|
| 活动集更新：swept CCD、CPU 下载/筛选 | 18.507 s | 27.6% |
| PCG 方向求解 | 15.416 s | 23.0% |
| 梯度/Hessian 装配 | 11.814 s | 17.6% |
| safe 状态更新/再检查 | 6.625 s | 9.9% |
| safe step 完整 swept CCD | 3.525 s | 5.3% |
| 内层线搜索 | 1.431 s | 2.1% |
| 其余外层操作（差额） | 9.811 s | 14.6% |

profile 外层累计 `67.130 s`，表内各项通过非重叠时间戳近似划分。活动集更新与 safe step CCD 两列覆盖 CCD 查询；safe 状态更新还包含 BVH、候选、相交验证。故碰撞相关工作量至少贯穿这三列，不能把 `3.525 s` 单独当成全部碰撞耗时。

## 判断与下一步

1. 最大尖峰来自接触阶段反复小步造成的数百次内层 Newton/PCG；先审计 `r<1` 时的收敛与退出规则，并保留完整 CCD 和物理质量门槛。
2. 持续负担来自每轮活动集更新、全场景 swept CCD、GPU→CPU 候选下载/筛选、以及 safe 状态再检查。优先减少无谓的全量查询与读回，按接触区间计数验证。
3. 在质量门槛通过前，不宣布 TOI 加速。当前 base/TOI 轨迹接触区间明显不同，时间比较只能作为瓶颈诊断。

原始非 profile 运行：`autodl_v17_boxes_base100`、`autodl_v17_boxes_toi_host100`，均在同一远端 `v17/runs/autodl/` 下。分段汇总另见本地 `reports/autodl_v17_boxes_segment_diagnostics.json`。
