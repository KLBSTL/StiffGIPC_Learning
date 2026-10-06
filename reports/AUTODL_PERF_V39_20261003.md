# v39 MAS 运算精度集成与验证

## 结论

默认关闭的 MAS wide apply 已集成并在 AutoDL CUDA 12.8 / RTX 4090 构建验证。实际 MAS 类对旧固定系统的计算与 v38 候选及独立 CPU 层级计算一致，但兔子完整仿真仍失败：默认 rho 容差完成 31 帧，在第 32 帧触发 PCG 上限；rho=1e-14 完成 38 帧，在第 39 帧触发同类保护。不能宣布兔子修复，也不推广默认开关或认证加速。

## 实现与冻结身份

- 独立 `sources/stiff_perf_v39`，开关 `GIPC_MAS_WIDE_APPLY=1`；默认 0。float32 局部逆存储、原生停止规则和安全检查保持。
- 开启后层级 R、局部乘加、Z 及最终累加采用 double；新增缓冲区在组装时分配，避免 Graph 捕获中分配。精度模式和 R64/Z64 地址加入 Graph 身份。
- 实际 MAS 类增加诊断快照入口；同一对象执行 off/on/off、直接 apply 和三次 Graph 重放，并检查缓冲区增长后的身份变化及结果。
- runner 显式记录精度选项，增加 768 MiB 磁盘余量保护；本轮各场景最长 240 秒。
- 清单 `manifests/perf_v39.json`：559 文件，源码摘要 `bd8a515243603ef77edf5c45b1b72a28b803684cbbc6c7ce8a182ecee1f1924d`。
- AutoDL 二进制 SHA256：`221fb2a194194b76375c8f3da16a46a83b3598d2f2c8b637a0d449f832da5d77`。结束前复核 v36/v37 各 557 文件及对应二进制未变。

## 构建与固定算子

Release / sm89 构建成功，CTest 1/1；融合更新夹具 28 组、108108 值逐位一致。组件检查包含 6 个基础、7 个 warm history、7 个 port、17 个 volume case，总体通过。

| 旧快照 | v39 wide 对 CPU 层级相对差最大值 | 对 v38 captured_r64 | Graph 对直接 apply 最大相对差 |
|---|---:|---:|---:|
| 悬挂初始 | 3.460e-16 | 未运行该 v38 模式 | 1.870e-16 |
| 兔子初始 | 4.328e-15 | 4.070e-16 | 4.382e-16 |
| 兔子 v37 失败 | 1.149e-12 | 1.006e-12 | 1.234e-12 |

三个快照均通过 1e-10 的 CPU/Graph 检查；off/on/off 恢复原身份、切换改变身份、缓冲区增长改变身份的检查通过。增长后算子相对差最大 6.574e-17。此诊断验证实际类及缓存键，未将它等同于所有动态图变化的穷尽测试。

## 场景结果

均在隔离 v39 二进制上运行，dt=.01、tol=.01、suite=1、Robust velocity tol=.05、融合 diag 更新关闭，开启诊断审计；耗时不得作为正式性能收益。

| 运行 | 精度/预条件器 | 完成帧 | 结果 |
|---|---|---:|---|
| native_smoke | wide off / MAS | 2/2 | 通过 |
| wide_smoke | wide on / MAS | 2/2 | 通过 |
| diag_control | wide on / diag | 2/2 | 通过，diag 路径兼容 |
| wide_hang_paths | wide on / MAS | 100/100 | 通过，完整接受路径导出 |
| wide_bunny | wide on / MAS，rho=1e-4 | 31/100 | 第 32 帧 PCG 上限，拒绝未收敛 trial |
| wide_bunny_strict | wide on / MAS，rho=1e-14 | 38/100 | 第 39 帧 PCG 上限，拒绝未收敛 trial |

两个新失败系统的 CPU 真相对残差分别 9.0255e-4、3.1468e-4；MAS 重复作用相对差最大 4.7423e-13、1.0627e-10。严格配置的新系统重复差略超过固定夹具的 1e-10 门槛，不能外推算子在任意病态状态均满足该界。局部逆未检出相对尺度低于 -1e-7 的负特征值，但这不证明完整系统 SPD 或 CG 能可靠收敛。

原默认/严格阈值失败、5 个初始或失败 MAS 审计、完整 A/b 和局部缓冲区均保留。没有放宽 PCG 上限、关闭安全检查或隐藏失败。

## 独立连续碰撞与覆盖

| 路径 | 已完成帧 / 帧尝试 | 接受推进 | 静止桥接 | CCD 段数 | 碰撞标记 |
|---|---:|---:|---:|---:|---:|
| 悬挂布料 | 100 / 100 | 206 | 99 | 305 | 0 |
| 兔子默认 | 31 / 32 | 101 | 31 | 132 | 0 |
| 兔子严格 | 38 / 39 | 144 | 38 | 182 | 0 |

共 619 段。包括失败帧中已经接受的推进；未接受的失败 trial 不计作路径。每帧 safe 状态数量与统计中的接受步数+1相等，所有已完成帧首末状态逐位匹配，CCD 数量与全部导出相邻段一致。

悬挂使用 `validate_path.exe`；兔子使用既有表面限定 `diagnose_first_path.exe`，不启用遇首个标记提前停止。均为独立 CPU BVH + Tight-Inclusion，容差 1e-9，Stable NH1 材料策略。兔子默认/严格存在 1354 / 730 次 FEM 四面体翻转段事件（不是独立四面体数量），ABD 翻转为零；Stable NH1 容许翻转使该项不阻断材料策略下的 CCD 门槛，但这些结果不能证明物理质量合格，更不能代替兔子 100 帧验收。

## 数据、命令与下一步

结果包 563138695 字节，SHA256 `a16a5d2b5caa2039a7b37e1d021f468da0759a65d6d177462ecdc19c22e4682f`；固定夹具包 SHA256 `8fceb32279d062063d49605a8142eb6f48bb3720780593959a269bc5675bab16`。下载后完整校验、提取。保留本地压缩包及远端原始目录，删除远端重复传输包。

远端目录 `/root/autodl-tmp/stiff_toi_cudagraph_20260929/v39_20261003`；主要命令：

```text
bash tools/autodl_v39_build.sh
ctest --test-dir builds/autodl-stiff_perf_v39 --output-on-failure
GIPC_VALIDATE_COMPONENTS=$PWD/reports/components.json builds/autodl-stiff_perf_v39/gipc
python3 tools/benchmark_perf_v39.py --phase fixtures
python3 tools/benchmark_perf_v39.py --phase smoke
python3 tools/benchmark_perf_v39.py --phase paths
python3 tools/benchmark_perf_v39.py --phase bunny
python3 tools/benchmark_perf_v39.py --phase strict
```

各场景完整命令在下载目录 `reports/v39_*.json`；本地运行 `tools/analyze_fixture_v39.py`、`tools/analyze_perf_v37.py` 和 `tools/verify_v39.py`，产出 `MAS_FIXTURE_V39_20261003.json`、`MAS_FAILURES_V39_20261003.json`、`VERIFICATION_V39_20261003.json`。最后一项包含全部 CCD 精确命令和验证器 SHA256。复用 v37 分析器输出的空 window 项不代表开展了新状态窗口实验。

后续先用本轮两个新失败 A/b 做固定系统 host/Graph、残差演化和局部逆精度对照，检查病态状态下停滞的来源。保留默认关闭；共同物理参考、host 全轨迹分叉及质量匹配性能门槛仍未验收。

测试期间按用户要求清理历史空间，详见 `CLEANUP_20261003.md`。GPU 结束状态 0% / 24081 MiB free。
