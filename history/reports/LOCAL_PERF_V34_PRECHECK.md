# v34 正确性预检（2026-10-03）

已完成独立副本、Release 构建和冻结身份；新增 Graph PCG 的 x/r 更新与 3×3 块对角应用融合。默认开关为 0，使用 `--fused-diag-update 1` 显式启用。局部 ABD 12×12 预条件器仍按原顺序执行，MAS 自动回退。

- GPU fixture：28 组，108108 个 x/r/z 值逐位一致，CPU 最大相对差异 2.23977e-14。
- 实际布料–球：30 帧，116 个线性求解均使用融合；Graph/host 最大解相对差异 0.000108373，host 自身重复最大差异 0.00211991。
- MAS 回退：10 帧，20 个线性求解均未使用融合。
- 额外导出：100 帧完成，无 PCG/outer 上限或原生不安全接受步；逐帧桥接、接受步数量、终点与 trace 完全对应。
- 独立 CPU Tight-Inclusion：366 段全部通过，碰撞保守标志 0。
- 此导出相对先前 v32 diag r01 的布料最大 RMS 差异 0.9556%；这是不同运行之间的轨迹差异，不是收敛物理解误差。

当前 GPU 背景占用约 34–41%，超过 30% 门槛。以上运行均标记 quality-only/diagnostic，不用于速度比。正式矩阵尚待运行，不能宣称已获得额外加速。

源代码：sources/stiff_perf_v34。构建：builds/local-fused-v34。冻结身份：manifests/perf_v34.json。完整证据：LOCAL_PERF_V34_PRECHECK.json。

预检命令：

```text
E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase smoke
E:/Anaconda/envs/DL/python.exe tools/benchmark_perf_v34.py --phase paths
E:/Anaconda/envs/DL/python.exe tools/validate_perf_v34.py
E:/Anaconda/envs/DL/python.exe tools/precheck_perf_v34.py
```
