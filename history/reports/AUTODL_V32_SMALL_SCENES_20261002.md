# AutoDL 小场景 v32 对比（2026-10-02）

RTX 4090 24GB，CUDA 12.8，sm89 Release。每组 100 帧，dt=.01，Newton=.01，PCG=1e-4，suite=0；GPU 串行执行。快速成功组最多重复 3 次，以下时间取中位数。

**加速比为实测耗时比，未通过质量匹配认证。** 质量栏为与同 dt base 的布料轨迹差异，不能当作精确物理解误差；本轮没有另做独立 CPU 连续碰撞验证。Stable NH1 允许四面体翻转，负 Jacobian 单独记录。

计时：IPC_Solver 外围同步的 solver_ms 求和，排除进程启动和求解器外轨迹导出。接触/非接触共用 base 第一轮原生窄阶段候选对数 >0 的帧掩码；缺少某阶段时速度比显示 —。

## cloth_hang_l

base 掩码：非接触 91 帧，接触 9 帧。

| 配置 | 重复 | 总秒 | /base | 非接触秒 | /base | 接触秒 | /base | /base+Graph | 布料 RMS 最大差异 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3 | 1.985 | 1.000× | 1.498 | 1.000× | 0.487 | 1.000× | 0.788× | 0.00–0.00% |
| base+Graph | 3 | 1.565 | 1.268× | 1.156 | 1.296× | 0.405 | 1.201× | 1.000× | 0.00–0.03% |
| base+TOI (.05) | 3 | 2.273 | 0.873× | 1.848 | 0.810× | 0.425 | 1.146× | 0.689× | 1.09–1.18% |
| base+TOI (.05)+Graph | 3 | 1.679 | 1.182× | 1.342 | 1.116× | 0.341 | 1.430× | 0.932× | 1.12–1.14% |
| base+TOI (1.0) | 3 | 1.501 | 1.322× | 1.176 | 1.273× | 0.325 | 1.499× | 1.042× | 3.81–3.81% |
| base+TOI (1.0)+Graph | 3 | 1.214 | 1.634× | 0.948 | 1.580× | 0.266 | 1.828× | 1.288× | 3.81–3.81% |

| 配置 | Newton 方向中位数 | CG 中位数 | 最小地面间隙 m | 最小相对 tet Jacobian |
|---|---:|---:|---:|---:|
| base | 432 | 13748 | 2.757e-01 | — |
| base+Graph | 435 | 13863 | 2.757e-01 | — |
| base+TOI (.05) | 416 | 19079 | 2.757e-01 | — |
| base+TOI (.05)+Graph | 410 | 18610 | 2.757e-01 | — |
| base+TOI (1.0) | 247 | 9914 | 2.768e-01 | — |
| base+TOI (1.0)+Graph | 245 | 9829 | 2.768e-01 | — |


## cloth_sphere7_l

base 掩码：非接触 21 帧，接触 79 帧。

| 配置 | 重复 | 总秒 | /base | 非接触秒 | /base | 接触秒 | /base | /base+Graph | 布料 RMS 最大差异 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3 | 4.498 | 1.000× | 0.218 | 1.000× | 4.280 | 1.000× | 0.834× | 0.00–0.00% |
| base+Graph | 3 | 3.752 | 1.199× | 0.209 | 1.043× | 3.544 | 1.208× | 1.000× | 0.64–1.35% |
| base+TOI (.05) | 3 | 4.028 | 1.117× | 0.149 | 1.465× | 3.867 | 1.107× | 0.931× | 3.86–4.10% |
| base+TOI (.05)+Graph | 3 | 3.315 | 1.357× | 0.147 | 1.485× | 3.168 | 1.351× | 1.132× | 3.74–4.18% |
| base+TOI (1.0) | 3 | 2.899 | 1.551× | 0.139 | 1.574× | 2.761 | 1.550× | 1.294× | 4.49–4.84% |
| base+TOI (1.0)+Graph | 3 | 2.569 | 1.751× | 0.124 | 1.756× | 2.443 | 1.752× | 1.460× | 4.43–4.52% |

| 配置 | Newton 方向中位数 | CG 中位数 | 最小地面间隙 m | 最小相对 tet Jacobian |
|---|---:|---:|---:|---:|
| base | 582 | 24113 | 7.831e-02 | 1.000 |
| base+Graph | 577 | 23589 | 7.841e-02 | 1.000 |
| base+TOI (.05) | 665 | 24728 | 6.840e-02 | 1.000 |
| base+TOI (.05)+Graph | 655 | 24371 | 6.853e-02 | 1.000 |
| base+TOI (1.0) | 372 | 15488 | 7.195e-02 | 1.000 |
| base+TOI (1.0)+Graph | 383 | 15999 | 7.221e-02 | 1.000 |


## bunny_cloth_bunny_l

base 掩码：非接触 0 帧，接触 100 帧。

| 配置 | 重复 | 总秒 | /base | 非接触秒 | /base | 接触秒 | /base | /base+Graph | 布料 RMS 最大差异 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3 | 8.565 | 1.000× | 0.000 | — | 8.565 | 1.000× | 0.958× | 0.00–0.00% |
| base+Graph | 3 | 8.209 | 1.043× | 0.000 | — | 8.209 | 1.043× | 1.000× | 0.50–0.90% |

| 配置 | Newton 方向中位数 | CG 中位数 | 最小地面间隙 m | 最小相对 tet Jacobian |
|---|---:|---:|---:|---:|
| base | 604 | 20994 | 3.137e-05 | -0.221 |
| base+Graph | 618 | 21339 | 3.160e-05 | -0.530 |

- 未纳入速度汇总：toi005 r1，状态 failed，完成 33 帧，原因 PCG iteration limit；末次 PCG 18257 次，原始诊断见 JSON 和对应 run.log。
- 未纳入速度汇总：toi005_graph r1，状态 failed，完成 38 帧，原因 PCG iteration limit；末次 PCG 18257 次，原始诊断见 JSON 和对应 run.log。
- 未纳入速度汇总：toi1 r1，状态 failed，完成 37 帧，原因 PCG iteration limit；末次 PCG 18257 次，原始诊断见 JSON 和对应 run.log。
- 未纳入速度汇总：toi1_graph r1，状态 failed，完成 37 帧，原因 PCG iteration limit；末次 PCG 18257 次，原始诊断见 JSON 和对应 run.log。

## 兔子 TOI 失败诊断

四个正式 TOI 配置都触发 PCG 上限（18257 次），随后拒绝提交未收敛的试探步，退出码 4；均未得到完整 100 帧加速比。base 和 base+Graph 完整通过。

补充残差审计运行完成 29 帧后同样中止。末次真实相对残差 `71.0411`，预条件残差内积 rho `-778201`，表明该线性求解已发散，远高于请求的 1e-4。诊断运行改变了审计/同步开关，失败帧不同，未用于计时。

已确认直接原因是混合 ABD/FEM/布料场景的 TOI 线性求解未收敛。host 与 Graph 都失败，因此先检查 TOI 组装矩阵、MAS 预条件器的正定性/适配和接触刚度尺度；负 rho 是预条件系统的异常证据，尚不能单凭该标量定位到某个组装项。CCD 候选数增长记录在日志中，但没有出现显存不足退出。不能通过忽略 PCG 上限给出有效速度比。

## 验证与范围

- 完整且满足时序/功能/求解标志门槛的组：42/46。
- 已检查源摘要、Linux 可执行文件哈希、runner 哈希、参数、100 帧编号、有限值、PCG/outer 上限、TOI 原生安全步标志和 Graph 实际执行方式。
- 七项 Robust 适配 GPU 组件测试见下载目录 builds/toi_components.json。
- 显存每秒采样一次，观测增量为下界；极短运行出现 0 MiB 不表示没有使用显存。预估值也不代表实际峰值。
- 总时间、接触时间、非接触时间分别取三轮中位数，因此两个阶段的中位数之和可能与总时间中位数略有不同。
- 跳过组：8；具体原因见 JSON skipped。

完整轨迹在 AutoDL：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v32_20261002/runs/autodl`。
完整归档：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v32_20261002_results.tar.gz`，约 977MB，SHA256 `3c5f014b326254e4c75610facca268be09306caf7c16972831de7cdc1861e78c`。
完整归档的本机下载未完成；已下载部分保留为 `downloads/v32_20261002_results.partial.tar.gz`，不能作为完整归档使用。

本机计时、诊断和拓扑记录：[下载目录](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_v32_20261002)。逐帧质量计算在 AutoDL 上使用完整 101 个状态快照完成；本机核对记录包哈希、各组参数、日志哈希、100 帧计时和速度汇总。
源摘要：`e0d75fed3b32d319a42faa3c3cc344df1ba7cee8db5fa62417a9ad7419ba91ad`。

## 复现与核对命令

在独立 AutoDL 目录运行（构建脚本要求新的构建目录；已有目录可直接运行矩阵）：

```bash
bash tools/autodl_v32_build.sh
python3 tools/benchmark_v32_autodl.py --steps 30 --repeats 1 --arms base,toi005,toi1_graph --timeout 90 --budget 600
python3 tools/benchmark_v32_autodl.py --steps 100 --repeats 3 --timeout 300 --budget 1800
/root/miniconda3/bin/python tools/report_v32_autodl.py --data .
```

本机使用 DL Python 核对：

```text
python tools/verify_v32_autodl_results.py
python tools/plot_v32_autodl.py
```

[数据表](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/AUTODL_V32_SMALL_SCENES_20261002.csv) · [完整汇总 JSON](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/AUTODL_V32_SMALL_SCENES_20261002.json) · [加速比图](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/figures/AUTODL_V32_SMALL_SCENES_20261002.png)
