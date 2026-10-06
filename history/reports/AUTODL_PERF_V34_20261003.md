# AUTODL v34：Graph PCG 更新与块对角预条件融合

2026-10-03，RTX 4090 24 GiB; CUDA 12.8; Release sm89。布料–球小场景，每组 100 帧 × 3，交错执行；dt=.01，Newton=.01，PCG rho 比 1e-4，TOI velocity_tol=.05 m/s，suite=0。

新增可选融合 kernel：同时更新 x/r 并应用全局 3×3 块对角预条件器；随后仍按原顺序应用局部 ABD 12×12 预条件器。Graph cache key 包含实际融合状态。MAS 自动回退原路径。没有修改收敛阈值、TOI、摩擦、CCD、安全检查或矩阵。

源代码独立保存在 sources/stiff_perf_v34_cuda128，Release 构建 builds/autodl-stiff_perf_v34_cuda128，冻结身份 manifests/perf_v34_cuda128_autodl.json。CUDA 12.8 修正仅是独立 fixture 的显式 C++ 数组类型；运行时数值源与 Windows v34 一致。base/v32 使用此前已验证的 Linux 二进制并复核哈希。

| 配置 | 总秒中位数 | /base | /v32 diag | /v34 关闭融合 | CV | Newton 方向 | CG | 最大 RMS/base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 4.6762 | 1.000× | 0.679× | 0.675× | 3.18% | 585 | 23999 | 0.00% |
| v32_diag | 3.1730 | 1.474× | 1.000× | 0.994× | 1.28% | 600 | 47214 | 3.94% |
| v34_off | 3.1554 | 1.482× | 1.006× | 1.000× | 2.47% | 596 | 47099 | 4.29% |
| v34_on | 3.1894 | 1.466× | 0.995× | 0.989× | 3.65% | 625 | 48403 | 4.11% |

AutoDL 工作目录：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v34_20261003`。完整结果已下载到本任务的 `downloads/autodl_perf_v34_20261003`，压缩包 SHA256 校验通过。

同一 v34 二进制融合开关的三轮成对速度比：0.933×、1.032×、0.978×。

| 配置 | 非接触秒 | 接触秒 | 最小地面间隙 m | 最大边拉伸 | 最大 RMS/v34 off |
|---|---:|---:|---:|---:|---:|
| base | 0.2299 | 4.4409 | 7.857e-02 | 1.02410 | 4.29% |
| v32_diag | 0.1184 | 3.0546 | 7.145e-02 | 1.01769 | 1.64% |
| v34_off | 0.1191 | 3.0344 | 7.144e-02 | 1.01769 | 0.00% |
| v34_on | 0.1155 | 3.0750 | 7.142e-02 | 1.01769 | 1.52% |

| 配置 | 非接触 /base | 接触 /base | 非接触 /v34 off | 接触 /v34 off | 同配置三次最大布料 RMS 差 |
|---|---:|---:|---:|---:|---:|
| base | 1.000× | 1.000× | 0.518× | 0.683× | 0.57% |
| v32_diag | 1.942× | 1.454× | 1.006× | 0.993× | 1.32% |
| v34_off | 1.930× | 1.464× | 1.000× | 1.000× | 1.08% |
| v34_on | 1.991× | 1.444× | 1.032× | 0.987× | 1.43% |

阶段掩码采用当前 base r01 原生窄阶段候选数 >0：非接触 21 帧，接触候选 79 帧，各配置使用相同帧掩码。候选存在不等于实际施力。分别取中位数，因此阶段和可能与总中位数略有差异。

## 数值与安全验证

- GPU 融合 fixture：28 组，108108 个 x/r/z 值与原两 kernel 逐位一致；CPU 参考最大相对差异 2.240e-14。包含非对角块、奇数尺寸、256 边界、多种 alpha 与越界哨兵。
- 实际场景：30 帧融合 Graph/host 同一矩阵解审计及真实残差审计；10 帧 MAS 回退；正式 12/12 组完成，无求解上限、非有限值或原生不安全接受步。
- 30 帧固定系统审计记录：Graph/host 最大相对解差 9.745e-05，host 自身重复差 8.168e-04，最大真相对残差 1.073e-01。严格 1e-6 同解门槛仍未通过；原生 PCG rho 比停止与真残差不是同一指标，本轮没有放宽停止参数。
- 独立接受路径 CCD：通过，377 段，0 标志。逐帧桥接和终点核对；该检查覆盖额外导出运行。

计时只统计原生 solver_ms，排除进程启动和求解器外轨迹导出。GPU 测试串行运行，保留背景占用/温度/频率样本。每组运行前检查 GPU 利用率不高于 30%，并由 runner 拒绝与其他 CUDA 进程重叠。实测耗时比不代表收敛物理解误差一致，质量匹配标志仍为 false。fixture 的逐位一致证明局部运算保持一致；整段仿真还受浮点并行规约顺序影响。

## 本轮采用决策

成对速度比中位数为 0.978×。默认继续使用 --fused-diag-update 0；只有三轮全部加速且质量验证通过时，才有依据考虑开启。当前小场景的数据不支持稳定融合收益。相对 base 的速度来自此前 TOI、Graph 与预条件器组合，不能算成此次 kernel 融合的新增收益。

## 复现命令

```text
bash tools/autodl_v34_build.sh
python3 tools/benchmark_perf_v34.py --platform autodl --phase smoke
python3 tools/benchmark_perf_v34.py --platform autodl --phase matrix
python3 tools/benchmark_perf_v34.py --platform autodl --phase paths
python3 tools/export_perf_v34_autodl.py --version v34
E:/Anaconda/envs/DL/python.exe tools/validate_perf_v34.py --platform autodl --version v34 --data downloads/autodl_perf_v34_20261003
E:/Anaconda/envs/DL/python.exe tools/report_perf_v34.py --platform autodl --data downloads/autodl_perf_v34_20261003
```

复现实验需要新输出名称；脚本拒绝覆盖已有记录。独立 CCD 的精确命令保存在报告补充中。
