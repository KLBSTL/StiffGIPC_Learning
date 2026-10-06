# AutoDL v35：能量批处理与 BVH refit 组合

使用冻结的 v34 CUDA 12.8 二进制，没有新增数值内核。布料–球小场景，100 帧 × 3，suite=0/1 交错运行；固定 dt=.01、Newton=.01、PCG rho 比 1e-4、velocity_tol=.05、diag + conditional Graph、融合关闭。

AutoDL 工作目录：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v34_20261003`。完整结果已下载到本任务的 `downloads/autodl_perf_v35_20261003` 与 `downloads/autodl_perf_v35_profile_20261003`，两份压缩包 SHA256 校验通过。

suite=1 在该配置中启用能量批处理和 CCD BVH refit。IPC 能量复用只用于 IPC 分支，本轮 TOI 不使用；MAS 静态拓扑不适用于本轮对角预条件器。

| 配置 | 总秒中位数 | 非接触秒 | 接触秒 | /suite off | /近期 base | CV | 方向 | CG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| suite_off | 3.1668 | 0.1211 | 3.0463 | 1.000× | 1.477× | 2.64% | 604 | 47553 |
| suite_on | 3.1247 | 0.1082 | 3.0158 | 1.013× | 1.497× | 3.69% | 610 | 47625 |

三轮成对新增速度比：1.021×、1.013×、1.037×。

| 配置 | 非接触 /suite off | 接触 /suite off | 最大 RMS/off | 最大 RMS/base | 同配置重复 RMS | 最大边拉伸 | 最小地面间隙 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| suite_off | 1.000× | 1.000× | 0.00% | 4.05% | 1.12% | 1.01769 | 7.129e-02 |
| suite_on | 1.119× | 1.010× | 1.52% | 4.22% | 1.25% | 1.01769 | 7.151e-02 |

## 验证与范围

- 额外 100 帧逐帧 BVH refit/rebuild 第一次候选集合完全一致；所有能量查询批处理/原路径最大相对差 0.000e+00（门槛 1e-10）。
- 正式六组均完成 100 帧，参数、来源/二进制/runner 哈希、Graph 实际执行和计时总和核对；无 PCG/外层上限、非有限值或原生不安全接受步。
- 额外 100 帧完整接受路径：385 段独立 Tight-Inclusion 检查，0 个保守标记，通过=True。每帧起点/终点与 trace 逐字节核对。

阶段按 v34 base r01 候选数共同掩码划分；候选存在不等于实际受力。总/阶段分别取中位数。base 来自同实例刚完成的 v34 三轮矩阵，未参与本次 suite 配对，因此 /base 是近期原始耗时对照。

RMS 是同时间步轨迹差，不能视为对共同收敛物理解的误差；正式质量匹配速度资格仍为 false。组合只在本小场景评估，全局默认不变。Graph 融合开关仍为 0。v34 同系统审计的最大 Graph/host 相对解差 9.745e-5、真残差 0.107；严格 1e-6 同解门槛和完整物理参考仍未验收。

## 采用结论

总时间中位数比为 1.013×，成对比中位数为 1.021×。三轮均略快，但幅度仅 1.3%–3.7%，与跨运行方向/CG 数变化和计时 CV 同量级。因此记录为小场景的正向候选配置，尚不认证稳定、质量匹配的通用收益；不改全局默认。当前组合没有减少 CG 总数，收益主要应从每次能量查询与 BVH 更新成本解释。

## 诊断分项计时

每配置额外单次 100 帧，开启同步分项计时；本表只定位成本，不能代替正式三轮速度比。不同运行方向/CG 数不同。

| 配置 | 组装秒 | PCG 秒 | 线搜索秒 | 完整 CCD+活动集秒 | 安全更新秒 | CG | 线搜索 ms/方向 |
|---|---:|---:|---:|---:|---:|---:|---:|
| suite=0 | 0.4138 | 1.3515 | 0.1897 | 0.3393 | 0.7121 | 48874 | 0.310 |
| suite=1 | 0.4137 | 1.3083 | 0.1576 | 0.2850 | 0.6511 | 46884 | 0.262 |

开启组合后，已计时区段 PCG 占 46.5%，安全状态更新（含 BVH/候选与原生交叉检查）占 23.1%，组装占 14.7%；线搜索只占 5.6%。因此只降低能量计算成本，对总时间的影响有限。下一优先项为固定 A/b 下 PCG 及安全更新成本，先分离算子吞吐和跨运行工作量变化，再选择优化。

Robust 分支把完整 CCD 与活动集更新共享在 active_update 区段，safe_ccd 单独字段主要是复用 alpha 的 bookkeeping；不能据其接近零推断 CCD 开销接近零。上表已将二者合并。区段和不含初始组装、帧初始化及其他未计时开销。

## 复现

```text
python3 tools/benchmark_perf_v35.py --phase smoke
python3 tools/benchmark_perf_v35.py --phase matrix
python3 tools/benchmark_perf_v35.py --phase paths
python3 tools/export_perf_v34_autodl.py --version v35
python3 tools/profile_perf_v35.py
E:/Anaconda/envs/DL/python.exe tools/validate_perf_v34.py --platform autodl --version v35 --data downloads/autodl_perf_v35_20261003
E:/Anaconda/envs/DL/python.exe tools/report_perf_v35.py --data downloads/autodl_perf_v35_20261003 --base-data downloads/autodl_perf_v34_20261003 --profile-data downloads/autodl_perf_v35_profile_20261003
```

脚本拒绝覆盖已有记录；重新运行需新输出目录/名称。CPU CCD 精确命令和验证器哈希见 AUTODL_PERF_V35_PATH_VALIDATION.json。
