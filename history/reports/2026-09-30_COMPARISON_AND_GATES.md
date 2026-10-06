# Stiff-GIPC + TOI + CUDA Graph 阶段对比与质量门槛（2026-09-30）

## 结论

四组方法已在 `boxes1920_cloth_l` 的本机和 AutoDL CUDA 12.8 上分别跑完 100 帧，TOI 两组的已接受子步通过独立 CPU Tight-Inclusion 连续碰撞检查。v18 质量罚参数加跨帧活动集在 AutoDL 完成兔子 host/Graph 100 帧，host 还完成 `dt/2` 和 `dt/4` 时间步细化；但 host 的完整表面 CCD 仍有 2 个保守标记，`dt/4` 对较紧 base `dt/8` 参考的位置误差为 1.210%，Graph 100 帧等价门槛也未过。**融合加速尚不能验收**。下列计时全部是单次诊断，正式质量匹配加速比为 N/A。

## 可复现配置

- 官方 base 提交 `bb2849a7b292099581907937860d96ecfdf42588`；四组分别为官方 base、base+Graph、base+TOI(host)、base+TOI+Graph。v17 源码摘要 `cdc1e0533a3aa56af78b68bd8dfc022f5727757e7d24586d145d99a2047dce25`，AutoDL 使用独立 v17 目录和 SHA-256 `6b68e22fdfab7303d4d78f58861f78b2d0bfcfc923781b2f28acc72d0ee809b5` 的上传包。
- 本机为 RTX 3070 Laptop / CUDA 13.0；AutoDL 为 RTX 4090 / CUDA 12.8.93 / SM89。两个平台分别从 v17 源码编译，不能跨平台直接比较绝对秒数。
- `boxes1920_cloth_l` 有 16,449 个顶点、100 帧。接触分类固定使用官方 base 的窄相位几何判据：第 66–79 帧共 14 帧几何接触，其余 86 帧为非接触段。这个分类不是各方法自己的活动约束数量。
- 本机 v17 base 可执行文件 SHA-256 `1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205`，融合文件 `e86695866e33ba72d45a090c86efc4b8d08724f5b29e88a5122d87540b666546`。运行目录里的 `requested.json`、`result.json`、`trace/frames.csv` 和 `output/stats.json` 保存完整参数、状态、逐帧计时和失败原因。
- 当前本地完整源码/诊断归档为 `bundles/stiff_toi_cudagraph_source.tar.gz`，SHA-256 `4a08dc3766e6810c939c11d2e4abd08670f0a6394b81e59b78118236355dcf3a`；AutoDL 同哈希副本为 `/root/stiff_toi_cudagraph_20260929/bundles/v18-final-4a08dc3766e6810c.tar.gz`。v17/v18 构建与所有原始运行目录均保留，未覆盖其他对话文件。

## 100 帧计时：诊断值，非正式加速比

| 平台 | 方法 | 全程求解 s | 接触段 s | 非接触段 s | 完成 |
| --- | --- | ---: | ---: | ---: | --- |
| 本机 | base | 16.663 | 5.425 | 11.237 | 是 |
| 本机 | base+Graph | 13.146 | 5.033 | 8.114 | 是 |
| 本机 | base+TOI host | 183.389 | 111.787 | 71.601 | 是 |
| 本机 | base+TOI+Graph | 153.182 | 88.271 | 64.911 | 是 |
| AutoDL | base | 5.543 | 1.414 | 4.129 | 是 |
| AutoDL | base+Graph | 4.373 | 1.405 | 2.968 | 是 |
| AutoDL | base+TOI host | 72.820 | 46.532 | 26.288 | 是 |
| AutoDL | base+TOI+Graph | 65.815 | 38.047 | 27.767 | 是 |

AutoDL 的单次时间比：base / base+Graph 为 1.27×，接触段 1.01×、非接触段 1.39×；TOI host / TOI+Graph 为 1.11×，接触段 1.22×、非接触段 0.95×。TOI host 相比 base 明显更慢；没有通过物理质量验收前，不将这些数字解释成最终算法收益。分段原始统计在本机 `reports/boxes1920_v17_segment_diagnostics.json` 和 AutoDL `reports/autodl_v17_boxes_segment_diagnostics.json`。

## 质量检查

| 检查 | 证据 | 判定 |
| --- | --- | --- |
| TOI 接受路径连续碰撞 | 本机 host 1918 段、Graph 1881 段；AutoDL host 1873 段、Graph 1951 段，四组均零保守碰撞标记、零 FEM/ABD 翻转 | 已通过该路径检查 |
| Graph 与同后端普通求解等价，归一化质量加权最大 RMS ≤1e-6 | 本机 base+Graph 0.004504、TOI+Graph 0.004337；AutoDL base+Graph 0.003658、TOI+Graph 0.012021 | 未通过 |
| 同二进制重复波动 | 本机 base 重跑相对原 base 为 0.004470；初帧一致，差异从约 1e-15 增长，末段进入接触后放大 | 该场景不能用末帧差异单独定位 Graph 缺陷 |
| 严格线性求解下 Graph 局部等价 | `cloth_sphere7_l` 前 20 帧、PCG 容差 1e-16，归一化 RMS `6.9245e-19` | 仅限这 20 帧配置 |
| 共同物理参考 `dt,dt/2,dt/4` 和速度、能量、应变误差 | AutoDL bunny 已完成 base/TOI 跨帧分支时间步细化及较紧 base dt/8 参考；TOI dt/4 对该参考位置 1.210%、速度 35.36%，真残差/完整能量与应变误差尚未审计 | 未通过 |
| 兔子–布料–兔子 100 帧 | 默认罚参数的 AutoDL host 在第 23 帧后 CCD 候选对资源上限触发；本机小罚参数各消融在第 47–58 帧失效 | 未完成 |

`cloth_sphere7_l` 虽完成本机 100 帧且 TOI host/Graph 路径分别通过 1350/1292 段连续 CCD 检查，但官方 base 依据相同几何判据的真实接触帧为 **0/100**，故不能充当接触段加速样本。其 Graph 100 帧等价门槛也未过。盒子布料是当前完成的实际接触样本。

## TOI 停滞与修正分支

本机第 48 帧失败源于正常面积的 PT 三角形接触距离沿安全状态路径降至约 `6.67e-16 m`。该接触在活动集中持续存在，故不是简单的漏检。host、关闭摩擦、跨帧保存活动集、提前提高罚刚度等消融都没有使旧分支完成 100 帧；详细接触 ID、路径和日志见 `TOI_SINGULARITY_DIAGNOSIS.md`。

v18 新增可选 `GIPC_TOI_MU_MODE=mass`，根据公开 Robust 源码用 FEM 顶点质量与 ABD 刚体总质量估计逐接触罚刚度；默认对角估计不变。独立本机全量编译 SHA-256 `a04563aa738b4a5af2f7c8cddf1e5f27d7baa747ee748fce1844f2a47624757f`；AutoDL 上传包 SHA-256 `c82bfc05cc43d3194ff7999c78179bd4d6d4f0c086e040b86352780578accdff`，CUDA 12.8 可执行文件 SHA-256 `2a4c53cb6710d8fc182aa43c6b26b2fa21ea81aedf1a05d957443beff521d429`。两端 6 个接触与 17 个体积组件检查均通过，AutoDL 2 帧小场景正常退出。

AutoDL 的 1 倍质量罚参数兔子场景完成 **100/100 帧**，状态有限，求解时间 205.565 s（单次诊断）。这越过旧分支的完成性阻塞，但物理质量仍明显偏离：与相同初态/物理时间的官方 base 100 帧比较，质量加权位置 RMS/场景尺度 `0.04140`、速度相对误差 `0.5915`；TOI 末态有 48 个翻转四面体，而该次 base 为 0，布料最大边伸长分别为 1.3001 和 1.0404。逐帧审计在第 28 帧找到首个翻转单元 `94004`，相对初态 Jacobian 约 `-5.52e-5`；第 100 帧最差约 `-1.79`。base 是松容差对照，尚不是更细时间步严格参考；已有差异足以阻止把 v18 认作质量合格。

原独立 CPU Tight-Inclusion 检查全部 **2597 段已接受子步**，得到 `209` 个保守碰撞标记和累计 `22545` 次 FEM 四面体翻转。然而首个标记位于第 42 帧 `safe_0042_0008→0009`，点 `36807` 对三角形 `[36798,36801,36803]`；该点是**内部 FEM 顶点**，不属于原生求解器的自碰撞表面顶点集合。它与该面两个顶点共享四面体邻接，接受子步的有向距离由 `-2.958e-4 m` 变为 `1.298e-4 m`，与局部翻转一致。因此原报告的 209 个标记**不能直接解释成 209 次表面穿透**，保留为宽范围诊断。

使用同一 Tight-Inclusion、将 VF 和地面查询限定到导出表面顶点后，全部 2597 段仍有 **72 个保守 CCD 标记**，零标记门槛未过。首个表面标记在第 46 帧 `safe_0046_0022→0023`，点 `19762` 对面 `[39065,38442,39079]`；四点均为自由 FEM，互不共享四面体。有向距离从 `-3.191e-10 m` 变为 `-6.511e-11 m`，同侧且非常接近容差 `1e-9 m`；这是保守“可能碰撞”标记，**单条证据并不证明实际穿透**。完整报告在 `reports/autodl_v18_mass_bunny_surface_ccd.json`，首对几何在 `reports/autodl_v18_mass_bunny_first_surface_collision_geometry.json`。除表面 CCD 零标记门槛未过之外，翻转与物理偏差由状态审计确认，v18 不能作为质量匹配加速结果发布。

在 AutoDL 对该 PT 对单独复跑 47 帧并观察第 46 帧：122 次前后活动集观察中有 119 次处于活动集，从外层迭代 1 就加入。因此它不是简单的接触漏加。四个 FEM 顶点质量对应的初始逐接触罚刚度分别约 `11.919` 和三个 `3.906`，按最小值选用约 `3.906`；当帧全局对角估计约 `386.141`。弱罚刚度与间隙接近零相容，但 2 倍/10 倍消融又分别触发奇异/PCG 上限，不能仅凭这一对认定根因或直接提高罚刚度。时间线见 `reports/autodl_v18_watch_surface_pt_frame46.json`。

罚参数 10 倍分支在第 47 帧后 PCG 命中迭代上限；2 倍分支在第 55 帧后 PT 距离再次降到近机器精度。两者均未完成 100 帧，因此不能通过直接加大罚刚度修复。用户要求此后全部测试在 AutoDL；本机第二次延长试跑已主动停止，其约 62 帧记录不是算法失败。

为直接约束翻转，另在 AutoDL 将 1 倍质量罚参数与严格正体积限步组合：CCD 对容量 8M 时第 48 帧后需求 16.08M 而停止；独立放宽到 24M 后，第 49 帧后需求 41.00M 再次停止。两次显存峰值增量约 2.8 GiB，实际停止由显式候选对上限触发。这显示当前严格体积约束会使候选对数急剧增大，不能把单纯扩容视为已解决的质量修复。

五组消融的精确参数、完成帧数和原始受控失败文本见 `reports/autodl_v18_mass_ablation_summary.json`。本机与 AutoDL 均没有将这些失败运行计入正式加速比。

## v18 跨帧活动集：较好但仍未通过的分支

在 AutoDL 将 1 倍质量罚参数与 `--persist-contacts` 组合后，host 100 帧两次完成，末态均为 0 个翻转四面体，布料最大边伸长约 `1.052–1.062`。相对相同步长 base，第二次位置 RMS/场景尺度 `0.01735`，速度相对误差 `0.22584`。与不保留活动集的 `0.04140` 和 `0.59154` 相比有明显改善，但仍不能作为质量验收。该分支的时间均以 `--quality-only` 明确标记为诊断。

完整接受子步的独立表面 Tight-Inclusion 检查覆盖 `1404` 段，仍有 **2 个保守 CCD 标记**，累计 `1617` 次 FEM 体积翻转；Stable NH1 容忍翻转，零 CCD 标记门槛仍未过。首个表面标记为第 46 帧 `safe_0046_0017→0018`，点 `19592` 对面 `[38428,38445,39067]`；点面同侧，有向距离从 `-2.189e-10 m` 到 `-4.387e-11 m`，接近 `1e-9 m` 检查容差。它是保守近零标记，不能单凭这段认定实际穿透。见 `reports/autodl_v18_mass1_persist_bunny100_surface_ccd.json` 和 `reports/autodl_v18_mass1_persist_first_surface_collision_geometry.json`。

同一秒物理时间、同初态的 AutoDL 时间步细化结果如下。此表中的参考是官方 base `dt/4` 的 400 帧；它仍使用原 PCG 容差，属于物理趋势参照而非最终严格参考。

| 方法 | 步长与帧数 | 对 base dt/4 位置 RMS/尺度 | 速度相对误差 | 末态翻转数 |
| --- | --- | ---: | ---: | ---: |
| base | dt，100 帧 | 2.722% | 47.69% | 0 |
| base | dt/2，200 帧 | 1.232% | 26.28% | 0 |
| TOI host，质量罚参数 + 跨帧活动集 | dt，100 帧 | 2.704% | 52.41% | 0 |
| 同上 | dt/2，200 帧 | 1.578% | 41.31% | 0 |
| 同上 | dt/4，400 帧 | 1.034% | 26.47% | 0 |
| TOI+Graph，同一组合 | dt，100 帧 | 2.743% | 52.26% | 0 |

TOI 位置误差随步长细化下降，但 `dt/4` 仍略高于计划中建议的 1% 位置门槛，速度误差也高；需要更细时间步/严格线性求解参考和接触、能量、应变指标才能验收。Graph 100 帧完成，但相对 host 的归一化最大质量 RMS 为 `0.002456`，高于 `1e-6` 门槛；同二进制 host 两次也有 `0.002085` 的波动，说明不能直接把末帧差异单独归因于 Graph。Graph 该次只录制帧状态，未做全部接受子步 CCD；不据此声明 Graph 兔子路径安全。原始报告：`reports/autodl_v18_persist_bunny_dt4_vs_base_dt4_physics.json`、`reports/autodl_v18_mass1_persist_bunny_graph_equivalence.json`、`reports/autodl_v18_mass1_persist_bunny_host_repeat_equivalence.json`。

进一步在 AutoDL 以 base `dt/8`、800 帧、请求 Newton 容差 `1e-5` 与 PCG 容差 `1e-8` 建立较紧参考：800 帧完成，5942 次方向求解，零 PCG/Newton 上限。base `dt/4` 对该参考的位置差 `0.981%`、速度差 `25.89%`；TOI 跨帧 `dt/4` 则为 **`1.210%`、`35.36%`**，仍未达建议 1% 位置门槛。此次未启用逐次真残差审计，故“较紧参考”只表示输入容差和时间步更严，不宣称完全收敛参考。见 `reports/autodl_v17_bunny_base_dt4_vs_dt8strict_physics.json` 与 `reports/autodl_v18_persist_bunny_dt4_vs_base_dt8strict_physics.json`。

## 尚未满足的进入正式计时条件

1. 让兔子接触场景在与 base 可比的物理质量下完成 100 帧；v18 已完成但出现变形和翻转，需以严格时间步参考、独立连续 CCD 及物理指标确认修正。
2. 为接触场景建立共同严格物理参考，并对 Graph 使用可解释的局部同矩阵/RHS、真残差和多次基线波动评估。预注册的 `1e-6` 全程轨迹门槛未过，不能事后放宽。
3. 通过质量门槛后，再在独占 GPU 上进行配对三次计时、多个分辨率和场景的正式加速比汇总；现在的单次时间和诊断分段只说明潜在瓶颈。

## 关键复现命令（AutoDL v18 目录）

```bash
python3 tools/run_local.py --platform autodl --arm base_toi --scene bunny_cloth_bunny_l --steps 100 --name autodl_v18_mass_bunny_host100 --mu-mode mass --ccd-pair-limit 8000000 --trace --substeps --physics --diagnostic-toi --timeout 900
/root/stiff_toi_cudagraph_20260929/v17/builds/autodl-validator/validate_path runs/autodl/autodl_v18_mass_bunny_host100/trace reports/autodl_v18_mass_bunny_host100_ccd.json substeps --stable-nh1
/root/miniconda3/bin/python tools/physical_metrics.py runs/autodl/autodl_v18_mass_bunny_host100 --reference /root/stiff_toi_cudagraph_20260929/v17/runs/autodl/autodl_v17_bunny_base100_reference --output reports/autodl_v18_mass_bunny_vs_base_physics.json
```

原命令的验证器扫描全部顶点，因此其 209 个标记是宽范围诊断；表面限定版源码为 `tools/validator/diagnose_first_path.cpp`，可用相同参数复核 72 个表面标记。两版都使用独立 CPU BVH + Tight-Inclusion；`--stable-nh1` 只声明该材料允许翻转，不豁免连续碰撞。`trace/substeps` 占约 2.4 GiB；AutoDL 当前独立 v18 目录保留了原始路径。
