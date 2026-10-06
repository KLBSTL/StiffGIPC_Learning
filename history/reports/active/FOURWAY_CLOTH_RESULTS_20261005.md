# 两个布料场景的四配置加速比（2026-10-05）

本机 **RTX 3070 Laptop**，每组从零连续 **100帧、dt=.01**，每场景四组各三轮交错，共24次计时、2400物理帧；另8次两帧启动检查。加速比取每轮 `base秒数/该组秒数` 的三轮中位值，时间列为三轮求解耗时中位，二者不是强制用中位时间相除。启动和求解区间外导出不计入，重型诊断/分析/渲染在计时组中关闭。

这是共享桌面诊断结果，不是受控GPU、七对统计下界或同质量性能认证。此次没有使用AutoDL，不能与先前4090的21/45帧结果直接比较。

**本轮两个场景都是Graph-only最快；加入TOI后没有胜过Graph-only。** TOI的布料拉伸高于原版，继续保留先前用户选择的默认开关，但不据本轮宣布质量修复或最终2×通过。

|场景|base|base+CUDA Graph|base+TOI|base+CUDA Graph+TOI|
|---|---:|---:|---:|---:|
|悬挂布料 cloth_hang_l|1.000×|1.469×|1.028×|1.277×|
|固定兔子布料 cloth_fixed_bunny_l|1.000×|1.317×|0.472×|0.526×|

## 配置定义

base使用冻结的原版StiffGIPC；Graph-only为当前活动程序的IPC＋conditional PCG Graph，仍用legacy MAS。TOI host/Graph两组保留当前稳定Cholesky MAS、默认factor_inverse、默认world_block罚尺度、初值选择与safe重启守卫。restriction均为当前默认serial，**未开启warp**。四组关闭refit、能量批处理和energy reuse，Graph不是整个TOI外循环的捕获。

因此TOI相对原版base的总收益包含所需稳定MAS差异，不能称纯TOI算法收益；TOI host与Graph使用同一程序、同一算法配置，只有PCG执行方式不同。IPC Graph与冻结base是两个程序，使用一致输入和物理参数并核验legacy/host或Graph实际执行，未将软件身份差异隐藏。所有材料、ABD/FEM/布料、完整CCD和原停止条件保留：IPC/TOI剩余量=.01、trial速度=.05 m/s、PCG rho=1e-4。

当前活动程序SHA256：`57782cfeabdc35eb3151187a0f466911ef7b2ecf831b13ecc60ef221385b75f6`。每次运行保留requested/resolved配置、实际执行模式、源程序/链接身份及GPU采样。原版没有新resolved接口，其清单、有效场景参数和观测host执行单独核验。

## 耗时、工作量与组件收益

### 悬挂布料 cloth_hang_l

|配置|三次耗时 s|时间中位 s|配对加速比|PCG中位|方向数中位|
|---|---|---:|---:|---:|---:|
|base|5.032, 4.513, 4.675|4.675|1.000×|14022|442|
|base+CUDA Graph|3.386, 3.183, 3.183|3.183|1.469×|14022|442|
|base+TOI|4.722, 4.390, 4.625|4.625|1.028×|12457|340|
|base+CUDA Graph+TOI|3.834, 3.548, 3.662|3.662|1.277×|12473|340|

Graph在TOI内的配对中位收益：**1.237×**。在已经有Graph时，TOI相对IPC的配置收益：**0.883×**（小于1即更慢）。二者不以不同轨迹的迭代总量冒充同一线性系统的算子收益。

### 固定兔子布料 cloth_fixed_bunny_l

|配置|三次耗时 s|时间中位 s|配对加速比|PCG中位|方向数中位|
|---|---|---:|---:|---:|---:|
|base|14.240, 14.172, 14.328|14.240|1.000×|37629|592|
|base+CUDA Graph|10.507, 10.759, 11.212|10.759|1.317×|37770|593|
|base+TOI|28.590, 31.578, 30.366|30.366|0.472×|42102|793|
|base+CUDA Graph+TOI|27.827, 26.943, 26.371|26.943|0.526×|43098|791|

Graph在TOI内的配对中位收益：**1.151×**。在已经有Graph时，TOI相对IPC的配置收益：**0.399×**（小于1即更慢）。二者不以不同轨迹的迭代总量冒充同一线性系统的算子收益。

悬挂TOI host相对base的方向数减少23.1%、总PCG减少11.2%，但整场只获得约1.03×配置收益。固定兔子则方向数增加34.0%、总PCG增加11.9%，整场明显慢化。工作量和预条件器路径都必须计入；本轮没有单独隔离稳定MAS成本，不能把2.1倍慢化全部归给某一个kernel或全部归给PCG次数。

![耗时与布料拉伸](figures/fourway_cloth_timing_quality_20261005.png)

## 质量与限制

24组计时均完整完成100帧，所有导出位置有限，无PCG触顶或breakdown。各组初态、拓扑、质量、固定点及场景/材料参数核对一致。只有执行与输入检查通过，完整物理质量尚未认证。以下为接受的物理帧端点指标；纯布料无体积FEM，不把其占位J=1当成体积质量证明。

### 悬挂布料 cloth_hang_l

|配置|三轮最大布料边长比范围|最大固定点/物体漂移 m|严格端点重复范围通过次数|
|---|---|---:|---:|
|base|1.12887619–1.12887623|0|基线范围来源|
|base+CUDA Graph|1.12887619–1.12887620|0|0/3|
|base+TOI|1.13172608–1.13172608|0|0/3|
|base+CUDA Graph+TOI|1.13172608–1.13172608|0|0/3|

原版自身三对位置差的最大布料RMS：0.008178 mm。

|配置|三轮对原版的最大布料位置RMS mm|超出原版位置重复范围的帧数（逐轮）|
|---|---|---|
|base+CUDA Graph|0.010055, 0.008217, 0.008044|92, 75, 42|
|base+TOI|15.567588, 15.567627, 15.567633|100, 100, 100|
|base+CUDA Graph+TOI|15.567647, 15.567596, 15.567595|100, 100, 100|

![实际网格：hang](figures/fourway_cloth_hang_meshes_20261005.png)

### 固定兔子布料 cloth_fixed_bunny_l

|配置|三轮最大布料边长比范围|最大固定点/物体漂移 m|严格端点重复范围通过次数|
|---|---|---:|---:|
|base|1.03557094–1.03575319|6.21e-17|基线范围来源|
|base+CUDA Graph|1.03542725–1.03570530|5.72e-17|0/3|
|base+TOI|1.04038251–1.04041992|6.21e-17|0/3|
|base+CUDA Graph+TOI|1.04007570–1.04035421|6.21e-17|0/3|

原版自身三对位置差的最大布料RMS：25.416258 mm。

|配置|三轮对原版的最大布料位置RMS mm|超出原版位置重复范围的帧数（逐轮）|
|---|---|---|
|base+CUDA Graph|24.651301, 26.060889, 21.746038|38, 55, 59|
|base+TOI|112.479145, 106.832718, 100.233733|100, 100, 100|
|base+CUDA Graph+TOI|100.856759, 86.627418, 109.092534|100, 100, 100|

![实际网格：fixed_bunny](figures/fourway_cloth_fixed_bunny_meshes_20261005.png)

位置差不是物理真值误差。严格重复范围外的微小数值差异记为待确认；显著拉伸变化单列，不自动放宽门槛。图像采用第一轮真实网格、共同最终/峰值帧、相同相机/尺度/拉伸颜色；视觉相似也不能代替质量门禁。

Graph-only的悬挂逐帧边长比越界最大仅6.74e-7，最大位置RMS约10.05µm，原版自身约8.18µm；不将严格范围0/3直接表述为明显物理退化。固定兔子自身后期位置分岔已达25.42mm，Graph-only约26.06mm，质量等价仍待确认。TOI的悬挂位置差约15.57mm，固定兔子约87–112mm，且最大拉伸增加，不能继承Graph-only的近似形态作为同质量证明。

三张图已逐张视检，实际网格、相同尺度/相机、峰值与第100帧显示正常，计时图的重复范围和标注可读。

本轮没有导出/认证实际速度，没有追加接受子步的独立CPU CCD；原版历史CCD结果不能继承给新轨迹。因此这些数字只能称“该配置的诊断加速比”，不能称同质量加速或最终2×验收。

## 复现与证据

在 `E:/university_class/ComputerGraphics/GIPC`，使用 `E:/Anaconda/envs/DL/python.exe`：

```text
stiff_toi_cudagraph_20260929/tools/active/prepare_fourway_cloth.py
stiff_toi_cudagraph_20260929/tools/active/batch.py --plan configs/active/fourway_cloth_smoke_20261005.json
stiff_toi_cudagraph_20260929/tools/active/batch.py --plan configs/active/fourway_cloth_timing_20261005.json
stiff_toi_cudagraph_20260929/tools/active/analyze_fourway_cloth.py --output reports/active/FOURWAY_CLOTH_RESULTS_20261005.json
stiff_toi_cudagraph_20260929/tools/active/visualize_fourway_cloth.py
stiff_toi_cudagraph_20260929/tools/active/report_fourway_cloth.py
```

已有运行名/输出不可覆盖，复现须修改新plan中的运行名及报告名。完整记录：[预声明协议](FOURWAY_CLOTH_PROTOCOL_20261005.md)、[启动检查](FOURWAY_CLOTH_SMOKE_GATES_20261005.json)、[24组batch](FOURWAY_CLOTH_TIMING_BATCH_20261005.json)、[全部配置/几何/逐帧统计](FOURWAY_CLOTH_RESULTS_20261005.json)、[视觉记录](FOURWAY_CLOTH_VISUAL_20261005.json)。
