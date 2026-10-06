# Robust 可信供体隔离执行：v32

状态：Task A 已完成。真实 CUDA RED/GREEN、完整原版/修正版独立构建、两档 30 帧修正前后对照均已完成并保存身份与原始状态。未执行独立碰撞质量验收，未宣称速度资格。

## 范围与身份

- 所有源副本、构建和运行产物位于 `experiments/robust_reference_v32`；供体与其他任务目录只读。
- 公开提交：`389f8a52a29606ccaf5640607dd1e38987500988`。供体已有 `apps/AL_examples/animal_well/main.cpp` 修改已记录，未把 dirty worktree 当作纯提交。
- 公开源冻结 2,243 文件 / 32,276,569 字节；历史源冻结 2,265 文件 / 32,522,415 字节；旧 EXE/DLL 冻结 19 文件 / 42,210,816 字节。
- `FREEZE_MANIFEST.json` 保存全部逐文件 SHA-256。源快照排除 `.git`、assets、output、artifacts 与 Python cache；全部 src、include、CMake 和 bundled dependency 源文件保留。
- 公开/历史的 `al_contact_function.h`、`friction_utils.h`、`codim_ipc_contact_function.h`、`al_vertex_half_plane_contact_function.h` 逐字节一致。AL 头 SHA：`5683d5e4e44be5cf28e1cc01d947dc5be7b12aca0cb3d164462133de0f8d8dc2`。

## 实际函数回归

`friction_regression.cu` 直接包含冻结供体头，在 CUDA kernel 中调用实际 `PT_friction_energy` / `PT_friction_gradient_hessian`、`EE_friction_energy` / `EE_friction_gradient_hessian` 与 half-plane 对应函数。未转写公式，未移除 device qualifier。

9 个固定输入覆盖 PT / EE / ground 的非平滑区、平滑区与零运动。检查能量中心差分梯度、梯度中心差分 Hessian、对称与 PSD、刚体平移、几何零空间与非正摩擦功。中心差分只扰动当前坐标，previous geometry、normal force、mu、eps_vh 固定。非平滑步长 1e-7 m，其余 1e-9 m；梯度相对阈值 1e-6，Hessian 相对阈值 5e-5。零梯度使用绝对误差阈值，避免除零。

已执行并通过编译：

```powershell
& './experiments/robust_reference_v32/build_regression.ps1' -Variant original
```

已执行 GPU RED / GREEN：原版 raw exit 1，3 个 EE 用例失败，6 个 PT/ground 控制用例通过；修正版 raw exit 0，9/9 通过。两个 EXE 使用完全相同测试源（SHA `f0c2f3618760ffae9f9c6ecb497f35098818d543c62837579d4c54cd7be82d83`）。固定输入的非平滑 EE 梯度误差从 0.4625730204 降至 1.6148959e-10。

| EE 区域 | 原版梯度误差 | 修正版梯度误差 | 原版 Hessian 误差 | 修正版 Hessian 误差 |
|---|---:|---:|---:|---:|
| 非平滑 | 0.4625730204 | 1.6149e-10 | 0.4625730203 | 2.2319e-10 |
| 平滑 | 0.4625730268 | 2.3516e-8 | 0.4625730210 | 2.1206e-8 |
| 零运动 | 绝对 1.9085e-12 | 绝对 1.9085e-12 | 0.4625753335 | 3.2166e-6 |

Hessian 误差是实际 H 对实际 G 的有限差分导数，原版 G 本身已经与能量不一致；与上一报告的“两套 J 对同一二维解析 Hessian 的 sandwich 差异”不是同一个量。零梯度采用绝对误差判据；stdout 同时保留接近零分母下的诊断相对值，不能只据该相对值判断失败。

`test-results/{original,corrected}-first` 保存 raw exit、stdout、stderr、command、cwd 与 test/header/EXE SHA。为保留当时正在编译的完整基线，GREEN 先使用独立 `corrected_headers` 副本。基线完成并冻结后，将同一个修正施加于完整源；两份修正头 SHA 均为 `262f1026090ba417b3eadd45c04a8318482580f958dedc40308f97e4767c10d0`，只有 `point_triangle_jacobi(basis, gamma, J)` → `edge_edge_jacobi(basis, gamma, J)` 一处替换。

## 独立构建

旧 donor build cache 指向原机器 `D:/code_2` 和 CUDA 12.6；当前隔离工程使用 CUDA 13.0.88、MSVC 19.44.35222、C++20、sm86、Ninja Release。

完整 configure 已成功：

```powershell
& './experiments/robust_reference_v32/build_standalone.ps1' -ConfigureOnly
```

隔离 build glue 仅改变：顶层 CMake 使用旧已安装 vcpkg dependencies 只读、跳过 manifest/install；`stiff4_cloth_cases/CMakeLists.txt` 使用原资产绝对路径。全局 toolchain 被此调用显式置空，developer mode 避免 donor submodule 更新。供体 build tree 未重新配置。

完整未修正 engine / runner 构建已通过（exit 0，491/491 tasks，205 CUDA translation units），通过 `freeze_built_runtime.py --variant original` 冻结 14 个新 runtime 文件。随后修改单一 Jacobian 调用，5/5 项增量重建与链接通过（exit 0），再冻结 14 个修正版 runtime 文件。原始历史 DLL 与此次 CUDA 13 重建原版保留独立身份。

两组新 runtime 中，只有 `uipc_backend_cuda.dll` 的 SHA 改变，scene runner、core、geometry、constitution、io、none backend、sanity check 与第三方 DLL 均逐字节一致。原版 backend SHA 为 `32f2699ec4aadaeda9845bfa5fca065874957d199e4e0813348117c6b5e3b44c`；修正版为 `6ffbaaa4852b90996f13b78137f6fd8a10ab58a85770e7313ef7aac61f54bb5b`。

旧场景 runner 已支持 `STIFF4_ROBUST_STATE_EXPORT=1`，每帧 `world.retrieve` 后导出位置、拓扑与 fixed flags。新 `run_standalone.py` 会从冻结运行副本启动，把工作目录、engine workspace、输出放在本实验下；检查每帧覆盖、CSV、有限状态与文件长度。逐帧导出诊断运行不作为计时对照。

## 同参数修正前后场景对照

RTX 3070 Laptop、driver 581.57、8 GiB；开跑前 6,372 MiB VRAM 与 17.02 GiB / 31.84 GiB RAM 空闲。RAM 的第一次 WMI 查询被沙箱拒绝，按环境要求获准在沙箱外重试后取得上述读数。主任务明确授予串行 GPU 时隙，顺序为原版 0.05 → 修正版 0.05 → 原版 1.0 → 修正版 1.0；运行完成后释放 GPU。

布落球 `cloth_sphere7_l`，dt=0.01 s；每档同参数 30 帧、各一次；共 120 帧。两个 pair 的完整有效配置、材料/初始 mesh、初始 surface 与 topology 逐字节一致。PCG 为 `fused_pcg`、tol_rate=1e-3；Newton min_iter=6、max_iter=1024；累计 TOI threshold=0.001，decay=0.9，d_hat=0.0025029782260339384 m。旧 donor 控制中的 `E<=E0+1e-12 || newton_converged` 保持原样；该参考并非主任务的严格能量接受派生路径。

每次输出 30 帧 native retrieved surface，3,083 顶点、5,960 三角形、2,219,760 bytes 状态；全部有限，CSV 与帧标记完整、CG log 数与方向数对应。状态导出与 `world.retrieve` 开启，因此此处 raw 时间只作诊断，单次结果不能作为计时验收。

| velocity_tol | 原版 / 修正版方向数 | 原版 / 修正版 CG | 原版 / 修正版 advance s | 第 30 帧布料 RMS 差 / mm | 布料最大差 / mm |
|---|---:|---:|---:|---:|---:|
| 0.05 | 114 / 114 | 3,850 / 3,820 | 1.772554 / 1.961684 | 0.252143 | 4.015434 |
| 1.0 | 55 / 55 | 2,060 / 2,065 | 1.122955 / 1.111878 | 2.220363 | 18.090509 |

这些小窗口中，两条运行的 Newton 方向数相同，CG 数改变较小；布料轨迹存在毫米到厘米量级的局部差异，旧未修正轨迹不能作为修正版的同状态目标。未统计场景中错误 EE 调用的触发次数，也未将轨迹差异解释为物理误差。每组仅一次运行，未独立估计相同 binary 重复运行的 CUDA 非确定性，因此不把全部差异量归因于单行修正。

**布料 mask 必须使用显式映射。** native merged `surface_fixed.i32` 全为 0，未反映 fixed ABD sphere instance，不能据其把全部 3,083 顶点当作布料。通过初态精确坐标匹配，2,601 个布料顶点一一映射至 surface indices 482–3082，最大坐标距离为 0；5,000 个 cloth triangles 也与 surface topology 匹配。两档 `cloth_surface_indices_v*.i32` 和 `SCENE_PATCH_COMPARISON.json` 保存映射与 SHA。表中的误差仅取这些布料顶点；其余 482 个 sphere surface 顶点修正前后所有帧差为 0。

`advance_seconds` 是 native `advance_al` pipeline、同步与计时报表前 bookkeeping 的窗口；逐帧 retrieve/二进制导出在其后进行，但运行仍有诊断插桩。没有跨引擎速度比较、独立 CCD 或质量资格。

## 最终复核与解释限制

```powershell
& 'E:/Anaconda/envs/DL/python.exe' experiments/robust_reference_v32/verify_reference.py
& 'E:/Anaconda/envs/DL/python.exe' experiments/robust_reference_v32/compare_scenes.py
```

复核通过：4,527 个 donor/snapshot 文件身份保持一致；完整修正版源仅有 3 个文件变化（2 个 CMake build glue、1 个 EE 调用），无额外源文件；真实 CUDA 回归 stdout、原版/修正版 raw exit、EXE/test/header SHA 与测试的修正头一致。两组新 runtime 逐文件核对，只有 CUDA backend 变化；两档 scene config/初态/拓扑一致，120 帧导出完整。

既有 CPU 转写公式的结果与此次真实 CUDA 执行保留独立证据。函数差分通过不证明场景碰撞质量、修正影响频率或速度资格。历史 DLL 与当前源的编译对应关系未由旧 SHA 单独证明；此次重建原版由新的 runtime manifest 记录。完整 engine 的材料/生命周期仍沿用历史 donor，主任务的 Stiff 适配需要独立门槛。

主要产物：`FREEZE_MANIFEST.json`、`REFERENCE_VERIFICATION.json`、`test-results/{original,corrected}-first/*`、`runtimes/rebuilt_{original,corrected}/RUNTIME_MANIFEST.json`、`SCENE_GPU_PREFLIGHT.json`、`SCENE_PATCH_COMPARISON.json`、`scene-results/*-30f-*/`。源码/运行/回归脚本及说明位于本实验目录。供体与其他任务没有源码修改、构建重配置或 Git 提交。
