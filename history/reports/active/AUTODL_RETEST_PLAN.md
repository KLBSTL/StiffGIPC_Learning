# AutoDL 复测准备与场景选择

2026-10-04。**当前仅完成离线准备；未连接 AutoDL，未上传、未编译、未运行远端 GPU。最终候选须先完成本机测试，再显式导出。** 本轮准备将 `factor_inverse` 作为实验候选，保留 `triangular+warp` 参考；这不意味着默认推广或已经通过质量门槛。因子作用接口 `mas_factor_action` 在导出时仍须显式指定。

## 场景与问题

| 场景 | 选择原因 | 要回答的问题 |
|---|---|---|
| `bunny_cloth_bunny_l` | 混合模型主目标，也是当前质量和工作量难点 | TOI 是否减少整场工作量，FEM/ABD/布料是否相对 Stiff 不退化？失败也保留，不能用布料成功替代它。 |
| `cloth_hang_l` | 正向对照；已有本机收益 | 简单布料上收益是否在独占服务器重现，Graph 和 TOI 分别贡献多少？ |
| `cloth_fixed_bunny_l` | 大规模固定障碍，当前负向对照 | 稳定 MAS 优化能否改善线性成本；额外方向数量是否仍使 TOI 慢于 IPC？ |
| `cloth_sphere7_l`（备用） | 当前 TOI 相对同组件 IPC 接近持平 | 若前三场不能区分候选，再增加此场；默认不加入运行矩阵。 |

旧布料测试中部分场景使用对角预条件器、不同 dt，本轮不得使用旧秒数替代同轮 Stiff 基线。

## 固定比较矩阵

五个主组：官方 Stiff、同组件 IPC+host、IPC+Graph、TOI+host、TOI+Graph。四个活动程序组共用选定候选的稳定 Cholesky MAS、限制算子及 refit / batch / reuse 设置，仅改变接触后端和 PCG 执行方式。所有组 dt=.01，IPC/PCG 停止参数一致，TOI 两项停止参数分开记录。官方 Stiff 保留原生 MAS。

若最终选择 `factor_inverse`，可选第六组 `toi_graph_triangular`，使用 `mas_factor_action=triangular, mas_restrict=warp`，用于隔离新局部作用收益。不开启这组时不宣称已分离新旧因子作用收益。所有研究探针默认关闭，不根据远端性能修改停止条件。

| 阶段 | 默认运行数 | 单次预算 | 进入条件 |
|---|---:|---|---|
| smoke | 3场×5组=15 | 3帧，30秒 | 新鲜编译、组件及守卫检查通过 |
| window | 3场×5组×3轮=45 | 悬挂21帧、固定兔子45帧、混合兔子35帧；每次120秒 | 对应组smoke完成；无需预先质量gate，先做同窗诊断 |
| pilot | 每个通过场景5组+2次额外Stiff；最多21 | 100帧，120秒 | 三轮window完成，短窗质量及基线重复范围实际证据审核通过 |
| audit | 每个通过场景Stiff/TOI+Graph；最多6 | 100帧，120秒 | 同样要求window质量gate；独立导出速度、物理量和接受路径，不作为计时 |
| paired | 每个通过场景5组×7轮；最多105 | 100帧，120秒 | 完整pilot/audit质量、基线重复范围、负载控制证据审核通过 |

可选三角参考使 smoke / window / pilot / paired 最多增加3 / 9 / 3 / 21次；audit不增加。窗口三轮按组旋转顺序，包含速度、物理量和接受路径导出，用于质量诊断；其耗时不能混入随后关闭诊断的正式计时。GPU全程串行；重型CPU分析在GPU运行结束后进行。某组超时或失败，不延长预算、不自动重跑，后续同组重复跳过。**尚未通过短窗质量的场景不能进入任何100帧阶段。** gate是代理根据实际证据填写的审核产物，不是要求用户重复确认；没有证据就保留失败或待确认，不伪造通过文件。

## 实现与保护

- `tools/active/autodl_prepare.py`：复用公共 `config.py`；检查当前活动编译身份及冻结 Stiff 源码，按需导出源码、完整资源、明确支持的分析入口、验证器、候选配置、本机报告和独立清单。不含旧运行、下载、二进制构建目录；冻结源码路径与文件哈希。生成但不执行五个阶段的计划。
- `tools/active/autodl_linux.py`：Linux适配器复用配置解析和 `validate_run.py`，独立从源码构建 Stiff / active / CPU CCD validator。要求新目录，不复用历史对象或服务器旧二进制；保存编译命令、真实对象哈希、链接命令/响应文件、工具链、GPU与程序身份。默认构建并行度2，构建超时1800秒。
- 运行共享 `.gpu.lock`，记录 requested_config、resolved_config、GPU UUID、环境、程序哈希和实际帧数。选择明确的物理GPU，启动和运行期间检查其他compute进程；正式配对开始前两次GPU利用率必须≤5%。保留至少4 GiB磁盘启动余量与GPU内存余量；不终止其他程序，只停止自己启动的进程组。
- 官方 Stiff 无 resolved-config 接口，核验其实际 `scene.effective_run` dt / Newton / PCG 及原生 MAS；四活动组按公共运行契约核验。初态、拓扑、材料、质量、运动学及轨迹质量仍须独立分析。
- 正式计时关闭成本探针、Nsight、接受路径、物理量与速度导出。TOI普通 `phase_ms` 的全零占位不是0耗时；沿用已修正的V2语义，分项成本缺失则记为未知。
- 适配器始终输出 `performance_certified=false`。通过启动负载检查不等于完成同质量/2×认证。

### 交付入口范围与本机历史数值验收的区别

工具采用显式白名单，不再整体复制 `tools/active/*.py`：交付 AutoDL 的准备/运行/自测脚本，以及 `config.py`、`validate_run.py`、通用 `analyze.py`、`cost.py`、`compare_costs.py`。`cost.py` 所需的 `tools/summarize_cost_trace_active.py` 一并提供。通用分析入口从显式传入的远端运行路径读取数据，不能把质量报告中的 `quality_certified=false` 当作已通过。

明确不交付 Windows 专用 `build.py/run.py/batch.py`，以及绑定旧目录或旧矩阵的布料渲染/审计入口。`verify_systems.py`、`fixtures.py`、`verify_fixture_identity.py`、`cross_rhs.py`、`contact_shadow.py`、`test_contracts.py` 等历史或本机构建分析入口也不交付；因此无需为它们打包旧历史运行器或 `downloads`。未来确有需要时，应单独提供数据与依赖闭包，不能仅复制入口脚本。

**远端组件/守卫fixture不等同于本机六份历史数值系统验收。** 当前远端包不带那六份历史A/b/M输入，也不运行其数值重放；不能宣称已在AutoDL重复该验收。新因子作用候选的本机历史系统结论以本机报告为准。远端重新构建后的架构数值差异仍由组件/守卫、短测和真实轨迹检查发现；若需要同系统跨架构认证，必须另行移植六份输入及对应数值检验工具后执行。

### Linux 路径复核

官方Stiff的资源根由其CMake编译为 `sources/stiff_base/Assets/`；活动程序继承v50的定义，资源根为 `sources/stiff_perf_v50/Assets/`，不是 `stiff_active/Assets/`。两套完整资源均在导出范围内。Linux适配器还会核对真实编译命令中的 `GIPC_ASSETS_DIR`，不只依赖此静态说明。

活动目标虽然定义在 `inherited-v50` 子项目，但 overlay 将 `RUNTIME_OUTPUT_DIRECTORY` 明确设置为活动构建根，因此程序路径为 `builds/autodl-active/active/gipc`；官方基线为 `builds/autodl-active/base/gipc`。CPU验证器在 `builds/autodl-active/validator/` 输出 `validate_path` 和 `diagnose_first_path`，其 Tight-Inclusion 相对源码依赖一并交付。构建后逐个检查这些目标存在，并记录两个验证器的哈希。尚未真实运行Linux构建；这些是静态路径检查及将来的构建保护，不是远端成功证据。

## 本机候选完成后才执行的导出

以下命令中的候选和报告路径须替换为本轮完成本机测试的实际文件。本轮携带 `factor_inverse` 作为实验候选并添加 `--triangular-reference`，先进行远端smoke和短窗诊断；实际证据过线后才能运行100帧。若改选旧稳定作用可显式使用 `--factor-action triangular` 并去掉参考组。导出工具会打印实际归档文件、字节数及SHA256；包内清单记录本机报告和所用源码身份。

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/active/autodl_prepare.py --candidate-config configs/active/FINAL_CANDIDATE.json --factor-action factor_inverse --triangular-reference --inspect
& 'E:/Anaconda/envs/DL/python.exe' tools/active/autodl_prepare.py --candidate-config configs/active/FINAL_CANDIDATE.json --factor-action factor_inverse --triangular-reference --local-report reports/active/FINAL_LOCAL_REPORT.md --output packages/autodl_final_candidate.tar.gz
```

需要事先具备：Python≥3.10、CMake≥3.22、CUDA开发工具及与GPU兼容的驱动、Eigen3、GLEW、GLUT、OpenGL、nlohmann-json；分析工具还用到NumPy。旧AutoDL环境曾使用CUDA12.8 / sm89，但本轮必须现场确认，不能硬编码继承。构建脚本不安装依赖或修改服务器配置。

## 用户打开 AutoDL 后的顺序

先检查磁盘和GPU负载，将归档解压到用户数据盘的新专用目录，核对归档 SHA256；避免历史系统盘满的问题。以实际GPU能力填写 `--arch`，下例的89只是旧机器示例。

```bash
python3 tools/active/autodl_linux.py verify
python3 tools/active/autodl_linux.py build --gpu 0 --arch 89 --jobs 2
xvfb-run -a python3 tools/active/autodl_linux.py run --gpu 0 --stage smoke
xvfb-run -a python3 tools/active/autodl_linux.py run --gpu 0 --stage window
```

GLUT仍需显示环境，使用已有Xvfb；如果已有有效DISPLAY则可直接运行Python。组件/守卫fixture不创建窗口。远端Linux实际编译、GLUT/驱动兼容、fixture与运行尚待验证。

window和audit轨迹须核验每帧safe编号连续、首末状态分别等于该帧导出端点，再运行构建的 `builds/autodl-active/validator/diagnose_first_path TRACE REPORT substeps --stable-nh1`，核对实际检查段数。不得把少量safe文件或零标记单独当作覆盖完整。

在三次远端Stiff重复范围基础上，分别审查布料拉伸、FEM J/负体积、ABD及固定点漂移、位置和速度、接受路径CCD、PCG异常；轨迹差异超出自身重复范围记为待确认，不自动放宽质量门槛。混合模型不得用整体RMS掩盖局部问题。

质量/负载审核文件仅对通过场景设置真值，包含相同的 `source_digest`、`candidate_sha256`，以及实际证据文件的路径和SHA256。每个场景记录 `quality_passed`、`baseline_repeat_range_frozen`、`reviewed_stage`、非空 `evidence` 数组（每项 `path` / `sha256`）。短窗审核用 `reviewed_stage=window`，只允许通过的场景进入pilot/audit；未过场景继续停留在短窗诊断。完整pilot/audit审核后才用 `reviewed_stage=pilot_audit`，且正式七对还要求 `load_controlled=true`。不提供预先填好的“全部通过”文件。代理完成实际短窗证据审核后才运行：

```bash
xvfb-run -a python3 tools/active/autodl_linux.py run --gpu 0 --stage pilot --gate reports/active/AUTODL_WINDOW_REVIEWED_GATE.json
xvfb-run -a python3 tools/active/autodl_linux.py run --gpu 0 --stage audit --gate reports/active/AUTODL_WINDOW_REVIEWED_GATE.json
```

再完成完整场景与负载审核后才运行七对；window gate不能用于授权此阶段：

```bash
xvfb-run -a python3 tools/active/autodl_linux.py run --gpu 0 --stage paired --gate reports/active/AUTODL_REVIEWED_GATE.json
```

正式报告先在每轮计算Stiff/候选、IPC同执行/TOI同执行、host/Graph，再报告七轮配对中位数及单侧95%配对自助法下界。仅所有质量与负载要求成立、融合相对Stiff的中位数和下界均≥2×，才可宣称达到原目标。300帧稳定性仍是之后单独预声明的验收，不在本次默认矩阵中自动展开。

## 离线验证

已执行 `python tools/active/autodl_selftest.py`，验证阶段预算、唯一运行名、五组单变量关系、诊断隔离、三次基线、轮次旋转、显式因子候选/三角参考、证据门禁、交付工具本地导入闭包与Linux资源/程序/验证器路径约定。已执行 `py_compile`、CLI帮助及inspect模式。这些离线自测不运行远端构建、GPU或SSH；归档由独立导出动作产生。远端脚本只有离线契约验证，不能据此声称AutoDL验证通过。
