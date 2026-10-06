# FP64 因子逆局部作用：本机实现、测试与 AutoDL 复测交付

2026-10-04。**本轮实现与本机测试已完成；候选有明确执行收益，但未通过轨迹质量门槛，不默认推广，不启动100/300帧长测。AutoDL只完成离线准备，等待用户开启实例。原混合兔子同质量2×目标仍未实现。**

## 实现与结论

保留现有FP64 Cholesky因子L，预计算B=L^-1，以Bᵀ(Br)替代每次PCG中的顺序三角代入。未形成低精度显式逆矩阵，未改变材料、ABD/FEM/布料、停止参数、CCD或迭代上限。新开关为 `mas_factor_action=factor_inverse`；程序默认仍为 `triangular`。两个比较臂均使用上一版 `mas_restrict=warp`，因此本轮隔离的是局部因子作用收益。

实际局部维度为48（BANKSIZE=16），每块增加18,432字节FP64因子缓冲。Graph签名包含作用模式和B缓冲地址；扩容、重捕获、异常因子处理和生命周期已接入。一次PCG内模式固定，不静默回退。

| 判断 | 当前证据 |
|---|---|
| 已证明：测试范围内固定算子检查通过 | 六历史系统、五新系统；CPU参考、B/L关系、R/P结构、对称/正二次型探针通过；默认关闭时66份旧缓冲逐位一致 |
| 已证明：本机诊断范围有执行收益 | 五个同A/b系统PCG约1.37–1.46×；含准备成本的线性窗口1.50×/1.53× |
| 已排除：本轮候选可直接推广为同质量加速版本 | 固定兔子布料逐帧拉伸越界；混合体FEM J、负体积及非正单元出现旧重复范围外变化 |
| 待验证 | 同GPU独占性能、AutoDL架构复现、完整100帧质量、七对置信下界和300帧稳定性；本轮均未认证 |

## 构建身份

新程序SHA256：`2ad9c03b4bd38bab48742397fb48c9d52ac8057a14328e22e32a54003d409c80`。

干净编译37个单元成功，但CUDA 13的MSBuild `GenerateDeps` 扫描器把5个单元的依赖写成了冻结层头文件。没有忽略该冲突：新增active PCG头/源的明确包含路径，并用记录的实际NVCC命令重新生成依赖。NVCC确认这5个单元读取的是active头；保留原始tlog、衍生命令、fresh依赖和哈希。若fresh依赖仍绕过active，构建继续拒绝。

最终验证构建记录：`builds/active/provenance/20261004T145849_519480Z_factor_action_verified/manifest.json`。后续无改动构建记录：`20261004T150443_710925Z_factor_action_noop/manifest.json`，重编对象数0、程序哈希不变。冻结历史未修改。

## 数值检查与固定系统

两种作用模式各通过43组件、15守卫及六历史fixture。新模式覆盖60个历史输入向量与CPU作用核对、host/Graph重放、缓冲增长与缓存签名、零RHS、异常pivot和非有限B；主缓冲故障探针前后不变。六份历史A/b准确CPU参考的最大真残差为4.6853e-11，低于1e-8；五份新系统最大为5.6761e-13。该门槛约束准确参考，不是默认PCG的停止阈值。

固定系统来自连续推进的f2_n1、f25_n1、f34_n1、f25_n7、f25_n8。每系统host/Graph各三对交错、每臂预热，A/b/M保持不变，原生产解逐位恢复。60个导出解的独立CPU真残差与GPU报告一致。

| 系统 | host配对中位比 | Graph配对中位比 |
|---|---:|---:|
| f2_n1 | 1.396× | 1.378× |
| f25_n1 | 1.384× | 1.431× |
| f34_n1 | 1.393× | 1.369× |
| f25_n7 | 1.398× | 1.458× |
| f25_n8 | 1.371× | 1.427× |

这些比值仅为同系统PCG，尚未包含生成B的成本。前四个系统新旧解差约1e-14或更小。困难f25_n8需要单独说明：Graph旧PCG真残差为0.02056–0.02308，新为0.02373–0.03582；迭代221变为222–223，解重复范围也扩大。但新解到准确CPU参考的相对误差为6.412–6.502%，旧为6.544–6.563%，三次新解均稍更接近准确解。不能把残差增加直接等同于前向误差恶化，也不能据此宣布轨迹无退化。默认rho仍为1e-4，没有全局收紧或放宽。

因此在固定算子检查与成本门槛通过后，继续了预声明短窗诊断；没有把它当作数值等价或正式质量验收。短窗后来确实发现了轨迹质量越界，依赖长测停止。

证据：`FACTOR_DEFAULT_IDENTITY.json`、`FACTOR_CPU_HISTORICAL.json`、`FACTOR_CPU_WINDOWS.json`、`FACTOR_CPU_DIFFICULT.json`、`FACTOR_FIXED_PAIRS.json`、`FACTOR_PAIR_RESIDUALS.json`、`FACTOR_FIXED_CONFIG_CHECKS.json`。

## 含准备成本的线性窗口

混合兔子从0连续推进35帧，跟踪1–3、24–26、33–35帧。顺序为旧1、新1、新2、旧2；未开启重型算子探针。

| 配对 | 旧线性累计 | 新线性累计 | 旧/新 | 每PCG迭代归一化比 | 新B准备成本 |
|---|---:|---:|---:|---:|---:|
| 1 | 10.2047s | 6.8144s | 1.4975× | 1.3503× | 86.85ms |
| 2 | 11.0262s | 7.2038s | 1.5306× | 1.3720× | 87.26ms |

总线性成本下降33.2%/34.7%，超过预设15%门槛。B准备已经包含在累计值中。窗口系统数分别79→75、82→79，PCG次数12004→10824、12786→11461；因此累计收益也包含轨迹/工作量变化，不能全部归因给kernel。CUDA事件区间含host间隙和诊断开销，只作为诊断。普通TOI `phase_ms` 全零仍是占位，分析器报告为不可用。

证据：`FACTOR_COST_PAIR1.json`、`FACTOR_COST_PAIR2.json`、`FACTOR_COST_BATCH.json`、`FACTOR_COST_CONFIG_CHECKS.json`。

## 三场景短窗口：27次交错计时

本机RTX 3070 Laptop，共享桌面负载；dt=.01。每场Stiff、旧TOI、新TOI三组，各三轮旋转顺序；关闭重型诊断。全部从零推进，同场景初态/材料/拓扑相同。下表为各臂耗时中位数和同轮配对比值中位数；二者不能直接互除替代。

| 场景 / 帧数 | Stiff中位 | 旧TOI中位 | 新TOI中位 | 相对旧TOI | 相对同轮Stiff |
|---|---:|---:|---:|---:|---:|
| 悬挂布料 / 21 | 0.8744s | 0.4863s | 0.4076s | 1.2435× | 2.2212× |
| 固定兔子布料 / 45 | 5.1995s | 6.2388s | 5.0087s | 1.2149× | 1.0352× |
| 混合兔子 / 35 | 5.8665s | 20.2659s | 14.7342s | 1.3684× | 0.3761× |

**不是100帧加速比，不与上一轮100帧秒数混用。悬挂的2.22×不是原混合目标达标。** 混合体仍比Stiff慢约2.66倍（配对耗时比倒数），PCG中位21058对3654，工作量约5.76倍；方向中位171对142。优化局部执行后，求解工作量和物理质量仍是主障碍。

## 质量判定：未通过

27组全部有限，无PCG触顶或breakdown。固定兔子新候选固定点漂移为3.10e-17–6.21e-17米。但这不抵消下述退化。

| 指标 | Stiff三重复 | 旧TOI三重复 | 新TOI三重复 |
|---|---:|---:|---:|
| 悬挂峰值边长比 | 1.12887618–1.12887621 | 1.131726081115378–1.131726081115381 | 1.131726081115376–1.131726081115379 |
| 固定兔子布料峰值边长比 | 1.034944–1.035106 | 1.040037–1.040296 | 1.040295–1.040425 |
| 混合FEM最小J | 0.49786891–0.49786908 | −0.0146204 至 −0.0054648 | −0.0532178 至 −0.0056250 |
| 混合峰值负体积 | 0 | 2.547e-10–4.456e-9 | 2.622e-10–1.622e-8 |

悬挂相对旧TOI的标量范围通过，新旧位置最大RMS仅2.61e-14–2.90e-14米（旧自身上界1.93e-14米）；严格位置范围仍标待确认，不放宽门槛。相对Stiff的既有峰值伸长差继续存在。

固定兔子布料的新候选三次分别有5/6/4帧拉伸超过旧TOI逐帧范围，最大超出2.301e-4。新旧位置最大RMS为2.026–3.588mm，旧重复上界2.067mm；相对Stiff约33.38–33.54mm，Stiff重复上界1.806mm。

混合体r1/r3在f35的J分别−0.03304/−0.05322，超过旧TOI最差范围；r3在f34新增一个非正FEM单元，而旧TOI三次该帧均为0。不能只看“峰值非正数量都为1”。这里仍是相对基线范围检查，没有新加“绝对禁止FEM翻转”的物理要求。

| 混合分组位置最大RMS，mm | 新对Stiff | Stiff自身重复上界 | 新对旧TOI | 旧TOI自身重复上界 |
|---|---:|---:|---:|---:|
| 布料 | 1.108–1.520 | 0.009748 | 0.656–1.068 | 0.2344 |
| FEM | 15.909–16.109 | 0.001065 | 0.662–1.273 | 0.6489 |
| ABD | 0.968–1.380 | 0.000190 | 0.656–1.068 | 0.2345 |

位置差异不是物理真值误差，但拉伸、J、负体积及非正单元已足以阻止推广。计时组未导出速度，未用有限差分伪造速度质量证明。纯布料不存在体积FEM，其J=1是缺失组占位；固定兔子没有动态ABD位置组，固定点单独检查。

另6次接受路径诊断共661个真实接受段，加196个帧间静止连接，独立CPU CCD实际检查857条路径，零标记；首尾与帧状态逐位匹配，编号连续，无ABD翻转或非有限值，固定点检查通过。这些是独立运行，不能代替计时轨迹的全部物理验收。

证据：`FACTOR_WINDOW_QUALITY_TIMING.json`、`FACTOR_WINDOW_CCD.json`及各CCD子报告；`FACTOR_WINDOW_BATCH.json`与`FACTOR_WINDOW_AUDIT_BATCH.json`记录原始请求与运行身份。

三张实际网格图已视检（`figures/factor_window_hang.png`、`factor_window_fixed_bunny.png`、`factor_window_mixed.png`），共用相机、帧、尺度和色标。悬挂新旧外观一致，固定兔子褶皱/接触发展有差异；混合图的布料色标只放大了1附近约1e-7的变化，不能把颜色理解为大拉伸，内部FEM质量以数值报告为准。

## AutoDL：选定场景与进入条件

保留三个有价值场景：混合兔子（主目标、困难系统和FEM退化），悬挂布料（执行收益正对照），固定兔子布料（大固定障碍、局部质量差异）。落球仅备用，不扩大矩阵。

`configs/active/factor_retest_candidate.json` 指定的是**待复现的实验候选，不是通过验收的推荐配置**。导出包含新旧作用开关，增加同组件TOI Graph triangular参考臂；默认C++仍是triangular。

用户开机后先确认磁盘/GPU/CUDA，使用数据盘新目录从源码构建，再做组件/守卫、3帧smoke以及21/45/35帧三轮window。窗口组为Stiff、IPC/TOI×host/Graph、TOI Graph旧作用；window含速度/物理/接受路径诊断，时间仅作诊断。所有100帧阶段均需实际质量证据和冻结重复范围；七对阶段另需完整场景及负载证据。当前没有预填通过gate，不能绕过本轮失败直接跑100/300帧。

包的本机历史系统结论不等于远端同系统认证：没有把六份历史A/b/M输入放进包。远端实际Linux编译和架构兼容也尚未运行。运行流程见 `AUTODL_RETEST_PLAN.md`。本轮没有SSH、上传、远端清理或远端GPU运行。

## 下一步方向

1. 保留新算子为显式诊断开关，暂停推广和新kernel网格；默认稳定三角路径保持。
2. 优先对混合f25到f34/35的接触推进、默认PCG误差传播和FEM状态做已有轨迹/保存系统的因果核查。不能把一次PCG的参考误差改善直接等价为更好的非线性轨迹。
3. AutoDL打开后先跨架构重现上述三个短窗，区分执行收益是否可复现、质量越界是否仍在。质量门槛没有通过之前，维持完整验收停止状态。

## 实际验证命令

工作目录为 `E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929`，Python为 `E:/Anaconda/envs/DL/python.exe`。以下为本轮已执行入口；重复实验需预先改用新的输出名称，工具不会覆盖证据。

```text
tools/active/build.py build --label factor_action_verified
tools/active/build.py build --label factor_action_noop
tools/active/fixtures.py --name factor_tri_fixtures --mas-restrict warp --mas-factor-action triangular
tools/active/fixtures.py --name factor_inverse_fixtures --mas-restrict warp --mas-factor-action factor_inverse
tools/active/verify_fixture_identity.py runs/active/restrict_final_fixtures runs/active/factor_tri_fixtures --output reports/active/FACTOR_DEFAULT_IDENTITY.json
tools/active/verify_systems.py --historical-fixtures runs/active/factor_inverse_fixtures --output reports/active/FACTOR_CPU_HISTORICAL.json
tools/active/run.py --config configs/active/factor_systems.json --name factor_protected_systems
tools/active/run.py --config configs/active/factor_difficult.json --name factor_difficult_systems
tools/active/verify_factor_pairs.py runs/active/factor_protected_systems/fixed runs/active/factor_difficult_systems/fixed --output reports/active/FACTOR_PAIR_RESIDUALS.json
tools/active/batch.py --plan configs/active/factor_cost.json
tools/active/batch.py --plan configs/active/factor_windows.json
tools/active/analyze_factor_windows.py --output reports/active/FACTOR_WINDOW_QUALITY_TIMING.json
tools/active/batch.py --plan configs/active/factor_window_audit.json
tools/active/audit_factor_windows.py --output reports/active/FACTOR_WINDOW_CCD.json
tools/active/render_factor_windows.py
```

五份新CPU参考另用 `verify_systems.py --prefix` 检查，报告为 `FACTOR_CPU_WINDOWS.json` / `FACTOR_CPU_DIFFICULT.json`；两个成本比较使用 `compare_costs.py`。配置单测5项通过，AutoDL离线契约单测10项通过。共39次场景运行（2保存系统、4成本、27计时、6路径诊断）和两套fixture完成；未认证完整质量或2×目标。
