# 接触布料：候选池优化与阶段成本

日期：2026-10-06。本轮从零测试 `cloth_hang_l`、`cloth_fixed_bunny_l`，保留材料、legacy累计停止 `.01`、最少6更新、PCG rho `1e-4` 和完整安全CCD。目标仍为相对官方Stiff至少2×；本机共享GPU只作诊断。新的 `contact_pool` 与 `contact_pool_validate` 独立、默认关闭，不由combined继承。

## 成本依据与选择

以下是前一轮完整100帧原始实测，不是本轮新候选的结果；原数据与程序身份保留于 `CONTACT_OPTIMIZATION_RESULTS_20261006.md`。

| 100帧阶段 | 悬挂/s（占总时间） | 固定兔子/s（占总时间） |
|---|---:|---:|
| Hessian等装配 | 0.5261（9.93%） | 2.4394（12.27%） |
| 转换、MAS准备、PCG、解分发 | 2.4809（46.80%） | 9.3711（47.13%） |
| CCD | 0.6233（11.76%） | 1.7979（9.04%） |
| 线搜索 | 1.2460（23.51%） | 5.5767（28.05%） |
| 状态更新 | 0.1177（2.22%） | 0.1235（0.62%） |
| 退出轮装配 | 0.0472（0.89%） | 0.0449（0.23%） |
| 其余未分类时间 | 0.2595（4.90%） | 0.5306（2.67%） |
| solver合计 | 5.3006 | 19.8840 |
| 同轮官方Stiff | 9.1698 | 29.5393 |
| 相对Stiff | 1.730× | 1.486× |

达到2×还需从当前组合净减13.50%（0.7157s）／25.72%（5.1144s）。线性阶段不是纯PCG kernel时间；阶段CUDA事件含提交间隙。CPU等待与Graph GPU时间不相加，退出轮装配不重复计入。

前轮Nsight实际hang41/fixed49普通VF+EE查询22.855/32.644ms，占诊断帧wall 7.32%/12.57%。本轮选择同一Newton方向已生成的FullCCD swept候选作为来源，重新分类每次trial，减少普通树遍历。仅两个诊断帧不能推出整场收益，更不能预设单组件达2×。

## 实现与检错边界

- `collision/ipc_contact_pool.h/.inl/.md`：配置、原始primitive身份、generation生命周期、内容/包含守卫、窄相、私有对照与fixture分开。inl仅由mlbvh.cu包含一次。
- `mlbvh.cu/.cuh`：capture-only swept入口，保留原disabled kernel；VF保留vertex/face，EE保留query/leaf edge ID、方向与重复。复用原PT/EE距离和mollifier编码。
- `core/GIPC.cu`：只在一条Newton方向内开启来源，buildFullCP容量重试后seal，trial更新alpha，lineSearch结束立即失效；只替换buildCP的两种普通自接触query。
- 每trial核对body、boundary、rest、direction、face/edge/surface映射内容和地址，实际顶点及普通primitive bounds必须在缓存swept bounds内。任何失效回退旧query；不靠经验epsilon补证明。
- 普通BVH、FullCCD、ground和isIntersected保持。诊断只用私有旧/新输出；总能量、自接触能量及实际Armijo判定在同状态核对，并恢复生产DCD/CCD/MatIndex及5项计数。`1e-10`仅为能量数值比较，不是材料宽限。
- 生产容量不足pass只计数并重试，单列validation可观测overflow；这种pass不冒充进入能量审计的完整trial。
- CPU配置/分析器检查、GPU小型fixture、真实同状态guard、速度组分别记录。fixture中的PT/EE类型计数不能冒充所有细子类已覆盖。

## 预先冻结的有限测试

1. 18配置契约＋分析器自测；独立新GPU fixture与8个既有数值回归。
2. hang on51、fixed on59、mixed off35/on35。validate只在on guards开启；120s及既有显存预算不变。
3. 正确性守卫允许后，三轮交错off/on hang51/fixed59，重型诊断关闭。冻结接触窗hang41–50、fixed22–59及45–59，mixed24–26/33–35。
4. 两布料全前缀及接触窗配对中位均≥1.05、材料支持成立后才测4份事件成本；扣除swept sidecar、prepare、guard/classify及旧查询回退，父子范围去重。
5. 完整100帧3轮Stiff/off/on需上述净收益证据，缺证据不启动。最多一次有证据修订，不扩展参数网格。

冻结质量协议SHA `1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f` 不变；旧固定基线失败保留。位置/实际速度按cloth/FEM/ABD分别报告，Stiff缺实际速度不从位置差分补造。mixed单对无冻结FEM重复范围，只作有界兼容检查。私有旧query对照不是独立接受路径CPU CCD认证。

## 本轮结果

**已验证：组件实现、构建身份、小型GPU检查及实际同状态查询/能量核对完成。待验证：净成本、配对速度、完整质量和2×。** 新组件保持默认关闭。本轮没有性能负收益结论，也没有推广依据。

程序SHA `faedd6f85f3c97356b2ad4bc607dfff7f0cd9c9ede79c6615336e0a679693b10`；37编译单元、11改变对象、源码/对象/链接输入核验完成，0编译错误。不可变构建记录 `builds/active/provenance/20261006T063929_873984Z_ipc_contact_pool_20261006/manifest.json`。18项配置契约和分析器CPU自测通过；9组GPU测试通过，其中新fixture53条case、16次私有旧新比较、875条接触记录。ABD反例实测origin=2、预测端点=2、实际q更新=4，两个规模均按预期回退。

四次实际场景均从零完整结束，包含两布料51/59帧、mixed off/on各35帧。三份on共**717次**旧/新同状态比较、**846,113条**typed接触记录通过，均保留方向、符号与重数，MatIndex类型内置换有效。没有PCG触顶/breakdown、非有限状态、ABD翻转或异常固定漂移。

| 守卫场景 | 尝试/复用/回退 | 复用率 | 比较typed记录 | pool本体峰值/MiB | Newton方向/PCG |
|---|---:|---:|---:|---:|---:|
| hang on51 | 298/295/3 | 99.0% | 208,711 | 24.60 | 297/9,104 |
| fixed on59 | 324/317/7 | 97.8% | 173,230 | 38.83 | 324/18,990 |
| mixed on35 | 142/105/37 | 73.9% | 464,172 | 45.40 | 142/3,654 |
| mixed off35 | 无pool准备或复用 | — | — | 0 | 142/3,658 |

实际回退均为bounds不包含，证明不能删掉机器守卫。pool峰值不包含能量审计快照、私有对照输出等重诊断缓冲。能量数值比较采用 `|Enew-Eold|/max(1,|Enew|,|Eold|)`，小于1的能量对应绝对误差口径。hang/fixed总能量最大归一化差分别 `1.6941e-21/3.3881e-21`，self-barrier分别 `2.6470e-23/8.4703e-22`；mixed总能量差0、self-barrier最大 `1.6941e-21`。717条审计的实际Armijo分支一致，生产输出在继续前恢复。

两布料守卫材料指标均通过冻结界限：hang最大伸长1.128876241、p99 1.032409344；fixed最大伸长1.035340186、p99 1.013270817。fixed固定ABD漂移 `3.10e-17m`。这些是短前缀数据，不替代100帧或独立接受路径CCD审计。

mixed实际复用覆盖24–26和33–35接触窗口，但质量只有单对，**待确认**。off/on布料位置最大逐帧RMS `1.094μm`、FEM `0.534μm`、ABD `0.050μm`；实际速度对应 `7.566e-6/2.115e-5/3.746e-7m/s`。FEM最小J整段off/on为 `0.4978689446/0.4978688634`，逐帧差范围 `[-1.595e-6,1.987e-6]`；非正单元和负体积均为0。布料整段最大伸长略小，逐帧仍有最高 `2.893e-9` 的正差。自动兼容门禁采用严格逐帧非恶化，因此返回false；它既含ulp级差，也含约ppm的FEM差，不能只说“全是ulp”，更不能直接解释成已证明的物理劣化。没有冻结mixed重复差异范围，本轮保留门禁及原始差分，不自动扩大容差或改判通过。

## 为什么没有新加速比

三轮交错速度计划16项中的4项守卫已完成；12项screen的首项 **hang-off-r1** 在41/51帧触发既有显存预算保护，自动停止，后11项未启动。没有一组完整off/on配对。**不得拿守卫时间、失败前缀、旧版本时间或另一种诊断开关拼出新加速比。** 当前新组件收益未知。

首项开跑free3462MiB，按原公式预算1926MiB；采样最低free1458MiB，总下降2004MiB，超预算78MiB。最低值高于768MiB硬底线；这是1.5GiB保留量触发的保护，未证明cudaMalloc OOM。失败程序contact_pool=false、pool本体峰值0。四项先前守卫起始free3967/3958/3968/3964MiB，预算余量392/242/98/42MiB；不能因它们成功而忽视现在低约505MiB的起始空间。未关闭用户应用、修改预算、延长超时或用失败数据认证速度。

资源报告 `contact_pool_resource_analysis_20261006.json` 核对19份输入。布料按观测最大drop2180MiB＋512MiB扰动余量＋1536MiB保留，建议启动free≥4352MiB（4.25GiB）；mixed建议≥4608MiB（4.5GiB）。这是规划余量，不是保证。已请求用户释放GPU应用显存，同时完成离线分析。4份事件成本、18份100帧、300帧、AutoDL均未启动。

## 组件分析与下一轮方向

1. **支持但未证明：**两布料复用率高、typed等价和能量/接受判定支持此执行路径，机器包含守卫有真实必要。现有数据没有支持新增停止规则、改变材料或减少必要CCD。
2. **待验证净收益：**pool准备增加原始ID sidecar写、内容快照、swept包围盒保存、每trial内容/包含检查，以及seal与guard读回。原两次树遍历减少也可能被维护/同步抵消。不能因复用99%就宣布更快，更不能把原线性47%和线搜索28%全部归给查询。
3. **先完成测试，暂不再写第二个kernel候选：**显存条件恢复后，保留首项失败证据，按明确的新恢复计划完成交错短筛，仍用120s和原资源公式。完整配对、原输入身份、全前缀及冻结接触窗各自核对，材料范围不扩大。
4. 若短筛有明确≥5%净增益，才做原计划4份cost记录；包含swept sidecar、prepare及discrete query父范围，去除guard/classify嵌套重复，并报告工作量和CPU等待。新旧轨迹未冻结时只作归属诊断，不叫同系统因果速度。
5. mixed门禁须有预先声明的整段指标和重复比较依据；本轮逐帧“任何正差”原结果保留，不能看过候选后自动重设。mixed pending不发展成布料性能工作的长期前置条件，但默认融合、全模型或同质量认证仍不得绕过该缺口。
6. 净收益未达门槛则封存pool；只在已有WHILE间隙经未profiled固定A/b/M重放验证后考虑Graph分块，计入停止后的尾轮成本并保留原停止时序。没有新成本依据，不重启MAS小kernel/SpMV小融合或AL-TOI参数网格。

**目标结论：2×尚未实现。** 目前相对Stiff的完整100帧背景仍为上一轮1.730×/1.486×，本轮只证明新执行组件可进入有界性能探索。材料门槛、独立CCD缺口、共享桌面限制和旧失败证据均未改。

## 复现与证据

本轮命令均在项目目录下，Python为 `E:/Anaconda/envs/DL/python.exe`；GPU串行：

```text
python tools/active/build.py build --label ipc_contact_pool_20261006 --jobs 2
python -m unittest discover -s tools/active -p test_contracts.py -q
python tools/active/contact_pool.py self-test
python tools/active/fixtures.py --name ipc_contact_pool_20261006_fixtures --contact-pool
python tools/active/batch.py --plan configs/active/ipc_contact_pool_20261006_guards.json
python tools/active/contact_pool.py analyze --stage guards
python tools/active/batch.py --plan configs/active/ipc_contact_pool_20261006_screen.json
python tools/active/contact_pool.py analyze --stage screen
```

原始数据 `runs/active/ipc_contact_pool_20261006_*`，冻结计划 `configs/active/ipc_contact_pool_20261006_{guards,screen}.json`，分析 `reports/active/ipc_contact_pool_20261006_{guards,screen}_analysis.json`。screen命令因资源保护返回1，结果记录保留。唯一索引维护保留616项历史条目及全部历史决定；本轮验证清单单列已尝试、完成、失败和未启动项目。
