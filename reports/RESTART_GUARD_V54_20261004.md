# v54 重选初值后的满步退出修正

> 后续核验更正：本文原固定点位移0统计使用FEM边界标记，未覆盖固定ABD障碍。后续按固定物体语义补查v54相关38组/3657端点，最大障碍位移6.2063e-17m；见 `V54_FIXED_ABD_SUPPLEMENT.json`。几何结论未发现异常，但“严格0且已覆盖固定障碍”的旧检查表述不准确。

2026-10-04，本机 RTX3070 Laptop / CUDA13 / Release。**已消除固定兔子布料第24帧可重复的13.2%局部伸长尖峰；尚未证明所有布料质量不退化，也没有获得通用加速。** 候选保持默认关闭。此次完成一项求解器修改及三个布料场景的三臂、两轮100帧验证，没有继续堆叠求解器变体。

## 修改与对照

v53在重新线性化后比较完整目标能量，可能将旧trial替换为safe。原满步退出仍使用整帧累计的至少六次Newton历史，因此新子问题可能只算一次就退出。v54只在**确实选中safe**时，要求该重启子问题至少执行六次迭代才能使用满步启发式退出；小方向收敛仍可提前退出。这是对启发式退出历史的局部修正，不是严格收敛证明。

源文件 `sources/stiff_perf_v54/StiffGIPC/solver/toi_solver.cu`，新开关 `GIPC_TOI_RESTART_FULL_STEP_GUARD` 默认0；初值选择也仍默认0。材料、时间步、线搜索、PCG容差/上限、CCD保护均不变。冻结v50/v53及其程序保留，v54复用v50构建，只覆盖这一源文件。

同一v54二进制三臂：OFF=`choose-start 0, restart-guard 0`；LEGACY=`1,0`；GUARD=`1,1`。各场景按OFF r1、LEGACY r1、GUARD r1、GUARD r2、LEGACY r2、OFF r2串行，每组从0运行100帧，180秒预算。Graph、native、rho=1e-4、velocity_tol=.05、suite=1、完整CCD、候选上限1000万。追踪/物理能量/PCG审计开启，计时仅作诊断。同场景材料、初始状态、质量、拓扑、边界及其余求解参数逐项一致，friction_compiled=true。

悬挂dt=.04/MAS Cholesky；落球和固定兔子dt=.01/diag。三个场景总共18组100帧全部正常完成。

## 先验初筛与完整形变

运行前在 `WORK_PLAN_V54_20261004.md` 声明固定兔子30帧初筛：最大边长比<=1.045（历史OFF完整100帧包络），退出/有限/固定点/CCD通过。结果OFF=1.034412、LEGACY=1.132352、GUARD=1.032997；GUARD阻止9次原本可满步退出的尝试，初筛通过，才扩展完整场景。

下表为100帧期间所有布料边的最大当前长度/初始长度，每项列出两次重复。这个指标不是相对于收敛物理解的误差。

| 场景 | OFF | LEGACY | GUARD |
|---|---|---|---|
| 悬挂 | 1.097296 / 1.097296 | 1.097296 / 1.097296 | 1.097296 / 1.097296 |
| 落球 | 1.017690 / 1.017690 | 1.055387 / 1.017365 | 1.018914 / 1.017450 |
| 固定兔子 | 1.044019 / 1.041862 | 1.132352 / 1.132352 | 1.042325 / 1.045064 |

固定兔子明显尖峰消失；GUARD后期最高1.045064，仍略高于本轮OFF最高1.044019及历史OFF1.044475。落球GUARD一轮峰值也比OFF高约0.001224。不能事后放宽门槛，把这些差异直接认定为全部无退化；30帧初筛通过不等于100帧质量验收。LEGACY落球两轮差异很大，再次说明单次峰值不足以评价稳定性。

所有固定顶点位移0，导出顶点和能量有限，无PCG触顶/breakdown，所有端点三角形面积比为正。GUARD阻止原满步退出次数：悬挂2/2，落球30/38，固定兔子73/70，分支实际激活。全部完整目标选点、能量接受、原退出与新增重启条件审计通过。

自身重复轨迹最大质量加权RMS/布料初始包围盒对角线：

| 场景 | OFF | LEGACY | GUARD |
|---|---:|---:|---:|
| 悬挂 | .0513% | .0007% | .0002% |
| 落球 | .9781% | 1.4557% | 1.4036% |
| 固定兔子 | 1.2629% | .8529% | 1.4570% |

GUARD的重复稳定性没有在所有场景优于OFF。不同初值/终止历史会改变轨迹，不能将低能量起点视作轨迹等价保证。

## 第24帧的修复证据

固定兔子布料同一条边 `[23850,23857]`，初长.0213333333333米。两次GUARD均将原LEGACY的约1.132352边长比降为约1.01112；此时全布最大边长比约1.02964。标准库math.dist独立抽查与向量化统计一致。

| 第24帧 | Newton / outer | 最后原始方向速度 m/s | cloth_potential |
|---|---:|---:|---:|
| OFF两轮 | 18 / 13 | .066947 | .0248704 |
| LEGACY两轮 | 7 / 2 | 2.407259 | .415950 |
| GUARD两轮 | 13 / 3 | .10762–.10777 | .0223125–.0223166 |

这支持“重选初值后沿用旧累计历史使满步退出过早”是该短回归的有效修复方向。它仍不能证明所有混合体停滞/形变差异都来自这一原因。

## 诊断耗时

两次solver_seconds的中位数；加速比=OFF/GUARD。参照是同v54关闭初值选择，不是官方Stiff。

| 场景 | OFF秒 | LEGACY秒 | GUARD秒 | OFF/GUARD |
|---|---:|---:|---:|---:|
| 悬挂 | 5.09149 | 3.29892 | 3.32578 | 1.531× |
| 落球 | 6.36238 | 6.58921 | 7.17537 | .887×（慢12.8%） |
| 固定兔子 | 12.21341 | 11.32399 | 12.21198 | 1.000× |

修复后悬挂保留约1.53倍诊断收益，固定兔子基本回到OFF耗时，落球有额外成本。本轮是质量修复验证，含导出与审计、只有两次重复，没有进行正式性能认证。

## 验证与视觉

Release成功。初次沙箱构建因SDK读取权限失败，保留 `V54_build.log`；取得允许后同配置构建成功，日志 `V54_build_release.log`。已知D9002及MSB8012构建警告保留，实际输出gipc_v54.exe已复制并冻结。43项组件、15项守卫检查通过。未重跑CTest。

无safe重启的悬挂3帧smoke两臂都没有触发新分支，最大坐标差1.0474e-13米，12次PCG且最大38迭代，未要求逐位相同。初筛及smoke共255段独立CCD通过；完整18组共6227段接受路径/跨帧桥接经独立CPU BVH + Tight-Inclusion检查，碰撞标记均0，所有验证器exit_code=0、路径覆盖核对通过。两阶段合计6482段，原始结果及断言汇总在 `V54_FINAL_CHECK.json`。全部仿真和验证进程已结束。

程序SHA256：`5b59e32eab424daea91c932d944ff17a840477804afedd29d65b7ad7550a8395`；源摘要：`01b4a1009d2cbff4868af5f46b085e8d8fb570c627226c555f25b4d4961d1280`。294项依赖、二进制及每组runner身份核验。

三张同帧热图与三张各组峰值图均已视检：固定兔子LEGACY第24帧明显局部热点在GUARD中消失；落球最终褶皱与OFF存在差异，没有爆炸/断裂外观。峰值图每一格是该组自己的峰值时刻，不能当作同一时刻的位置比较。灰色障碍三角面显示受三维绘图遮挡排序影响，碰撞判断以独立CCD为准。

![固定兔子同帧对比](V54_CLOTH_VISUAL_cloth_fixed_bunny_l.png)
![三臂两轮各自峰值](V54_PEAK_cloth_fixed_bunny_l.png)

## 复现与产物

任务根目录下，Python为 `E:/Anaconda/envs/DL/python.exe`，PowerShell7。以下是实际执行入口；脚本拒绝覆盖旧产物，重跑需新运行名称和输出路径。

```powershell
$overlay=(Resolve-Path tools/v54_overlay.targets).Path
& 'D:/vs2022/MSBuild/Current/Bin/amd64/MSBuild.exe' builds/local-v50/gipc.vcxproj /p:Configuration=Release /p:Platform=x64 "/p:ForceImportAfterCppTargets=$overlay" /m:2 /v:minimal
python tools/freeze_local_v54.py
# clean GIPC environment, then separately:
$env:GIPC_VALIDATE_COMPONENTS='reports/V54_components.json'
& builds/local-v54/Release/gipc.exe
Remove-Item Env:GIPC_VALIDATE_COMPONENTS
$env:GIPC_PCG_GUARD_FIXTURE='reports/V54_guard.json'
& builds/local-v54/Release/gipc.exe
Remove-Item Env:GIPC_PCG_GUARD_FIXTURE
python tools/run_v54_tests.py smoke
python tools/run_v54_tests.py screen
python tools/analyze_v54_cloth.py --matrix reports/V54_SCREEN_MATRIX.json --output reports/V54_SCREEN_QUALITY.json
python tools/analyze_local_v54.py v54_smoke_legacy_r1 v54_smoke_guard_r1 v54_screen_off_r1 v54_screen_legacy_r1 v54_screen_guard_r1 --ccd --output reports/V54_SCREEN_VERIFICATION.json
python tools/run_v54_tests.py cloth
python tools/analyze_v54_cloth.py --output reports/V54_CLOTH_QUALITY.json
$names=(Get-Content -Raw reports/V54_CLOTH_MATRIX.json | ConvertFrom-Json).name
python tools/analyze_local_v54.py @names --ccd --output reports/V54_CLOTH_VERIFICATION.json
python tools/render_v54_cloth.py
python tools/render_v54_peaks.py
python tools/summarize_v54.py
```

原始命令/参数/身份见各 `runs/local/v54_*/requested.json` 和三个 `V54_*_MATRIX.json`。形变/轨迹在 `V54_CLOTH_QUALITY.json`，CCD与覆盖在 `V54_CLOTH_VERIFICATION.json`，紧凑汇总在 `V54_FINAL_CHECK.json`。初筛质量JSON沿用了“两轮”通用scope描述，实际初筛每臂一轮，以matrix与本报告为准；完整cloth确为两轮。

## 后续边界

本轮接受“固定兔子第24帧明显尖峰被抑制”的结论，**暂不接受全局质量等价、稳定性修复或通用加速**。不开启默认开关，不扩展混合体，也不继续试新的迭代次数。

下一步冻结本候选，优先对落球第25–48帧与固定兔子第44–49帧进行相同物理时长dt/2、dt/4参考，比较拉伸、能量和轨迹的收敛趋势；必要时做同状态退出充分性检查。质量有依据后才回到混合体完成性和Graph耗时分项，原任务整体仍未验收。
