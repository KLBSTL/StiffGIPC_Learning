# v50–v52 修复候选未通过验收

2026-10-04，本机 RTX3070 Laptop 8GiB / CUDA13 / VS2022 Release。

**实际修改并测试了三个候选，但没有可靠修复。v52一次完成40帧，独立重启的100帧组却在第38帧受控失败。三个候选均默认关闭，不把单次成功宣传为修复或加速。**

## 依据与改动

参考advance_al.cu及global_active_set_manager.cu与移植的slack/乘子/满步退出语义总体相符。参考同样保留正slack接触的固定slack曲率，不能直接认定其为移植bug。参考抑制近平行EE而移植缺少该分支，但实际v49未完成outer快照的6166个EE没有符合阈值的样本，暂不把它当成根因。

该快照7728/8825个接触为正slack；原方向曲率中它们贡献0.181993，活跃接触0.137901。由此测试两个算法候选：

- v50解析消元slack，同时修改能量、梯度和Hessian。目标为gamma/2*(mu*min(c-lambda/mu,0)^2-lambda^2/mu)。
- v51仅在线搜索使用消元目标，保留原固定slack正定曲率作为上界模型。装配点梯度仍相同。
- 两者失败后，v52关闭slack改动，以min(1,100*velocity_tol/raw_velocity)限制首次试探比例，然后照常回溯。**仍用原始方向判收敛；r<1不能作为满步退出**。不改变物理dt、PCG、乘子或完整CCD。

## 连续运行

均从第0帧开始，dt=.01、rho=1e-4、速度阈值=.05、MAS Cholesky、原生inner退出、Graph、CCD候选上限1000万。有追踪/审计开销，非性能认证。

| 运行 | 完成/目标帧 | 墙钟 | 结果 |
|---|---:|---:|---|
| v50_bunny40 | 27/40 | 180.609s | 超时，拒绝推广 |
| v51_bunny40 | 34/40 | 180.469s | 超时，拒绝推广 |
| v52_bunny40 | 40/40 | 99.109s | 单次完成，重复可靠性未通过 |
| v52_bunny100 | 37/100 | 309.094s | 第38帧受控失败，退出码4 |
| v52_bunny40_off | 34/40 | 180.610s | 同程序关闭改动，超时 |

100帧总预算预先按帧数设为450秒，不是延长失败40帧组。实际309.094秒自行退出：`CCD pair resource limit: 13061914 > 10000000`。随后尝试停止进程时进程已不存在；未执行人工终止，也不是450秒超时。

100帧组第38帧记录1092次Newton、24个outer，1062次方向受限幅；最后outer有519次inner。全组已记录1368次PCG无上限，单次最多607迭代，最低采样空闲显存3872MiB。最终故障是几何候选数量保护，不是PCG上限或显存耗尽。单步限幅没有防止完整试探轨迹的累计畸变。

## 验证与限制

- v50 Release构建、CTest 2/2通过；三个程序各43项GPU组件检查通过。新增6项覆盖slack活跃/松弛/切换边界及逐接触惩罚比例。1000组CPU消元恒等式最大能量差7.33e-15。
- 三个候选均完成悬挂布3帧开关对照；v50最大位置差2.26e-15。
- v52成功40帧和失败100帧组通过逐方向限幅、能量接受及原生退出条件检查，没有把限幅后的方向当成已收敛。
- CPU独立CCD：v50 135段、v51 123段、v52成功组189段、v52失败组177段，均无碰撞标记，仅覆盖实际导出的接受路径及桥接。
- v52成功组端点最小FEM J=-4.7164，单帧最多125个非正J四面体，最大布边拉伸比1.30473；失败组已完成37帧最小J=-1.4781，最多41个非正J四面体。Stable NH1允许翻转，但这些结果没有获得物理质量认证，CCD通过不能替代质量。
- v49–v52源文件及程序最终哈希核验通过。v51/v52只覆盖一个CUDA源文件并复用v50其余对象/资产，避免重复全量构建。首次v51构建覆盖误作用于依赖项目，编译即失败；限定gipc主项目后成功，失败日志保留。最终v50原程序哈希未变。

## 决策

`GIPC_TOI_REDUCED_SLACK`和`GIPC_TOI_BOUNDED_TRIAL`仍默认关闭。三个候选均不作可靠修复推广，不提高PCG/CCD上限，不记录正式加速比。原任务仍未完成。

后续需要检验完整AL试探轨迹的累计畸变、更新接触几何的时机与Stiff材料非线性求解的关系。目前没有证据确认其中哪项是充分根因；继续增加局部限幅或收紧PCG不是已验证的解决方案。

## 重现与证据

Python为E:/Anaconda/envs/DL/python.exe。运行入口拒绝覆盖已有同名输出；复跑请通过run_perf_vXX.py另取name。

```text
cmake --build builds/local-v50 --config Release --target gipc diag_fused_update_tests --parallel 2
ctest --test-dir builds/local-v50 -C Release --output-on-failure
python tools/validate_v50.py smoke
python tools/validate_v50.py bunny
python tools/validate_v51.py smoke
python tools/validate_v51.py bunny
python tools/validate_v52.py smoke
python tools/validate_v52.py bunny
python tools/validate_v52.py long
python tools/validate_v52.py control
python tools/audit_bounded_v52.py v52_bunny40
python tools/audit_bounded_v52.py v52_bunny100
```

v51/v52使用MSBuild编译v50生成的gipc.vcxproj，Release/x64，传入`/p:ForceImportAfterCppTargets=<tools/v51_overlay.targets或v52_overlay.targets绝对路径>`。输出gipc_v51.exe/gipc_v52.exe，复制到独立build/Release目录后冻结；不要全局传TargetName影响依赖项目。

证据：V50_VERIFICATION.json、V51_VERIFICATION.json、V52_VERIFICATION_40.json、V52_VERIFICATION_100.json、V52_VERIFICATION_OFF.json、AUDIT_v52_bunny40.json、AUDIT_v52_bunny100.json、各requested/result/stats及构建日志。规划与逐次决策见WORK_PLAN_V50_20261004.md。
