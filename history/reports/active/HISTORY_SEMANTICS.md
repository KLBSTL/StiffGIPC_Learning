# 混合兔子历史语义与证据边界

本表依据本地冻结源码和历史报告，2026-10-04复核。时间属于各自硬件、配置、窗口；没有把跨版本原始时间重新认证为同质量加速。

|版本/来源|保留的机制与证据|不能作为当前结论的内容|
|---|---|---|
|v18 mass + persist|混合兔子两次100帧；末态0翻转；最大布边1.052–1.062；值得作局部罚权机制对照|仍有2个保守表面CCD标记；dt/4对较紧dt/8参考位置1.210%、速度35.36%；不是已验收快版本|
|v29 paper路径|小球TOI方向/CG约base的3.4–3.5倍，已慢3.57倍|不是后来Cholesky才产生全部TOI慢问题|
|公开Robust及修正供体|固定提交389f8a5；修正实际EE摩擦Jacobian，9项真实CUDA函数回归通过|公开libuipc不等于论文原始全套实现；旧未修正DLL不能作为物理真值|
|v32 robust核心适配|跨帧C/λ/γ、释放年龄、帧级摩擦、当前trial共用CCD、排除固定DOF的mu范数已移植|小球1.55–2.08倍原始时间优势未质量认证；AutoDL混合兔子四TOI臂33–38帧PCG失败|
|v43–v50稳定MAS|Cholesky作用修复旧浮点逆矩阵的不稳定；大方向在准确解中仍存在|为速度直接撤销稳定性修复、全局收紧PCG、按候选数推断CCD耗时主因均无充分依据|
|v53初值选择|新接触模型内比较warm与safe完整能量，选较低者；改善长inner连续推进|改变非凸轨迹；100帧442–499秒不构成加速或质量通过|
|v54重启计数|只有实际选safe后，满步退出再要求本子问题6次；小方向退出仍可更早|不能把所有正常outer都改成重新6步；混合兔子目前120秒43帧|
|active阶段A|配置/观察逻辑提取，明确四种容差；新TOI剩余量参数默认继承原Newton值；统一上下文ID|尚不能因编译或组件通过就称迁移轨迹等价；需本轮窗口/冻结算子验证|

## 已确认的当前语义

- `robust` 强制保留跨帧C、lambda、gamma；`GIPC_TOI_WARM_* = 0`不会覆盖该策略。当前解析保留原短路行为，resolved配置显式列出覆盖。不要再以“补跨帧历史”作为缺失功能。
- 公开实现每Newton调用线性化，但锚点来自safe/nonpenetrate；当前每outer在safe线性化，inner内safe不变，不能仅按循环外形认定语义错误。
- 公开供体允许小方向绕过能量下降，当前Stiff移植仍要求能量接受。因此velocity阈值1.0的旧Robust潜力与当前.05、严格接受路径不可互换。
- 旧参考运行累计TOI阈值.001；当前历史配置使用Newton命名参数.01兼作剩余量阈值。active分离接口但保留.01默认，未借整理改变停止要求。
- Cholesky会强制wide apply并覆盖inverse64。fused-diag只适用于可用的对角预条件路径，不继承acceleration suite；混合兔子MAS不能计算它的收益。

## 首次工作量增长的当前线索

最新`bunny_components_toi_toi_r1`记录：第24帧加入988个ground接触，PCG仍约29–33次；第25帧outer1选safe后，六次方向的PCG为30、223、339、246、31、31。接触模型未变时，第二个方向先变贵，说明应保存该outer的inner0→1，而非只保存每帧第一系统。v53同窗口只走一方向29次，但下一帧第二方向起出现222、308、237次，故“重启计数制造全部难系统”尚无依据。

最新第33帧开始1525个历史接触均lambda=0、gamma<1；第35帧按原26次释放年龄批量删除最早988个ground。这支持检查松弛接触的固定slack曲率、材料变化与MAS效果的耦合，但不证明应删除所有正slack曲率：v50全局消元候选已经失败。

下一步只在选定窗口补`c/δ`、slack、`lambda/(mu δ)`、互补量、更新差、分组方向曲率、局部contact-free顺应度。它们是接触子问题诊断，不应命名为完整物理KKT验收。先建立证据再修改罚权，不重复全局mass或mu倍率扫描。

## 证据位置

- `reports/2026-09-30_COMPARISON_AND_GATES.md:49–82`
- `reports/LOCAL_STATE_AND_SMALL_BENCHMARK_V29.md:48–89`
- `reports/ROBUST_REFERENCE_V32.md:25–35,59–83`
- `reports/ROBUST_DIRECT_PORT_FEASIBILITY_20261001.md:35–64,139–143`
- `reports/ROBUST_PORT_IMPLEMENTATION_20261001.md:9–15,54–83`
- `reports/AUTODL_V32_SMALL_SCENES_20261002.md:55–80`
- `reports/DIRECTION_REVIEW_V46_20261004.md:7–12`
- `reports/FULL_ENERGY_V49_20261004.md:61–83`
- `reports/REPAIR_ATTEMPTS_V50_V52_20261004.md:19–46`
- `reports/BUNNY_COMPONENTS_20261004.md:35–37,55–74`
- `runs/local/bunny_components_toi_toi_r1/output/stats.json`、`runs/local/v53_bunny100_confirm/output/stats.json`（本文首次增长对照来源；没有新GPU测试）
