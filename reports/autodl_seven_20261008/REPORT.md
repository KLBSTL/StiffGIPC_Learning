# AutoDL 七场景 120 帧单次测试

原版 Stiff 为分母；all = conditional Graph + FullCCD refit + 批量能量 + 能量复用 + 普通 BVH refit（周期 8）。
rho=1e-4，dt=.01，legacy MAS/停止规则，完整 CCD。每项一次；无同质量或统计性能认证。

| 场景 | base 秒（1×） | Graph 秒 | Graph/base 加速 | all 秒 | all/base 加速 | 其他组件增量 |
|---|---:|---:|---:|---:|---:|---:|
| cloth_hang_l | 2.2876 | 1.8789 | 1.218× | 1.6180 | 1.414× | 1.161× |
| cloth_hang_m | 4.7361 | 3.8885 | 1.218× | 3.5797 | 1.323× | 1.086× |
| cloth_sphere7_l | 5.2871 | 4.5845 | 1.153× | 4.3459 | 1.217× | 1.055× |
| cloth_sphere7_m | 8.3087 | 6.8719 | 1.209× | 6.6191 | 1.255× | 1.038× |
| cloth_fixed_bunny_l | 6.5606 | 5.1714 | 1.269× | 4.8759 | 1.346× | 1.061× |
| cloth_fixed_bunny_m | 12.0072 | 9.9698 | 1.204× | 9.6854 | 1.240× | 1.029× |
| bunny_cloth_bunny_l | 10.6731 | 10.1766 | 1.049× | 9.9258 | 1.075× | 1.025× |

求解时间为 frames.csv 的 CPU 同步包络；core event 嵌套于其中，进程 wall 含加载和不对称导出成本，详见 TIMINGS.csv。
装配、线性、CCD、线搜索、状态更新是原生阶段计时，退出轮装配单列；阶段覆盖差异不能当作完整 wall 分解。
Graph/base 比含活动实现与冻结原版差异，本轮无活动 host 臂，不能严格隔离纯 Graph 的因果收益。
材料指标及按布料/FEM/ABD 分组的轨迹差异见 RESULTS.json；没有新基线重复标定、独立接受路径 CCD 或原版真实速度，不宣称质量已认证。
来源：原生 solver commit 3d8c15e；编译、实际对象、链接、输入和配置身份见 BUILD_IDENTITY.json、SEAL.json、runs/*.json。
