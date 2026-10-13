# AutoDL 三场景复测（2026-10-13）

这是已测代码的 Git 历史快照。18份数据均已保存；逐 Newton 数据为无损的 `data/per_newton.csv.gz`，其余 CSV 保持原字节。完整原始归档已取回本机、核对 SHA256，并补齐全部逐帧位置和速度；详见 `DATA_MANIFEST.json` 和 `evidence/`。新环境复现步骤见仓库根目录 [HISTORY_SNAPSHOT.md](../../HISTORY_SNAPSHOT.md)。

RTX 4090 / Ubuntu 22.04 / CUDA 12.8。355个仿真源码与输入文件与本机记录版逐字节相同；未修改数值算法。Bunny、Mixed、Boxes使用同一当前执行组合，比较三种停止规则，并测较严数值参考。

dt=0.04、候选PCG rho=1e-4、105帧×3轮；性能汇总使用前100帧，状态表取物理第100帧。参考使用Movement only、rho=1e-6，外层阈值仍为0.01，非精确解。

| 场景 | MOV时间(s) | 纯累计TOI时间(s) | TOI倍率 | 补偿＋门控时间(s) | 门控倍率 |
|---|---:|---:|---:|---:|---:|
| Boxes | 64.663762 | 22.648293 | 2.855127× | 59.749192 | 1.082253× |
| Mixed | 24.738401 | 10.338492 | 2.392844× | 14.361656 | 1.722531× |
| Bunny | 9.216805 | 5.193676 | 1.774620× | 8.091610 | 1.139057× |

时间为前100帧三轮中位数，S为同平台MOV时间中位数/当前规则时间中位数。MOV基线是当前记录程序的Movement only，区别于冻结StiffGIPC Base。所选三场景时间中位数之和的倍率：TOI 2.582970×，门控1.199708×。

Boxes同输入第100步中，纯TOI：N=8、P=192、t=0.203721s，RMS差异1.966769mm、最大点差14.033194mm；门控：N=15、P=311、t=0.297273s，RMS差异0.015503mm、最大点差0.288912mm。位置差异相对较严参考，非绝对误差。完整轨迹长期差异还需结合参考自身重复性，不能把纯TOI倍率当作通过精度约束的认证。

## 报告和视觉图

- [主报告：6页](AutoDL_3Scene_Main.pdf)：配置、本机/AutoDL对照、分项性能、实际第100帧、共同检查点/独立轨迹差异、PCG成本、参考重复性、记录开销。
- [附录：15页](AutoDL_3Scene_Appendix.pdf)：三场景逐帧曲线、三轮关键帧0/25/50/75/100/105、三轮第100帧总览。
- [Boxes共同检查点](checkpoints/Checkpoint100_Boxes.pdf)
- [Mixed共同检查点](checkpoints/Checkpoint100_Mixed.pdf)
- [Bunny共同检查点](checkpoints/Checkpoint100_Bunny.pdf)

实际网格使用统一相机、变形比例1，无平滑。云端软件EGL绘图仅在数值测试完成后使用，未重跑仿真；162份OBJ坐标和拓扑与原始导出完全一致。共5个PDF、24页，均完成渲染检查。

## 数据与核验

data/保留本轮完整数值CSV：逐帧、逐Newton、实际第100帧、阶段汇总、轨迹位置/速度差异、质量指标、参考重复性、共同检查点、GPU采样、初始化/导出/写盘、记录开关、嵌套CostScope等。父子包含区间不能求和；残差观察与求解/组装区间重叠，不重复计入总时间。

本轮完成12次两帧验证、36次105帧轨迹、63次检查点/记录开关、9次额外成本诊断。生成3780条正式轨迹帧、67523条逐Newton、28620条误差记录，核对36个初态、63个检查点输入、162份无损OBJ。首次Linux适配在原生程序启动前因缺少元数据中止，0帧；该失败证据保留，不参与正式统计。

跨平台初始位置、速度、质量、边界类型和物体编号逐字节相同；三角形仅排列顺序不同，连接关系与方向、四面体数据一致。跨平台时间比同时受硬件、驱动、编译器和数值路径影响，不能全部归因GPU。

- [数值与网格核验](delivery_verification.json)
- [跨平台拓扑核对](cross_platform_topology.json)
- [PDF数据来源及SHA256](PDF_manifest.json)
- [PDF排版核验](PDF_verification.json)

完整原始位置和速度归档在服务器：`/root/autodl-tmp/codex_recording105_20261013/autodl_recording105_20261013_full.tar.gz`，4,570,828,245字节，SHA256：`b13b4f11cbeac3225d096c21079874515de7360cd31171d87904a70789534d49`。完整归档与复核包的本地取回凭据位于仓库transfers/。云端原始运行目录、编译证据、所有数值和视觉结果均保留。

## 复核命令

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/remote/verify_recording105_autodl.py --root runs/autodl_recording105_review_20261013 --receipt-name delivery_verification_local.json
& 'C:/Users/86189/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 -B tools/remote/report_autodl3_pdf_20261013.py --root runs/autodl_recording105_review_20261013
& 'C:/Users/86189/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 -B tools/remote/qa_autodl3_pdf_20261013.py
```

这些工具保留先前输出，重跑前需选择新的输出/凭据目录。GPU测试命令及其完整日志见取回控制状态和源包：`xvfb-run -a /root/miniconda3/bin/python -B -u tools/remote/run_recording105_autodl_v2.py run`。

按用户授权清理了9月26日、27日旧StiffGIPC实验和对应归档；另一个旧重复归档在22262个成员逐一与保留目录匹配后清理。10月3日之后的原始实验数据保留，清理凭据位于transfers/。
