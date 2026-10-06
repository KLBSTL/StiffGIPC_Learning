# CUDA Graph 小场景单独检查

## 结论

仅启用 conditional CUDA Graph，关闭 v3 A 的能量批处理、复用、拓扑缓存和 BVH refit。悬挂布料 L（1939 顶点，20 帧，无几何接触）完成了 base、同一融合程序的 host 后端，以及 Graph 后端对照。

共同 PCG rho 容差为 `1e-16` 时，Graph 的位置和同矩阵/RHS 解差两项检查通过预定 `1e-6` 门槛。默认 rho 容差 `1e-4` 时检查未通过，而且同一矩阵的 host 两次重复也有差异。因此暂未发现小场景中明显的 Graph 循环语义错误；不能据此宣布默认容差、复杂接触或全部缓存路径均已验收。

## 检查结果

所有组使用相同初态、20 帧、dt=0.01、Newton tol=0.01、MAS 预条件器，suite=0。

| 共同 PCG rho 容差 | Graph/base 归一化最大质量加权位置 RMS | 同 A/b Graph/host 最大相对解差 | 同 A/b host/host 最大相对解差 | Graph 最大真实欧氏相对残差 |
|---|---:|---:|---:|---:|
| 1e-4 | 2.019e-4，失败 | 3.496e-3 | 4.384e-3 | 8.499e-1 |
| 1e-12 | 1.136e-6，失败 | 2.561e-6 | 7.496e-7 | 9.668e-5 |
| 1e-16 | 4.426e-9，通过 | 1.730e-8，通过 | 7.904e-9 | 1.090e-6 |

严格组 base 最大真实欧氏相对残差为 1.074e-6，融合 host 为 1.109e-6。PCG 配置比较的是预条件内积 rho，不能把配置 `1e-16` 解释为欧氏残差 `1e-16`。这里报告实测残差，没有以新的宽松残差门槛补判成功。

严格 Graph 组 128 次线性求解、52 次 capture、76 次 cache hit、51 次 invalidation，未命中 PCG 上限。Graph 审计在每次求解后额外执行两次 host 求解，并恢复原 Graph 解；其计时包含这些重复工作，**不能计算性能加速比**。

默认容差下 base 和关闭全部新功能的融合 host 之间也有位置差异。MAS 内部使用浮点向量和原子加法，是需要继续检查的数值非确定性来源；当前结果尚未证明它是唯一原因。

## 可复核文件

### 关闭审计后的速度

在上述严格容差 `1e-16` 下，编译结束后独占 GPU，按 base/Graph、Graph/base、base/Graph 顺序测 3 次。计时为 20 帧 solver 窗口，包含捕获/实例化/重建，排除初始化与导出。只启用 Graph，suite=0。

| 重复 | base 秒 | Graph 秒 | 配对 base/Graph |
|---|---:|---:|---:|
| 1 | 5.587729 | 1.918447 | 2.913× |
| 2 | 4.940984 | 1.858931 | 2.658× |
| 3 | 5.053670 | 1.854934 | 2.724× |

配对速度比中位数 **2.724×**；base/Graph 时间中位数 5.053670/1.858931 秒，减少约 63%。三个配对的全 21 状态位置等价检查均通过，最差归一化 RMS 为 7.156e-9。该数字只描述当前 1939 顶点、20 帧、严格容差小场景，不是默认参数或复杂接触场景的总体结论。独立连续 CCD、物理参考和全部缓存扩容门槛仍需分项验收。

- `GRAPH_SPEED20.json`：三次原始时间、配置及配对速度比。
- `graph_speed20_r01.json` 至 `r03.json`：三个轨迹配对。
- `../runs/local/graph_speed20_v4/`：矩阵状态和各运行索引。

- `../runs/local/graph_isolation20_{base,fused_host,base_graph}/`：默认容差。
- `../runs/local/graph_isolation20_tight_{base,fused_host,base_graph}/`：1e-12。
- `../runs/local/graph_isolation20_strict_{base,fused_host,base_graph}/`：1e-16。
- `graph_isolation20_strict_graph_equivalence.json`：全 21 个状态的轨迹检查。
- 各组 `requested.json` 包含配置、可执行文件哈希，`output/stats.json` 包含逐次同系统解差及残差。

检查使用保留的 v4 二进制（主矩阵原版本），与 TOI 修复的 v5 构建分开。随后继续诊断 TOI 的 trial 非翻转限步问题；不据此放宽原质量门槛。
