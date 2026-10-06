# 等待用户开启AutoDL：因子逆候选复测

本机测试和交付检查完成；没有SSH、上传或远端运行。因子逆候选因轨迹质量越界未获推广，程序默认仍为triangular。

- 归档：`packages/autodl_factor_diagnostic_20261004.tar.gz`
- 大小：44,010,023字节（约41.97 MiB）
- SHA256：`659cef89983a818f0988973c1a96038e40c24ab81e9bcc2b9ad13149f20f4862`
- 已逐项核验647个输入文件、653个tar成员，无额外成员或路径越界；源文件哈希与本机一致。
- 校验收据：`reports/active/FACTOR_DELIVERY_CHECK.json`，命令 `E:/Anaconda/envs/DL/python.exe tools/active/check_factor_delivery.py`。
- 本机报告：`reports/active/FACTOR_ACTION_RESULTS.md`；运行流程：`reports/active/AUTODL_RETEST_PLAN.md`。

三个场景为混合兔子、悬挂布料、固定兔子布料。归档中的候选配置是实验复现用途，含Stiff、同组件IPC/TOI×host/Graph及TOI Graph旧triangular作用参考；没有预填质量通过文件。

用户开启实例后再确认当前SSH地址、磁盘与GPU状态，不沿用旧机器假设。将归档放在数据盘的新专用目录，核对SHA，检测CUDA/算力后从源码构建。先执行18个3帧smoke及54个三轮window（21/45/35帧）。window带完整诊断，不能用其时间认证性能。运行后检查接受路径、位置/速度、拉伸、FEM J/负体积和固定点。

所有100帧pilot/audit以及七对阶段均被实际质量gate保护；七对另要求完整pilot/audit阶段与受控负载证据。当前质量未过，不能直接跳到100/300帧。本次不包含六份历史A/b/M的远端数值重放，不宣称Linux构建或架构兼容已验证。
