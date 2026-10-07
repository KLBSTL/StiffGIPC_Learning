# 新报告执行优化：本机阶段结果

rho=1e-4，100帧，dt=.01。共享WDDM桌面诊断；未认证同质量或2×。
原版未导出实际速度；缺口保留，不以位置差分代替。

| 场景/阶段/配置 | 求解秒 | 方向数 | PCG总量 | 最大拉伸 |
|---|---:|---:|---:|---:|
| ablation_fixed_full_batch_1 | 19.8147 | 583 | 37221 | 1.035313600 |
| ablation_fixed_full_batch_2 | 19.9946 | 589 | 37443 | 1.035529639 |
| ablation_fixed_full_batch_3 | 20.1554 | 588 | 37421 | 1.035384830 |
| ablation_fixed_full_discrete_1 | 19.9511 | 581 | 37572 | 1.035429453 |
| ablation_fixed_full_discrete_2 | 19.8961 | 585 | 37273 | 1.035367897 |
| ablation_fixed_full_discrete_3 | 20.1887 | 598 | 38192 | 1.035584802 |
| ablation_fixed_full_refit_1 | 19.7656 | 579 | 37206 | 1.035765189 |
| ablation_fixed_full_refit_2 | 19.7152 | 586 | 37322 | 1.035890367 |
| ablation_fixed_full_refit_3 | 19.9969 | 585 | 37135 | 1.035635250 |
| ablation_fixed_full_reuse_1 | 20.3592 | 598 | 38159 | 1.035717012 |
| ablation_fixed_full_reuse_2 | 19.9338 | 587 | 37058 | 1.035625981 |
| ablation_fixed_full_reuse_3 | 19.1998 | 580 | 36463 | 1.035497256 |
| ablation_fixed_minus_batch_1 | 20.5955 | 590 | 37801 | 1.035246370 |
| ablation_fixed_minus_batch_2 | 19.8885 | 576 | 37070 | 1.036069230 |
| ablation_fixed_minus_batch_3 | 20.4786 | 597 | 37558 | 1.037757704 |
| ablation_fixed_minus_discrete_1 | 20.5095 | 595 | 37230 | 1.037381852 |
| ablation_fixed_minus_discrete_2 | 21.0644 | 604 | 38656 | 1.037148794 |
| ablation_fixed_minus_discrete_3 | 20.8939 | 607 | 38250 | 1.037114073 |
| ablation_fixed_minus_refit_1 | 21.1013 | 600 | 38189 | 1.035302406 |
| ablation_fixed_minus_refit_2 | 20.8447 | 594 | 38063 | 1.037557173 |
| ablation_fixed_minus_refit_3 | 20.5472 | 586 | 37282 | 1.035663671 |
| ablation_fixed_minus_reuse_1 | 20.1448 | 579 | 37032 | 1.035791632 |
| ablation_fixed_minus_reuse_2 | 19.6116 | 586 | 37159 | 1.035853180 |
| ablation_fixed_minus_reuse_3 | 20.2362 | 587 | 37680 | 1.035299145 |
| ablation_sphere_full_batch_1 | 11.8616 | 565 | 24344 | 1.018952713 |
| ablation_sphere_full_batch_2 | 13.0185 | 554 | 23472 | 1.019227035 |
| ablation_sphere_full_batch_3 | 12.6957 | 574 | 25513 | 1.019138380 |
| ablation_sphere_full_discrete_1 | 12.0799 | 558 | 24191 | 1.019036207 |
| ablation_sphere_full_discrete_2 | 12.0829 | 575 | 25531 | 1.018994149 |
| ablation_sphere_full_discrete_3 | 13.1305 | 574 | 25098 | 1.019399124 |
| ablation_sphere_full_refit_1 | 12.0479 | 585 | 25605 | 1.019156676 |
| ablation_sphere_full_refit_2 | 11.2632 | 572 | 25086 | 1.018953383 |
| ablation_sphere_full_refit_3 | 12.3902 | 587 | 25169 | 1.019178711 |
| ablation_sphere_full_reuse_1 | 11.9632 | 563 | 24236 | 1.018971754 |
| ablation_sphere_full_reuse_2 | 11.4091 | 576 | 24638 | 1.018962128 |
| ablation_sphere_full_reuse_3 | 12.8623 | 565 | 24109 | 1.019069781 |
| ablation_sphere_minus_batch_1 | 11.4932 | 552 | 23813 | 1.019384739 |
| ablation_sphere_minus_batch_2 | 12.2489 | 569 | 24672 | 1.019746438 |
| ablation_sphere_minus_batch_3 | 12.4844 | 555 | 23371 | 1.026039284 |
| ablation_sphere_minus_discrete_1 | 12.1902 | 565 | 24753 | 1.019135347 |
| ablation_sphere_minus_discrete_2 | 12.8520 | 564 | 24560 | 1.018990851 |
| ablation_sphere_minus_discrete_3 | 12.7983 | 551 | 23337 | 1.019377471 |
| ablation_sphere_minus_refit_1 | 12.6234 | 580 | 25513 | 1.019018020 |
| ablation_sphere_minus_refit_2 | 14.2082 | 567 | 25040 | 1.031021412 |
| ablation_sphere_minus_refit_3 | 11.6027 | 566 | 24018 | 1.019324966 |
| ablation_sphere_minus_reuse_1 | 12.3104 | 561 | 24215 | 1.019042716 |
| ablation_sphere_minus_reuse_2 | 11.9009 | 553 | 23580 | 1.018955424 |
| ablation_sphere_minus_reuse_3 | 12.3760 | 546 | 22931 | 1.019612821 |
| calibration_fixed_base_1 | 26.6181 | 580 | 36994 | 1.035610993 |
| calibration_fixed_base_2 | 24.4796 | 591 | 38235 | 1.035761916 |
| calibration_fixed_base_3 | 22.4717 | 584 | 37151 | 1.053267225 |
| calibration_sphere_base_1 | 17.5126 | 549 | 23261 | 1.019362029 |
| calibration_sphere_base_2 | 18.3386 | 562 | 24206 | 1.019347155 |
| calibration_sphere_base_3 | 17.9851 | 552 | 23438 | 1.019885888 |
| holdout_fixed_base_1 | 21.5270 | 588 | 37802 | 1.035769885 |
| holdout_fixed_base_2 | 21.2860 | 586 | 37235 | 1.035711981 |
| holdout_sphere_base_1 | 19.3839 | 568 | 24826 | 1.019151433 |
| holdout_sphere_base_2 | 19.1343 | 575 | 25201 | 1.018978764 |
| screen_fixed_combined_1 | 14.2183 | 591 | 37825 | 1.035590307 |
| screen_fixed_combined_2 | 14.0828 | 590 | 37506 | 1.036800632 |
| screen_fixed_combined_3 | 13.9341 | 578 | 37451 | 1.041334120 |
| screen_fixed_graph_1 | 15.7471 | 590 | 38058 | 1.035724776 |
| screen_fixed_graph_2 | 15.3926 | 587 | 37369 | 1.035457704 |
| screen_fixed_graph_3 | 15.5150 | 587 | 38073 | 1.035830419 |
| screen_sphere_combined_1 | 9.4171 | 570 | 24752 | 1.018953377 |
| screen_sphere_combined_2 | 9.2534 | 559 | 23980 | 1.018953605 |
| screen_sphere_combined_3 | 9.4044 | 566 | 24094 | 1.019994212 |
| screen_sphere_graph_1 | 10.4709 | 556 | 23983 | 1.026978103 |
| screen_sphere_graph_2 | 11.1972 | 579 | 25476 | 1.018953060 |
| screen_sphere_graph_3 | 10.8314 | 578 | 24833 | 1.019582712 |

原始数据：`runs/report_execution_20261007/`。未完成和资源失败不得进入比率。
