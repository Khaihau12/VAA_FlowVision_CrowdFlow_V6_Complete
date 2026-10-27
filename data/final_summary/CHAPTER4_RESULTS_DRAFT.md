# WILDTRACK — TỔNG HỢP KẾT QUẢ THỰC NGHIỆM

## 1. Protocol cuối cùng

- Held-out TEST gồm 80 mốc thời gian đối với các thí nghiệm baseline, dropout, detection loss và các phân tích hành vi.
- Fusion tham chiếu được giữ cố định là Hungarian ground-plane matching với ngưỡng 100 cm và trung bình tọa độ của cặp match.
- Synchronization Shift primary chỉ gồm `0, ±0.5, ±1.0, ±2.0 s`; kết quả `±1.5 s`, nếu có, chỉ được giữ dưới dạng exploratory.
- Không điều chỉnh tham số dựa trên TEST sau khi TEST đã được mở.

## 2. Trạng thái các hạng mục

|   id | experiment                          | status   | primary_source                                                                        |
|-----:|:------------------------------------|:---------|:--------------------------------------------------------------------------------------|
|    1 | Single / Multi-camera baseline      | COMPLETE | final_system_performance.csv                                                          |
|    2 | Controlled camera-error correlation | COMPLETE | /kaggle/working/CONTROLLED_ERROR_CORRELATION/controlled_error_correlation_summary.csv |
|    3 | Camera observation dropout          | COMPLETE | /kaggle/working/FRAME_DROP_ROBUSTNESS/frame_drop_seed_results.csv                     |
|    4 | Synchronization shift               | COMPLETE | /kaggle/working/SYNC_SHIFT_ROBUSTNESS/sync_shift_results.csv                          |
|    5 | Controlled detection loss           | COMPLETE | /kaggle/working/DETECTION_LOSS_ROBUSTNESS/detection_loss_seed_results.csv             |
|    6 | Robustness master                   | COMPLETE | rebuilt from raw robustness experiments                                               |
|    7 | Fusion-method comparison            | COMPLETE | /kaggle/working/FUSION_METHOD_COMPARISON/fusion_paper_ready_table.csv                 |
|    8 | Zone analysis / occupancy           | COMPLETE | zone_occupancy_summary.csv                                                            |
|    9 | Global temporal ground tracking     | COMPLETE | test_tracking_metrics.csv                                                             |
|   10 | Entry / Exit                        | COMPLETE | test_entry_exit_pair_summary.csv                                                      |
|   11 | Dwell time                          | COMPLETE | dwell_pair_summary.csv                                                                |
|   12 | Zone-to-zone flow transition        | COMPLETE | flow_transition_pair_summary.csv                                                      |

## 3. Bảng tổng quan theo cấu hình camera

| pair   |   Baseline_Occupancy_MAE |   Baseline_Ground_F1 |   Baseline_Ground_Recall |   Tracking_IDF1 |   EntryExit_F1 |   Dwell_Visit_F1 |   Dwell_MAE_seconds |   Flow_Event_F1 |   Flow_Count_WAPE |   Flow_Distribution_TV |
|:-------|-------------------------:|---------------------:|-------------------------:|----------------:|---------------:|-----------------:|--------------------:|----------------:|------------------:|-----------------------:|
| Pair12 |                    5.075 |               0.7591 |                   0.6642 |          0.4684 |         0.6875 |           0.2727 |                 0.5 |          0.6875 |            0.2    |                 0.0686 |
| Pair34 |                    5.825 |               0.4706 |                   0.403  |          0.3101 |         0.3636 |           0      |               nan   |          0.2727 |            0.5333 |                 0.3333 |
| Pair56 |                    4.65  |               0.6931 |                   0.6138 |          0.3848 |         0.5    |           0.1481 |                 0.5 |          0.4062 |            0.7333 |                 0.3039 |

## 4. Baseline multi-camera

| pair   | system    |   occupancy_MAE |   ground_precision |   ground_recall |   ground_f1 |   ground_MAE_cm |   ground_RMSE_cm |
|:-------|:----------|----------------:|-------------------:|----------------:|------------:|----------------:|-----------------:|
| Pair12 | Cam1+Cam2 |           5.075 |             0.8856 |          0.6642 |      0.7591 |         25.7281 |          32.1713 |
| Pair34 | Cam3+Cam4 |           5.825 |             0.5654 |          0.403  |      0.4706 |         41.1308 |          47.9469 |
| Pair56 | Cam5+Cam6 |           4.65  |             0.796  |          0.6138 |      0.6931 |         26.6906 |          33.5093 |

## 5. Tương quan lỗi giữa hai camera

| pair   | camera_A   | camera_B   |   test_n |   raw_Pearson_r |   raw_Pearson_p |   raw_Spearman_rho |   relative_error_Pearson_r |   relative_error_Pearson_p |   relative_error_Spearman_rho |   controlled_residual_Pearson_r |   controlled_residual_Pearson_p |   controlled_residual_Spearman_rho |   controlled_residual_Spearman_p |   raw_minus_controlled_r |   correlation_reduction_pct |   Cam1_VAL_crowd_R2 |   Cam2_VAL_crowd_R2 |   Cam1_TEST_crowd_R2 |   Cam2_TEST_crowd_R2 |   joint_failure_observed |   joint_failure_expected_if_independent |   joint_failure_excess |   joint_failure_lift |   failure_phi |   P_Cam2_fail_given_Cam1_fail |   P_Cam1_fail_given_Cam2_fail |   Cam2_rescue_when_Cam1_fails |   Cam1_rescue_when_Cam2_fails |   Cam3_VAL_crowd_R2 |   Cam4_VAL_crowd_R2 |   Cam3_TEST_crowd_R2 |   Cam4_TEST_crowd_R2 |   P_Cam4_fail_given_Cam3_fail |   P_Cam3_fail_given_Cam4_fail |   Cam4_rescue_when_Cam3_fails |   Cam3_rescue_when_Cam4_fails |   Cam5_VAL_crowd_R2 |   Cam6_VAL_crowd_R2 |   Cam5_TEST_crowd_R2 |   Cam6_TEST_crowd_R2 |   P_Cam6_fail_given_Cam5_fail |   P_Cam5_fail_given_Cam6_fail |   Cam6_rescue_when_Cam5_fails |   Cam5_rescue_when_Cam6_fails | _result_source                                                                        |
|:-------|:-----------|:-----------|---------:|----------------:|----------------:|-------------------:|---------------------------:|---------------------------:|------------------------------:|--------------------------------:|--------------------------------:|-----------------------------------:|---------------------------------:|-------------------------:|----------------------------:|--------------------:|--------------------:|---------------------:|---------------------:|-------------------------:|----------------------------------------:|-----------------------:|---------------------:|--------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|--------------------:|--------------------:|---------------------:|---------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|--------------------:|--------------------:|---------------------:|---------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|:--------------------------------------------------------------------------------------|
| Pair12 | Cam1       | Cam2       |       80 |          0.9124 |               0 |             0.9182 |                     0.5598 |                     0      |                        0.6286 |                          0.0541 |                          0.6337 |                             0.1776 |                           0.115  |                   0.8583 |                     94.0711 |              0.8917 |              0.8272 |               0.9032 |               0.6093 |                   0.175  |                                  0.0422 |                 0.1328 |               4.1481 |        0.8149 |                        0.7778 |                        0.9333 |                        0.2222 |                        0.0667 |            nan      |            nan      |             nan      |             nan      |                      nan      |                      nan      |                      nan      |                      nan      |            nan      |            nan      |             nan      |             nan      |                      nan      |                      nan      |                      nan      |                      nan      | /kaggle/working/CONTROLLED_ERROR_CORRELATION/controlled_error_correlation_summary.csv |
| Pair34 | Cam3       | Cam4       |       80 |          0.9129 |               0 |             0.9161 |                     0.4535 |                     0      |                        0.4377 |                         -0.0978 |                          0.388  |                            -0.1141 |                           0.3134 |                   1.0108 |                    110.715  |            nan      |            nan      |             nan      |             nan      |                   0.125  |                                  0.0281 |                 0.0969 |               4.4444 |        0.6951 |                      nan      |                      nan      |                      nan      |                      nan      |              0.8917 |              0.8779 |               0.8975 |               0.8871 |                        0.6667 |                        0.8333 |                        0.3333 |                        0.1667 |            nan      |            nan      |             nan      |             nan      |                      nan      |                      nan      |                      nan      |                      nan      | /kaggle/working/CONTROLLED_ERROR_CORRELATION/controlled_error_correlation_summary.csv |
| Pair56 | Cam5       | Cam6       |       80 |          0.9045 |               0 |             0.8606 |                     0.3753 |                     0.0006 |                        0.4038 |                         -0.2309 |                          0.0394 |                            -0.1938 |                           0.0851 |                   1.1354 |                    125.525  |            nan      |            nan      |             nan      |             nan      |                   0.1125 |                                  0.0241 |                 0.0884 |               4.6753 |        0.6759 |                      nan      |                      nan      |                      nan      |                      nan      |            nan      |            nan      |             nan      |             nan      |                      nan      |                      nan      |                      nan      |                      nan      |              0.8946 |              0.7054 |               0.8085 |               0.8109 |                        0.6429 |                        0.8182 |                        0.3571 |                        0.1818 | /kaggle/working/CONTROLLED_ERROR_CORRELATION/controlled_error_correlation_summary.csv |

Sai số occupancy thô có thể tương quan cao do cả hai camera cùng chịu tác động của độ đông của cảnh. Vì vậy kết luận chính nên dựa đồng thời trên tương quan phần dư sau khi kiểm soát crowd size và xác suất đồng thời xuất hiện các failure state lớn.


## 6. Robustness

### 6.1 Maximum-severity endpoint

| perturbation          | pair   | physical_level   |   F1_retention |   F1_change_pct |   Recall_retention |   Recall_change_pct |   occupancy_MAE_ratio |   occupancy_MAE_change_pct |   Ground_MAE_ratio |
|:----------------------|:-------|:-----------------|---------------:|----------------:|-------------------:|--------------------:|----------------------:|---------------------------:|-------------------:|
| Detection Loss        | Pair12 | 40%              |         0.9135 |         -8.6532 |             0.8412 |            -15.8801 |                1.5387 |                    53.867  |             1.0214 |
| Detection Loss        | Pair34 | 40%              |         0.9275 |         -7.2548 |             0.854  |            -14.5988 |                1.4674 |                    46.7382 |             0.9766 |
| Detection Loss        | Pair56 | 40%              |         0.9435 |         -5.6498 |             0.8641 |            -13.5866 |                1.6374 |                    63.7366 |             0.9627 |
| Frame Drop            | Pair12 | 40%              |         0.9048 |         -9.5163 |             0.8369 |            -16.3109 |                1.5158 |                    51.5764 |             1.0117 |
| Frame Drop            | Pair34 | 40%              |         0.9013 |         -9.868  |             0.8309 |            -16.9136 |                1.4678 |                    46.7811 |             0.9922 |
| Frame Drop            | Pair56 | 40%              |         0.9223 |         -7.7707 |             0.8508 |            -14.924  |                1.5927 |                    59.2742 |             0.9811 |
| Synchronization Shift | Pair12 | |Δt|=2.0s        |         0.833  |        -16.7017 |             0.8494 |            -15.0571 |                0.8839 |                   -11.6147 |             1.0594 |
| Synchronization Shift | Pair34 | |Δt|=2.0s        |         0.8843 |        -11.5667 |             0.8881 |            -11.1864 |                0.9768 |                    -2.3227 |             1.0387 |
| Synchronization Shift | Pair56 | |Δt|=2.0s        |         0.7752 |        -22.4788 |             0.7804 |            -21.9621 |                0.9792 |                    -2.0767 |             1.1027 |

### 6.2 Area under robustness curves

| perturbation          | pair   |   F1_retention_AUC |   Recall_retention_AUC |   Occupancy_MAE_ratio_AUC |
|:----------------------|:-------|-------------------:|-----------------------:|--------------------------:|
| Detection Loss        | Pair12 |             0.9592 |                 0.9219 |                    1.2674 |
| Detection Loss        | Pair34 |             0.9694 |                 0.9315 |                    1.2313 |
| Detection Loss        | Pair56 |             0.9744 |                 0.934  |                    1.3146 |
| Frame Drop            | Pair12 |             0.955  |                 0.9203 |                    1.2542 |
| Frame Drop            | Pair34 |             0.9547 |                 0.9182 |                    1.232  |
| Frame Drop            | Pair56 |             0.9651 |                 0.9293 |                    1.2871 |
| Synchronization Shift | Pair12 |             0.905  |                 0.9147 |                    0.9347 |
| Synchronization Shift | Pair34 |             0.9463 |                 0.9492 |                    0.9844 |
| Synchronization Shift | Pair56 |             0.8756 |                 0.8782 |                    0.9836 |

Ground MAE được diễn giải cùng Recall/F1 vì metric này chỉ được tính trên các prediction–GT match thành công và có thể giảm do survivorship/selection effect.


## 7. So sánh phương pháp fusion

| pair   | method              |   occupancy_MAE |   ground_precision |   ground_recall |   ground_f1 |   ground_MAE_cm |   ground_RMSE_cm |   cross_camera_matches |
|:-------|:--------------------|----------------:|-------------------:|----------------:|------------:|----------------:|-----------------:|-----------------------:|
| Pair12 | Greedy Equal        |          5.55   |             0.9158 |          0.6629 |      0.7691 |         25.4405 |          31.4952 |                    199 |
| Pair12 | Hungarian Best-View |          5.075  |             0.8856 |          0.6642 |      0.7591 |         25.1244 |          31.6699 |                    157 |
| Pair12 | Hungarian Equal     |          5.075  |             0.8856 |          0.6642 |      0.7591 |         25.7281 |          32.1713 |                    157 |
| Pair12 | Hungarian Weighted  |          5.075  |             0.8864 |          0.6648 |      0.7598 |         25.2081 |          31.7573 |                    157 |
| Pair12 | Naive Union         |          4.0625 |             0.7821 |          0.6629 |      0.7176 |         27.1396 |          34.9714 |                      0 |
| Pair34 | Greedy Equal        |          5.9    |             0.5693 |          0.4036 |      0.4723 |         41.1147 |          47.9088 |                     46 |
| Pair34 | Hungarian Best-View |          5.825  |             0.5663 |          0.4036 |      0.4713 |         40.5901 |          47.3262 |                     40 |
| Pair34 | Hungarian Equal     |          5.825  |             0.5654 |          0.403  |      0.4706 |         41.1308 |          47.9469 |                     40 |
| Pair34 | Hungarian Weighted  |          5.825  |             0.5654 |          0.403  |      0.4706 |         41.0394 |          47.8079 |                     40 |
| Pair34 | Naive Union         |          5.475  |             0.5388 |          0.3974 |      0.4574 |         39.9755 |          46.705  |                      0 |
| Pair56 | Greedy Equal        |          5.6125 |             0.8525 |          0.6144 |      0.7141 |         24.6887 |          30.7372 |                    230 |
| Pair56 | Hungarian Best-View |          4.65   |             0.8    |          0.6169 |      0.6966 |         27.8482 |          35.7619 |                    149 |
| Pair56 | Hungarian Equal     |          4.65   |             0.796  |          0.6138 |      0.6931 |         26.6906 |          33.5093 |                    149 |
| Pair56 | Hungarian Weighted  |          4.65   |             0.796  |          0.6138 |      0.6931 |         26.7298 |          33.571  |                    149 |
| Pair56 | Naive Union         |          3.4875 |             0.7091 |          0.6126 |      0.6573 |         27.6901 |          35.462  |                      0 |

Naive Union có thể vô tình cải thiện occupancy count khi hệ thống đang undercount nhưng không xử lý duplicate. Hungarian Equal vẫn được giữ làm phương pháp tham chiếu đã khóa; không đổi phương pháp chính sau khi quan sát TEST.


## 8. Zone occupancy

### Fusion so với single-camera tốt nhất

| pair   | best_single_system   |   best_single_pooled_zone_MAE |   fusion_pooled_zone_MAE |   fusion_zone_MAE_improvement_pct |
|:-------|:---------------------|------------------------------:|-------------------------:|----------------------------------:|
| Pair12 | Single Cam2          |                        3.8542 |                    1.875 |                           51.3514 |
| Pair34 | Single Cam3          |                        3.8625 |                    2.475 |                           35.9223 |
| Pair56 | Single Cam5          |                        3.5458 |                    2.55  |                           28.0846 |

### Zone boundary diagnostic

| pair   |   matched_GT_pred |   correct_zone_matches |   zone_assignment_accuracy |   matched_ground_MAE_cm |
|:-------|------------------:|-----------------------:|---------------------------:|------------------------:|
| Pair12 |              1068 |                   1016 |                     0.9513 |                 25.7281 |
| Pair34 |               648 |                    633 |                     0.9769 |                 41.1308 |
| Pair56 |               987 |                    898 |                     0.9098 |                 26.6906 |

## 9. Global temporal ground tracking

|   total_GT_observations |   total_pred_observations |   detection_TP |   detection_FP |   detection_FN |   detection_precision |   detection_recall |   detection_f1 |   matched_ground_MAE_cm |   IDTP |   IDFP |   IDFN |   ID_precision |   ID_recall |   IDF1 |   ID_switches |   fragmentations |   MOTA |   weighted_track_purity |   predicted_trajectories | pair   |   selected_gate_cm |   selected_max_missed |
|------------------------:|--------------------------:|---------------:|---------------:|---------------:|----------------------:|-------------------:|---------------:|------------------------:|-------:|-------:|-------:|---------------:|------------:|-------:|--------------:|-----------------:|-------:|------------------------:|-------------------------:|:-------|-------------------:|----------------------:|
|                    1608 |                      1206 |           1068 |            138 |            540 |                0.8856 |             0.6642 |         0.7591 |                 25.7281 |    659 |    547 |    949 |         0.5464 |      0.4098 | 0.4684 |            64 |               83 | 0.5386 |                  0.8202 |                       90 | Pair12 |                125 |                     2 |
|                    1608 |                      1146 |            648 |            498 |            960 |                0.5654 |             0.403  |         0.4706 |                 41.1308 |    427 |    719 |   1181 |         0.3726 |      0.2655 | 0.3101 |            52 |              108 | 0.0609 |                  0.8194 |                      107 | Pair34 |                125 |                     2 |
|                    1608 |                      1240 |            987 |            253 |            621 |                0.796  |             0.6138 |         0.6931 |                 26.6906 |    548 |    692 |   1060 |         0.4419 |      0.3408 | 0.3848 |           114 |              132 | 0.3856 |                  0.7791 |                       79 | Pair56 |                125 |                     2 |

Tracking được xem là reference tracker cho phân tích luồng; IDF1 và fragmentation phải được nêu rõ khi diễn giải các metric cấp cá nhân như Dwell.


## 10. Entry / Exit

| pair   |   GT_events |   Pred_events |   TP |   FP |   FN |   Precision |   Recall |     F1 |   Timing_MAE_seconds |
|:-------|------------:|--------------:|-----:|-----:|-----:|------------:|---------:|-------:|---------------------:|
| Pair12 |          60 |            68 |   44 |   24 |   16 |      0.6471 |   0.7333 | 0.6875 |               0.2045 |
| Pair34 |          60 |            28 |   16 |   12 |   44 |      0.5714 |   0.2667 | 0.3636 |               0.4688 |
| Pair56 |          60 |            68 |   32 |   36 |   28 |      0.4706 |   0.5333 | 0.5    |               0.2656 |

## 11. Dwell time

| pair   |   GT_complete_visits |   Pred_complete_visits |   Matched_complete_visits |   FP_visits |   FN_visits |   Visit_Precision |   Visit_Recall |   Visit_F1 |   Dwell_MAE_seconds |   Dwell_RMSE_seconds |   Dwell_Median_AE_seconds |   Dwell_Bias_seconds |   Dwell_WAPE |   Entry_Timing_MAE_seconds |   Exit_Timing_MAE_seconds |   Matched_GT_mean_dwell_seconds |   Matched_Pred_mean_dwell_seconds |   Dwell_Pearson_r |
|:-------|---------------------:|-----------------------:|--------------------------:|------------:|------------:|------------------:|---------------:|-----------:|--------------------:|---------------------:|--------------------------:|---------------------:|-------------:|---------------------------:|--------------------------:|--------------------------------:|----------------------------------:|------------------:|
| Pair12 |                    9 |                     13 |                         3 |          10 |           6 |            0.2308 |         0.3333 |     0.2727 |                 0.5 |                  0.5 |                       0.5 |               0.1667 |       0.0448 |                     0.1667 |                    0.3333 |                         11.1667 |                           11.3333 |            0.9177 |
| Pair34 |                    9 |                      1 |                         0 |           1 |           9 |            0      |         0      |     0      |               nan   |                nan   |                     nan   |             nan      |     nan      |                   nan      |                  nan      |                        nan      |                          nan      |          nan      |
| Pair56 |                    9 |                     18 |                         2 |          16 |           7 |            0.1111 |         0.2222 |     0.1481 |                 0.5 |                  0.5 |                       0.5 |               0      |       0.0476 |                     0.25   |                    0.25   |                         10.5    |                           10.5    |          nan      |

Dwell MAE là metric có điều kiện trên các complete visit được match thành công. Vì số complete GT visit trên TEST nhỏ và Visit Recall thấp, không được dùng MAE nhỏ để tuyên bố toàn hệ thống ước lượng dwell chính xác cao.


## 12. Zone-to-zone flow transition

| pair   |   GT_transition_events |   Pred_transition_events |   TP |   FP |   FN |   Transition_Precision |   Transition_Recall |   Transition_F1 |   Timing_MAE_seconds |   Timing_Bias_seconds |   GT_total_flow |   Pred_total_flow |   Total_flow_error |   Total_flow_abs_error_pct |   Transition_Count_MAE |   Transition_Count_RMSE |   Transition_Count_WAPE |   Transition_Distribution_TV |   Max_abs_transition_count_error |   temporal_bin_seconds |   number_bins |   directed_bin_cells |   Temporal_Transition_Count_MAE |   Temporal_Transition_Count_RMSE |   Temporal_Transition_Count_WAPE |   Total_Flow_per_Bin_MAE |
|:-------|-----------------------:|-------------------------:|-----:|-----:|-----:|-----------------------:|--------------------:|----------------:|---------------------:|----------------------:|----------------:|------------------:|-------------------:|---------------------------:|-----------------------:|------------------------:|------------------------:|-----------------------------:|---------------------------------:|-----------------------:|--------------:|---------------------:|--------------------------------:|---------------------------------:|---------------------------------:|-------------------------:|
| Pair12 |                     30 |                       34 |   22 |   12 |    8 |                 0.6471 |              0.7333 |          0.6875 |               0.2045 |               -0.0682 |              30 |                34 |                  4 |                    13.3333 |                 1      |                  1.5275 |                  0.2    |                       0.0686 |                                3 |                      5 |             8 |                   48 |                          0.3333 |                           0.736  |                           0.5333 |                     2    |
| Pair34 |                     30 |                       14 |    6 |    8 |   24 |                 0.4286 |              0.2    |          0.2727 |               0.3333 |                0.3333 |              30 |                14 |                -16 |                    53.3333 |                 2.6667 |                  3.6056 |                  0.5333 |                       0.3333 |                                7 |                      5 |             8 |                   48 |                          0.5    |                           0.9354 |                           0.8    |                     2.25 |
| Pair56 |                     30 |                       34 |   13 |   21 |   17 |                 0.3824 |              0.4333 |          0.4062 |               0.2308 |               -0.1538 |              30 |                34 |                  4 |                    13.3333 |                 3.6667 |                  5      |                  0.7333 |                       0.3039 |                                9 |                      5 |             8 |                   48 |                          0.75   |                           1.354  |                           1.2    |                     2.25 |

### Directed transition count

| pair   | from_zone   | to_zone   | transition       |   GT_count |   Pred_count |   count_error |   abs_count_error |   squared_count_error |
|:-------|:------------|:----------|:-----------------|-----------:|-------------:|--------------:|------------------:|----------------------:|
| Pair12 | Zone A      | Zone B    | Zone A -> Zone B |          6 |            9 |             3 |                 3 |                     9 |
| Pair12 | Zone A      | Zone C    | Zone A -> Zone C |          0 |            0 |             0 |                 0 |                     0 |
| Pair12 | Zone B      | Zone A    | Zone B -> Zone A |         14 |           16 |             2 |                 2 |                     4 |
| Pair12 | Zone B      | Zone C    | Zone B -> Zone C |          7 |            6 |            -1 |                 1 |                     1 |
| Pair12 | Zone C      | Zone A    | Zone C -> Zone A |          0 |            0 |             0 |                 0 |                     0 |
| Pair12 | Zone C      | Zone B    | Zone C -> Zone B |          3 |            3 |             0 |                 0 |                     0 |
| Pair34 | Zone A      | Zone B    | Zone A -> Zone B |          6 |            4 |            -2 |                 2 |                     4 |
| Pair34 | Zone A      | Zone C    | Zone A -> Zone C |          0 |            0 |             0 |                 0 |                     0 |
| Pair34 | Zone B      | Zone A    | Zone B -> Zone A |         14 |           10 |            -4 |                 4 |                    16 |
| Pair34 | Zone B      | Zone C    | Zone B -> Zone C |          7 |            0 |            -7 |                 7 |                    49 |
| Pair34 | Zone C      | Zone A    | Zone C -> Zone A |          0 |            0 |             0 |                 0 |                     0 |
| Pair34 | Zone C      | Zone B    | Zone C -> Zone B |          3 |            0 |            -3 |                 3 |                     9 |
| Pair56 | Zone A      | Zone B    | Zone A -> Zone B |          6 |           15 |             9 |                 9 |                    81 |
| Pair56 | Zone A      | Zone C    | Zone A -> Zone C |          0 |            0 |             0 |                 0 |                     0 |
| Pair56 | Zone B      | Zone A    | Zone B -> Zone A |         14 |           18 |             4 |                 4 |                    16 |
| Pair56 | Zone B      | Zone C    | Zone B -> Zone C |          7 |            0 |            -7 |                 7 |                    49 |
| Pair56 | Zone C      | Zone A    | Zone C -> Zone A |          0 |            0 |             0 |                 0 |                     0 |
| Pair56 | Zone C      | Zone B    | Zone C -> Zone B |          3 |            1 |            -2 |                 2 |                     4 |

Flow event F1 và aggregate flow-count error đo hai khía cạnh khác nhau. Tổng số transition có thể gần đúng nhưng phân bố theo hướng vẫn sai đáng kể, vì vậy cần báo cáo cả transition event metrics và flow matrix/count metrics.


## 13. Các kết quả nổi bật tự động trích xuất

| criterion                                 | metric                 | best_pair   |   value |
|:------------------------------------------|:-----------------------|:------------|--------:|
| Best baseline ground-detection F1         | Baseline_Ground_F1     | Pair12      |  0.7591 |
| Lowest baseline occupancy MAE             | Baseline_Occupancy_MAE | Pair56      |  4.65   |
| Best global identity continuity           | Tracking_IDF1          | Pair12      |  0.4684 |
| Best Entry/Exit event recovery            | EntryExit_F1           | Pair12      |  0.6875 |
| Best complete-visit recovery              | Dwell_Visit_F1         | Pair12      |  0.2727 |
| Best directed flow-event recovery         | Flow_Event_F1          | Pair12      |  0.6875 |
| Lowest aggregate directed-flow count WAPE | Flow_Count_WAPE        | Pair12      |  0.2    |
| Closest flow-direction distribution       | Flow_Distribution_TV   | Pair12      |  0.0686 |

## 14. Giới hạn cần ghi rõ

1. WILDTRACK là dữ liệu public cross-view, không thay thế cho dữ liệu 2-camera tự thu trong môi trường gần bài toán nhà ga.
2. Các zone A/B/C là vùng không gian tổng quát, không được gọi là Security, Gate, Check-in hoặc các khu chức năng sân bay.
3. Global tracking chỉ là reference ground tracker và có identity fragmentation đáng kể.
4. Dwell-time evidence còn hạn chế do số complete visits nhỏ.
5. Không tiến hành tuning mới dựa trên held-out TEST.
