<!-- Surprise-only vs text-only vs combined, post_car_1_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 | cw_mean | cw_t_nw | cw_p | cw_seasons | oos_r2_vs_surprise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 2111 | (none) | surprise_only | surprise_only | post_car_1_ff4 | 2111 | 12 | 0.1567 | 10.82 | 1.0 | 0.0156 | 12.7168 | 12 | 0.56 | 0.5742 | 0.0223 |  |  |  |  |  |
| validation | 2111 | (none) | sue_rank (raw signal) | sue_rank | post_car_1_ff4 | 2111 | 12 | 0.1599 | 10.17 | 1.0 | 0.0148 | 5.3903 | 12 | 0.5554 | 0.5729 |  |  |  |  |  |  |
| validation | 2111 | lm_tone | text (raw signal) | lm_tone_rk | post_car_1_ff4 | 2111 | 12 | 0.1463 | 14.46 | 1.0 | 0.0143 | 7.0743 | 12 | 0.5595 | 0.5732 |  |  |  |  |  |  |
| validation | 2111 | lm_tone | text_only | text_only__lm_tone | post_car_1_ff4 | 2111 | 12 | 0.1463 | 14.46 | 1.0 | 0.0143 | 7.0743 | 12 | 0.5595 | 0.5732 | 0.0219 |  |  |  |  |  |
| validation | 2111 | lm_tone | combined | combined__lm_tone | post_car_1_ff4 | 2111 | 12 | 0.1771 | 12.05 | 1.0 | 0.0146 | 9.3206 | 12 | 0.5654 | 0.5873 | 0.0301 | 0.0 | 9.5609 | 0.0 | 12.0 | 0.008 |
| validation | 2111 | lm_negshare | text (raw signal) | lm_negshare_rk | post_car_1_ff4 | 2111 | 12 | 0.1456 | 12.32 | 1.0 | 0.0142 | 10.097 | 12 | 0.5527 | 0.5756 |  |  |  |  |  |  |
| validation | 2111 | lm_negshare | text_only | text_only__lm_negshare | post_car_1_ff4 | 2111 | 12 | 0.1456 | 12.32 | 1.0 | 0.0142 | 10.097 | 12 | 0.5527 | 0.5756 | 0.0212 |  |  |  |  |  |
| validation | 2111 | lm_negshare | combined | combined__lm_negshare | post_car_1_ff4 | 2111 | 12 | 0.1793 | 12.55 | 1.0 | 0.015 | 8.1703 | 12 | 0.5766 | 0.5894 | 0.0308 | 0.0 | 7.6374 | 0.0 | 12.0 | 0.0087 |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | text (raw signal) | tfidf_ridge__car_0_1_ff4_rk | post_car_1_ff4 | 2111 | 12 | 0.129 | 9.14 | 1.0 | 0.0131 | 5.6369 | 12 | 0.5465 | 0.567 |  |  |  |  |  |  |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | text_only | text_only__tfidf_ridge__car_0_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.129 | 9.14 | 1.0 | 0.0131 | 5.6369 | 12 | 0.5465 | 0.567 | 0.0169 |  |  |  |  |  |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | combined | combined__tfidf_ridge__car_0_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.1681 | 11.82 | 1.0 | 0.0157 | 5.5956 | 12 | 0.5678 | 0.5843 | 0.0259 | 0.0 | 3.9951 | 0.0011 | 12.0 | 0.0037 |
| validation | 2111 | tfidf_logit__up | text (raw signal) | tfidf_logit__up_rk | post_car_1_ff4 | 2111 | 12 | 0.1284 | 11.22 | 1.0 | 0.0142 | 4.2249 | 12 | 0.548 | 0.5671 |  |  |  |  |  |  |
| validation | 2111 | tfidf_logit__up | text_only | text_only__tfidf_logit__up | post_car_1_ff4 | 2111 | 12 | 0.1284 | 11.22 | 1.0 | 0.0142 | 4.2249 | 12 | 0.548 | 0.5671 | 0.0165 |  |  |  |  |  |
| validation | 2111 | tfidf_logit__up | combined | combined__tfidf_logit__up | post_car_1_ff4 | 2111 | 12 | 0.1678 | 11.21 | 1.0 | 0.0117 | 5.5501 | 12 | 0.572 | 0.5848 | 0.0259 | 0.0 | 5.0302 | 0.0002 | 12.0 | 0.0037 |
| validation | 2111 | llm__good_llm__tone_v1__raw | text (raw signal) | llm__good_llm__tone_v1__raw_rk | post_car_1_ff4 | 2111 | 12 | 0.1529 | 7.06 | 1.0 | 0.0146 | 4.1823 | 12 | 0.5565 | 0.5753 |  |  |  |  |  |  |
| validation | 2111 | llm__good_llm__tone_v1__raw | text_only | text_only__llm__good_llm__tone_v1__raw | post_car_1_ff4 | 2111 | 12 | 0.1529 | 7.06 | 1.0 | 0.0146 | 4.1823 | 12 | 0.5565 | 0.5753 | 0.0245 |  |  |  |  |  |
| validation | 2111 | llm__good_llm__tone_v1__raw | combined | combined__llm__good_llm__tone_v1__raw | post_car_1_ff4 | 2111 | 12 | 0.182 | 9.27 | 1.0 | 0.0104 | 3.0178 | 12 | 0.5692 | 0.5894 | 0.0319 | 0.0 | 5.8014 | 0.0001 | 12.0 | 0.0099 |
| validation | 2111 | llm__noise_llm__tone_v1__raw | text (raw signal) | llm__noise_llm__tone_v1__raw_rk | post_car_1_ff4 | 2111 | 12 | 0.0344 | 3.68 | 0.75 | 0.0059 | 4.1455 | 12 | 0.5043 | 0.5149 |  |  |  |  |  |  |
| validation | 2111 | llm__noise_llm__tone_v1__raw | text_only | text_only__llm__noise_llm__tone_v1__raw | post_car_1_ff4 | 2111 | 12 | -0.0344 | -3.68 | 0.25 | -0.0059 | -4.1455 | 12 | 0.4957 | 0.4851 | -0.0002 |  |  |  |  |  |
| validation | 2111 | llm__noise_llm__tone_v1__raw | combined | combined__llm__noise_llm__tone_v1__raw | post_car_1_ff4 | 2111 | 12 | 0.1536 | 11.47 | 1.0 | 0.0157 | 11.3107 | 12 | 0.5611 | 0.5731 | 0.0219 | -0.0 | -3.4466 | 0.9973 | 12.0 | -0.0003 |
