<!-- Surprise-only vs text-only vs combined, car_0_1_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 | cw_mean | cw_t_nw | cw_p | cw_seasons | oos_r2_vs_surprise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 2111 | (none) | surprise_only | surprise_only | car_0_1_ff4 | 2111 | 12 | 0.3496 | 15.88 | 1.0 | 0.0542 | 10.0843 | 12 | 0.6384 | 0.6717 | 0.1142 |  |  |  |  |  |
| validation | 2111 | (none) | sue_rank (raw signal) | sue_rank | car_0_1_ff4 | 2111 | 12 | 0.3509 | 15.57 | 1.0 | 0.0542 | 10.0843 | 12 | 0.6358 | 0.6747 |  |  |  |  |  |  |
| validation | 2111 | lm_tone | text (raw signal) | lm_tone_rk | car_0_1_ff4 | 2111 | 12 | 0.2949 | 16.93 | 1.0 | 0.0529 | 11.0811 | 12 | 0.6095 | 0.6442 |  |  |  |  |  |  |
| validation | 2111 | lm_tone | text_only | text_only__lm_tone | car_0_1_ff4 | 2111 | 12 | 0.2949 | 16.93 | 1.0 | 0.0529 | 11.0811 | 12 | 0.6095 | 0.6442 | 0.0838 |  |  |  |  |  |
| validation | 2111 | lm_tone | combined | combined__lm_tone | car_0_1_ff4 | 2111 | 12 | 0.3793 | 17.62 | 1.0 | 0.0643 | 15.1121 | 12 | 0.6488 | 0.6869 | 0.1405 | 0.0002 | 10.1932 | 0.0 | 12.0 | 0.0296 |
| validation | 2111 | lm_negshare | text (raw signal) | lm_negshare_rk | car_0_1_ff4 | 2111 | 12 | 0.2672 | 15.64 | 1.0 | 0.0491 | 12.3588 | 12 | 0.6005 | 0.6343 |  |  |  |  |  |  |
| validation | 2111 | lm_negshare | text_only | text_only__lm_negshare | car_0_1_ff4 | 2111 | 12 | 0.2672 | 15.64 | 1.0 | 0.0491 | 12.3588 | 12 | 0.6005 | 0.6343 | 0.0639 |  |  |  |  |  |
| validation | 2111 | lm_negshare | combined | combined__lm_negshare | car_0_1_ff4 | 2111 | 12 | 0.3756 | 18.27 | 1.0 | 0.0601 | 11.9594 | 12 | 0.6474 | 0.6864 | 0.1334 | 0.0001 | 10.1725 | 0.0 | 12.0 | 0.0216 |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | text (raw signal) | tfidf_ridge__car_0_1_ff4_rk | car_0_1_ff4 | 2111 | 12 | 0.2587 | 9.62 | 1.0 | 0.0438 | 8.8487 | 12 | 0.5945 | 0.6268 |  |  |  |  |  |  |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | text_only | text_only__tfidf_ridge__car_0_1_ff4 | car_0_1_ff4 | 2111 | 12 | 0.2587 | 9.62 | 1.0 | 0.0438 | 8.8487 | 12 | 0.5945 | 0.6268 | 0.0688 |  |  |  |  |  |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | combined | combined__tfidf_ridge__car_0_1_ff4 | car_0_1_ff4 | 2111 | 12 | 0.3697 | 14.81 | 1.0 | 0.0595 | 11.8428 | 12 | 0.6408 | 0.6827 | 0.1405 | 0.0001 | 5.9559 | 0.0 | 12.0 | 0.0297 |
| validation | 2111 | tfidf_logit__up | text (raw signal) | tfidf_logit__up_rk | car_0_1_ff4 | 2111 | 12 | 0.2506 | 9.36 | 1.0 | 0.0463 | 6.2924 | 12 | 0.5897 | 0.6252 |  |  |  |  |  |  |
| validation | 2111 | tfidf_logit__up | text_only | text_only__tfidf_logit__up | car_0_1_ff4 | 2111 | 12 | 0.2506 | 9.36 | 1.0 | 0.0463 | 6.2924 | 12 | 0.5897 | 0.6252 | 0.0636 |  |  |  |  |  |
| validation | 2111 | tfidf_logit__up | combined | combined__tfidf_logit__up | car_0_1_ff4 | 2111 | 12 | 0.3687 | 14.75 | 1.0 | 0.062 | 12.912 | 12 | 0.6427 | 0.6827 | 0.1388 | 0.0001 | 5.3829 | 0.0001 | 12.0 | 0.0278 |
| validation | 2111 | llm__good_llm__tone_v1__raw | text (raw signal) | llm__good_llm__tone_v1__raw_rk | car_0_1_ff4 | 2111 | 12 | 0.3141 | 25.19 | 1.0 | 0.0593 | 10.9578 | 12 | 0.6235 | 0.6562 |  |  |  |  |  |  |
| validation | 2111 | llm__good_llm__tone_v1__raw | text_only | text_only__llm__good_llm__tone_v1__raw | car_0_1_ff4 | 2111 | 12 | 0.3141 | 25.19 | 1.0 | 0.0593 | 10.9578 | 12 | 0.6235 | 0.6562 | 0.0946 |  |  |  |  |  |
| validation | 2111 | llm__good_llm__tone_v1__raw | combined | combined__llm__good_llm__tone_v1__raw | car_0_1_ff4 | 2111 | 12 | 0.39 | 19.55 | 1.0 | 0.0653 | 13.398 | 12 | 0.6384 | 0.6932 | 0.1475 | 0.0002 | 17.1287 | 0.0 | 12.0 | 0.0376 |
| validation | 2111 | llm__noise_llm__tone_v1__raw | text (raw signal) | llm__noise_llm__tone_v1__raw_rk | car_0_1_ff4 | 2111 | 12 | 0.0098 | 0.56 | 0.5 | -0.0011 | -0.2822 | 12 | 0.5223 | 0.5169 |  |  |  |  |  |  |
| validation | 2111 | llm__noise_llm__tone_v1__raw | text_only | text_only__llm__noise_llm__tone_v1__raw | car_0_1_ff4 | 2111 | 12 | 0.0098 | 0.56 | 0.5 | -0.0011 | -0.2822 | 12 | 0.5223 | 0.5169 | -0.0047 |  |  |  |  |  |
| validation | 2111 | llm__noise_llm__tone_v1__raw | combined | combined__llm__noise_llm__tone_v1__raw | car_0_1_ff4 | 2111 | 12 | 0.3479 | 16.12 | 1.0 | 0.0542 | 10.0843 | 12 | 0.6246 | 0.6734 | 0.1145 | 0.0 | 0.9147 | 0.19 | 12.0 | 0.0003 |
