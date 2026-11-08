<!-- Surprise-only vs text-only vs combined, car_1_60_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 | cw_mean | cw_t_nw | cw_p | cw_seasons | oos_r2_vs_surprise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 2111 | (none) | surprise_only | surprise_only | car_1_60_ff4 | 2111 | 12 | -0.0245 | -1.23 | 0.42 | -0.0113 | -0.9171 | 12 | 0.5003 | 0.4952 | -0.0042 |  |  |  |  |  |
| validation | 2111 | (none) | sue_rank (raw signal) | sue_rank | car_1_60_ff4 | 2111 | 12 | -0.02 | -1.21 | 0.33 | -0.0071 | -0.8179 | 12 | 0.4976 | 0.5 |  |  |  |  |  |  |
| validation | 2111 | lm_tone | text (raw signal) | lm_tone_rk | car_1_60_ff4 | 2111 | 12 | 0.0077 | 0.41 | 0.67 | 0.0044 | 0.3733 | 12 | 0.5095 | 0.4998 |  |  |  |  |  |  |
| validation | 2111 | lm_tone | text_only | text_only__lm_tone | car_1_60_ff4 | 2111 | 12 | 0.0077 | 0.41 | 0.67 | 0.0044 | 0.3733 | 12 | 0.5095 | 0.4998 | -0.0009 |  |  |  |  |  |
| validation | 2111 | lm_tone | combined | combined__lm_tone | car_1_60_ff4 | 2111 | 12 | -0.0104 | -0.74 | 0.5 | 0.0003 | 0.0451 | 12 | 0.5081 | 0.4961 | -0.0035 | 0.0 | 1.4122 | 0.0928 | 12.0 | 0.0007 |
| validation | 2111 | lm_negshare | text (raw signal) | lm_negshare_rk | car_1_60_ff4 | 2111 | 12 | -0.0084 | -0.42 | 0.75 | -0.0095 | -0.8918 | 12 | 0.4998 | 0.4968 |  |  |  |  |  |  |
| validation | 2111 | lm_negshare | text_only | text_only__lm_negshare | car_1_60_ff4 | 2111 | 12 | -0.0084 | -0.42 | 0.75 | -0.0095 | -0.8918 | 12 | 0.4998 | 0.4968 | -0.0017 |  |  |  |  |  |
| validation | 2111 | lm_negshare | combined | combined__lm_negshare | car_1_60_ff4 | 2111 | 12 | -0.0208 | -1.44 | 0.5 | -0.0004 | -0.0491 | 12 | 0.5078 | 0.493 | -0.0044 | 0.0 | 0.2524 | 0.4027 | 12.0 | -0.0002 |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | text (raw signal) | tfidf_ridge__car_0_1_ff4_rk | car_1_60_ff4 | 2111 | 12 | 0.0126 | 0.64 | 0.83 | 0.0119 | 1.2396 | 12 | 0.509 | 0.5065 |  |  |  |  |  |  |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | text_only | text_only__tfidf_ridge__car_0_1_ff4 | car_1_60_ff4 | 2111 | 12 | 0.0126 | 0.64 | 0.83 | 0.0119 | 1.2396 | 12 | 0.509 | 0.5065 | -0.0004 |  |  |  |  |  |
| validation | 2111 | tfidf_ridge__car_0_1_ff4 | combined | combined__tfidf_ridge__car_0_1_ff4 | car_1_60_ff4 | 2111 | 12 | 0.006 | 0.41 | 0.5 | 0.0052 | 0.5186 | 12 | 0.5005 | 0.4993 | -0.0029 | 0.0 | 2.1651 | 0.0266 | 12.0 | 0.0012 |
| validation | 2111 | tfidf_logit__up | text (raw signal) | tfidf_logit__up_rk | car_1_60_ff4 | 2111 | 12 | 0.0151 | 0.7 | 0.75 | 0.0078 | 0.496 | 12 | 0.51 | 0.5037 |  |  |  |  |  |  |
| validation | 2111 | tfidf_logit__up | text_only | text_only__tfidf_logit__up | car_1_60_ff4 | 2111 | 12 | 0.0151 | 0.7 | 0.75 | 0.0078 | 0.496 | 12 | 0.51 | 0.5037 | -0.0003 |  |  |  |  |  |
| validation | 2111 | tfidf_logit__up | combined | combined__tfidf_logit__up | car_1_60_ff4 | 2111 | 12 | -0.0011 | -0.07 | 0.5 | 0.0 | 0.0006 | 12 | 0.4981 | 0.4962 | -0.003 | 0.0 | 1.7539 | 0.0536 | 12.0 | 0.0012 |
| validation | 2111 | llm__good_llm__tone_v1__raw | text (raw signal) | llm__good_llm__tone_v1__raw_rk | car_1_60_ff4 | 2111 | 12 | 0.0011 | 0.06 | 0.67 | -0.0008 | -0.067 | 12 | 0.4991 | 0.4936 |  |  |  |  |  |  |
| validation | 2111 | llm__good_llm__tone_v1__raw | text_only | text_only__llm__good_llm__tone_v1__raw | car_1_60_ff4 | 2111 | 12 | 0.0011 | 0.06 | 0.67 | -0.0008 | -0.067 | 12 | 0.4991 | 0.4936 | -0.0019 |  |  |  |  |  |
| validation | 2111 | llm__good_llm__tone_v1__raw | combined | combined__llm__good_llm__tone_v1__raw | car_1_60_ff4 | 2111 | 12 | -0.0076 | -0.44 | 0.58 | -0.007 | -0.56 | 12 | 0.4915 | 0.4934 | -0.0043 | 0.0 | 0.7874 | 0.2238 | 12.0 | -0.0001 |
| validation | 2111 | llm__noise_llm__tone_v1__raw | text (raw signal) | llm__noise_llm__tone_v1__raw_rk | car_1_60_ff4 | 2111 | 12 | -0.0159 | -0.45 | 0.5 | 0.008 | 0.416 | 12 | 0.4805 | 0.4851 |  |  |  |  |  |  |
| validation | 2111 | llm__noise_llm__tone_v1__raw | text_only | text_only__llm__noise_llm__tone_v1__raw | car_1_60_ff4 | 2111 | 12 | -0.0159 | -0.45 | 0.5 | 0.008 | 0.416 | 12 | 0.4805 | 0.4851 | -0.0015 |  |  |  |  |  |
| validation | 2111 | llm__noise_llm__tone_v1__raw | combined | combined__llm__noise_llm__tone_v1__raw | car_1_60_ff4 | 2111 | 12 | -0.024 | -0.9 | 0.33 | -0.012 | -0.8076 | 12 | 0.4882 | 0.4894 | -0.0047 | -0.0 | -0.4104 | 0.6553 | 12.0 | -0.0005 |
