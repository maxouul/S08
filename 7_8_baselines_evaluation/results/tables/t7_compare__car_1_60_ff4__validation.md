<!-- Surprise-only vs text-only vs combined, car_1_60_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 11375 | (none) | surprise_only | surprise_only | car_1_60_ff4 | 11375 | 12 | -0.021 | -1.92 | 0.17 | -0.0019 | -0.2877 | 12 | 0.488 | 0.4883 | -0.001 |
| validation | 11375 | (none) | sue_rank (raw signal) | sue_rank | car_1_60_ff4 | 11375 | 12 | -0.0201 | -1.29 | 0.42 | -0.0166 | -2.3815 | 12 | 0.4985 | 0.4947 |  |
