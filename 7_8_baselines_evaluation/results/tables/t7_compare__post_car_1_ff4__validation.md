<!-- Surprise-only vs text-only vs combined, post_car_1_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 11375 | (none) | surprise_only | surprise_only | post_car_1_ff4 | 11375 | 12 | 0.0247 | 2.52 | 0.75 | 0.0022 | 0.5919 | 12 | 0.509 | 0.5139 | -0.0041 |
| validation | 11375 | (none) | sue_rank (raw signal) | sue_rank | post_car_1_ff4 | 11375 | 12 | 0.0255 | 2.47 | 0.75 | 0.0022 | 0.5919 | 12 | 0.5093 | 0.5143 |  |
