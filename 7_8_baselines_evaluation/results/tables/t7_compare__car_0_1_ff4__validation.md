<!-- Surprise-only vs text-only vs combined, car_0_1_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 11375 | (none) | surprise_only | surprise_only | car_0_1_ff4 | 11375 | 12 | 0.312 | 22.12 | 1.0 | 0.0752 | 13.3293 | 12 | 0.6326 | 0.6531 | 0.0873 |
| validation | 11375 | (none) | sue_rank (raw signal) | sue_rank | car_0_1_ff4 | 11375 | 12 | 0.3136 | 22.26 | 1.0 | 0.0752 | 13.3293 | 12 | 0.6173 | 0.6546 |  |
