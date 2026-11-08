<!-- Surprise-only vs text-only vs combined, post_car_60_ff4, validation split -->
| eval_split | n_events | text_signal | approach | signal | outcome | n | seasons | mean_ic | ic_t_nw | ic_share_positive | ls_mean | ls_t_nw | ls_seasons | hit_rate | auc | oos_r2_vs0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | 11375 | (none) | surprise_only | surprise_only | post_car_60_ff4 | 11375 | 12 | -0.019 | -0.99 | 0.33 | -0.0172 | -1.7475 | 12 | 0.4921 | 0.4925 | -0.0017 |
| validation | 11375 | (none) | sue_rank (raw signal) | sue_rank | post_car_60_ff4 | 11375 | 12 | -0.0114 | -0.73 | 0.5 | -0.0135 | -1.8579 | 12 | 0.5017 | 0.4972 |  |
