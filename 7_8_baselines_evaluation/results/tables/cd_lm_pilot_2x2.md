<!-- EXPLORATORY: LM tone on the 115 events with a text (dev 105, validation 10); below the 5% coverage rule, not a result; text sign = above / below median tone -->
| signal | outcome | text_rule | cell | n | share | mean | se_naive | se_cluster | t_cluster | ci95_lo | ci95_hi | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lm_tone | car_0_1_ff4 | season_median | beat_textpos | 40 | 0.4082 | 0.0208 | 0.0087 | 0.009 | 2.3124 | 0.0032 | 0.0384 | 0.0267 |
| lm_tone | car_0_1_ff4 | season_median | beat_textneg | 35 | 0.3571 | 0.0036 | 0.0093 | 0.0086 | 0.4222 | -0.0132 | 0.0205 | 0.004 |
| lm_tone | car_0_1_ff4 | season_median | miss_textpos | 5 | 0.051 | -0.0119 | 0.0247 | 0.0125 | -0.9495 | -0.0364 | 0.0126 | -0.0027 |
| lm_tone | car_0_1_ff4 | season_median | miss_textneg | 18 | 0.1837 | -0.0467 | 0.013 | 0.0148 | -3.1558 | -0.0757 | -0.0177 | -0.0347 |
