<!-- Robustness grid -->
| text_signal | dimension | group | outcome | n | seasons | coef_text | t_text | p_text | mean_ic | ic_t_nw | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lm_tone_rk | baseline | all | car_0_1_ff4 | 2111 | 12 | 0.0337 | 7.4337 | 0.0 | 0.2949 | 16.93 |  |
| lm_tone_rk | sector (FF12) | BusEq | car_0_1_ff4 | 182 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Chems | car_0_1_ff4 | 194 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Durbl | car_0_1_ff4 | 208 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Enrgy | car_0_1_ff4 | 199 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Hlth | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Manuf | car_0_1_ff4 | 158 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Money | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | NoDur | car_0_1_ff4 | 198 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Shops | car_0_1_ff4 | 191 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Telcm | car_0_1_ff4 | 206 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | sector (FF12) | Utils | car_0_1_ff4 | 197 | 12 |  |  |  |  |  | too few events or seasons |
| lm_tone_rk | size tercile (within season) | small | car_0_1_ff4 | 709 | 12 | 0.0357 | 4.9169 | 0.0 | 0.3159 | 14.14 |  |
| lm_tone_rk | size tercile (within season) | mid | car_0_1_ff4 | 699 | 12 | 0.0332 | 4.4328 | 0.0 | 0.2642 | 9.15 |  |
| lm_tone_rk | size tercile (within season) | large | car_0_1_ff4 | 703 | 12 | 0.0332 | 3.7232 | 0.0002 | 0.2798 | 9.72 |  |
| lm_tone_rk | NYSE 20th pct flag | below | car_0_1_ff4 | 433 | 12 | 0.0386 | 3.7378 | 0.0002 | 0.3477 | 8.45 |  |
| lm_tone_rk | NYSE 20th pct flag | above | car_0_1_ff4 | 1678 | 12 | 0.032 | 6.7105 | 0.0 | 0.2776 | 17.36 |  |
| lm_tone_rk | period | 2017-19 | car_0_1_ff4 | 2111 | 12 | 0.0337 | 7.4337 | 0.0 | 0.2949 | 16.93 |  |
| lm_tone_rk | Llama 3.1 cutoff | before 2024 | car_0_1_ff4 | 2111 | 12 | 0.0337 | 7.4337 | 0.0 | 0.2949 | 16.93 |  |
| lm_tone_rk | event window | post_car_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.01 | 5.2821 | 0.0 | 0.1463 | 14.46 |  |
| lm_tone_rk | event window | post_car_60_ff4 | post_car_60_ff4 | 2111 | 12 | 0.0218 | 2.0756 | 0.0379 | 0.0294 | 1.18 |  |
| lm_tone_rk | event window | car_1_60_ff4 | car_1_60_ff4 | 2111 | 12 | 0.0134 | 1.3731 | 0.1697 | 0.0077 | 0.41 |  |
| lm_tone_rk | event window | car_0_1_mkt | car_0_1_mkt | 2111 | 12 | 0.0338 | 7.8543 | 0.0 | 0.2895 | 16.11 |  |
| lm_tone_rk | event window | post_car_1_mkt | post_car_1_mkt | 2111 | 12 | 0.0105 | 5.557 | 0.0 | 0.1419 | 11.64 |  |
| lm_tone_rk | event window | post_car_60_mkt | post_car_60_mkt | 2111 | 12 | 0.0217 | 2.1031 | 0.0355 | 0.0288 | 1.13 |  |
| lm_tone_rk | placebo (should be ~0) | ar_m1 | ar_m1 | 2111 | 12 | -0.0011 | -0.6828 | 0.4947 | -0.0132 | -0.56 |  |
| lm_tone_rk | sample | excluding basis_suspect | car_0_1_ff4 | 2049 | 12 | 0.0324 | 6.6431 | 0.0 | 0.2869 | 16.55 |  |
| lm_tone_rk | sample | excluding timing_suspect | car_0_1_ff4 | 2111 | 12 | 0.0337 | 7.4337 | 0.0 | 0.2949 | 16.93 |  |
| lm_tone_rk | surprise definition | SUE scaled by dispersion (3+ analysts) | car_0_1_ff4 | 2111 | 12 | 0.0351 | 6.8765 | 0.0 | 0.2949 | 16.93 |  |
| lm_negshare_rk | baseline | all | car_0_1_ff4 | 2111 | 12 | 0.0283 | 7.457 | 0.0 | 0.2672 | 15.64 |  |
| lm_negshare_rk | sector (FF12) | BusEq | car_0_1_ff4 | 182 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Chems | car_0_1_ff4 | 194 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Durbl | car_0_1_ff4 | 208 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Enrgy | car_0_1_ff4 | 199 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Hlth | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Manuf | car_0_1_ff4 | 158 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Money | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | NoDur | car_0_1_ff4 | 198 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Shops | car_0_1_ff4 | 191 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Telcm | car_0_1_ff4 | 206 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | sector (FF12) | Utils | car_0_1_ff4 | 197 | 12 |  |  |  |  |  | too few events or seasons |
| lm_negshare_rk | size tercile (within season) | small | car_0_1_ff4 | 709 | 12 | 0.0314 | 3.8299 | 0.0001 | 0.2993 | 10.65 |  |
| lm_negshare_rk | size tercile (within season) | mid | car_0_1_ff4 | 699 | 12 | 0.0274 | 3.5661 | 0.0004 | 0.239 | 7.12 |  |
| lm_negshare_rk | size tercile (within season) | large | car_0_1_ff4 | 703 | 12 | 0.027 | 3.7855 | 0.0002 | 0.2379 | 10.62 |  |
| lm_negshare_rk | NYSE 20th pct flag | below | car_0_1_ff4 | 433 | 12 | 0.0386 | 3.5879 | 0.0003 | 0.352 | 8.94 |  |
| lm_negshare_rk | NYSE 20th pct flag | above | car_0_1_ff4 | 1678 | 12 | 0.025 | 6.0137 | 0.0 | 0.241 | 14.15 |  |
| lm_negshare_rk | period | 2017-19 | car_0_1_ff4 | 2111 | 12 | 0.0283 | 7.457 | 0.0 | 0.2672 | 15.64 |  |
| lm_negshare_rk | Llama 3.1 cutoff | before 2024 | car_0_1_ff4 | 2111 | 12 | 0.0283 | 7.457 | 0.0 | 0.2672 | 15.64 |  |
| lm_negshare_rk | event window | post_car_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.0102 | 3.9191 | 0.0001 | 0.1456 | 12.32 |  |
| lm_negshare_rk | event window | post_car_60_ff4 | post_car_60_ff4 | 2111 | 12 | 0.0195 | 2.2741 | 0.023 | 0.0251 | 1.33 |  |
| lm_negshare_rk | event window | car_1_60_ff4 | car_1_60_ff4 | 2111 | 12 | 0.0027 | 0.2545 | 0.7991 | -0.0084 | -0.42 |  |
| lm_negshare_rk | event window | car_0_1_mkt | car_0_1_mkt | 2111 | 12 | 0.0286 | 7.7373 | 0.0 | 0.2636 | 14.38 |  |
| lm_negshare_rk | event window | post_car_1_mkt | post_car_1_mkt | 2111 | 12 | 0.0096 | 3.8228 | 0.0001 | 0.1302 | 8.52 |  |
| lm_negshare_rk | event window | post_car_60_mkt | post_car_60_mkt | 2111 | 12 | 0.0199 | 2.4303 | 0.0151 | 0.0264 | 1.33 |  |
| lm_negshare_rk | placebo (should be ~0) | ar_m1 | ar_m1 | 2111 | 12 | -0.0004 | -0.2857 | 0.7751 | -0.0059 | -0.32 |  |
| lm_negshare_rk | sample | excluding basis_suspect | car_0_1_ff4 | 2049 | 12 | 0.0271 | 6.9232 | 0.0 | 0.261 | 14.8 |  |
| lm_negshare_rk | sample | excluding timing_suspect | car_0_1_ff4 | 2111 | 12 | 0.0283 | 7.457 | 0.0 | 0.2672 | 15.64 |  |
| lm_negshare_rk | surprise definition | SUE scaled by dispersion (3+ analysts) | car_0_1_ff4 | 2111 | 12 | 0.0295 | 6.6127 | 0.0 | 0.2672 | 15.64 |  |
| tfidf_ridge__car_0_1_ff4_rk | baseline | all | car_0_1_ff4 | 2111 | 12 | 0.0265 | 5.1592 | 0.0 | 0.2587 | 9.62 |  |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | BusEq | car_0_1_ff4 | 182 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Chems | car_0_1_ff4 | 194 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Durbl | car_0_1_ff4 | 208 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Enrgy | car_0_1_ff4 | 199 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Hlth | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Manuf | car_0_1_ff4 | 158 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Money | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | NoDur | car_0_1_ff4 | 198 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Shops | car_0_1_ff4 | 191 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Telcm | car_0_1_ff4 | 206 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | sector (FF12) | Utils | car_0_1_ff4 | 197 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_ridge__car_0_1_ff4_rk | size tercile (within season) | small | car_0_1_ff4 | 709 | 12 | 0.0277 | 3.2593 | 0.0011 | 0.274 | 7.4 |  |
| tfidf_ridge__car_0_1_ff4_rk | size tercile (within season) | mid | car_0_1_ff4 | 699 | 12 | 0.0248 | 2.8802 | 0.004 | 0.2317 | 5.51 |  |
| tfidf_ridge__car_0_1_ff4_rk | size tercile (within season) | large | car_0_1_ff4 | 703 | 12 | 0.0278 | 3.2762 | 0.0011 | 0.2515 | 7.66 |  |
| tfidf_ridge__car_0_1_ff4_rk | NYSE 20th pct flag | below | car_0_1_ff4 | 433 | 12 | 0.0307 | 2.698 | 0.007 | 0.3068 | 6.18 |  |
| tfidf_ridge__car_0_1_ff4_rk | NYSE 20th pct flag | above | car_0_1_ff4 | 1678 | 12 | 0.0248 | 4.6522 | 0.0 | 0.2428 | 9.42 |  |
| tfidf_ridge__car_0_1_ff4_rk | period | 2017-19 | car_0_1_ff4 | 2111 | 12 | 0.0265 | 5.1592 | 0.0 | 0.2587 | 9.62 |  |
| tfidf_ridge__car_0_1_ff4_rk | Llama 3.1 cutoff | before 2024 | car_0_1_ff4 | 2111 | 12 | 0.0265 | 5.1592 | 0.0 | 0.2587 | 9.62 |  |
| tfidf_ridge__car_0_1_ff4_rk | event window | post_car_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.008 | 3.4517 | 0.0006 | 0.129 | 9.14 |  |
| tfidf_ridge__car_0_1_ff4_rk | event window | post_car_60_ff4 | post_car_60_ff4 | 2111 | 12 | 0.0133 | 1.2089 | 0.2267 | 0.0131 | 0.5 |  |
| tfidf_ridge__car_0_1_ff4_rk | event window | car_1_60_ff4 | car_1_60_ff4 | 2111 | 12 | 0.013 | 1.7579 | 0.0788 | 0.0126 | 0.64 |  |
| tfidf_ridge__car_0_1_ff4_rk | event window | car_0_1_mkt | car_0_1_mkt | 2111 | 12 | 0.0265 | 5.4766 | 0.0 | 0.254 | 9.46 |  |
| tfidf_ridge__car_0_1_ff4_rk | event window | post_car_1_mkt | post_car_1_mkt | 2111 | 12 | 0.0084 | 3.4341 | 0.0006 | 0.1255 | 6.52 |  |
| tfidf_ridge__car_0_1_ff4_rk | event window | post_car_60_mkt | post_car_60_mkt | 2111 | 12 | 0.0129 | 1.1808 | 0.2377 | 0.0119 | 0.45 |  |
| tfidf_ridge__car_0_1_ff4_rk | placebo (should be ~0) | ar_m1 | ar_m1 | 2111 | 12 | -0.0002 | -0.1118 | 0.911 | -0.0018 | -0.1 |  |
| tfidf_ridge__car_0_1_ff4_rk | sample | excluding basis_suspect | car_0_1_ff4 | 2049 | 12 | 0.0247 | 4.6959 | 0.0 | 0.2484 | 8.96 |  |
| tfidf_ridge__car_0_1_ff4_rk | sample | excluding timing_suspect | car_0_1_ff4 | 2111 | 12 | 0.0265 | 5.1592 | 0.0 | 0.2587 | 9.62 |  |
| tfidf_ridge__car_0_1_ff4_rk | surprise definition | SUE scaled by dispersion (3+ analysts) | car_0_1_ff4 | 2111 | 12 | 0.028 | 4.919 | 0.0 | 0.2587 | 9.62 |  |
| tfidf_logit__up_rk | baseline | all | car_0_1_ff4 | 2111 | 12 | 0.0254 | 4.8578 | 0.0 | 0.2506 | 9.36 |  |
| tfidf_logit__up_rk | sector (FF12) | BusEq | car_0_1_ff4 | 182 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Chems | car_0_1_ff4 | 194 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Durbl | car_0_1_ff4 | 208 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Enrgy | car_0_1_ff4 | 199 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Hlth | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Manuf | car_0_1_ff4 | 158 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Money | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | NoDur | car_0_1_ff4 | 198 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Shops | car_0_1_ff4 | 191 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Telcm | car_0_1_ff4 | 206 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | sector (FF12) | Utils | car_0_1_ff4 | 197 | 12 |  |  |  |  |  | too few events or seasons |
| tfidf_logit__up_rk | size tercile (within season) | small | car_0_1_ff4 | 709 | 12 | 0.0281 | 2.88 | 0.004 | 0.2664 | 6.59 |  |
| tfidf_logit__up_rk | size tercile (within season) | mid | car_0_1_ff4 | 699 | 12 | 0.0217 | 2.457 | 0.014 | 0.2063 | 4.79 |  |
| tfidf_logit__up_rk | size tercile (within season) | large | car_0_1_ff4 | 703 | 12 | 0.0274 | 3.4937 | 0.0005 | 0.2518 | 8.52 |  |
| tfidf_logit__up_rk | NYSE 20th pct flag | below | car_0_1_ff4 | 433 | 12 | 0.0296 | 2.2829 | 0.0224 | 0.2931 | 5.7 |  |
| tfidf_logit__up_rk | NYSE 20th pct flag | above | car_0_1_ff4 | 1678 | 12 | 0.0237 | 4.8036 | 0.0 | 0.2361 | 9.63 |  |
| tfidf_logit__up_rk | period | 2017-19 | car_0_1_ff4 | 2111 | 12 | 0.0254 | 4.8578 | 0.0 | 0.2506 | 9.36 |  |
| tfidf_logit__up_rk | Llama 3.1 cutoff | before 2024 | car_0_1_ff4 | 2111 | 12 | 0.0254 | 4.8578 | 0.0 | 0.2506 | 9.36 |  |
| tfidf_logit__up_rk | event window | post_car_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.008 | 3.4919 | 0.0005 | 0.1284 | 11.22 |  |
| tfidf_logit__up_rk | event window | post_car_60_ff4 | post_car_60_ff4 | 2111 | 12 | 0.0104 | 0.9302 | 0.3523 | 0.0059 | 0.25 |  |
| tfidf_logit__up_rk | event window | car_1_60_ff4 | car_1_60_ff4 | 2111 | 12 | 0.0142 | 1.6959 | 0.0899 | 0.0151 | 0.7 |  |
| tfidf_logit__up_rk | event window | car_0_1_mkt | car_0_1_mkt | 2111 | 12 | 0.0254 | 5.2123 | 0.0 | 0.2468 | 9.57 |  |
| tfidf_logit__up_rk | event window | post_car_1_mkt | post_car_1_mkt | 2111 | 12 | 0.0081 | 3.4017 | 0.0007 | 0.1206 | 7.65 |  |
| tfidf_logit__up_rk | event window | post_car_60_mkt | post_car_60_mkt | 2111 | 12 | 0.0102 | 0.915 | 0.3602 | 0.0053 | 0.22 |  |
| tfidf_logit__up_rk | placebo (should be ~0) | ar_m1 | ar_m1 | 2111 | 12 | 0.0008 | 0.5797 | 0.5621 | 0.0105 | 0.62 |  |
| tfidf_logit__up_rk | sample | excluding basis_suspect | car_0_1_ff4 | 2049 | 12 | 0.0236 | 4.4186 | 0.0 | 0.2392 | 8.72 |  |
| tfidf_logit__up_rk | sample | excluding timing_suspect | car_0_1_ff4 | 2111 | 12 | 0.0254 | 4.8578 | 0.0 | 0.2506 | 9.36 |  |
| tfidf_logit__up_rk | surprise definition | SUE scaled by dispersion (3+ analysts) | car_0_1_ff4 | 2111 | 12 | 0.0269 | 4.6367 | 0.0 | 0.2506 | 9.36 |  |
| llm__good_llm__tone_v1__raw_rk | baseline | all | car_0_1_ff4 | 2111 | 12 | 0.038 | 12.8287 | 0.0 | 0.3141 | 25.19 |  |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | BusEq | car_0_1_ff4 | 182 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Chems | car_0_1_ff4 | 194 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Durbl | car_0_1_ff4 | 208 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Enrgy | car_0_1_ff4 | 199 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Hlth | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Manuf | car_0_1_ff4 | 158 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Money | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | NoDur | car_0_1_ff4 | 198 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Shops | car_0_1_ff4 | 191 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Telcm | car_0_1_ff4 | 206 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | sector (FF12) | Utils | car_0_1_ff4 | 197 | 12 |  |  |  |  |  | too few events or seasons |
| llm__good_llm__tone_v1__raw_rk | size tercile (within season) | small | car_0_1_ff4 | 709 | 12 | 0.0398 | 9.1846 | 0.0 | 0.3432 | 18.5 |  |
| llm__good_llm__tone_v1__raw_rk | size tercile (within season) | mid | car_0_1_ff4 | 699 | 12 | 0.0422 | 6.5275 | 0.0 | 0.3105 | 15.31 |  |
| llm__good_llm__tone_v1__raw_rk | size tercile (within season) | large | car_0_1_ff4 | 703 | 12 | 0.0309 | 5.0754 | 0.0 | 0.267 | 10.31 |  |
| llm__good_llm__tone_v1__raw_rk | NYSE 20th pct flag | below | car_0_1_ff4 | 433 | 12 | 0.0394 | 4.5593 | 0.0 | 0.3527 | 8.09 |  |
| llm__good_llm__tone_v1__raw_rk | NYSE 20th pct flag | above | car_0_1_ff4 | 1678 | 12 | 0.037 | 11.2537 | 0.0 | 0.304 | 18.18 |  |
| llm__good_llm__tone_v1__raw_rk | period | 2017-19 | car_0_1_ff4 | 2111 | 12 | 0.038 | 12.8287 | 0.0 | 0.3141 | 25.19 |  |
| llm__good_llm__tone_v1__raw_rk | Llama 3.1 cutoff | before 2024 | car_0_1_ff4 | 2111 | 12 | 0.038 | 12.8287 | 0.0 | 0.3141 | 25.19 |  |
| llm__good_llm__tone_v1__raw_rk | event window | post_car_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.0112 | 3.9833 | 0.0001 | 0.1529 | 7.06 |  |
| llm__good_llm__tone_v1__raw_rk | event window | post_car_60_ff4 | post_car_60_ff4 | 2111 | 12 | 0.0059 | 0.6395 | 0.5225 | -0.0061 | -0.23 |  |
| llm__good_llm__tone_v1__raw_rk | event window | car_1_60_ff4 | car_1_60_ff4 | 2111 | 12 | 0.0069 | 0.7199 | 0.4716 | 0.0011 | 0.06 |  |
| llm__good_llm__tone_v1__raw_rk | event window | car_0_1_mkt | car_0_1_mkt | 2111 | 12 | 0.0386 | 12.2275 | 0.0 | 0.3109 | 18.96 |  |
| llm__good_llm__tone_v1__raw_rk | event window | post_car_1_mkt | post_car_1_mkt | 2111 | 12 | 0.0121 | 4.5894 | 0.0 | 0.1536 | 9.36 |  |
| llm__good_llm__tone_v1__raw_rk | event window | post_car_60_mkt | post_car_60_mkt | 2111 | 12 | 0.0058 | 0.644 | 0.5196 | -0.0067 | -0.25 |  |
| llm__good_llm__tone_v1__raw_rk | placebo (should be ~0) | ar_m1 | ar_m1 | 2111 | 12 | -0.0018 | -1.1251 | 0.2606 | -0.0267 | -1.39 |  |
| llm__good_llm__tone_v1__raw_rk | sample | excluding basis_suspect | car_0_1_ff4 | 2049 | 12 | 0.0374 | 12.8945 | 0.0 | 0.3083 | 23.46 |  |
| llm__good_llm__tone_v1__raw_rk | sample | excluding timing_suspect | car_0_1_ff4 | 2111 | 12 | 0.038 | 12.8287 | 0.0 | 0.3141 | 25.19 |  |
| llm__good_llm__tone_v1__raw_rk | surprise definition | SUE scaled by dispersion (3+ analysts) | car_0_1_ff4 | 2111 | 12 | 0.0397 | 11.9262 | 0.0 | 0.3141 | 25.19 |  |
| llm__noise_llm__tone_v1__raw_rk | baseline | all | car_0_1_ff4 | 2111 | 12 | 0.0025 | 0.7078 | 0.4791 | 0.0098 | 0.56 |  |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | BusEq | car_0_1_ff4 | 182 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Chems | car_0_1_ff4 | 194 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Durbl | car_0_1_ff4 | 208 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Enrgy | car_0_1_ff4 | 199 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Hlth | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Manuf | car_0_1_ff4 | 158 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Money | car_0_1_ff4 | 189 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | NoDur | car_0_1_ff4 | 198 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Shops | car_0_1_ff4 | 191 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Telcm | car_0_1_ff4 | 206 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | sector (FF12) | Utils | car_0_1_ff4 | 197 | 12 |  |  |  |  |  | too few events or seasons |
| llm__noise_llm__tone_v1__raw_rk | size tercile (within season) | small | car_0_1_ff4 | 709 | 12 | 0.0069 | 0.8536 | 0.3933 | 0.0186 | 0.49 |  |
| llm__noise_llm__tone_v1__raw_rk | size tercile (within season) | mid | car_0_1_ff4 | 699 | 12 | 0.0045 | 0.7493 | 0.4536 | 0.0091 | 0.33 |  |
| llm__noise_llm__tone_v1__raw_rk | size tercile (within season) | large | car_0_1_ff4 | 703 | 12 | -0.0003 | -0.0722 | 0.9424 | -0.0084 | -0.46 |  |
| llm__noise_llm__tone_v1__raw_rk | NYSE 20th pct flag | below | car_0_1_ff4 | 433 | 12 | 0.0122 | 1.2144 | 0.2246 | 0.0702 | 2.08 |  |
| llm__noise_llm__tone_v1__raw_rk | NYSE 20th pct flag | above | car_0_1_ff4 | 1678 | 12 | 0.0001 | 0.0203 | 0.9838 | -0.006 | -0.38 |  |
| llm__noise_llm__tone_v1__raw_rk | period | 2017-19 | car_0_1_ff4 | 2111 | 12 | 0.0025 | 0.7078 | 0.4791 | 0.0098 | 0.56 |  |
| llm__noise_llm__tone_v1__raw_rk | Llama 3.1 cutoff | before 2024 | car_0_1_ff4 | 2111 | 12 | 0.0025 | 0.7078 | 0.4791 | 0.0098 | 0.56 |  |
| llm__noise_llm__tone_v1__raw_rk | event window | post_car_1_ff4 | post_car_1_ff4 | 2111 | 12 | 0.0036 | 2.4048 | 0.0162 | 0.0344 | 3.68 |  |
| llm__noise_llm__tone_v1__raw_rk | event window | post_car_60_ff4 | post_car_60_ff4 | 2111 | 12 | -0.003 | -0.3513 | 0.7254 | -0.0123 | -1.01 |  |
| llm__noise_llm__tone_v1__raw_rk | event window | car_1_60_ff4 | car_1_60_ff4 | 2111 | 12 | -0.0047 | -0.3685 | 0.7125 | -0.0159 | -0.45 |  |
| llm__noise_llm__tone_v1__raw_rk | event window | car_0_1_mkt | car_0_1_mkt | 2111 | 12 | 0.0027 | 0.7727 | 0.4397 | 0.0098 | 0.59 |  |
| llm__noise_llm__tone_v1__raw_rk | event window | post_car_1_mkt | post_car_1_mkt | 2111 | 12 | 0.0025 | 1.559 | 0.119 | 0.0255 | 3.2 |  |
| llm__noise_llm__tone_v1__raw_rk | event window | post_car_60_mkt | post_car_60_mkt | 2111 | 12 | -0.002 | -0.2269 | 0.8205 | -0.0105 | -0.84 |  |
| llm__noise_llm__tone_v1__raw_rk | placebo (should be ~0) | ar_m1 | ar_m1 | 2111 | 12 | 0.001 | 0.7407 | 0.4589 | 0.0149 | 1.08 |  |
| llm__noise_llm__tone_v1__raw_rk | sample | excluding basis_suspect | car_0_1_ff4 | 2049 | 12 | 0.0025 | 0.7106 | 0.4773 | 0.009 | 0.49 |  |
| llm__noise_llm__tone_v1__raw_rk | sample | excluding timing_suspect | car_0_1_ff4 | 2111 | 12 | 0.0025 | 0.7078 | 0.4791 | 0.0098 | 0.56 |  |
| llm__noise_llm__tone_v1__raw_rk | surprise definition | SUE scaled by dispersion (3+ analysts) | car_0_1_ff4 | 2111 | 12 | 0.0022 | 0.7115 | 0.4768 | 0.0098 | 0.56 |  |
