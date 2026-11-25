| Model | Norm. Qini (IPW) | AUUC | PEHE ($) | Uplift@20% ($) | Latency ms/1k | Fit s | Device | 95% CI cover. of true tau |
|---|---|---|---|---|---|---|---|---|
| S-Learner | 0.94 | -0.518 | 16.151 | 10.663 | 2.287 | 3.037 | GPU | 0.134 |
| T-Learner | 0.942 | -0.506 | 15.622 | 10.712 | 2.5 | 6.031 | GPU | 0.389 |
| X-Learner | 0.951 | -0.441 | 15.199 | 11.533 | 3.413 | 13.668 | GPU | 0.245 |
| R-Learner | 0.944 | -0.495 | 15.574 | 11.497 | 1.374 | 17.582 | GPU | 0.386 |
| DR-Learner | 0.934 | -0.561 | 15.785 | 11.786 | 2.384 | 8.305 | GPU | 0.31 |
| LinearDML | 0.93 | -0.597 | 15.745 | 9.784 | 2.079 | 1.925 | CPU | 0.707 |
| NonParamDML | 0.942 | -0.503 | 15.774 | 11.338 | 2.99 | 7.99 | GPU | 0.345 |
| CausalForestDML | 0.952 | -0.43 | 14.941 | 11.384 | 31.498 | 2.648 | GPU | 0.46 |
| UpliftRF | 0.907 | -0.76 | 17.488 | 5.53 | 34.127 | 6.093 | CPU | 0.084 |
| TARNet | 0.94 | -0.521 | 15.143 | 12.25 | 1.369 | 8.667 | GPU | 0.505 |
| DragonNet | 0.941 | -0.515 | 15.18 | 11.908 | 1.495 | 6.829 | GPU | 0.459 |

| Policy | Targeted | Net incr. margin ($, test set) | Discount cannibalization ($) | Incr. bookings |
|---|---|---|---|---|
| Mass Blast | 1.0 | -165699.9 | 224858.7 | 888.2 |
| No Promo | 0.0 | 0.0 | 0.0 | 0.0 |
| Propensity Targeting | 0.2 | -127405.1 | 152647.8 | 290.4 |
| Causal Uplift (CausalForestDML) | 0.2 | 28925.2 | 28927.5 | 824.6 |

| persona | Always-Taker | Lost Cause | Persuadable | Sleeping Dog |
|---|---|---|---|---|
| Always-Taker | 0.96 | 0.0 | 0.0 | 0.03 |
| Lost Cause | 0.0 | 0.94 | 0.06 | 0.0 |
| Neutral | 0.0 | 0.96 | 0.04 | 0.0 |
| Persuadable | 0.0 | 0.04 | 0.96 | 0.0 |
| Sleeping Dog | 0.0 | 0.02 | 0.0 | 0.98 |

| base | model | pehe | qini_norm_ipw |
|---|---|---|---|
| xgboost | S-Learner | 16.151 | 0.94 |
| xgboost | T-Learner | 15.622 | 0.942 |
| xgboost | X-Learner | 15.199 | 0.951 |
| lightgbm | S-Learner | 15.753 | 0.941 |
| lightgbm | T-Learner | 19.456 | 0.915 |
| lightgbm | X-Learner | 17.878 | 0.926 |
| catboost | S-Learner | 15.156 | 0.942 |
| catboost | T-Learner | 16.742 | 0.935 |
| catboost | X-Learner | 15.796 | 0.947 |