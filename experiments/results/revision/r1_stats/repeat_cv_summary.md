# E1 三轮重复测量 CV 汇总（ddof=1 统一口径）

- 口径：CV_i = std(3 轮, ddof=1) / mean(3 轮)，百分比；样本须恰 3 轮、同模式、无 error；
- P90 为 numpy 默认线性插值；
- 输入文件 SHA-256：
  - `experiments/results/revision/r1_stats/repeat_r1/exp1_results_20260819_054319.json` : `975ea2c8dc48a3b8572669a21c7aca558d71a072dde2521fcef4f2ed07941bad`
  - `experiments/results/revision/r1_stats/repeat_r2/exp1_results_20260819_063135.json` : `32dc30290a92f9148196100314c754ad2307933f14a0ed0a27416d8355e04de9`
  - `experiments/results/revision/r1_stats/repeat_r3/exp1_results_20260819_072058.json` : `4726e368454b279b44f5f689571c786e359c33e658a5023ca236f9baf3ccd39b`

| mode | n | mean CV% | median CV% | P90 CV% | max CV% | CV>5% n(%) |
|---|---|---|---|---|---|---|
| streaming | 50 | 5.1935 | 4.0492 | 10.7303 | 18.9577 | 19 (38.00%) |
| non-streaming | 50 | 5.2317 | 4.6530 | 9.9213 | 14.0090 | 23 (46.00%) |
