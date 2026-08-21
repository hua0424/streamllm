# 成对统计推断（W5 冻结协议）

- bootstrap：paired、10000 次、seed=20260821、percentile 95% CI；
- Wilcoxon：双侧、zero_method='wilcox'、correction=False、method='auto'；
- 差值方向：延迟类 A−B（正=B 更快）；R5 为 B−A；改善率=(mean(A)−mean(B))/mean(A)；
- scipy 1.15.3；

## 输入 SHA-256

- `experiments/results/exp1_latency/exp1_results_20251210_024430.json` : `39fd55d89c0a84b0c21fd7ce8379505a22acf420b3efda0c5a67d72fe580ce36`
- `experiments/results/revision/r1_stats/table3_filter_manifest.json` : `0d31461a5c004da57923860d282dd1fa8ca203e17db2a30d51f7473107a6fc61`
- `experiments/results/revision/r3_baseline_la/system_ab_rerun/exp1_results_20260820_035759.json` : `d492f0a2f60b321b024d5e5541907fd75bc2d2d1b17b2c54680fdc5dd84811b0`
- `experiments/results/revision/r3_baseline_la/la_results_20260821_074150.json` : `cdc648ca94b590dfb6324800d233fe8f3a15755155528a85eb96027a4c1af792`
- `experiments\results\revision\r2_real_speech\librispeech_clean\exp1_results_20260819_132753.json` : `a0ebba6834a13eb3c2666f1243b81903ba49dd7b4f71309638e1b2b31ec8e610`
- `experiments\results\revision\r2_real_speech\aishell1_clean\exp1_results_20260819_142217.json` : `4bcbcfa0d52b171d8ae5e298d4dbd33124c50062bc812897d04b253c6c7c7c30`
- `experiments\results\revision\r2_real_speech\librispeech_snr10\exp1_results_20260819_152110.json` : `d66f73f6b34d9919e576d41b47761b558d48fcd983a16a22c10c4c70703ea23a`
- `experiments\results\revision\r2_real_speech\aishell1_snr10\exp1_results_20260819_171410.json` : `9ceb3634444873042d87ac9f2414cfc25ea9bd5263352238abf9d463728823dd`
- `experiments\results\revision\r2_real_speech\librispeech_snr15\exp1_results_20260819_150204.json` : `048c793b2aa9dbdb9eaf43560f21bcb05ae2d2093998be5e56c33804905ca4f3`
- `experiments\results\revision\r2_real_speech\aishell1_snr15\exp1_results_20260819_165612.json` : `7f92b375232923739c97ca68676689a925ded138eec8e642b9c81343a91b26fa`
- `experiments\results\revision\r2_real_speech\librispeech_snr20\exp1_results_20260819_144225.json` : `0b588d05a7aec6ec765ef25a5b7f5436c97e6324a91b29876d3d3f4e020fe5c8`
- `experiments\results\revision\r2_real_speech\aishell1_snr20\exp1_results_20260819_163808.json` : `babd2380f861c264727eb74880c293ec37728abf8223b5f4c6656245aada44d9`
- `experiments\results\revision\r2_real_speech\librispeech_babble\exp1_results_20260819_161959.json` : `0f6a1d98c043637a301a1c63553c661d7fb687db772b58d4ec1465b69079c64d`
- `experiments\results\revision\r2_real_speech\aishell1_babble\exp1_results_20260819_180825.json` : `35ba724eaa45a084950be2779cebf19b4aa5220beeb558aedaeb7cbd4a70b23d`
- `experiments\results\revision\r2_real_speech\librispeech_speed09\exp1_results_20260819_154155.json` : `37b0ff1f2b42d759a64860fe7c14730edb8c2aff4eee8119df58c5c208ff3e12`
- `experiments\results\revision\r2_real_speech\aishell1_speed09\exp1_results_20260819_173323.json` : `74d4a7a5b52b2188bc48db11d4bf0cf9bb91316c2394f0d690a31bd4915de2a6`
- `experiments\results\revision\r2_real_speech\librispeech_speed11\exp1_results_20260819_155948.json` : `b557f701281d9cd84d455b1b0674f14cbbe7396b7ebc080e170b93b6eb794581`
- `experiments\results\revision\r2_real_speech\aishell1_speed11\exp1_results_20260819_175019.json` : `2b2f40a41d9fa7a1bb43b8f1570caa07517eab18e3a2a895e7ade74a13e50469`
- `experiments/results/revision/r5_semantic/semantic_consistency.csv` : `7852d689a200fdebd9106d5b495ed6661a32b9f02f75557318cc1d9576a52a04`

## 结果

| comparison | family | n | diff mean | diff CI95 | improvement [CI] | p_raw | p_holm | rank-biserial | dz |
|---|---|---|---|---|---|---|---|---|---|
| table3_long | table3_groups | 121 | 595.4089 | [515.0288, 674.4433] | 0.3458 [0.3059, 0.3829] | 2.005e-19 | 2.005e-19 | 0.9442 | 1.3107 |
| table3_very_long | table3_groups | 208 | 2092.8078 | [1988.5999, 2196.6905] | 0.6556 [0.6375, 0.6728] | 7.219e-36 | 1.444e-35 | 0.9998 | 2.7425 |
| table3_extra_long | table3_groups | 679 | 5657.8728 | [5502.3915, 5819.9489] | 0.8388 [0.8331, 0.8441] | 7.654e-113 | 2.296e-112 | 1.0000 | 2.6746 |
| table3_overall | standalone | 1008 | 4314.5279 | [4149.7706, 4481.5641] | 0.7976 [0.7899, 0.8050] | 3.350e-166 |  | 0.9991 | 1.6243 |
| table7_a_vs_b | standalone | 498 | 3736.8325 | [3498.7311, 3978.8110] | 0.7036 [0.6889, 0.7173] | 1.331e-82 |  | 0.9957 | 1.3423 |
| table7_b_vs_la | standalone | 498 | 541.1114 | [485.2529, 599.9460] | 0.2558 [0.2345, 0.2772] | 1.241e-70 |  | 0.9189 | 0.8365 |
| r2_librispeech_clean | standalone | 75 | 1133.7906 | [832.7578, 1436.3608] | 0.4041 [0.3221, 0.4737] | 8.357e-09 |  | 0.7656 | 0.8352 |
| r2_aishell1_clean | standalone | 75 | 1172.8803 | [888.6692, 1473.5489] | 0.4084 [0.3414, 0.4677] | 1.746e-09 |  | 0.8000 | 0.8958 |
| r2_librispeech_snr10 | r2_augmented | 30 | 547.1204 | [299.5981, 797.8881] | 0.2485 [0.1480, 0.3361] | 6.084e-04 | 5.476e-03 | 0.6860 | 0.7632 |
| r2_aishell1_snr10 | r2_augmented | 30 | 657.0452 | [394.2809, 909.6558] | 0.2842 [0.1898, 0.3592] | 1.886e-04 | 1.886e-03 | 0.7376 | 0.8992 |
| r2_librispeech_snr15 | r2_augmented | 30 | 532.8175 | [225.9412, 821.3729] | 0.2283 [0.1047, 0.3308] | 3.223e-03 | 1.289e-02 | 0.6000 | 0.6206 |
| r2_aishell1_snr15 | r2_augmented | 30 | 659.9005 | [393.7578, 916.9908] | 0.2881 [0.1893, 0.3680] | 7.911e-05 | 8.702e-04 | 0.7720 | 0.8798 |
| r2_librispeech_snr20 | r2_augmented | 30 | 595.7777 | [287.7630, 893.1749] | 0.2673 [0.1420, 0.3699] | 1.583e-03 | 7.916e-03 | 0.6387 | 0.6977 |
| r2_aishell1_snr20 | r2_augmented | 30 | 598.5837 | [313.5769, 882.4947] | 0.2609 [0.1501, 0.3515] | 8.718e-04 | 5.585e-03 | 0.6688 | 0.7414 |
| r2_librispeech_babble | r2_augmented | 29 | -516.8716 | [-1087.6618, 2.3033] | -0.2246 [-0.4727, 0.0011] | 2.291e-02 | 4.582e-02 | -0.4805 | -0.3385 |
| r2_aishell1_babble | r2_augmented | 30 | 145.5010 | [-279.7619, 522.6361] | 0.0653 [-0.1282, 0.2268] | 2.710e-01 | 2.710e-01 | 0.2344 | 0.1282 |
| r2_librispeech_speed09 | r2_augmented | 30 | 739.4660 | [445.4360, 1013.1848] | 0.3158 [0.2062, 0.4023] | 1.598e-05 | 1.917e-04 | 0.8280 | 0.9185 |
| r2_aishell1_speed09 | r2_augmented | 30 | 697.4125 | [408.2744, 986.1367] | 0.3118 [0.2029, 0.4024] | 6.666e-04 | 5.476e-03 | 0.6817 | 0.8311 |
| r2_librispeech_speed11 | r2_augmented | 30 | 524.7003 | [263.2825, 786.3401] | 0.2405 [0.1330, 0.3313] | 3.475e-03 | 1.289e-02 | 0.5957 | 0.6951 |
| r2_aishell1_speed11 | r2_augmented | 30 | 501.8047 | [271.4152, 737.1680] | 0.2361 [0.1403, 0.3187] | 7.979e-04 | 5.585e-03 | 0.6731 | 0.7554 |
| r5_solo_b_minus_a | standalone | 50 | -0.0600 | [-0.3400, 0.2200] |   |  |  |  |  |
