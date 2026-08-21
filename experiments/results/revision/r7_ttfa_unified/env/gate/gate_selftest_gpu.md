# GPU clean 树 self-test 归档（§2c，Gate r3）

- 命令: `uv run python -m experiments.scripts.run_ttfa_unified --self-test`
- 期望: 90 PASS / 0 FAIL
- 结果: 90 PASS / 0 FAIL，exit 0
- git HEAD: b8893d63782b32a36eeb08584720993247fe0312
- 环境: python 3.10.18, torch 2.5.1+cu121
- 输出文件: `gate_selftest_gpu.log`
- 输出 sha256: `ee762a3c38ae54c9e29a81727e5eb96d1cdd9fa26891f9ac0cf3c3a73f674c00`
