# GPU clean 树 self-test 归档（§2c，Gate 第 11 项）

- 命令: `uv run python -m experiments.scripts.run_ttfa_unified --self-test`
- 期望: 90 PASS / 0 FAIL
- 结果: 90 PASS / 0 FAIL，exit 0
- git HEAD: 2e54ac297d51076f10c2445c9485b107544cf16f
- 环境: python 3.10.18, torch 2.5.1+cu121
- 输出文件: `selftest_gpu_20260822.log`
- 输出 sha256: `a44d54b88651c3aa10e3cc0dbaa84ca136c6e34e156d6185c372650f3826f461`
