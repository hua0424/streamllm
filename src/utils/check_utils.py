# src/utils/check_utils.py
"""smoke 测试共用的断言小工具（原先在 8 个 run_* 脚本中重复定义，审查后收拢）。"""


def make_check(logger):
    """返回绑定了 logger 的 check 函数：PASS/FAIL 记日志，FAIL 抛 AssertionError。"""
    def _check(name: str, cond: bool):
        logger.info(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            raise AssertionError(name)
    return _check
