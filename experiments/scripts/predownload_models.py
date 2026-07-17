#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预下载实验所需模型（断点续传 + 自动重试——不稳网络下下载大模型必备）。

背景：TEN 7B 等 15GB 级模型在不稳网络下常见 ChunkedEncodingError/IncompleteRead
（下载中途断流）。huggingface_hub 会保留 .incomplete 断点，重试即续传；
本脚本自动循环重试直到完成。走 src.config —— 自动带上 .env 的 HF_ENDPOINT
（镜像修复，2026-07-17）与 HF_HOME。

运行（实验机，项目根目录；大模型建议 nohup 后台）：
    HF_TOKEN= uv run python -m experiments.scripts.predownload_models          # 默认三件套
    HF_TOKEN= uv run python -m experiments.scripts.predownload_models \
        --models mistralai/Mistral-7B-Instruct-v0.3                            # 追加裁判等
"""

import argparse
import time

from src.config import (HF_HOME, P2_LLM_MODEL_NAME, P2_REWRITER_MODEL_NAME,
                        P2_TRIGGER_MODEL_NAME)
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


def download_with_retry(repo_id: str, max_retries: int, sleep_s: float) -> bool:
    from huggingface_hub import snapshot_download   # 延迟 import：确保 config 的 endpoint 回补已生效
    for attempt in range(1, max_retries + 1):
        try:
            path = snapshot_download(repo_id, cache_dir=HF_HOME)
            logger.info(f"✓ {repo_id} 完成 → {path}")
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f"[{attempt}/{max_retries}] {repo_id} 中断（{type(e).__name__}: "
                           f"{str(e)[:120]}），{sleep_s}s 后续传重试…")
            time.sleep(sleep_s)
    logger.error(f"✗ {repo_id} 重试 {max_retries} 次仍未完成")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="追加下载的模型（默认已含 主LLM/软触发/重写 三件套）")
    ap.add_argument("--skip-defaults", action="store_true", help="只下 --models 指定的")
    ap.add_argument("--max-retries", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=10.0)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    targets = [] if args.skip_defaults else [P2_LLM_MODEL_NAME, P2_TRIGGER_MODEL_NAME,
                                             P2_REWRITER_MODEL_NAME]
    targets += args.models
    # 去重保序
    seen, todo = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            todo.append(t)
    logger.info(f"预下载 {len(todo)} 个模型 → HF_HOME={HF_HOME}")
    ok = all(download_with_retry(t, args.max_retries, args.sleep) for t in todo)
    logger.info("ALL DONE ✓" if ok else "有模型未完成，重跑本命令即续传")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
