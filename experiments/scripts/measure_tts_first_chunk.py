#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEV-5 / E6：TTS 首包延迟测量（R6.2，意见2）

测量 CosyVoice 流式 TTS 服务的 time-to-first-chunk（TTFC）与整段合成 RTF，
用于 TTFA 端到端预算表（语音结束 → 首个可听音频帧）。

请求口径与论文数据生成一致：POST /inference_sft，spk_id="晓伊"，speed=0.8，
返回流式 PCM（22050 Hz / 16bit / mono）。

输入（二选一）：
- --input texts.jsonl：每行 {"sample_id": ..., "text": ..., "language": "zh"|"en"}
- --from-e4 <E4结果目录>：从 exp1_results_*.json 抽取 streaming 模式的 full_response，
  按 sample_id 前缀判定语言（multiwoz→en, crosswoz→zh），每语言取 --n-zh/--n-en 条（seed=42）

输出 CSV：sample_id,language,n_chars,ttfc_ms,total_ms,audio_sec,rtf,error

使用方式：
    uv run python -m experiments.scripts.measure_tts_first_chunk \
        --from-e4 experiments/results/revision/r4_commit --n-zh 25 --n-en 25 \
        --output experiments/results/revision/r6_ttfa/tts_first_chunk.csv
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 与论文数据生成一致的请求参数默认值（可用 TTS_* 环境变量或 CLI 覆盖）
DEFAULT_SPK_ID = os.getenv("TTS_SPK_ID", "晓伊")
DEFAULT_SPEED = float(os.getenv("TTS_SPEED", "0.8"))
DEFAULT_TTS_URL = os.getenv("TTS_URL", "http://host.docker.internal:20401")
# CosyVoice 流式输出 PCM 格式
PCM_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "22050"))
PCM_BYTES_PER_SAMPLE = int(os.getenv("TTS_BITS_PER_SAMPLE", "16")) // 8  # mono 16bit


def probe_service(cfg: dict, timeout: int = 10) -> bool:
    """服务探活：用短文本发一次请求，确认服务可用"""
    try:
        response = requests.post(
            f"{cfg['url'].rstrip('/')}/inference_sft",
            data={"tts_text": "探活", "spk_id": cfg["spk_id"], "stream": True, "speed": cfg["speed"]},
            timeout=timeout,
            stream=True,
        )
        if response.status_code != 200:
            logger.error(f"探活失败: HTTP {response.status_code}")
            return False
        first = next(response.iter_content(chunk_size=16000), None)
        ok = first is not None and len(first) > 0
        logger.info(f"服务探活{'通过' if ok else '失败（无音频数据）'}: {cfg['url']}")
        return ok
    except Exception as e:
        logger.error(f"服务不可达 {cfg['url']}: {e}")
        return False


def measure_one(cfg: dict, text: str, timeout: int = 120) -> dict:
    """
    对一条文本测量 TTFC 与整段合成时间。

    Returns:
        dict(ttfc_ms, total_ms, audio_sec, rtf, error)
    """
    out = {"ttfc_ms": "", "total_ms": "", "audio_sec": "", "rtf": "", "error": ""}
    try:
        t_request = time.perf_counter()
        response = requests.post(
            f"{cfg['url'].rstrip('/')}/inference_sft",
            data={"tts_text": text, "spk_id": cfg["spk_id"], "stream": True, "speed": cfg["speed"]},
            timeout=timeout,
            stream=True,
        )
        if response.status_code != 200:
            out["error"] = f"http_{response.status_code}"
            return out

        total_bytes = 0
        t_first = None
        for chunk in response.iter_content(chunk_size=16000):
            if chunk:
                if t_first is None:
                    t_first = time.perf_counter()
                total_bytes += len(chunk)
        t_done = time.perf_counter()

        if t_first is None or total_bytes == 0:
            out["error"] = "empty_audio"
            return out

        audio_sec = total_bytes / (cfg["pcm_sample_rate"] * cfg["pcm_bytes_per_sample"])
        total_s = t_done - t_request
        out.update({
            "ttfc_ms": (t_first - t_request) * 1000,
            "total_ms": total_s * 1000,
            "audio_sec": audio_sec,
            "rtf": total_s / audio_sec if audio_sec > 0 else "",
        })
        return out
    except requests.exceptions.Timeout:
        out["error"] = f"timeout_{timeout}s"
        return out
    except Exception as e:
        out["error"] = str(e)[:120]
        return out


def lang_of_sample(sample_id: str) -> str:
    if sample_id.startswith("multiwoz"):
        return "en"
    if sample_id.startswith("crosswoz"):
        return "zh"
    return "unknown"


def load_from_jsonl(path: Path) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                items.append({"sample_id": d["sample_id"], "text": d["text"],
                              "language": d.get("language", lang_of_sample(d["sample_id"]))})
    return items


def load_from_e4(e4_dir: Path, n_zh: int, n_en: int, seed: int = 42,
                 allow_preview_fallback: bool = False) -> list:
    """从 E4 结果目录抽取 streaming 模式 full_response，每语言取 n 条（seed 固定）

    P0-4：默认要求 full_response 存在；缺失/为空时除非显式允许 preview fallback，
    否则报错退出，避免把 100 字符截断预览当完整回复送 TTS。
    """
    result_files = sorted(glob.glob(str(e4_dir / "exp1_results_*.json")))
    if not result_files:
        raise FileNotFoundError(f"E4 目录下没有 exp1_results_*.json: {e4_dir}")
    data = json.load(open(result_files[-1], encoding="utf-8"))

    pool = {"zh": [], "en": []}
    missing = 0
    for r in data["results"]:
        if r["mode"] != "streaming" or r.get("error"):
            continue
        lang = lang_of_sample(r["sample_id"])
        if lang not in pool:
            continue
        full = (r.get("full_response") or "").strip()
        if full:
            pool[lang].append({"sample_id": r["sample_id"], "text": full,
                               "language": lang, "text_source": "full_response"})
        elif "full_response" in r:
            # 键存在但为空：视为无效记录（E4 运行异常），即使允许 fallback 也不采用
            missing += 1
        elif allow_preview_fallback and r.get("response_preview", "").strip():
            # 键缺失（旧版 E4 未开启 --save-full-response）：仅显式允许时可用 preview
            pool[lang].append({"sample_id": r["sample_id"], "text": r["response_preview"].strip(),
                               "language": lang, "text_source": "preview"})
        else:
            missing += 1

    if missing:
        msg = (f"E4 结果中 {missing} 条 streaming 记录缺少 full_response；"
               f"请用 --save-full-response 重跑 E4，或显式加 --allow-preview-fallback")
        if not allow_preview_fallback:
            raise SystemExit(msg)
        logger.warning(msg + "（已按 preview fallback 处理）")

    rng = random.Random(seed)
    items = []
    for lang, n in [("zh", n_zh), ("en", n_en)]:
        candidates = pool[lang]
        if len(candidates) > n:
            candidates = sorted(rng.sample(candidates, n), key=lambda x: x["sample_id"])
        else:
            logger.warning(f"{lang} 可用回复仅 {len(candidates)} 条（要求 {n}）")
        items.extend(candidates)
    n_preview = sum(1 for i in items if i["text_source"] == "preview")
    logger.info(f"从 {result_files[-1]} 抽取 {len(items)} 条回复 "
                f"(zh={sum(1 for i in items if i['language']=='zh')}, "
                f"en={sum(1 for i in items if i['language']=='en')}"
                + (f", 其中 preview fallback {n_preview} 条" if n_preview else "") + ")")
    return items


def main():
    parser = argparse.ArgumentParser(description="TTS 首包延迟测量（R6.2）")
    parser.add_argument('--url', type=str, default=DEFAULT_TTS_URL,
                        help='TTS 服务地址（可用 TTS_URL 环境变量）')
    parser.add_argument('--spk-id', type=str, default=DEFAULT_SPK_ID,
                        help='说话人 ID（可用 TTS_SPK_ID 环境变量，默认与论文数据生成一致）')
    parser.add_argument('--speed', type=float, default=DEFAULT_SPEED,
                        help='语速系数（可用 TTS_SPEED 环境变量，默认 0.8 与数据生成一致）')
    parser.add_argument('--input', type=str, default=None,
                        help='输入 JSONL：{"sample_id","text","language"}')
    parser.add_argument('--from-e4', type=str, default=None,
                        help='E4 结果目录（自动抽取 full_response）')
    parser.add_argument('--n-zh', type=int, default=25)
    parser.add_argument('--n-en', type=int, default=25)
    parser.add_argument('--output', type=str, required=True, help='输出 CSV 路径')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='请求间隔秒数（默认 1.0，避免压测语义）')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--allow-preview-fallback', action='store_true',
                        help='E4 结果缺少 full_response 时允许用 response_preview 截断文本代替'
                             '（默认禁止；使用时 CSV 中 text_source=preview）')
    args = parser.parse_args()

    if not args.input and not args.from_e4:
        parser.error("需要 --input 或 --from-e4 之一")

    if args.input:
        items = load_from_jsonl(Path(args.input))
        for it in items:
            it["text_source"] = "jsonl"
    else:
        items = load_from_e4(Path(args.from_e4), args.n_zh, args.n_en,
                             allow_preview_fallback=args.allow_preview_fallback)

    logger.info(f"共 {len(items)} 条待测量，服务: {args.url}")

    cfg = {"url": args.url, "spk_id": args.spk_id, "speed": args.speed,
           "pcm_sample_rate": PCM_SAMPLE_RATE, "pcm_bytes_per_sample": PCM_BYTES_PER_SAMPLE}

    if not probe_service(cfg):
        logger.error("TTS 服务不可达，停止。请确认 CosyVoice 服务已启动后重试。")
        sys.exit(2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # RUNINFO：记录本次测量的完整配置与输入来源（P1-5）
    runinfo = output_path.with_suffix('.runinfo.md')
    runinfo.write_text(
        f"# TTS 首包测量 RUNINFO\n\n"
        f"- 命令参数: {' '.join(sys.argv)}\n"
        f"- 服务: {args.url}\n- spk_id: {args.spk_id}\n- speed: {args.speed}\n"
        f"- PCM: {PCM_SAMPLE_RATE} Hz / {PCM_BYTES_PER_SAMPLE * 8} bit / mono\n"
        f"- 输入: {args.input or args.from_e4}\n"
        f"- preview fallback: {'允许' if args.allow_preview_fallback else '禁止'}\n"
        f"- 条数: {len(items)}\n",
        encoding="utf-8")

    n_ok = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "language", "n_chars", "ttfc_ms",
                         "total_ms", "audio_sec", "rtf", "text_source", "error"])
        for i, item in enumerate(items):
            m = measure_one(cfg, item["text"], timeout=args.timeout)
            writer.writerow([
                item["sample_id"], item["language"], len(item["text"]),
                f"{m['ttfc_ms']:.2f}" if m['ttfc_ms'] != "" else "",
                f"{m['total_ms']:.2f}" if m['total_ms'] != "" else "",
                f"{m['audio_sec']:.3f}" if m['audio_sec'] != "" else "",
                f"{m['rtf']:.3f}" if m['rtf'] != "" else "",
                item.get("text_source", ""),
                m["error"],
            ])
            f.flush()
            status = f"TTFC={m['ttfc_ms']:.0f}ms" if not m["error"] else f"error={m['error']}"
            logger.info(f"[{i + 1}/{len(items)}] {item['sample_id']} ({item['language']}, "
                        f"{len(item['text'])}字符): {status}")
            if not m["error"]:
                n_ok += 1
            if i < len(items) - 1:
                time.sleep(args.interval)

    logger.info(f"完成: {n_ok}/{len(items)} 成功 -> {output_path}")
    if n_ok == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
