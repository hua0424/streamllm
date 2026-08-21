# experiments/scripts/qa_real_speech.py
"""
R2 真实语音评测集 QA：静态校验 + 可选 Whisper 转写 sanity。

静态校验（默认）：
- 每条样本：WAV 存在、16 kHz 单声道、文件时长与 JSON audio_duration 一致（±50ms）、
  参考文本非空、RMS 能量 >= 1e-4；
- 干净集（librispeech/aishell1）另查分组配额（默认 long=30/very_long=30/extra_long=15）；
- 增强变体目录按 --expected-variant-count 查总数（默认 30）。

转写 sanity（--transcribe）：
- 用 Whisper（--asr-model-size，正式为 turbo）按 System A 相同解码参数
  （beam_size=5, temperature=0.0, condition_on_previous_text=False, word_timestamps=False）
  转写全部（或 --limit N）样本，计算 WER（英文）/ CER（中文），
  复用 run_exp_quality 的 normalize_text/wer/cer/zh_to_word_seq，与 exp3 口径一致；
- 验收线（--wer-limit，默认 10%）：干净集错误率超线 → 退出码 2，说明拼接/转写对齐有 bug，
  须先修复再跑 E2（对应 CISR_REVISION_PLAN §3.2 QA 要求）。

用法（GPU 主机，E2-0 验收步骤）：
  uv run python -m experiments.scripts.qa_real_speech --datasets librispeech,aishell1 \
      --transcribe --asr-model-size turbo --device cuda:0
  uv run python -m experiments.scripts.qa_real_speech \
      --datasets librispeech_snr20,librispeech_snr15,librispeech_snr10,librispeech_speed09,librispeech_speed11,librispeech_babble,aishell1_snr20,aishell1_snr15,aishell1_snr10,aishell1_speed09,aishell1_speed11,aishell1_babble

退出码：0=通过；1=静态校验失败；2=转写错误率超线。
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

SR = 16000
BASE_DATASETS = ("librispeech", "aishell1")
DURATION_TOL_S = 0.05
MIN_RMS = 1e-4
LANG_BY_DATASET = {"librispeech": "en", "aishell1": "zh"}


def language_of(dataset: str) -> str:
    for base, lang in LANG_BY_DATASET.items():
        if dataset == base or dataset.startswith(base + "_"):
            return lang
    raise SystemExit(f"无法推断数据集语言：{dataset}")


def check_static(dataset: str, json_dir: Path, audio_dir: Path,
                 expected_quota: dict, expected_variant_count: Optional[int],
                 report_rows: List[dict]) -> bool:
    metas = [json.load(open(p, encoding="utf-8")) for p in sorted((json_dir / dataset).glob("*.json"))]
    if not metas:
        print(f"[fail] {dataset}: 无 JSON 文件")
        return False

    ok = True
    group_counts = {}
    for m in metas:
        sample_id = m.get("sample_id", "?")
        wav = audio_dir / dataset / m["audio_file"]
        problems = []
        if not wav.exists():
            problems.append("wav_missing")
        else:
            info = sf.info(wav)
            if info.samplerate != SR:
                problems.append(f"sr_{info.samplerate}")
            if info.channels != 1:
                problems.append(f"channels_{info.channels}")
            if abs(info.frames / info.samplerate - m["audio_duration"]) > DURATION_TOL_S:
                problems.append(f"duration_mismatch({info.frames / info.samplerate:.3f} vs {m['audio_duration']})")
            audio, _ = sf.read(wav, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio[:, 0]
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            if rms < MIN_RMS:
                problems.append(f"low_rms({rms:.2e})")
        if not str(m.get("text", "")).strip():
            problems.append("empty_text")
        group = m.get("duration_group", "")
        group_counts[group] = group_counts.get(group, 0) + 1
        report_rows.append({
            "dataset": dataset, "sample_id": sample_id, "duration_group": group,
            "json_duration_s": m["audio_duration"], "problems": ";".join(problems) or "ok",
        })
        if problems:
            ok = False
            print(f"[fail] {dataset}/{sample_id}: {problems}")

    if dataset in BASE_DATASETS:
        for group, want in expected_quota.items():
            got = group_counts.get(group, 0)
            status = "ok" if got == want else "FAIL"
            print(f"[quota] {dataset} {group}: {got}/{want} {status}")
            if got != want:
                ok = False
    elif expected_variant_count is not None:
        got = len(metas)
        status = "ok" if got == expected_variant_count else "FAIL"
        print(f"[count] {dataset}: {got}/{expected_variant_count} {status}")
        if got != expected_variant_count:
            ok = False
    return ok


def run_transcribe(dataset: str, json_dir: Path, audio_dir: Path,
                   model, limit: Optional[int], report_rows: List[dict]) -> float:
    from experiments.scripts.run_exp_quality import cer, normalize_text, wer, zh_to_word_seq

    metas = [json.load(open(p, encoding="utf-8")) for p in sorted((json_dir / dataset).glob("*.json"))]
    if limit:
        metas = metas[:limit]
    lang = language_of(dataset)

    errors = []
    for m in metas:
        wav = audio_dir / dataset / m["audio_file"]
        # 与 System A 相同解码参数（faster_whisper_streamer.DEFAULT_*）
        result = model.transcribe(
            str(wav), language=lang, beam_size=5, temperature=0.0,
            condition_on_previous_text=False, word_timestamps=False,
        )
        hyp = result["text"].strip()
        if lang == "en":
            # LibriSpeech 参考文本为全大写、Whisper 输出混合大小写：
            # 在 exp3 归一化（去标点）之上补大小写折叠（LibriSpeech 官方评估惯例）。
            # run_exp_quality.normalize_text 保持不动（中文场景不受影响）。
            err = wer(normalize_text(m["text"]).lower(),
                      normalize_text(hyp).lower(), normalize=False)
        else:
            # exp3 原生口径（run_exp_quality.py:606）：cer 直接吃原文。
            # 不得先 zh_to_word_seq——逐字空格会污染 cer 的字符分母（2026-08-21 修正，
            # 此前口径把分母 inflate 约 2n-1，CER 被稀释；旧值以修正后重算为准）。
            err = cer(m["text"], hyp)
        errors.append(err)
        report_rows.append({
            "dataset": dataset, "sample_id": m["sample_id"],
            "duration_group": m.get("duration_group", ""),
            "error_rate": round(err, 4), "metric": "WER" if lang == "en" else "CER",
            "hypothesis": hyp, "reference": m["text"][:120],
        })
        print(f"[transcribe] {m['sample_id']}: {'WER' if lang == 'en' else 'CER'}={err:.3f}")

    mean_err = float(np.mean(errors)) if errors else 0.0
    print(f"[summary] {dataset}: mean {'WER' if lang == 'en' else 'CER'} = {mean_err:.4f} "
          f"({len(errors)} 条, max={max(errors):.4f})")
    return mean_err


def recompute_from_csv(csv_path: Path, json_dir: Path, wer_limit: float) -> int:
    """用修正口径从既有 qa_transcribe.csv 重算（2026-08-21 中文 CER 口径修正）。

    CSV 里的 reference 列是截断的（前 120 字符），完整参考从样本 JSON 取。
    英文 WER 口径未变（重算值应与原值一致，作为交叉验证）。
    """
    from experiments.scripts.run_exp_quality import cer, normalize_text, wer

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if not rows:
        print(f"[recompute] {csv_path} 为空")
        return 1
    out_rows = []
    by_dataset = {}
    for r in rows:
        meta_path = json_dir / r["dataset"] / f"{r['sample_id']}.json"
        meta = json.load(open(meta_path, encoding="utf-8"))
        hyp = r["hypothesis"].strip()
        if r["metric"] == "WER":
            new_err = wer(normalize_text(meta["text"]).lower(),
                          normalize_text(hyp).lower(), normalize=False)
        else:
            new_err = cer(meta["text"], hyp)
        out_rows.append({**r, "reference_full": meta["text"],
                         "error_rate_corrected": round(new_err, 4)})
        by_dataset.setdefault(r["dataset"], []).append((float(r["error_rate"]), new_err))

    out_path = csv_path.with_suffix(".corrected.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    worst = 0.0
    for ds, pairs in sorted(by_dataset.items()):
        old = [p[0] for p in pairs]
        new = [p[1] for p in pairs]
        worst = max(worst, float(np.mean(new)))
        print(f"[recompute] {ds}: 旧 mean={np.mean(old):.4f} → 修正 mean={np.mean(new):.4f} "
              f"(max={max(new):.4f}, n={len(new)})")
    print(f"[recompute] 已保存: {out_path}")
    return 0 if worst <= wer_limit else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 真实语音数据集 QA")
    parser.add_argument("--datasets", required=False, default=None,
                        help="逗号分隔目录名，如 librispeech,aishell1 或变体目录 librispeech_snr15")
    parser.add_argument("--recompute-from-csv", type=Path, default=None,
                        help="从既有 qa_transcribe.csv 用修正口径重算（无需模型；2026-08-21 中文 CER 修正）")
    parser.add_argument("--json-dir", type=Path, default=Path("experiments/datasets/processed/json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("experiments/datasets/processed/audio"))
    parser.add_argument("--report-dir", type=Path,
                        default=Path("experiments/results/revision/r2_real_speech"))
    parser.add_argument("--expected-quota", type=str, default="long=30,very_long=30,extra_long=15",
                        help="干净集分组配额")
    parser.add_argument("--expected-variant-count", type=int, default=30)
    parser.add_argument("--transcribe", action="store_true", help="运行 Whisper 转写 sanity")
    parser.add_argument("--asr-model-size", type=str, default="turbo")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="转写条数上限（调试用）")
    parser.add_argument("--wer-limit", type=float, default=0.10,
                        help="干净集错误率验收线（默认 0.10）")
    args = parser.parse_args()

    if args.recompute_from_csv:
        raise SystemExit(recompute_from_csv(args.recompute_from_csv, args.json_dir,
                                            args.wer_limit))
    if not args.datasets:
        parser.error("需要 --datasets 或 --recompute-from-csv")

    expected_quota = dict(
        (kv.split("=")[0], int(kv.split("=")[1]))
        for kv in args.expected_quota.split(",")
    )
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    static_ok = True
    all_static_rows: List[dict] = []
    for dataset in datasets:
        print(f"=== 静态校验: {dataset} ===")
        static_ok &= check_static(dataset, args.json_dir, args.audio_dir,
                                  expected_quota, args.expected_variant_count, all_static_rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    static_csv = args.report_dir / "qa_static.csv"
    if all_static_rows:
        with open(static_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_static_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_static_rows)
        print(f"[report] {static_csv}")

    if not static_ok:
        raise SystemExit(1)

    if args.transcribe:
        import whisper
        print(f"[load] whisper {args.asr_model_size} on {args.device}")
        model = whisper.load_model(args.asr_model_size, device=args.device)
        transcribe_ok = True
        all_transcribe_rows: List[dict] = []
        for dataset in datasets:
            if dataset not in BASE_DATASETS:
                continue  # 转写 sanity 只对干净集做验收判定
            print(f"=== 转写 sanity: {dataset} ===")
            mean_err = run_transcribe(dataset, args.json_dir, args.audio_dir,
                                      model, args.limit, all_transcribe_rows)
            if mean_err > args.wer_limit:
                print(f"[fail] {dataset} 错误率 {mean_err:.4f} > 验收线 {args.wer_limit}，"
                      f"拼接/转写对齐可能有 bug，先修复再跑 E2")
                transcribe_ok = False
        if all_transcribe_rows:
            transcribe_csv = args.report_dir / "qa_transcribe.csv"
            with open(transcribe_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_transcribe_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_transcribe_rows)
            print(f"[report] {transcribe_csv}")
        if not transcribe_ok:
            raise SystemExit(2)

    print("[qa] 全部通过")


if __name__ == "__main__":
    main()
