# experiments/scripts/build_augmented_variants.py
"""
R2 增强变体构建：对 build_real_speech_set 产出的干净集生成噪声/变速变体。

变体定义（CISR_REVISION_PLAN §3.3 / handoff E2b）：
- snr20 / snr15 / snr10：MUSAN noise 子集（环境噪声）按 RMS 归一叠加到目标 SNR；
- babble：MUSAN speech 子集作为 babble 噪声（同 SNR 混合逻辑，默认 15 dB，可 --babble-snr 调整）；
- speed09 / speed11：librosa.effects.time_stretch（保音高变速），播放速度 0.9x / 1.1x，
  时长变为 1/0.9 / 1/1.1 倍，JSON 的 audio_duration 与 duration_group 同步重判。

子集规则：每变体默认抽 30 条，优先 long(≤15) + very_long(≤15)，不足从 extra_long 补；
随机抽样固定种子，同一干净集 + 同一 seed 结果确定。

输出：processed/{json,audio}/{dataset}_{variant}/，schema 与干净集一致，额外字段：
variant_of（源 sample_id）、augmentation、achieved_snr_db（噪声类）、playback_rate（变速类）。
mix 后写盘前做 [-1,1] 裁剪，裁剪比例 >1% 时告警（正常 SNR 档不应触发）。

用法（GPU 主机）：
  uv run python -m experiments.scripts.build_augmented_variants --dataset librispeech \
      --variants snr20 snr15 snr10 speed09 speed11 babble
  uv run python -m experiments.scripts.build_augmented_variants --dataset aishell1 \
      --variants snr20 snr15 snr10 speed09 speed11 babble
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf

from experiments.scripts.build_real_speech_set import SR, classify_group

VALID_VARIANTS = ("snr20", "snr15", "snr10", "speed09", "speed11", "babble")
SPEED_RATE = {"speed09": 0.9, "speed11": 1.1}   # 播放速度；librosa rate<1 → 拉长
CLIP_WARN_FRAC = 0.01


def select_subset(metas: List[dict], subset_n: int, rng: random.Random) -> List[dict]:
    """优先 long/very_long 各半（上限 ceil(subset/2)），不足从 extra_long 补，再不足从剩余全部补。"""
    by_group: Dict[str, List[dict]] = {}
    for m in metas:
        by_group.setdefault(m.get("duration_group", classify_group(m["audio_duration"])), []).append(m)
    chosen: List[dict] = []
    half = (subset_n + 1) // 2
    for group in ("long", "very_long"):
        pool = sorted(by_group.get(group, []), key=lambda m: m["sample_id"])
        take_n = min(len(pool), half, subset_n - len(chosen))
        take = rng.sample(pool, take_n) if take_n > 0 else []
        chosen += take
    chosen_ids = {m["sample_id"] for m in chosen}
    for group in ("extra_long",):
        pool = [m for m in by_group.get(group, []) if m["sample_id"] not in chosen_ids]
        take = rng.sample(pool, min(len(pool), subset_n - len(chosen))) if pool else []
        chosen += take
    if len(chosen) < subset_n:
        chosen_ids = {m["sample_id"] for m in chosen}
        rest = [m for m in metas if m["sample_id"] not in chosen_ids]
        chosen += rng.sample(rest, min(len(rest), subset_n - len(chosen)))
    return sorted(chosen, key=lambda m: m["sample_id"])


def load_noise_pool(musan_root: Path, variant: str) -> List[Path]:
    sub = "speech" if variant == "babble" else "noise"
    pool = sorted((musan_root / sub).rglob("*.wav"))
    if not pool:
        raise SystemExit(f"未找到 MUSAN {sub} 子集的 wav（期望 {musan_root / sub}；"
                         f"先 tar -xzf musan.tar.gz）")
    return pool


def read_noise(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def pick_noise_segment(pool: List[Path], n_samples: int, rng: random.Random) -> np.ndarray:
    """随机选噪声文件与起点；不足整条语音长度时循环铺满。"""
    for _ in range(5):
        noise = read_noise(rng.choice(pool))
        if float(np.sqrt(np.mean(np.square(noise, dtype=np.float64)))) < 1e-6:
            continue
        if len(noise) < n_samples:
            noise = np.tile(noise, int(np.ceil(n_samples / len(noise))))
        start = rng.randint(0, len(noise) - n_samples)
        return noise[start:start + n_samples]
    raise SystemExit("MUSAN 噪声池连续 5 次取到无效（近静音）文件，检查 noise 子集")


def mix_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
    p_sig = float(np.mean(np.square(signal, dtype=np.float64)))
    p_noise = float(np.mean(np.square(noise, dtype=np.float64)))
    if p_noise <= 0:
        raise SystemExit("噪声段能量为 0，无法按 SNR 混合")
    gain = float(np.sqrt(p_sig / (10 ** (snr_db / 10) * p_noise)))
    mixed = signal + noise * gain
    achieved = 10.0 * float(np.log10(p_sig / (p_noise * gain * gain)))
    if abs(achieved - snr_db) > 0.1:
        raise SystemExit(f"SNR 混合校验失败: 目标 {snr_db} dB, 实测 {achieved:.3f} dB")
    clip_frac = float(np.mean(np.abs(mixed) > 1.0))
    if clip_frac > CLIP_WARN_FRAC:
        print(f"[warn] 混合后裁剪比例 {clip_frac:.2%}（>1%），检查信号电平")
    return np.clip(mixed, -1.0, 1.0).astype(np.float32), achieved


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 R2 噪声/变速增强变体")
    parser.add_argument("--dataset", required=True, help="干净集目录名，如 librispeech / aishell1")
    parser.add_argument("--variants", nargs="+", default=["snr20", "snr15", "snr10", "speed09", "speed11", "babble"],
                        choices=VALID_VARIANTS)
    parser.add_argument("--subset", type=int, default=30, help="每变体抽样条数")
    parser.add_argument("--babble-snr", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-root", type=Path, default=Path("experiments/datasets/raw_data"))
    parser.add_argument("--json-dir", type=Path, default=Path("experiments/datasets/processed/json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("experiments/datasets/processed/audio"))
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("experiments/results/revision/r2_real_speech"))
    args = parser.parse_args()

    clean_json_dir = args.json_dir / args.dataset
    clean_audio_dir = args.audio_dir / args.dataset
    metas = []
    for p in sorted(clean_json_dir.glob("*.json")):
        metas.append(json.load(open(p, encoding="utf-8")))
    if not metas:
        raise SystemExit(f"干净集为空：{clean_json_dir}（先运行 build_real_speech_set）")

    need_musan = any(v in ("snr20", "snr15", "snr10", "babble") for v in args.variants)
    noise_pools: Dict[str, List[Path]] = {}
    if need_musan:
        musan_root = args.raw_root / "musan"
        if not musan_root.exists():
            raise SystemExit(f"未找到 {musan_root}（先下载并解压 MUSAN，见 handoff E2-0）")
        for v in set(args.variants) & {"snr20", "snr15", "snr10", "babble"}:
            noise_pools[v] = load_noise_pool(musan_root, v)

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "subset": args.subset,
        "variants": {},
        "cmd": f"python -m experiments.scripts.build_augmented_variants " +
               " ".join(f"--{k} {v}" for k, v in vars(args).items() if k != "manifest_dir"),
    }

    for variant in args.variants:
        # 变体名做确定性偏移（不能用 hash()，跨进程有 PYTHONHASHSEED 随机性）
        rng = random.Random(args.seed + sum(ord(c) for c in variant))
        chosen = select_subset(metas, args.subset, rng)
        out_name = f"{args.dataset}_{variant}"
        out_json = args.json_dir / out_name
        out_audio = args.audio_dir / out_name
        out_json.mkdir(parents=True, exist_ok=True)
        out_audio.mkdir(parents=True, exist_ok=True)

        variant_records = []
        for m in chosen:
            audio, sr = sf.read(clean_audio_dir / m["audio_file"], dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio[:, 0]
            if sr != SR:
                raise SystemExit(f"{m['sample_id']}: 采样率 {sr} != {SR}")

            new_meta = dict(m)
            new_meta["dataset"] = out_name
            new_meta["variant_of"] = m["sample_id"]
            new_meta["augmentation"] = variant

            if variant in SPEED_RATE:
                rate = SPEED_RATE[variant]
                stretched = librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
                new_audio = stretched
                new_meta["playback_rate"] = rate
                snr_info = None
            else:
                snr_db = args.babble_snr if variant == "babble" else float(variant.replace("snr", ""))
                noise = pick_noise_segment(noise_pools[variant], len(audio), rng)
                new_audio, achieved = mix_snr(audio, noise, snr_db)
                new_meta["achieved_snr_db"] = round(achieved, 3)
                snr_info = round(achieved, 3)

            duration = len(new_audio) / SR
            wav_name = f"{m['sample_id']}.wav"
            sf.write(out_audio / wav_name, new_audio, SR, subtype="PCM_16")
            new_meta["audio_file"] = wav_name
            new_meta["audio_duration"] = round(duration, 3)
            new_meta["duration_group"] = classify_group(duration)
            with open(out_json / f"{m['sample_id']}.json", "w", encoding="utf-8") as f:
                json.dump(new_meta, f, ensure_ascii=False, indent=2)
            variant_records.append({
                "sample_id": m["sample_id"],
                "source_group": m.get("duration_group"),
                "new_group": new_meta["duration_group"],
                "duration_s": round(duration, 3),
                "achieved_snr_db": snr_info,
            })

        manifest["variants"][variant] = {
            "output": out_name,
            "n_samples": len(variant_records),
            "records": variant_records,
        }
        print(f"[variant] {out_name}: {len(variant_records)} 条")

    manifest_path = args.manifest_dir / f"{args.dataset}_augment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
