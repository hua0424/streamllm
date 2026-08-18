# experiments/scripts/build_real_speech_set.py
"""
R2 真实语音评测集构建：LibriSpeech（英文）/ AISHELL-1（中文）。

设计要点（与 CISR_REVISION_PLAN §三 一致）：
- 同章节（LibriSpeech）或同说话人（AISHELL-1）内按 utterance 顺序拼接，
  句间插入随机静音 U(0.2, 1.0) s（RNG 固定种子，全流程确定性可重建）；
- 目标规模默认每集合 75 条：long 30（15-30s）/ very_long 30（30-60s）/ extra_long 15（60-150s），
  分组区间与 run_exp_ablation.DURATION_GROUPS 口径一致；
- 输出与现有管线完全一致的 JSON+WAV schema（load_samples 免改）；
- 逐条 QA：时长重读校验（±50ms）、参考文本非空、RMS 能量下限；
- 生成构建 manifest（含逐样本来源 utterance、配额、种子、输入路径）供复查。

输入布局（experiments/datasets/raw_data/ 下，兼容压缩包与解压目录，压缩包自动解压）：
- librispeech/test-clean.tar.gz 或解压后的 LibriSpeech/test-clean/（test-other 可选备用）
- aishell1/data_aishell.tgz 或解压后的 AISHELL-1/（wav/S*.tar.gz 内层包直接懒读，无需再解压；
  transcript/aishell_transcript_v0.8.txt 为字间带空格格式，读取后去空格）

用法（GPU 主机正式构建）：
  uv run python -m experiments.scripts.build_real_speech_set --source librispeech
  uv run python -m experiments.scripts.build_real_speech_set --source aishell1

本机小规模验证：
  uv run python -m experiments.scripts.build_real_speech_set --source librispeech \
      --quota long:3,very_long:2,extra_long:1 \
      --json-dir <tmp>/json --audio-dir <tmp>/audio --manifest-dir <tmp>
"""
import argparse
import io
import json
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf

SR = 16000
# 构建目标区间（秒）：分组名与 run_exp_ablation.DURATION_GROUPS 一致
GROUP_TARGETS = {"long": (15.0, 30.0), "very_long": (30.0, 60.0), "extra_long": (60.0, 150.0)}
DEFAULT_QUOTA = "long:30,very_long:30,extra_long:15"
GAP_RANGE_S = (0.2, 1.0)   # 句间静音间隔 U(0.2, 1.0)
MIN_RMS = 1e-4             # 全样本 RMS 下限，排除静音/损坏样本
DURATION_TOL_S = 0.05      # 写盘后重读时长校验容差


def parse_quota(spec: str) -> Dict[str, int]:
    quota: Dict[str, int] = {}
    for part in spec.split(","):
        name, _, cnt = part.strip().partition(":")
        if name not in GROUP_TARGETS or not cnt:
            raise SystemExit(f"非法配额项 '{part}'，可用分组：{list(GROUP_TARGETS)}")
        quota[name] = int(cnt)
    return quota


def classify_group(duration: float) -> str:
    """与 run_exp_ablation.get_duration_group 相同口径的时长分组。"""
    if duration < 5:
        return "short"
    if duration < 15:
        return "medium"
    if duration < 30:
        return "long"
    if duration < 60:
        return "very_long"
    return "extra_long"


# ---------------------------------------------------------------------------
# 素材单元：LibriSpeech 章节 / AISHELL 说话人
# ---------------------------------------------------------------------------

@dataclass
class SourceUnit:
    """一个可拼接的素材单元。

    stream_utterances(start) 从第 start 句开始顺序产出 (utt_id, text, audio)；
    生成器结束时负责关闭内部资源（如内层 tar 句柄）。
    """
    source_id: str                                   # dialog_id
    kind: str                                        # librispeech | aishell
    items: List[Tuple[str, str]] = field(default_factory=list)  # (utt_id, text)，已过滤无转写句
    loader: Optional[callable] = None                # aishell: (start) -> iterator

    def stream_utterances(self, start: int) -> Iterator[Tuple[str, str, np.ndarray]]:
        if self.kind == "librispeech":
            paths = self._paths
            for utt_id, text in self.items[start:]:
                yield utt_id, text, _read_audio_bytes((paths[utt_id]).read_bytes(), utt_id)
        else:
            yield from self.loader(start)


def _read_audio_bytes(data: bytes, utt_id: str) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if sr != SR:
        raise SystemExit(f"{utt_id}: 采样率 {sr} != {SR}，语料应为 16 kHz")
    return audio


def _safe_extract(tar_path: Path, dest: Path) -> None:
    """解压 openslr 官方包到 dest（拒绝绝对路径/.. 成员，防止路径逃逸）。"""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise SystemExit(f"{tar_path}: 拒绝不安全成员名 {m.name}")
        tf.extractall(dest)
    print(f"[extract] {tar_path.name} -> {dest}")


# --------------------------- LibriSpeech ----------------------------------

def collect_librispeech(raw_root: Path, subsets: List[str]) -> List[SourceUnit]:
    base = raw_root / "librispeech"
    if not base.exists():
        raise SystemExit(f"未找到 {base}，先下载 LibriSpeech（见 GPU_EXPERIMENT_HANDOFF E2-0）")

    for tar_path in sorted(base.glob("*.tar.gz")):
        subset = tar_path.stem.replace(".tar", "")  # test-clean.tar.gz -> test-clean
        out = base / subset
        if not any(out.rglob("*.flac")):
            print(f"[extract] {tar_path.name} 尚未解压，解压到 {out}")
            _safe_extract(tar_path, out)

    units: List[SourceUnit] = []
    for subset in subsets:
        roots = [base / subset, base / "LibriSpeech" / subset]
        root = next((r for r in roots if r.exists()), None)
        if root is None:
            print(f"[warn] 子集目录不存在，跳过：{subset}（可选下载）")
            continue
        for trans in sorted(root.rglob("*.trans.txt")):
            texts: Dict[str, str] = {}
            for line in trans.read_text(encoding="utf-8").splitlines():
                utt_id, _, text = line.partition(" ")
                if utt_id and text:
                    texts[utt_id.strip()] = text.strip()
            chapter_dir = trans.parent
            flacs = sorted(chapter_dir.glob("*.flac"))
            items = [(p.stem, texts[p.stem]) for p in flacs if p.stem in texts]
            if not items:
                continue
            unit = SourceUnit(
                source_id=f"{chapter_dir.parent.name}-{chapter_dir.name}",  # spk-chap
                kind="librispeech",
                items=items,
            )
            unit._paths = {p.stem: p for p in flacs}  # type: ignore[attr-defined]
            units.append(unit)
    units.sort(key=lambda u: u.source_id)
    if not units:
        raise SystemExit("LibriSpeech 未收集到任何章节（检查 raw_data/librispeech 布局）")
    return units


# --------------------------- AISHELL-1 -------------------------------------

@dataclass
class _AishellIndex:
    cache: Dict[Path, List[Tuple[str, str]]] = field(default_factory=dict)


_AISHELL_INDEX = _AishellIndex()


def _aishell_members(tar_path: Path) -> List[Tuple[str, str]]:
    if tar_path not in _AISHELL_INDEX.cache:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [
                (Path(m.name).stem, m.name)
                for m in tf.getmembers()
                if m.isfile() and m.name.endswith(".wav")
            ]
        members.sort(key=lambda x: x[0])
        _AISHELL_INDEX.cache[tar_path] = members
    return _AISHELL_INDEX.cache[tar_path]


def collect_aishell(raw_root: Path, speakers: Optional[List[str]]) -> List[SourceUnit]:
    base = raw_root / "aishell1"
    if not base.exists():
        raise SystemExit(f"未找到 {base}，先下载 AISHELL-1（见 GPU_EXPERIMENT_HANDOFF E2-0）")

    tgz = sorted(base.rglob("data_aishell.tgz")) or sorted(base.glob("*.tgz"))
    if not any(base.rglob("aishell_transcript_v0.8.txt")) and tgz:
        print(f"[extract] {tgz[0]} 尚未解压，解压外层包到 {base}（内层 wav/S*.tar.gz 保持懒读）")
        _safe_extract(tgz[0], base)

    trans_files = sorted(base.rglob("aishell_transcript_v0.8.txt"))
    if not trans_files:
        raise SystemExit("未找到 aishell_transcript_v0.8.txt（检查 raw_data/aishell1 布局）")
    trans_file = trans_files[0]

    trans: Dict[str, str] = {}
    for line in trans_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            # 官方转写为字间带空格格式，读取后去空格
            trans[parts[0]] = "".join(parts[1:])
    if not trans:
        raise SystemExit(f"转写文件为空：{trans_file}")

    # 官方布局：AISHELL-1/transcript/aishell_transcript_v0.8.txt 与 AISHELL-1/wav/ 平级
    wav_root = trans_file.parent.parent / "wav"
    if not wav_root.exists():
        wav_candidates = sorted(trans_file.parent.parent.rglob("wav"))
        wav_root = next((w for w in wav_candidates if w.is_dir()), None)
    if not wav_root or not wav_root.exists():
        raise SystemExit(f"未找到 wav 目录（期望与 transcript/ 平级）：{trans_file.parent.parent / 'wav'}")

    units: List[SourceUnit] = []
    for entry in sorted(wav_root.iterdir()):
        spk = entry.name.split(".")[0]
        if speakers and spk not in speakers:
            continue
        if entry.is_dir():
            wavs = sorted(entry.glob("*.wav"))
            items = [(p.stem, trans[p.stem]) for p in wavs if p.stem in trans]
            if not items:
                continue

            def make_stream(dir_entry: Path, items: List[Tuple[str, str]]):
                def stream(start: int):
                    for utt_id, text in items[start:]:
                        audio, sr = sf.read(dir_entry / f"{utt_id}.wav", dtype="float32", always_2d=False)
                        if audio.ndim == 2:
                            audio = audio[:, 0]
                        if sr != SR:
                            raise SystemExit(f"{utt_id}: 采样率 {sr} != {SR}")
                        yield utt_id, text, audio
                return stream

            units.append(SourceUnit(source_id=spk, kind="aishell", items=items,
                                    loader=make_stream(entry, items)))
        elif entry.is_file() and entry.name.endswith(".tar.gz"):
            members = [(u, m) for u, m in _aishell_members(entry) if u in trans]
            if not members:
                continue
            items = [(u, trans[u]) for u, _ in members]
            member_names = dict(members)

            def make_tar_stream(tar_entry: Path, names: Dict[str, str],
                                bound_items: List[Tuple[str, str]]):
                def stream(start: int):
                    with tarfile.open(tar_entry, "r:gz") as tf:
                        for utt_id, _ in bound_items[start:]:
                            f = tf.extractfile(names[utt_id])
                            if f is None:
                                continue
                            yield utt_id, trans[utt_id], _read_audio_bytes(f.read(), utt_id)
                return stream

            units.append(SourceUnit(source_id=spk, kind="aishell", items=items,
                                    loader=make_tar_stream(entry, member_names, items)))
        else:
            continue

    if speakers:
        missing = set(speakers) - {u.source_id for u in units}
        if missing:
            raise SystemExit(f"指定的说话人不存在：{sorted(missing)}")
    units.sort(key=lambda u: u.source_id)
    if not units:
        raise SystemExit("AISHELL-1 未收集到任何说话人（检查 wav/ 目录）")
    return units


# ---------------------------------------------------------------------------
# 拼接构建
# ---------------------------------------------------------------------------

def _try_build_one(unit: SourceUnit, start: int, lo: float, hi: float,
                   rng: random.Random) -> Optional[Tuple[np.ndarray, float, List[str], List[str]]]:
    """从 start 句起贪心拼接：达到 lo 即收手；任何一句会越过 hi 则整体放弃（不消费素材）。"""
    pieces: List[np.ndarray] = []
    texts: List[str] = []
    utt_ids: List[str] = []
    total = 0.0
    stream = unit.stream_utterances(start)
    try:
        nxt = next(stream, None)
        while nxt is not None:
            if total >= lo:
                break  # 已达目标时长，收手（未消费 nxt）
            utt_id, text, audio = nxt
            dur = len(audio) / SR
            gap = rng.uniform(*GAP_RANGE_S) if pieces else 0.0
            if total + gap + dur > hi:
                return None
            if pieces:
                pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
            pieces.append(audio)
            texts.append(text)
            utt_ids.append(utt_id)
            total += gap + dur
            nxt = next(stream, None)
    finally:
        stream.close()
    if total < lo:
        return None
    full = np.concatenate(pieces)
    if float(np.sqrt(np.mean(np.square(full, dtype=np.float64)))) < MIN_RMS:
        return None
    return full, total, utt_ids, texts


def build_dataset(units: List[SourceUnit], quota: Dict[str, int], rng: random.Random,
                  dataset: str, language: str, json_dir: Path, audio_dir: Path) -> List[dict]:
    needs: List[str] = []
    for group in ("long", "very_long", "extra_long"):
        needs += [group] * quota.get(group, 0)

    cursors = {u.source_id: 0 for u in units}
    built_counts = {u.source_id: 0 for u in units}
    pos = 0
    n_units = len(units)
    records: List[dict] = []

    for group in needs:
        lo, hi = GROUP_TARGETS[group]
        result = None
        for _ in range(n_units):
            unit = units[pos % n_units]
            pos += 1
            cand = _try_build_one(unit, cursors[unit.source_id], lo, hi, rng)
            if cand is None:
                continue
            audio, total, utt_ids, texts = cand
            cursors[unit.source_id] += len(utt_ids)
            result = (unit, audio, total, utt_ids, texts)
            break
        if result is None:
            print(f"[warn] 分组 {group}：{n_units} 个素材单元均无法满足 {lo}-{hi}s，跳过 1 条")
            continue

        unit, audio, total, utt_ids, texts = result
        built_counts[unit.source_id] += 1
        seq = built_counts[unit.source_id]
        if dataset == "librispeech":
            sample_id = unit.source_id if seq == 1 else f"{unit.source_id}_p{seq}"
            sample_id = f"librispeech_{sample_id}"
            text = " ".join(texts)
        else:
            sample_id = f"aishell1_{unit.source_id}_{seq:02d}"
            text = "".join(texts)

        audio_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / f"{sample_id}.wav"
        sf.write(wav_path, audio, SR, subtype="PCM_16")
        info = sf.info(wav_path)
        if abs(info.frames / info.samplerate - total) > DURATION_TOL_S:
            raise SystemExit(f"{sample_id}: 写盘时长校验失败 "
                             f"({info.frames / info.samplerate:.3f}s vs {total:.3f}s)")
        got_group = classify_group(total)
        if got_group != group:
            raise SystemExit(f"{sample_id}: 目标分组 {group} 与实际 {got_group} 不一致（{total:.2f}s）")

        meta = {
            "sample_id": sample_id,
            "dialog_id": unit.source_id,
            "turn_index": 0,
            "text": text,
            "text_length": len(text),
            "audio_file": wav_path.name,
            "audio_duration": round(total, 3),
            "language": language,
            "dataset": dataset,
            "duration_group": group,
            "source_utterances": utt_ids,
        }
        with open(json_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        records.append({"sample_id": sample_id, "dialog_id": unit.source_id,
                        "duration_group": group, "audio_duration_s": round(total, 3),
                        "n_utterances": len(utt_ids), "source_utterances": utt_ids})
        print(f"[build] {sample_id}: {group} {total:.2f}s x{len(utt_ids)}utt")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 R2 真实语音评测集")
    parser.add_argument("--source", required=True, choices=["librispeech", "aishell1"])
    parser.add_argument("--raw-root", type=Path, default=Path("experiments/datasets/raw_data"))
    parser.add_argument("--json-dir", type=Path, default=Path("experiments/datasets/processed/json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("experiments/datasets/processed/audio"))
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("experiments/results/revision/r2_real_speech"))
    parser.add_argument("--quota", type=str, default=DEFAULT_QUOTA,
                        help=f"分组配额，默认 {DEFAULT_QUOTA}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--librispeech-subsets", type=str, default="test-clean",
                        help="逗号分隔，默认 test-clean；素材不足时可加 test-other")
    parser.add_argument("--aishell-speakers", type=str, default=None,
                        help="逗号分隔说话人列表（如官方 test split）；缺省=解压出的全部说话人按序取用")
    args = parser.parse_args()

    quota = parse_quota(args.quota)
    rng = random.Random(args.seed)

    if args.source == "librispeech":
        units = collect_librispeech(args.raw_root, args.librispeech_subsets.split(","))
        language = "en"
    else:
        speakers = args.aishell_speakers.split(",") if args.aishell_speakers else None
        units = collect_aishell(args.raw_root, speakers)
        language = "zh"
    print(f"[collect] {args.source}: {len(units)} 个素材单元（章节/说话人）")

    json_dir = args.json_dir / args.source
    audio_dir = args.audio_dir / args.source
    records = build_dataset(units, quota, rng, args.source, language, json_dir, audio_dir)

    counts = {g: sum(1 for r in records if r["duration_group"] == g)
              for g in ("long", "very_long", "extra_long")}
    total_dur = sum(r["audio_duration_s"] for r in records)
    print(f"[done] {args.source}: {len(records)} 条 ({counts})，总时长 {total_dur / 60:.1f} min")

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.source,
        "seed": args.seed,
        "quota": quota,
        "produced": len(records),
        "group_counts": counts,
        "total_duration_min": round(total_dur / 60, 2),
        "gap_policy": f"U({GAP_RANGE_S[0]}, {GAP_RANGE_S[1]}) s between utterances",
        "raw_root": str(args.raw_root),
        "samples": records,
        "cmd": f"python -m experiments.scripts.build_real_speech_set " +
               " ".join(f"--{k} {v}" for k, v in vars(args).items() if k != "manifest_dir"),
    }
    manifest_path = args.manifest_dir / f"{args.source}_build_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] {manifest_path}")

    if len(records) < sum(quota.values()):
        print(f"[warn] 产出 {len(records)} 条 < 配额 {sum(quota.values())} 条，"
              f"考虑追加子集（--librispeech-subsets test-clean,test-other）或检查素材")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
