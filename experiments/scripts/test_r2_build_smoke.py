# experiments/scripts/test_r2_build_smoke.py
"""
R2 数据集构建链路冒烟测试（纯 CPU，不依赖真实语料与网络）。

伪造迷你 LibriSpeech（FLAC + trans.txt，打包为 test-clean.tar.gz 走自动解压路径）、
迷你 AISHELL-1（一半说话人为内层 tar.gz、一半为解压目录）、迷你 MUSAN（noise/speech），
然后跑完整链路：

  [S1] 伪造迷你语料
  [S2] build_real_speech_set --source librispeech（配额 long:3,very_long:2,extra_long:1）
  [S3] build_real_speech_set --source aishell1
  [S4] schema/内容断言（字段齐全、16k 单声道、时长一致、中文转写无空格、句数与来源一致）
  [S5] 确定性：同种子重建 → 全部 wav/json 字节级一致
  [S6] build_augmented_variants（snr20 + speed09）：条数、独立复算 SNR、变速时长比
  [S7] qa_real_speech 静态验收通过；篡改一条时长后必须失败（负例）

运行：uv run python -m experiments.scripts.test_r2_build_smoke
退出码：0=全部通过；1=有失败项。
"""
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.scripts.build_real_speech_set import classify_group

REPO_ROOT = Path(__file__).resolve().parents[2]
SR = 16000
LIBRI_DURS = [3.4, 5.1, 6.8, 4.2, 7.5, 5.9, 6.1, 4.8, 3.9, 7.2, 5.5, 6.6]
AISHELL_DURS = [4.1, 5.3, 6.2, 4.7, 5.8, 6.5, 4.4, 5.6, 6.9, 4.9,
                5.2, 6.3, 5.9, 4.6, 6.1, 5.4]
LIBRI_SPK = ["61", "62", "63"]
LIBRI_CHAP = ["0001", "0002"]
AISHELL_SPK = ["S0001", "S0002", "S0003", "S0004"]

_results = []


def check(case: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    _results.append((case, cond))
    print(f"[{case}] {status}" + (f"  {detail}" if detail else ""))
    return cond


def run_cmd(args, expect_rc=0):
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True,
                          text=True, cwd=str(REPO_ROOT))
    if proc.returncode != expect_rc:
        print(f"--- 命令输出（期望 rc={expect_rc}，实际 rc={proc.returncode}）---")
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
    return proc


def tone(duration_s: float, freq: float, amp: float = 0.25) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    sig = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
    ramp = min(int(0.03 * SR), n // 2)
    if ramp > 0:
        env = np.ones(n, dtype=np.float32)
        env[:ramp] = np.linspace(0, 1, ramp, dtype=np.float32)
        env[-ramp:] = np.linspace(1, 0, ramp, dtype=np.float32)
        sig *= env
    return sig


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------- S1 伪造语料

def fabricate(raw_root: Path) -> dict:
    libri_dir = raw_root / "librispeech"
    libri_dir.mkdir(parents=True)
    staging = raw_root / "_libri_staging" / "LibriSpeech" / "test-clean"
    chapters = {}
    for spk in LIBRI_SPK:
        for chap in LIBRI_CHAP:
            chapter_dir = staging / spk / chap
            chapter_dir.mkdir(parents=True)
            trans_lines, texts = [], []
            for i, dur in enumerate(LIBRI_DURS):
                utt_id = f"{spk}-{chap}-{i:04d}"
                text = f"SMOKE TEXT {spk} {chap} LINE {i}"
                sf.write(chapter_dir / f"{utt_id}.flac", tone(dur, 220 + int(spk) + i * 31),
                         SR, format="FLAC")
                trans_lines.append(f"{utt_id} {text}")
                texts.append(text)
            (chapter_dir / f"{spk}-{chap}.trans.txt").write_text(
                "\n".join(trans_lines) + "\n", encoding="utf-8")
            chapters[f"{spk}-{chap}"] = texts
    # 打包为官方布局的 tar.gz，验证自动解压路径
    with tarfile.open(libri_dir / "test-clean.tar.gz", "w:gz") as tf:
        tf.add(staging.parent.parent / "LibriSpeech", arcname="LibriSpeech")

    aishell_base = raw_root / "aishell1" / "AISHELL-1"
    (aishell_base / "transcript").mkdir(parents=True)
    trans_lines = []
    for spk in AISHELL_SPK:
        for i, dur in enumerate(AISHELL_DURS):
            utt_id = f"BAC009{spk}W{i:04d}"
            trans_lines.append(f"{utt_id} 这 是 迷 你 语 料 {spk} 第 {i} 句")
    (aishell_base / "transcript" / "aishell_transcript_v0.8.txt").write_text(
        "\n".join(trans_lines) + "\n", encoding="utf-8")
    wav_root = aishell_base / "wav"
    wav_root.mkdir()
    for spk in AISHELL_SPK:
        base_freq = 180 + int(spk[-2:]) * 7
        utts = [f"BAC009{spk}W{i:04d}" for i in range(len(AISHELL_DURS))]
        if spk in ("S0001", "S0003"):  # 内层 tar.gz 布局
            with tarfile.open(wav_root / f"{spk}.tar.gz", "w:gz") as tf:
                for i, utt_id in enumerate(utts):
                    data = wav_bytes(tone(AISHELL_DURS[i], base_freq + i * 13))
                    info = tarfile.TarInfo(name=f"{spk}/{utt_id}.wav")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        else:  # 解压目录布局
            spk_dir = wav_root / spk
            spk_dir.mkdir()
            for i, utt_id in enumerate(utts):
                sf.write(spk_dir / f"{utt_id}.wav",
                         tone(AISHELL_DURS[i], base_freq + i * 13), SR)

    musan = raw_root / "musan"
    (musan / "noise" / "free-sound").mkdir(parents=True)
    (musan / "speech" / "us").mkdir(parents=True)
    rng = np.random.default_rng(7)
    sf.write(musan / "noise" / "free-sound" / "mini_noise.wav",
             (0.05 * rng.standard_normal(10 * SR)).astype(np.float32), SR)
    t = np.arange(12 * SR) / SR
    babble = (0.1 * np.sin(2 * np.pi * 300 * t) + 0.08 * np.sin(2 * np.pi * 700 * t)
              + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    sf.write(musan / "speech" / "us" / "mini_babble.wav", babble, SR)
    return chapters


# ---------------------------------------------------------------- S2/S3 构建

def run_builder(source: str, raw_root: Path, json_dir: Path, audio_dir: Path, manifest_dir: Path):
    proc = run_cmd([
        "experiments.scripts.build_real_speech_set",
        "--source", source,
        "--raw-root", str(raw_root),
        "--json-dir", str(json_dir),
        "--audio-dir", str(audio_dir),
        "--manifest-dir", str(manifest_dir),
        "--quota", "long:3,very_long:2,extra_long:1",
    ])
    return proc.returncode == 0


def load_built(json_dir: Path, dataset: str):
    metas = []
    for p in sorted((json_dir / dataset).glob("*.json")):
        metas.append((p, json.load(open(p, encoding="utf-8"))))
    return metas


def verify_schema(json_dir: Path, audio_dir: Path, dataset: str, tag: str):
    metas = load_built(json_dir, dataset)
    ok = check(f"{tag}-count", len(metas) == 6, f"{len(metas)} 条")
    fields = {"sample_id", "dialog_id", "turn_index", "text", "text_length",
              "audio_file", "audio_duration", "language", "dataset",
              "duration_group", "source_utterances"}
    groups = {}
    for _, m in metas:
        if not fields <= set(m.keys()):
            ok &= check(f"{tag}-fields-{m.get('sample_id')}", False, f"缺字段: {fields - set(m.keys())}")
            continue
        wav = audio_dir / dataset / m["audio_file"]
        info = sf.info(wav)
        dur_ok = abs(info.frames / info.samplerate - m["audio_duration"]) <= 0.05
        fmt_ok = info.samplerate == SR and info.channels == 1
        text_ok = bool(m["text"].strip()) and m["text_length"] == len(m["text"])
        if not (dur_ok and fmt_ok and text_ok):
            ok &= check(f"{tag}-sample-{m['sample_id']}", False,
                        f"dur_ok={dur_ok} fmt_ok={fmt_ok} text_ok={text_ok}")
        groups[m["duration_group"]] = groups.get(m["duration_group"], 0) + 1
    ok &= check(f"{tag}-groups", groups == {"long": 3, "very_long": 2, "extra_long": 1}, str(groups))
    return ok, metas


# ---------------------------------------------------------------- 主流程

def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="r2_smoke_"))
    raw_root = tmp / "raw_data"
    out1_json, out1_audio = tmp / "out1" / "json", tmp / "out1" / "audio"
    manifest_dir = tmp / "manifest"
    try:
        chapters = fabricate(raw_root)
        check("S1-fabricate", (raw_root / "librispeech" / "test-clean.tar.gz").exists()
              and (raw_root / "aishell1" / "AISHELL-1" / "wav").exists())

        ok = run_builder("librispeech", raw_root, out1_json, out1_audio, manifest_dir)
        check("S2-build-librispeech", ok)
        ok = run_builder("aishell1", raw_root, out1_json, out1_audio, manifest_dir)
        check("S3-build-aishell1", ok)

        ok_l, libri_metas = verify_schema(out1_json, out1_audio, "librispeech", "S4-libri")
        ok_a, aishell_metas = verify_schema(out1_json, out1_audio, "aishell1", "S4-aishell")
        zh_ok = all(" " not in m["text"] for _, m in aishell_metas)
        check("S4-zh-no-space", zh_ok)
        en_ok = all(m["text"] == " ".join(chapters[m["dialog_id"]][:len(m["source_utterances"])])
                    for _, m in libri_metas)
        check("S4-en-text-concat", en_ok, "参考文本=章节前 N 句按序拼接")

        # S5 确定性
        out2_json, out2_audio = tmp / "out2" / "json", tmp / "out2" / "audio"
        run_builder("librispeech", raw_root, out2_json, out2_audio, tmp / "manifest2")

        def digest(root: Path, dataset: str):
            h = hashlib.md5()
            for p in sorted((root / dataset).iterdir()):
                h.update(p.name.encode())
                h.update(p.read_bytes())
            return h.hexdigest()

        same = (digest(out1_json, "librispeech") == digest(out2_json, "librispeech")
                and digest(out1_audio, "librispeech") == digest(out2_audio, "librispeech"))
        check("S5-determinism", same)

        # S6 增强
        proc = run_cmd([
            "experiments.scripts.build_augmented_variants",
            "--dataset", "librispeech",
            "--variants", "snr20", "speed09",
            "--raw-root", str(raw_root),
            "--json-dir", str(out1_json),
            "--audio-dir", str(out1_audio),
            "--manifest-dir", str(manifest_dir),
        ])
        check("S6-augment-run", proc.returncode == 0)
        snr_files = list((out1_json / "librispeech_snr20").glob("*.json"))
        spd_files = list((out1_json / "librispeech_speed09").glob("*.json"))
        check("S6-counts", len(snr_files) == 6 and len(spd_files) == 6,
              f"snr20={len(snr_files)} speed09={len(spd_files)}")

        snr_ok, ratio_ok = True, True
        for jp in snr_files:
            m = json.load(open(jp, encoding="utf-8"))
            clean, _ = sf.read(out1_audio / "librispeech" / f"{m['variant_of']}.wav",
                               dtype="float32", always_2d=False)
            aug, _ = sf.read(out1_audio / "librispeech_snr20" / m["audio_file"],
                             dtype="float32", always_2d=False)
            noise = aug - clean  # 同为 PCM_16 读回，量化误差可忽略
            p_sig = float(np.mean(clean ** 2))
            p_noise = float(np.mean(noise ** 2))
            meas = 10 * np.log10(p_sig / p_noise)
            if abs(meas - 20.0) > 0.25 or abs(m["achieved_snr_db"] - 20.0) > 0.1:
                snr_ok = False
                print(f"  [S6] {m['sample_id']}: 实测 {meas:.2f} dB, manifest {m['achieved_snr_db']} dB")
        check("S6-snr-20db", snr_ok)
        for jp in spd_files:
            m = json.load(open(jp, encoding="utf-8"))
            src_m = json.load(open(out1_json / "librispeech" / f"{m['variant_of']}.json",
                                   encoding="utf-8"))
            ratio = m["audio_duration"] / src_m["audio_duration"]
            if abs(ratio - 1 / 0.9) > 0.05 * (1 / 0.9):
                ratio_ok = False
                print(f"  [S6] {m['sample_id']}: 时长比 {ratio:.3f} (期望 ~{1 / 0.9:.3f})")
            if m["duration_group"] != classify_group(m["audio_duration"]):
                ratio_ok = False
        check("S6-speed-ratio", ratio_ok)

        # S7 QA 正例 + 负例
        proc = run_cmd([
            "experiments.scripts.qa_real_speech",
            "--datasets", "librispeech,aishell1",
            "--json-dir", str(out1_json), "--audio-dir", str(out1_audio),
            "--report-dir", str(tmp / "qa_report"),
            "--expected-quota", "long=3,very_long=2,extra_long=1",
            "--expected-variant-count", "6",
        ])
        check("S7-qa-pass", proc.returncode == 0)

        bad_root = tmp / "bad"
        shutil.copytree(out1_json, bad_root / "json")
        shutil.copytree(out1_audio, bad_root / "audio")
        victim = next((bad_root / "json" / "librispeech").glob("*.json"))
        vm = json.load(open(victim, encoding="utf-8"))
        vm["audio_duration"] = round(vm["audio_duration"] + 1.0, 3)
        victim.write_text(json.dumps(vm, ensure_ascii=False), encoding="utf-8")
        proc = run_cmd([
            "experiments.scripts.qa_real_speech",
            "--datasets", "librispeech",
            "--json-dir", str(bad_root / "json"), "--audio-dir", str(bad_root / "audio"),
            "--report-dir", str(tmp / "qa_report_bad"),
            "--expected-quota", "long=3,very_long=2,extra_long=1",
        ], expect_rc=1)
        check("S7-qa-negative", proc.returncode == 1)

        passed = sum(1 for _, c in _results if c)
        print(f"\n=== 冒烟结果: {passed}/{len(_results)} 通过 ===")
        return 0 if passed == len(_results) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
