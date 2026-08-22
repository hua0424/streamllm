# -*- coding: utf-8 -*-
"""r7_main / r7_tts_control 本地结果级 QA（一次性核验脚本，随核验文档归档）。

用法: uv run python experiments/results/revision/r7_ttfa_unified/handoff/_qa_r7_local_20260822.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scripts.run_ttfa_unified import (  # noqa: E402
    build_schedule, schedule_hash as sched_hash_fn, canonical_json, validate_record)

R7 = PROJECT_ROOT / "experiments/results/revision/r7_ttfa_unified"
MAIN = R7 / "r7_main"
CTRL = R7 / "tts_control"
SAMPLE_LIST = PROJECT_ROOT / "experiments/results/revision/r1_stats/repeat_subset_ids.json"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def sha256_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_ckpt(p: Path):
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[0], lines[1:]


# ============ r7_main ============
hdr, recs = load_ckpt(MAIN / "checkpoint_r7_main.jsonl")

# 1) header/binding
check("r7_main 记录数=140", len(recs) == 140, f"actual={len(recs)}")
check("r7_main header schema=ttfa_unified/1", hdr.get("schema_version") == "ttfa_unified/1")
check("r7_main header run_id=r7_main", hdr.get("run_id") == "r7_main")
b = hdr.get("binding", hdr)
check("r7_main git_commit=c9437c3", str(b.get("git_commit", "")).startswith("c9437c3a"),
      b.get("git_commit", ""))
check("r7_main platform_conditions_sha256=a4c40057…",
      str(b.get("tts_config", {}).get("platform_conditions_sha256", "")).startswith("a4c400576b"),
      str(b.get("tts_config", {}).get("platform_conditions_sha256"))[:16])
check("r7_main silero artifact=e1122837…（PSE/分段器同源）",
      str(b.get("silero_meta", {}).get("artifact_sha256", "")).startswith("e1122837f4"))

# 2) 终态/错误/fatal + 脚本自带 validate_record 全量重放
states = Counter(r["terminal_state"] for r in recs)
check("r7_main 全部 success", set(states) == {"success"}, str(dict(states)))
check("r7_main error 全空", all(not r.get("error") for r in recs))
check("r7_main fatal 全 False", all(r.get("fatal") is False for r in recs))
v_errs = []
for r in recs:
    e = validate_record(r, expected_config_hash=hdr["config_hash"],
                        expected_schedule_hash=hdr["schedule_hash"])
    if e:
        v_errs.append((r["sample_id"], r["mode"], r["repeat_idx"], e))
check("validate_record 140 条全过（schema+因果偏序+闭合恒等式）",
      not v_errs, str(v_errs[:3]))

# 3) 任务唯一性 + 子集三轮
keys = [(r["sample_id"], r["mode"], r["repeat_idx"]) for r in recs]
check("r7_main (sample,mode,repeat) 唯一", len(set(keys)) == 140)
per_sample = defaultdict(list)
for r in recs:
    per_sample[r["sample_id"]].append(r["repeat_idx"])
subset_actual = sorted(s for s, v in per_sample.items() if len(v) == 6)
n_single = [s for s, v in per_sample.items() if len(v) == 2]
check("r7_main 10 子集样本恰 3 轮×2 模式", len(subset_actual) == 10 and
      all(sorted(v) == [0, 0, 1, 1, 2, 2] for s, v in per_sample.items() if s in subset_actual),
      f"subset={len(subset_actual)}")
check("r7_main 40 非子集样本仅 repeat 0",
      len(n_single) == 40 and all(per_sample[s] == [0, 0] for s in n_single))
check("r7_main 样本总数=50", len(per_sample) == 50)
lang_of = {r["sample_id"]: r["language"] for r in recs}
check("实际子集=5zh+5en",
      Counter(lang_of[s] for s in subset_actual) == {"zh": 5, "en": 5})

# 4) 计划复算：load_samples 复现运行时加载序（crosswoz→multiwoz）
from experiments.scripts.run_exp_latency import load_samples  # noqa: E402
sl = json.loads(SAMPLE_LIST.read_text(encoding="utf-8"))
sample_ids = sl if isinstance(sl, list) else sl["sample_ids"]
samples = []
for ds in ("crosswoz", "multiwoz"):
    for s in load_samples(PROJECT_ROOT / "experiments/datasets/processed/json",
                          PROJECT_ROOT / "experiments/datasets/processed/audio",
                          dataset_filter=ds):
        if s.sample_id in set(sample_ids):
            samples.append({"sample_id": s.sample_id, "language": s.language,
                            "duration_group": s.duration_group})
check("复现加载序命中 50 样本", len(samples) == 50, f"actual={len(samples)}")
zh = [s["sample_id"] for s in samples if s["language"] == "zh"][:5]
en = [s["sample_id"] for s in samples if s["language"] == "en"][:5]
subset_derived = zh + en
check("推导子集与实际执行子集一致", sorted(subset_derived) == subset_actual)
tasks = build_schedule(samples, subset_derived)
check("复算任务数=140", len(tasks) == 140, f"actual={len(tasks)}")
check("复算 schedule_hash 一致",
      sched_hash_fn(tasks) == hdr["schedule_hash"],
      f"recomputed={sched_hash_fn(tasks)[:12]} recorded={hdr['schedule_hash'][:12]}")
check("执行序与 schedule 全序一致",
      [(t["sample_id"], t["mode"], t["repeat_idx"]) for t in tasks] == keys)
check("subset_sha256 复算一致",
      hashlib.sha256(canonical_json(sorted(subset_derived)).encode()).hexdigest()
      == b.get("subset_sha256", ""))
check("sample_list_sha256 复算一致（LF 归一化）",
      sha256_lf(SAMPLE_LIST) == b.get("sample_list_sha256", ""))

# 5) A/B 配对与全局平衡
file_seq = defaultdict(list)
for i, r in enumerate(recs):
    file_seq[(r["sample_id"], r["repeat_idx"])].append((i, r["mode"]))
pair_orders = {}
for k, seq in file_seq.items():
    if len(seq) != 2 or {m for _, m in seq} != {"streaming", "non-streaming"}:
        FAIL.append(f"配对异常 {k}")
        continue
    pair_orders[k] = "AB" if seq[0][1] == "non-streaming" else "BA"
check("r7_main 70 对配对齐全", len(pair_orders) == 70)
cnt = Counter(pair_orders.values())
check("r7_main 全局 AB/BA=35/35", cnt["AB"] == 35 and cnt["BA"] == 35, str(dict(cnt)))
r0 = Counter(o for (s, ri), o in pair_orders.items() if ri == 0)
check("repeat0 AB/BA=25/25", r0["AB"] == 25 and r0["BA"] == 25, str(dict(r0)))
r12 = Counter(o for (s, ri), o in pair_orders.items() if ri > 0)
check("补轮 AB/BA=10/10", r12["AB"] == 10 and r12["BA"] == 10, str(dict(r12)))
sched_order = {t["sample_id"] + "|" + t["mode"] + "|" + str(t["repeat_idx"]): t["order"]
               for t in tasks}
mismatch = [k for k, o in pair_orders.items()
            if sched_order[f"{k[0]}|{'non-streaming' if o=='AB' else 'streaming'}|{k[1]}"] != o]
check("记录执行方向与 schedule order 一致", not mismatch, str(mismatch[:3]))

# 6) TTFA（正确字段：first_playable_pcm − physical_speech_end，见脚本 line 642）
def ttfa_ms_of(r):
    ev = r["events"]
    return (ev["first_playable_pcm_ns"] - ev["physical_speech_end_ns"]) / 1e6

st = [ttfa_ms_of(r) for r in recs if r["mode"] == "streaming"]
nst = [ttfa_ms_of(r) for r in recs if r["mode"] == "non-streaming"]
check("TTFA 全部非负", all(v >= 0 for v in st + nst))
st_s = sorted(st)
print(f"  [INFO] streaming TTFA ms: min={st_s[0]:.0f} p50={st_s[len(st_s)//2]:.0f} "
      f"max={st_s[-1]:.0f} n={len(st_s)}")
nst_s = sorted(nst)
print(f"  [INFO] non-streaming TTFA ms: p50={nst_s[len(nst_s)//2]:.0f} "
      f"max={nst_s[-1]:.0f} n={len(nst_s)}")

# 7) 分层覆盖
lang = Counter(s["language"] for s in samples)
check("语言覆盖 zh=25/en=25", lang["zh"] == 25 and lang["en"] == 25, str(dict(lang)))
groups = Counter(s["duration_group"] for s in samples)
check("时长组覆盖=very_long×50", groups == {"very_long": 50}, str(dict(groups)))

# 8) audio_map_sha256 复算
wav_map = {}
for r in recs:
    sid = r["sample_id"]
    if sid in wav_map and wav_map[sid] != r["wav_sha256"]:
        FAIL.append(f"{sid} wav_sha256 不一致")
    wav_map[sid] = r["wav_sha256"]
check("audio_map_sha256 复算一致",
      hashlib.sha256(canonical_json(wav_map).encode()).hexdigest()
      == b.get("audio_map_sha256", ""))

# 9) config_hash 复算（从 RUNINFO config JSON 重放）
runinfo = (MAIN / "RUNINFO_r7_main.md").read_text(encoding="utf-8")
cfg = json.loads(next(l for l in runinfo.splitlines() if l.startswith("- config: "))
                 [len("- config: "):])
check("config_hash 复算一致",
      hashlib.sha256(canonical_json(cfg).encode()).hexdigest() == hdr["config_hash"])

# 10) 信息性分布
epm = Counter(r.get("endpoint_mode") for r in recs)
print(f"  [INFO] endpoint_mode: {dict(epm)}（streaming=explicit_flush, A=full_input）")
gstop = Counter(r.get("generation_stop_reason") for r in recs)
print(f"  [INFO] generation_stop_reason: {dict(gstop)}")
check("endpoint_mode 与 mode 对应",
      all((r["mode"] == "streaming") == (r["endpoint_mode"] == "explicit_flush")
          for r in recs))
check("response_token_count≤128（max_tokens 上限）",
      all(r.get("response_token_count", 0) <= 128 for r in recs))

# ============ r7_tts_control ============
chdr, crecs = load_ckpt(CTRL / "checkpoint_r7_tts_control.jsonl")
check("tts_control 记录数=32", len(crecs) == 32, f"actual={len(crecs)}")
check("tts_control 全 success", all(r["terminal_state"] == "success" for r in crecs))
check("tts_control error 全空", all(not r.get("error") for r in crecs))
check("tts_control fatal 全 False", all(r.get("fatal") is False for r in crecs))
check("control_from hash=主 checkpoint sha256（LF 归一化）",
      sha256_lf(MAIN / "checkpoint_r7_main.jsonl").startswith("4edcd6ec28189d00")
      or hashlib.sha256((MAIN / "checkpoint_r7_main.jsonl").read_bytes()).hexdigest()
      .startswith("4edcd6ec28189d00"),
      sha256_lf(MAIN / "checkpoint_r7_main.jsonl")[:16])
tsrc = Counter(r["text_source"] for r in crecs)
print(f"  [INFO] text_source: {dict(tsrc)}")
per_ctrl = Counter(r["sample_id"] for r in crecs if r["text_source"] != "calibration")
check("10 样本×3 调用 + 校准 2",
      set(per_ctrl.values()) == {3} and len(per_ctrl) == 10
      and tsrc.get("calibration") == 2)
# 设计规则（run_tts_control line 2483-2485）：主 checkpoint repeat-0 完整配对中
# sorted() 后 zh 前 5 + en 前 5（与主实验子集的 load 序选样不同，属设计）
pairs0 = defaultdict(set)
for r in recs:
    if r["repeat_idx"] == 0:
        pairs0[r["sample_id"]].add(r["mode"])
complete0 = [sid for sid, m in pairs0.items() if m == {"streaming", "non-streaming"}]
exp_zh = sorted(sid for sid in complete0 if lang_of[sid] == "zh")[:5]
exp_en = sorted(sid for sid in complete0 if lang_of[sid] == "en")[:5]
check("控制样本集=sorted 前 5zh+5en（脚本设计规则）",
      sorted(per_ctrl) == sorted(exp_zh + exp_en))
cb = chdr.get("binding", chdr)
print(f"  [INFO] control binding: git_commit={str(cb.get('git_commit'))[:9]} "
      f"platform={str(cb.get('tts_config', {}).get('platform_conditions_sha256'))[:9]}…")
check("control git_commit=c9437c3", str(cb.get("git_commit", "")).startswith("c9437c3a"),
      str(cb.get("git_commit"))[:9])
check("control platform_conditions 绑定=a4c40057",
      str(cb.get("tts_config", {}).get("platform_conditions_sha256", "")).startswith("a4c400576b"))
cbad = []
for r in crecs:
    t = r.get("tts", {})
    chain = [t.get("tts_request_start_ns"), t.get("tts_response_headers_ns"),
             t.get("first_pcm_byte_ns")]
    if any(x is None for x in chain) or not all(chain[i] <= chain[i + 1]
                                                for i in range(len(chain) - 1)):
        cbad.append(r["sample_id"])
check("tts_control 请求链单调（req≤headers≤first_pcm）", not cbad, str(cbad[:3]))
lat = [(r["tts"]["first_pcm_byte_ns"] - r["tts"]["tts_request_start_ns"]) / 1e6
       for r in crecs]
mean_lat = sum(lat) / len(lat)
print(f"  [INFO] tts req→first_pcm ms: mean={mean_lat:.0f} n={len(lat)}")
check("tts 延迟均值复算=7076（RUNINFO 一致）", round(mean_lat) == 7076,
      f"recomputed={mean_lat:.1f}")

# 文本匹配：非校准条目的 tts_text_sha256 必须能在主 checkpoint 同样本文本/首句中找到
import re  # noqa: E402
def first_sentence(t: str) -> str:
    m = re.split(r"(?<=[。！？!?；;])", t.strip())
    return m[0] if m and m[0].strip() else t.strip()[:50]

main_full, main_first = defaultdict(set), defaultdict(set)
for r in recs:
    main_full[r["sample_id"]].add(r["tts_text_sha256"])
    main_first[r["sample_id"]].add(hashlib.sha256(
        first_sentence(r["tts_text"]).encode("utf-8")).hexdigest())
unmatched = []
for r in crecs:
    if r["text_source"] == "calibration":
        continue
    sid, h = r["sample_id"], r["tts_text_sha256"]
    if h not in main_full[sid] and h not in main_first[sid]:
        unmatched.append((sid, r["text_source"]))
check("匹配文本与主实验文本/首句哈希对应", not unmatched, str(unmatched[:5]))

print("\n========== 汇总 ==========")
print(f"PASS={len(PASS)} FAIL={len(FAIL)}")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
