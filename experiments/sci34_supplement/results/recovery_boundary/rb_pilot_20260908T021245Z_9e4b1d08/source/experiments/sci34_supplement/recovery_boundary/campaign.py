"""Run, independently validate/analyze, and seal a finite supplement campaign.

Usage: uv run --no-sync python -m experiments.sci34_supplement.recovery_boundary.campaign --help
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tarfile
import time
import uuid

from .protocol import ROOT, BASE, PROTOCOL, C2, cases, digest, expected_grid, file_hash, order


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode().strip()


def source_identity():
    # Include the entire tracked tree as a Git identity, plus byte identities for
    # executable dependencies and protocol. Ignore result artifacts only for byte list.
    paths = git("ls-files", "src", "experiments/sci34_supplement", "uv.lock", "pyproject.toml").splitlines()
    paths = [p for p in paths if "/results/" not in p]
    paths += [str(p.relative_to(ROOT)).replace("\\", "/") for p in BASE.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "files": {p: file_hash(ROOT/p) for p in sorted(set(paths))}}


def offline():
    # Must happen before importing config/transformers. Never print credentials.
    os.environ["HF_TOKEN"] = ""
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["LOG_FILE"] = ""
    os.environ["LOG_LEVEL"] = "WARNING"


def require_source(manifest):
    now = source_identity()
    assert now["commit"] == manifest["source"]["commit"], "Source commit changed"
    assert now["files"] == manifest["source"]["files"], "Source bytes changed"
    if not manifest["pilot"]:
        assert not now["dirty"], "Formal requires exact clean source tree (including untracked files)"


def session(run_dir, index, model):
    offline()
    import numpy as np
    from .runtime import Runtime
    manifest = read(run_dir / "manifest.json")
    require_source(manifest)
    target = run_dir / f"session_{index}"
    target.mkdir(exist_ok=False)
    start = time.perf_counter()
    write(target / "started.json", {"session": index, "process_id": os.getpid(), "process_start_id": uuid.uuid4().hex,
          "manifest_hash": file_hash(run_dir / "manifest.json")})
    runtime = Runtime(model)
    torch = runtime.torch
    torch.manual_seed(PROTOCOL["seed"] + index)
    random.seed(PROTOCOL["seed"] + index)
    np.random.seed(PROTOCOL["seed"] + index)
    write(target / "environment.json", {"runtime": runtime.backend.runtime_metadata,
          "model": runtime.backend.identity, "torch": torch.__version__,
          "transformers": __import__("transformers").__version__, "cuda": torch.version.cuda,
          "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
          "python": sys.version, "platform": platform.platform(), "strict_offline": True,
          "gpu_free_total_bytes": list(torch.cuda.mem_get_info(0))})
    repetitions = 1 if manifest["pilot"] else PROTOCOL["pairs_per_case"]
    with (target / "records.jsonl").open("x", encoding="utf-8", newline="\n") as out:
        for ci, case in enumerate(cases(manifest["pilot"])):
            # Independent reconstruction outside every arm timing; no shared mutable KV.
            for repeat in range(-PROTOCOL["warmup_pairs_per_case"], repetitions):
                for ordinal, arm in enumerate(order(index, ci, repeat)):
                    for row, logits in runtime.run_arm(case, arm):
                        if repeat < 0:
                            continue
                        row.update({"session": index, "case_id": case.id, "repeat": repeat,
                                    "arm_order": ordinal, "manifest_hash": file_hash(run_dir / "manifest.json")})
                        filename = f"{case.id}_{repeat}_{row['event']}_{arm}.npy"
                        with (target / filename).open("xb") as sidecar:
                            np.save(sidecar, logits, allow_pickle=False)
                        row["logits_file"] = f"session_{index}/{filename}"
                        row["logits_sha256"] = file_hash(target / filename)
                        out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        out.flush()
                        os.fsync(out.fileno())
    require_source(manifest)
    write(target / "complete.json", {"elapsed_s": time.perf_counter()-start})


def records(run_dir):
    result = []
    for path in sorted(run_dir.glob("session_*/records.jsonl")):
        result.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return result


def validate(run_dir):
    import numpy as np
    m = read(run_dir / "manifest.json")
    assert m["protocol"] == PROTOCOL and m["protocol_hash"] == digest(PROTOCOL)
    assert m["cases"] == [asdict(c) for c in cases(m["pilot"])]
    assert read(run_dir / "legacy_before.json") == read(run_dir / "legacy_after.json")
    for relative, sha in m["source"]["files"].items():
        assert file_hash(run_dir / "source" / relative) == sha
    assert m["pilot"] or not m["source"]["dirty"]
    from experiments.sci34_supplement.c2_equivalence.protocol import EXPECTED_MODEL_ARTIFACT_HASH
    from experiments.sci34_supplement.common import config_hash
    assert config_hash({k: m["model_identity"][k] for k in ("schema_version", "file_count", "total_bytes", "files")}) == EXPECTED_MODEL_ARTIFACT_HASH
    for env_file in run_dir.glob("session_*/environment.json"):
        env = read(env_file)
        assert env["runtime"]["attention_backend"] == "sdpa"
        assert env["runtime"]["resolved_dtype"] == "torch.bfloat16"
        assert env["runtime"]["accepted_model_artifact_hash"] == EXPECTED_MODEL_ARTIFACT_HASH
        assert env["strict_offline"] is True
        assert env["torch"] == "2.8.0+cu128" and env["transformers"] == "4.57.1"
    rows = records(run_dir)
    grid = [(r["session"], r["case_id"], r["repeat"], r["event"], r["arm"]) for r in rows]
    assert len(grid) == len(set(grid)) and set(grid) == set(expected_grid(m["pilot"])), "Missing/duplicate/unexpected record"
    started = [read(p) for p in sorted(run_dir.glob("session_*/started.json"))]
    assert len(started) == (1 if m["pilot"] else PROTOCOL["sessions"])
    assert len({s["process_start_id"] for s in started}) == len(started)
    for s in started:
        assert (run_dir / f"session_{s['session']}/environment.json").is_file()
        assert (run_dir / f"session_{s['session']}/complete.json").is_file()
        assert s["manifest_hash"] == file_hash(run_dir / "manifest.json")
    grouped = {}
    for row in rows:
        assert row["manifest_hash"] == file_hash(run_dir / "manifest.json")
        case_index = [c.id for c in cases(m["pilot"])].index(row["case_id"])
        assert order(row["session"], case_index, row["repeat"])[row["arm_order"]] == row["arm"]
        for field in ("recovery_ns", "suffix_ready_ns", "ready_total_ns", "decode_first_ns"):
            assert math.isfinite(row[field]) and row[field] >= 0
        assert row["ready_total_ns"] == row["recovery_ns"] + row["suffix_ready_ns"]
        assert row["first_deliverable_ns"] == (row["ready_total_ns"] + row["decode_first_ns"] if row["delivered"] else None)
        st = row["ready_state"]
        n = len(row["ready_ids_expected"])
        assert st["token_ids"] == row["ready_ids_expected"]
        assert st["seq_length"] == st["kv_length"] == n and st["mask"] == [[1]*n]
        assert st["role_phase"] == "assistant_open" and st["end_reason"] == "none"
        assert st["assistant_token_ids"] == [] and st["assistant_content_start"] == n
        # The canonical structural transitions are snapshotted BEFORE measurement.
        parts = m["chat_parts"]
        case = next(c for c in cases(m["pilot"]) if c.id == row["case_id"])
        token_plan = m["case_tokens"][case.id]
        if row["event"] == 0:
            start = case.context_tokens
            assert row["pre_ids"][start:] == token_plan["assistant"]
            if case.scenario == "speculation_full_invalidation":
                expected_keep = start - len(parts["user_to_assistant"])
            elif case.scenario == "reply_tail_noop":
                expected_keep = len(row["pre_ids"])
            else:
                expected_keep = start + (token_plan["fragment_ends"][case.retain_fragment_count-1] if case.retain_fragment_count else 0)
        else:
            start = len(row["pre_ids"]) - len(token_plan["second"])
            assert row["pre_ids"][start:] == token_plan["second"]
            expected_keep = start + max(1, int(len(token_plan["second"]) * case.second_crop_fraction))
        assert row["keep"] == expected_keep, "Independent keep derivation differs"
        ids = row["pre_ids"][:expected_keep]
        invalid = row["event"] == 0 and "invalidate" in row["case_id"]
        if not invalid:
            close = parts["assistant_eot"] + parts["assistant_to_user"]
            assert close.count(parts["eot_token_id"]) == 1
            ids += close
        assert ids == row["recovered_ids"]
        pre = row["pre_state"]
        assert pre["token_ids"] == row["pre_ids"]
        assert pre["seq_length"] == pre["kv_length"] == len(row["pre_ids"])
        assert pre["mask"] == [[1]*len(row["pre_ids"])]
        boundaries = list(pre["role_boundaries"])
        if not invalid:
            boundaries.append({"role_header_start": pre["assistant_role_start"],
                "content_start": pre["assistant_content_start"], "content_end": expected_keep,
                "role_end": expected_keep + len(parts["assistant_eot"]),
                "next_user_content_start": len(ids),
                "end_reason": pre["end_reason"] if expected_keep == len(row["pre_ids"]) else "cropped"})
        assert st["role_boundaries"] == boundaries
        assert st["assistant_role_start"] == n-len(parts["user_to_assistant"])
        assert st["assistant_content_end"] is None and st["assistant_role_end"] is None
        suffix_ids = m["suffix_tokens"][row["suffix"]]
        assert ids + suffix_ids + parts["user_to_assistant"] == st["token_ids"]
        final = row["final_state"]
        assert final["token_ids"] == st["token_ids"] + row["continuation_ids"]
        assert final["seq_length"] == final["kv_length"] == len(final["token_ids"])
        assert final["mask"] == [[1]*final["seq_length"]]
        assert len(row["selected_ids"]) <= PROTOCOL["continuation_cap"]
        assert parts["eot_token_id"] not in row["continuation_ids"]
        assert row["delivered"] == bool(row["continuation_ids"])
        selected = row["selected_ids"]
        ended_eot = bool(selected) and selected[-1] == parts["eot_token_id"]
        assert selected == row["continuation_ids"] + ([parts["eot_token_id"]] if ended_eot else [])
        assert final["role_phase"] == ("assistant_eot_pending" if ended_eot else "assistant_open")
        assert final["end_reason"] == ("eos" if ended_eot else "max_tokens")
        assert final["assistant_token_ids"] == row["continuation_ids"]
        assert final["role_boundaries"] == st["role_boundaries"]
        path = run_dir / row["logits_file"]
        assert path.resolve().is_relative_to(run_dir.resolve())
        assert file_hash(path) == row["logits_sha256"]
        array = np.load(path, allow_pickle=False)
        assert array.dtype == np.float32 and np.isfinite(array).all()
        key = (row["session"], row["case_id"], row["repeat"], row["event"])
        grouped.setdefault(key, {})[row["arm"]] = row
    diagnostics = []
    for key, pair in grouped.items():
        a, b = pair["crop"], pair["rebuild"]
        for field in ("pre_ids", "pre_state", "keep", "recovered_ids", "suffix", "ready_state"):
            assert a[field] == b[field], (key, field)
        x = np.load(run_dir / a["logits_file"], allow_pickle=False)
        y = np.load(run_dir / b["logits_file"], allow_pickle=False)
        assert x.shape == y.shape
        if key[3] == 1:
            previous = grouped[(key[0], key[1], key[2], 0)]
            second = m["case_tokens"][key[1]]["second"]
            for arm in ("crop", "rebuild"):
                assert pair[arm]["pre_ids"] == previous[arm]["ready_state"]["token_ids"] + second
        diagnostics.append({"key": list(key), "max_abs_logit_difference": float(np.max(np.abs(x-y))),
             "mean_abs_logit_difference": float(np.mean(np.abs(x-y))),
             "top1_same": bool(x.argmax() == y.argmax()),
             "continuation_same": a["continuation_ids"] == b["continuation_ids"]})
    boundary = read(run_dir / "boundary.json")
    from .boundary import run_boundary
    assert boundary == run_boundary(), "Independent B replay differs"
    return {"ok": True, "formal_evidence_eligible": not m["pilot"], "records": len(rows),
            "pairs": len(grouped), "diagnostics": diagnostics,
            "no_equivalence_claim": True, "boundary_ok": True}


def analyze(run_dir):
    check = validate(run_dir)
    rows = records(run_dir)
    cells = {}
    for row in rows:
        cells.setdefault((row["case_id"], row["event"]), {}).setdefault(row["session"], {}).setdefault(row["repeat"], {})[row["arm"]] = row
    rng = random.Random(PROTOCOL["seed"])
    output = []
    for (case_id, event), sessions in sorted(cells.items()):
        entry = {"case_id": case_id, "event": event, "metrics": {}}
        for metric in ("recovery_ns", "ready_total_ns", "first_deliverable_ns"):
            means, missing = [], 0
            for pairs in sessions.values():
                differences = []
                for pair in pairs.values():
                    a, b = pair["crop"][metric], pair["rebuild"][metric]
                    if a is None or b is None:
                        missing += 1
                    else:
                        differences.append((b-a)/1e6)
                if differences:
                    means.append(statistics.mean(differences))
            # No favorable complete-case latency estimate when delivery differs.
            result = {"missing_pairs": missing, "session_mean_differences_ms": means}
            if means and not missing:
                draws = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(PROTOCOL["bootstrap_draws"]))
                result.update({"mean_rebuild_minus_crop_ms": statistics.mean(means),
                               "session_cluster_percentile_ci95_ms": [draws[249], draws[9749]] if len(means) > 1 else None})
            entry["metrics"][metric] = result
        output.append(entry)
    return {"pilot": read(run_dir/"manifest.json")["pilot"], "cells": output,
            "uncertainty": "conditional on fixed fixtures; independent process-session clusters; repeats/events are not independent samples; 4-session CI is coarse",
            "delivery_counts": {a: sum(r["delivered"] for r in rows if r["arm"] == a) for a in ("crop", "rebuild")},
            "diagnostics": check["diagnostics"], "verdict": "descriptive; not superiority/equivalence acceptance"}


def inventory(run_dir):
    return {str(p.relative_to(run_dir)).replace("\\", "/"): file_hash(p)
            for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "seal.json"}


def verify(run_dir):
    sealed = read(run_dir / "seal.json")
    assert sealed["files"] == inventory(run_dir), "Seal inventory/hash mismatch"
    assert sealed["inventory_hash"] == digest(sealed["files"])
    return {"ok": True, "files": len(sealed["files"]), "seal_sha256": file_hash(run_dir / "seal.json")}


def seal(run_dir):
    check = validate(run_dir)
    assert read(run_dir / "validation.json") == check
    assert read(run_dir / "analysis.json") == analyze(run_dir)
    files = inventory(run_dir)
    write(run_dir / "seal.json", {"files": files, "inventory_hash": digest(files)})
    verify(run_dir)
    archive = run_dir.with_suffix(".tar.gz")
    with archive.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            tar.add(run_dir, arcname=run_dir.name)
    with Path(str(archive)+".sha256").open("x", encoding="utf-8") as f:
        f.write(file_hash(archive)+"  "+archive.name+"\n")
    return verify(run_dir)


def run(args):
    offline()
    source = source_identity()
    if not args.pilot:
        assert args.source_commit and args.source_commit == source["commit"], "Pass exact full source commit"
        assert not source["dirty"], "Commit/push changes explicitly before formal launch"
    out = Path(args.output_root).resolve()
    assert not out.is_relative_to(ROOT), "Keep run output outside source tree for clean-source gate"
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / ("rb_pilot_" if args.pilot else "rb_formal_")
    run_dir = run_dir.with_name(run_dir.name + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(exist_ok=False)
    print(f"RUN_DIR={run_dir}", flush=True)
    try:
        # Validate strong identity BEFORE loading weights; no fallback/download.
        from experiments.sci34_supplement.e1e2_confirmatory.strong_identity import strong_model_identity
        identity = strong_model_identity(args.model)
        from experiments.sci34_supplement.c2_equivalence.protocol import EXPECTED_MODEL_ARTIFACT_HASH
        payload = {k: identity[k] for k in ("schema_version", "file_count", "total_bytes", "files")}
        from experiments.sci34_supplement.common import config_hash
        assert config_hash(payload) == EXPECTED_MODEL_ARTIFACT_HASH
        from transformers import AutoTokenizer
        from experiments.sci34_supplement.c2_crop_integrity.canonical_chat import ChatTemplateParts
        from experiments.sci34_supplement.c2_equivalence.protocol import SYSTEM_PROMPT
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, token=False)
        parts = asdict(ChatTemplateParts.from_tokenizer(tokenizer, SYSTEM_PROMPT))
        parts = json.loads(json.dumps(parts))
        suffixes = {c.next_user for c in cases(args.pilot)} | {"State only the confirmed next step."}
        manifest = {"protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL), "pilot": args.pilot,
                    "source": source, "model_identity": identity, "chat_parts": parts,
                    "suffix_tokens": {s: tokenizer.encode(s, add_special_tokens=False) for s in sorted(suffixes)},
                    "case_tokens": {c.id: {"assistant": tokenizer.encode(c.assistant_text, add_special_tokens=False),
                        "fragment_ends": [len(tokenizer.encode("".join(c.fragments[:i+1]), add_special_tokens=False)) for i in range(len(c.fragments))],
                        "second": tokenizer.encode(c.second_assistant_text or "", add_special_tokens=False)} for c in cases(args.pilot)},
                    "cases": [asdict(c) for c in cases(args.pilot)],
                    "historical_manifest_sha256": file_hash(C2 / "campaign_manifest.json")}
        write(run_dir / "manifest.json", manifest)
        import shutil
        for relative in source["files"]:
            destination = run_dir / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        legacy_paths = git("ls-files", "experiments/results", "experiments/sci34_supplement/results").splitlines()
        legacy = {p: file_hash(ROOT/p) for p in legacy_paths}
        write(run_dir / "legacy_before.json", legacy)
        from .boundary import run_boundary
        write(run_dir / "boundary.json", run_boundary())
        count = 1 if args.pilot else PROTOCOL["sessions"]
        for index in range(count):
            with (run_dir / f"session_{index}.log").open("xb") as log:
                subprocess.run([sys.executable, "-m", __package__+".campaign", "session", "--run-dir", str(run_dir),
                                "--session-index", str(index), "--model", args.model],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        require_source(manifest)
        after = {p: file_hash(ROOT/p) for p in legacy_paths}
        write(run_dir / "legacy_after.json", after)
        assert legacy == after, "Legacy artifacts changed"
        write(run_dir / "validation.json", validate(run_dir))
        write(run_dir / "analysis.json", analyze(run_dir))
        print(json.dumps(seal(run_dir)), flush=True)
        print(f"ARCHIVE={run_dir}.tar.gz", flush=True)
    except BaseException as exc:
        if not (run_dir / "seal.json").exists():
            # Do not record exception strings that could contain credentials/URLs.
            write(run_dir / "FAILED.json", {"exception_type": type(exc).__name__, "retained": True,
                                           "instruction": "Stop; preserve all files. No retry/resume/quantization or protocol expansion."})
        raise


def main():
    if sys.flags.optimize:
        raise RuntimeError("Assertions are required; do not use python -O/PYTHONOPTIMIZE")
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "session", "validate", "analyze", "seal", "verify", "boundary"])
    parser.add_argument("--model", default="/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct")
    parser.add_argument("--output-root", default="/root/autodl-tmp/recovery_boundary_runs")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "session":
        session(args.run_dir, args.session_index, args.model)
    elif args.command == "boundary":
        from .boundary import run_boundary
        value = run_boundary()
        print(json.dumps({k: v for k, v in value.items() if k not in ("boundary_checks", "c2_closure_checks")}, indent=2))
    else:
        print(json.dumps(globals()[args.command](args.run_dir), indent=2))


if __name__ == "__main__":
    main()
