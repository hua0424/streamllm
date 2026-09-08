"""Model-independent assertions, using real production role/crop with tiny random CPU model.

No pretrained model/tokenizer download. Random model output is software test evidence only.
"""
import copy
from dataclasses import asdict
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from .protocol import PROTOCOL, cases, expected_grid, order
from .campaign import write, verify, inventory, digest, offline
from .boundary import run_boundary, reference, replay


def rejected(call):
    try:
        call()
    except (AssertionError, ValueError, FileExistsError):
        return
    raise AssertionError("Negative control was not rejected")


def production_cpu():
    offline()
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from src.llm.stream_llm_inference import StreamLLMInference as LLM
    from .runtime import Runtime, state
    llm = LLM.__new__(LLM)
    llm.device = "cpu"
    llm.model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
          num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).eval()
    llm._assistant_eot_id = 2
    llm._assistant_close_ids = [2, 3]
    llm._assistant_to_user_ids = [2, 3, 4]
    llm._user_to_assistant_ids = [2, 3, 5]
    class Tokenizer:
        def decode(self, ids, **kw):
            return "x"
        def __call__(self, text, **kw):
            return SimpleNamespace(input_ids=[8, 9])
    llm.tokenizer = Tokenizer()
    llm._decode_logits = lambda *a: torch.tensor([[7]])
    runtime = Runtime.__new__(Runtime)
    runtime.llm, runtime.torch = llm, torch
    runtime.backend = SimpleNamespace(parts=SimpleNamespace(assistant_eot=[2, 3], assistant_to_user=[4], user_to_assistant=[2, 3, 5]),
                                     _encode=lambda text: [8, 9])
    for invalid in (False, True):
        ids = [10, 11, 2, 3, 5, 6, 7]
        tensor = torch.tensor([ids])
        with torch.no_grad():
            result = llm.model(input_ids=tensor, use_cache=True)
        cache = llm.AccumKVCache(result.past_key_values, torch.ones_like(tensor), result.logits[:, -1, :],
                      len(ids), token_ids=ids, assistant_role_start=2, assistant_content_start=5,
                      assistant_token_ids=[6, 7], role_phase=llm.RolePhase.ASSISTANT_OPEN)
        case = SimpleNamespace(scenario="speculation_full_invalidation" if invalid else "reply_tail_noop", next_user="u")
        plan = runtime.plan(cache, case, 0)
        llm.crop_to_token(cache, plan["keep"])
        if not invalid:
            llm.reopen_user_role(cache)
        # Same rebuild implementation, map tensor placement to CPU for local test.
        real_tensor, real_arange = torch.tensor, torch.arange
        def tensor_cpu(*a, **kw):
            kw["device"] = "cpu"
            return real_tensor(*a, **kw)
        def arange_cpu(*a, **kw):
            kw["device"] = "cpu"
            return real_arange(*a, **kw)
        with patch.object(torch, "tensor", tensor_cpu), patch.object(torch, "arange", arange_cpu):
            clean = runtime.rebuild(plan)
        if invalid:
            clean.generation_end_reason = llm.GenerationEndReason.CROPPED
        assert state(cache) == state(clean)
        for target in (cache, clean):
            llm._prefill_ids_p2(target, [8, 9])
            target.generation_end_reason = llm.GenerationEndReason.NONE
            llm.open_assistant_role(target)
            gen = llm.generate_accumulating(target, max_new_tokens=2)
            before = target.seq_length
            assert next(gen) == ("x", 0)
            assert target.seq_length == before + 1  # consumer sees committed token
            gen.close()
            llm._assert_accum_consistent(target)
        assert state(cache) == state(clean)
        # Exercise the actual timed execute path on CPU with only device/timer
        # synchronization shimmed, not the production cache operations.
        runtime.sync = lambda: None
        plan2 = runtime.plan(cache, case, 0)
        arms = []
        for arm in ("crop", "rebuild"):
            source = copy.deepcopy(cache)
            with patch.object(torch, "tensor", tensor_cpu), patch.object(torch, "arange", arange_cpu), \
                 patch.object(torch.cuda, "reset_peak_memory_stats", lambda: None), \
                 patch.object(torch.cuda, "max_memory_allocated", lambda: 0), \
                 patch.object(torch.cuda, "max_memory_reserved", lambda: 0):
                _, row, logits = runtime.execute(source, plan2, arm)
            assert row["delivered"] and len(row["continuation_ids"]) == 8
            assert row["first_deliverable_ns"] == row["ready_total_ns"] + row["decode_first_ns"]
            arms.append(row)
        assert arms[0]["ready_state"] == arms[1]["ready_state"]


def campaign_validation_cpu():
    """Full on-disk validator regression; synthetic CPU artifacts, not GPU evidence.

    Genuine frozen cases and historical model identity, but deliberately synthetic
    token/state/timing/environment fixtures. No validator gates or B replay mocked.
    """
    import numpy as np
    from .campaign import read, validate
    from .protocol import C2, file_hash
    historical = read(C2 / "campaign_manifest.json")["config"]
    boundary = run_boundary()
    for pilot in (True, False):
        selected_cases = cases(pilot)
        with tempfile.TemporaryDirectory(prefix="rb_validate_cpu_") as temp:
            path = Path(temp)
            source_file = Path(__file__).with_name("campaign.py")
            (path / "source").mkdir()
            (path / "source/campaign.py").write_bytes(source_file.read_bytes())
            parts = {"assistant_eot": [2, 3], "assistant_to_user": [4],
                     "user_to_assistant": [2, 3, 5], "eot_token_id": 2}
            tokens = {c.id: {"assistant": [7] * (2 * len(c.fragments)),
                      "fragment_ends": [2 * (i+1) for i in range(len(c.fragments))],
                      "second": [8, 9, 10, 11]} for c in selected_cases}
            suffixes = {c.next_user for c in selected_cases} | {"State only the confirmed next step."}
            manifest = {"protocol": PROTOCOL, "protocol_hash": digest(PROTOCOL),
                        "pilot": pilot, "cases": [asdict(c) for c in selected_cases],
                        "source": {"dirty": False, "files": {"campaign.py": file_hash(source_file)}},
                        "model_identity": historical["model_identity"], "chat_parts": parts,
                        "case_tokens": tokens, "suffix_tokens": {s: [12, 13] for s in suffixes}}
            write(path / "manifest.json", manifest)
            # Prove we cross the same tuple -> JSON array boundary as run().
            loaded = read(path / "manifest.json")
            assert isinstance(manifest["cases"][0]["fragments"], tuple)
            assert isinstance(loaded["cases"][0]["fragments"], list)
            assert loaded["cases"] == json.loads(json.dumps(manifest["cases"]))
            manifest_hash = file_hash(path / "manifest.json")
            write(path / "legacy_before.json", {"synthetic": "unchanged"})
            write(path / "legacy_after.json", {"synthetic": "unchanged"})
            write(path / "boundary.json", boundary)
            for session in range(1 if pilot else PROTOCOL["sessions"]):
                target = path / f"session_{session}"
                target.mkdir()
                write(target / "started.json", {"session": session,
                      "process_start_id": f"synthetic-cpu-{session}", "manifest_hash": manifest_hash})
                write(target / "environment.json", {"runtime": historical["runtime_metadata"],
                      "strict_offline": True, "torch": "2.8.0+cu128", "transformers": "4.57.1"})
                write(target / "complete.json", {"synthetic_cpu": True})
                rows = []
                for ci, case in enumerate(selected_cases):
                    for repeat in range(1 if pilot else PROTOCOL["pairs_per_case"]):
                        pre_ids = [6] * (case.context_tokens - 3) + parts["user_to_assistant"] + tokens[case.id]["assistant"]
                        boundaries = []
                        start = case.context_tokens
                        for event in range(2 if case.second_crop_fraction is not None else 1):
                            invalid = event == 0 and case.scenario == "speculation_full_invalidation"
                            keep = (start + max(1, int(4 * case.second_crop_fraction)) if event else
                                    start - 3 if invalid else len(pre_ids) if case.scenario == "reply_tail_noop" else
                                    start + 2 * case.retain_fragment_count)
                            def state(ids, role, reason, content_start, history, content):
                                return {"token_ids": list(ids), "seq_length": len(ids), "kv_length": len(ids),
                                        "mask": [[1]*len(ids)], "role_phase": role, "end_reason": reason,
                                        "assistant_role_start": content_start-3, "assistant_content_start": content_start,
                                        "assistant_content_end": None, "assistant_role_end": None,
                                        "assistant_token_ids": content, "role_boundaries": copy.deepcopy(history)}
                            pre = state(pre_ids, "assistant_open", "none", start, boundaries, pre_ids[start:])
                            recovered = pre_ids[:keep] + ([] if invalid else [2, 3, 4])
                            if not invalid:
                                boundaries.append({"role_header_start": start-3, "content_start": start,
                                    "content_end": keep, "role_end": keep+2, "next_user_content_start": len(recovered),
                                    "end_reason": "none" if keep == len(pre_ids) else "cropped"})
                            suffix = case.next_user if event == 0 else "State only the confirmed next step."
                            ready_ids = recovered + [12, 13] + parts["user_to_assistant"]
                            ready = state(ready_ids, "assistant_open", "none", len(ready_ids), boundaries, [])
                            continuation = [14] * PROTOCOL["continuation_cap"]
                            final = state(ready_ids + continuation, "assistant_open", "max_tokens", len(ready_ids), boundaries, continuation)
                            for ordinal, arm in enumerate(order(session, ci, repeat)):
                                sidecar = target / f"{case.id}_{repeat}_{event}_{arm}.npy"
                                np.save(sidecar, np.arange(16, dtype=np.float32), allow_pickle=False)
                                rows.append({"session": session, "case_id": case.id, "repeat": repeat,
                                    "event": event, "arm": arm, "arm_order": ordinal, "manifest_hash": manifest_hash,
                                    "recovery_ns": 1, "suffix_ready_ns": 2, "ready_total_ns": 3,
                                    "decode_first_ns": 4, "first_deliverable_ns": 7, "delivered": True,
                                    "pre_ids": pre_ids, "pre_state": pre, "keep": keep, "recovered_ids": recovered,
                                    "suffix": suffix, "ready_ids_expected": ready_ids, "ready_state": ready,
                                    "final_state": final, "selected_ids": continuation, "continuation_ids": continuation,
                                    "logits_file": sidecar.relative_to(path).as_posix(), "logits_sha256": file_hash(sidecar)})
                            start = len(ready_ids)
                            pre_ids = ready_ids + tokens[case.id]["second"]
                (target / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            result = validate(path)
            assert result["ok"] and result["boundary_ok"]
            assert result["records"] == (8 if pilot else 320)
            assert result["pairs"] == (4 if pilot else 160)
            assert result["formal_evidence_eligible"] is (not pilot)
            original = (path / "manifest.json").read_bytes()
            # Require rejection at the cases gate, not a later stale manifest hash.
            for tamper in ("fragment", "order", "missing"):
                altered = copy.deepcopy(loaded)
                if tamper == "fragment":
                    altered["cases"][0]["fragments"][0] += " tampered"
                elif tamper == "order":
                    altered["cases"].reverse()
                else:
                    altered["cases"].pop()
                (path / "manifest.json").write_text(json.dumps(altered), encoding="utf-8")
                try:
                    validate(path)
                except AssertionError as exc:
                    import traceback
                    assert 'm["cases"]' in traceback.extract_tb(exc.__traceback__)[-1].line
                else:
                    raise AssertionError(f"Tampered cases accepted: {tamper}")
            (path / "manifest.json").write_bytes(original)
            assert validate(path) == result
            print(f"PASS: full CPU synthetic validate pilot={pilot}, records={result['records']}, pairs={result['pairs']}; case content/order/missing rejected at cases gate")


def main():
    logging.disable(logging.CRITICAL)
    assert len(cases()) == 9 and len(expected_grid()) == 320
    for ci in range(9):
        assert sum(order(s, ci, r)[0] == "crop" for s in range(4) for r in range(4)) == 8
    b = run_boundary()
    assert b["trajectories"] == 100 and b["stored_records_checked"] == 800
    assert len(b["c2_closure_checks"]) == 27
    f = [{"fragment_id": 0, "text": "a", "token_start": 0, "token_end": 2, "sample_start": 0, "sample_end": 10},
         {"fragment_id": 1, "text": "b", "token_start": 2, "token_end": 5, "sample_start": 10, "sample_end": 20}]
    assert reference(f, 10)["crop_token_end"] == 2
    assert reference(f, 11)["crop_token_end"] == 5
    for p in (0, 1, 10, 11, 20, 21):
        replay(f, p)
    with tempfile.TemporaryDirectory(prefix="rb_smoke_") as temp:
        path = Path(temp)
        write(path/"raw.json", {"fixture": True})
        rejected(lambda: write(path/"raw.json", {}))
        files = inventory(path)
        write(path/"seal.json", {"files": files, "inventory_hash": digest(files)})
        assert verify(path)["ok"]
        (path/"raw.json").write_text("tampered")
        rejected(lambda: verify(path))
    campaign_validation_cpu()
    production_cpu()
    print("PASS: 320-record grid/order; E3 100 trajectories/800 records/1118 cursors; C2 27 closures; unique writes/seal tamper; random CPU Qwen production crop/rebuild/role/consumer commit")


if __name__ == "__main__":
    main()
