"""Model-independent assertions, using real production role/crop with tiny random CPU model.

No pretrained model/tokenizer download. Random model output is software test evidence only.
"""
import copy
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
    production_cpu()
    print("PASS: 320-record grid/order; E3 100 trajectories/800 records/1118 cursors; C2 27 closures; unique writes/seal tamper; random CPU Qwen production crop/rebuild/role/consumer commit")


if __name__ == "__main__":
    main()
