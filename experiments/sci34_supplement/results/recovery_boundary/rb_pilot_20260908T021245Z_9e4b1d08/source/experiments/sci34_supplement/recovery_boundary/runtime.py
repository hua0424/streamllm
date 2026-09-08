"""R: executable production crop versus one-shot full token-ID rebuild."""
import copy
from dataclasses import asdict
import time
from .protocol import PROTOCOL


def state(cache):
    return {"token_ids": list(cache.token_ids), "mask": cache.attention_mask.cpu().tolist(),
            "seq_length": cache.seq_length, "kv_length": int(cache.past_key_values.get_seq_length()),
            "role_phase": cache.role_phase.value, "end_reason": cache.generation_end_reason.value,
            "assistant_role_start": cache.assistant_role_start,
            "assistant_content_start": cache.assistant_content_start,
            "assistant_content_end": cache.assistant_content_end,
            "assistant_role_end": cache.assistant_role_end,
            "assistant_token_ids": list(cache.assistant_token_ids),
            "role_boundaries": [{**asdict(b), "end_reason": b.end_reason.value} for b in cache.role_boundaries]}


class Runtime:
    def __init__(self, model_path):
        import torch
        import transformers
        if torch.__version__ != "2.8.0+cu128" or transformers.__version__ != "4.57.1":
            raise RuntimeError("Frozen backend requires torch 2.8.0+cu128 and transformers 4.57.1; stop rather than silently change backend")
        from experiments.sci34_supplement.c2_crop_integrity.runtime import TransformersBackend
        self.backend = TransformersBackend(model_path, device="cuda:0")
        self.llm = self.backend.llm
        self.torch = self.backend.torch
        assert self.backend.runtime_metadata["attention_backend"] == "sdpa"
        assert self.torch.cuda.device_count() == 1, "Expose exactly one GPU"

    def sync(self):
        self.torch.cuda.synchronize(0)

    def pending_eot(self, cache):
        original = self.llm._decode_logits
        self.llm._decode_logits = lambda *args: self.torch.tensor([[self.llm._assistant_eot_id]], device="cuda:0")
        try:
            assert list(self.llm.generate_accumulating(cache, max_new_tokens=1)) == []
        finally:
            self.llm._decode_logits = original
        assert cache.role_phase == self.llm.RolePhase.ASSISTANT_EOT_PENDING

    def prepare(self, case):
        b = self.backend
        cache = b._initial_cache(b._context_user_text(case))
        b._append_fixture_tokenwise(cache, b._encode(case.assistant_text))
        # Controlled EOT is a software state fixture, not a natural termination claim.
        if case.scenario in ("crop_pending_eot", "reply_tail_noop"):
            self.pending_eot(cache)
        return cache

    def plan(self, cache, case, event):
        b, llm = self.backend, self.llm
        if event:
            keep = cache.assistant_content_start + max(1, int(len(cache.assistant_token_ids) * case.second_crop_fraction))
        elif case.scenario == "speculation_full_invalidation":
            keep = cache.assistant_role_start
        elif case.scenario == "reply_tail_noop":
            keep = cache.seq_length
        else:
            partition = b._fragment_partition(case, b._encode(case.assistant_text))
            keep = cache.assistant_content_start + sum(map(len, partition[:case.retain_fragment_count]))
        invalidation = not event and case.scenario == "speculation_full_invalidation"
        close = [] if invalidation else list(b.parts.assistant_eot) + list(b.parts.assistant_to_user)
        expected = list(cache.token_ids[:keep]) + close
        boundaries = copy.deepcopy(cache.role_boundaries)
        reason = cache.generation_end_reason if keep == cache.seq_length else llm.GenerationEndReason.CROPPED
        if not invalidation:
            boundaries.append(llm.RoleBoundary(cache.assistant_role_start, cache.assistant_content_start,
                              keep, keep + len(b.parts.assistant_eot), len(expected), reason))
        suffix = case.next_user if not event else "State only the confirmed next step."
        ready_ids = expected + b._encode(suffix) + list(b.parts.user_to_assistant)
        return {"keep": keep, "invalidation": invalidation, "recovered_ids": expected,
                "ready_ids": ready_ids, "boundaries": boundaries, "suffix": suffix,
                "pre_ids": list(cache.token_ids), "pre_state": state(cache)}

    def rebuild(self, plan):
        llm, torch = self.llm, self.torch
        ids = torch.tensor([plan["recovered_ids"]], dtype=torch.long, device="cuda:0")
        mask = torch.ones_like(ids)
        with torch.no_grad():
            output = llm.model(input_ids=ids, attention_mask=mask,
                               position_ids=torch.arange(ids.shape[1], device="cuda:0").unsqueeze(0),
                               use_cache=True, return_dict=True)
        cache = llm.AccumKVCache(past_key_values=llm._as_dynamic_cache(output.past_key_values),
                                attention_mask=mask, next_token_logits=output.logits[:, -1, :],
                                seq_length=ids.shape[1], token_ids=list(plan["recovered_ids"]),
                                role_phase=llm.RolePhase.USER_OPEN,
                                generation_end_reason=llm.GenerationEndReason.NONE,
                                role_boundaries=copy.deepcopy(plan["boundaries"]))
        llm._assert_accum_consistent(cache)
        return cache

    def execute(self, cache, plan, arm):
        llm = self.llm
        self.sync()
        self.torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter_ns()
        if arm == "crop":
            llm.crop_to_token(cache, plan["keep"])
            if not plan["invalidation"]:
                llm.reopen_user_role(cache)
            # Invalidation keeps CROPPED until the suffix arrives (production contract).
        else:
            cache = self.rebuild(plan)
            if plan["invalidation"]:
                cache.generation_end_reason = llm.GenerationEndReason.CROPPED
        self.sync()
        recovered = time.perf_counter_ns()
        llm.prefill_user_text(cache, plan["suffix"])
        llm.open_assistant_role(cache)
        self.sync()
        ready = time.perf_counter_ns()
        # Capture checks/logits only AFTER the endpoint timer, then start a separate
        # decode interval. End-to-end is interval sum, not an instrumented wall span.
        ready_state = state(cache)
        assert ready_state["token_ids"] == plan["ready_ids"]
        assert ready_state["mask"] == [[1] * len(plan["ready_ids"])]
        assert ready_state["seq_length"] == ready_state["kv_length"] == len(plan["ready_ids"])
        assert cache.role_phase == llm.RolePhase.ASSISTANT_OPEN
        assert cache.generation_end_reason == llm.GenerationEndReason.NONE
        assert cache.role_boundaries == plan["boundaries"]
        logits = cache.next_token_logits.detach().float().cpu().numpy().copy()
        assert bool(self.torch.isfinite(cache.next_token_logits).all())
        self.sync()
        decode_start = time.perf_counter_ns()
        selected = []
        selection_ns = []
        generator = llm.generate_accumulating(cache, max_new_tokens=PROTOCOL["continuation_cap"],
                     temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                     on_token_decoded=lambda text, idx, token: (selected.append(token), selection_ns.append(time.perf_counter_ns())))
        first = next(generator, None)
        self.sync()
        first_done = time.perf_counter_ns()
        # Consume bounded diagnostics outside the primary first-deliverable interval.
        for _ in generator:
            pass
        self.sync()
        llm._assert_accum_consistent(cache)
        row = {"arm": arm, "keep": plan["keep"], "pre_ids": plan["pre_ids"],
               "recovered_ids": plan["recovered_ids"], "pre_state": plan["pre_state"], "suffix": plan["suffix"],
               "ready_state": ready_state, "ready_ids_expected": plan["ready_ids"],
               "recovery_ns": recovered-t0, "suffix_ready_ns": ready-recovered,
               "ready_total_ns": ready-t0, "decode_first_ns": first_done-decode_start,
               "first_deliverable_ns": None if first is None else ready-t0+first_done-decode_start,
               "selection_ns": None if not selection_ns else selection_ns[0]-decode_start,
               "delivered": first is not None, "selected_ids": selected,
               "continuation_ids": list(cache.assistant_token_ids),
               "final_state": state(cache), "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}
        return cache, row, logits

    def run_arm(self, case, arm):
        cache = self.prepare(case)
        rows = []
        for event in range(2 if case.second_crop_fraction is not None else 1):
            plan = self.plan(cache, case, event)
            cache, row, logits = self.execute(cache, plan, arm)
            row["event"] = event
            rows.append((row, logits))
            if event == 0 and case.second_crop_fraction is not None:
                # Restore the same fixed second-turn trajectory, irrespective of
                # numerical greedy divergence, before the second measured interruption.
                llm = self.llm
                llm.crop_to_token(cache, cache.assistant_content_start)
                # Empty natural diagnostic continuation leaves a pending no-op;
                # explicitly reset only this out-of-window controlled fixture state.
                cache.role_phase = llm.RolePhase.ASSISTANT_OPEN
                cache.generation_end_reason = llm.GenerationEndReason.NONE
                cache.assistant_content_end = None
                # The forced selector ignores this seed; each selected content
                # token still executes the real production forward/commit.
                cache.next_token_logits = self.torch.zeros((1, self.llm.model.config.vocab_size), device=self.llm.device)
                self.backend._append_fixture_tokenwise(cache, self.backend._encode(case.second_assistant_text))
        del cache
        return rows
