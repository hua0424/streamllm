"""Independent linear interval reference over saved E3 software trajectories."""
from bisect import bisect_left
from dataclasses import asdict
import json
from .protocol import E3, C2, file_hash


def reference(fragments, cursor):
    audio = [f for f in fragments if f.get("sample_end") is not None]
    if cursor <= 0 or not audio:
        hit = None
    else:
        hit = audio[min(bisect_left([f["sample_end"] for f in audio], cursor), len(audio) - 1)]
    end = 0 if hit is None else hit["token_end"]
    return {"crop_token_end": end, "interrupted_fragment_id": None if hit is None else hit["fragment_id"],
            "partial": hit is not None and cursor < hit["sample_end"],
            "heard_fragment_ids": [f["fragment_id"] for f in fragments if f["token_end"] <= end],
            "discarded_fragment_ids": [f["fragment_id"] for f in fragments if f["token_end"] > end]}


def replay(fragments, cursor):
    from src.dialogue.timeline import PlaybackTimeline
    timeline = PlaybackTimeline()
    for f in fragments:
        fid = timeline.add_fragment(f["text"], f["token_start"], f["token_end"])
        assert fid == f["fragment_id"]
        if f.get("sample_end") is not None:
            timeline.attach_chunk(fid, fid, f["sample_end"] - f["sample_start"])
    result = asdict(timeline.barge_in(cursor))
    assert result == reference(fragments, cursor), (result, reference(fragments, cursor))
    for rec in timeline._fragments:
        expected_status = ("PLAYING" if rec.fragment_id == result["interrupted_fragment_id"] and result["partial"]
                           else "PLAYED") if rec.fragment_id in result["heard_fragment_ids"] else "DISCARDED"
        assert rec.status.name == expected_status
    return result


def run_boundary():
    paths = [E3 / "trajectories.jsonl", E3 / "records.jsonl", C2 / "records.jsonl"]
    trajectories = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    stored = [json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()]
    checks = []
    by_id = {}
    for t in trajectories:
        assert t["trajectory_id"] not in by_id
        by_id[t["trajectory_id"]] = t
        token_end = sample_end = 0
        for f in t["fragments"]:
            assert f["token_start"] == token_end and f["token_end"] > token_end
            assert f["sample_start"] == sample_end and f["sample_end"] > sample_end
            token_end, sample_end = f["token_end"], f["sample_end"]
        assert token_end == len(t["token_ids"]) and sample_end == t["total_samples"]
        cursors = {0, sample_end, sample_end + 1}
        for f in t["fragments"]:
            cursors.update([f["sample_start"], f["sample_start"] + 1,
                            (f["sample_start"] + f["sample_end"]) // 2, f["sample_end"]])
        for cursor in sorted(cursors):
            checks.append({"trajectory_id": t["trajectory_id"], "cursor": cursor,
                           "expected": replay(t["fragments"], cursor)})
    for row in stored:
        t = by_id[row["trajectory_id"]]
        expected = reference(t["fragments"], row["played_samples"])
        assert row["heard_token_end"] == expected["crop_token_end"]
        assert row["partial"] == expected["partial"]
        assert row["interrupted_fragment_id"] == expected["interrupted_fragment_id"]
        if row["condition"] == "playback":
            assert row["history_token_end"] == expected["crop_token_end"]
    # Independently recompute saved C2 target and closure from recorded fragments,
    # not its keep value or its canonical_ledger. These are software fixtures.
    closure = []
    for row in map(json.loads, paths[2].read_text(encoding="utf-8").splitlines()):
        for event in row["crop_events"]:
            if event["event_id"] == "crop_2":
                ids = event["second_assistant_token_ids"]
                keep = event["assistant_content_start"] + max(1, int(len(ids) * event["second_crop_fraction"]))
            elif event["crop_target_semantics"] == "speculation_full_invalidation":
                keep = event["assistant_role_start"]
            elif event["crop_target_semantics"] == "reply_tail_noop":
                keep = len(event["pre_crop_token_ids"])
            else:
                keep = event["assistant_content_start"] + sum(map(len, event["fragment_token_ids"][:event["retain_fragment_count"]]))
            assert keep == event["keep_length"]
            expected = event["pre_crop_token_ids"][:keep]
            for chunk in event["expected_recovery_chunks"]:
                if chunk["operation"] == "reopen_user_role":
                    assert chunk["token_ids"].count(151645) == 1
                expected += chunk["token_ids"]
            assert event["final_token_ids"] == expected
            closure.append({"case_id": row["case_id"], "event_id": event["event_id"], "keep": keep})
    return {"ok": True, "scope": "E3 saved generated text + simulated sample geometry; C2 software fixtures, NOT real acoustic alignment",
            "inputs": {str(p.relative_to(E3.parents[2])): file_hash(p) for p in paths},
            "trajectories": len(trajectories), "stored_records_checked": len(stored),
            "boundary_checks": checks, "c2_closure_checks": closure,
            "missing_evidence": ["device loopback", "real TTS word alignment", "human acoustically-heard annotation"]}
