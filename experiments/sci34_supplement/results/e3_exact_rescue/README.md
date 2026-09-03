# Accepted E3 exact-input rescue bundle

This directory preserves the exact processed input referenced by accepted fixed-trajectory E3 run `sci34_f11ccba_20260901_e3`. It was recovered from the experiment machine during the C2 v1 archival round and added to Git in result commit `1a47ac1bb8a377a9cda8f3679e86ece63fc66488`.

## Contents and relation to E3

- `p2_turns.json`: the exact 100-dialogue input read by the accepted E3 run.
- `e3_manifest.json`: a copy of the accepted E3 manifest. Its `input.sha256` points to the same `p2_turns.json` byte stream.
- `e3_model_identity.json`: the model identity preserved from E3 plus a later full-file strong rehash of the same local Qwen2-7B-Instruct snapshot.
- `p2_turns.sha256`, `e3_manifest.sha256`, `provenance.sha256`, and `provenance_paths.txt`: experiment-machine provenance records. The absolute paths are historical and need not exist on another host.

The canonical experiment-machine/Git-blob SHA-256 of `p2_turns.json` is:

```text
a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c
```

The accepted E3 manifest Git-blob SHA-256 is:

```text
7690f1003109a37c6f216b674ff6df2b71a4bfac98f6992c1eb23b37f98967a4
```

## Source and deterministic builder

The rescued input was derived from MultiWOZ 2.1 with:

- source: `experiments/datasets/raw_data/MultiWOZ/MultiWOZ_2.1/data.json` on the experiment machine;
- source SHA-256: `8be37ba1cb5b5a35943f32d4dbe03c5017dd88e15716f74987f60e0ece37851c`;
- builder: `experiments/scripts/prepare_multiwoz_data.py`;
- builder SHA-256 at the experiment run: `ce001d07e4a65c6c44d9d1544f356ae447c6afb37b037cd9a670bd9aa710699d`;
- seed: `42`;
- `--max-dialogues 100`;
- default `--min-user-turns 3`.

Equivalent builder invocation, after obtaining MultiWOZ 2.1 under its own terms:

```bash
uv run --frozen python -m experiments.scripts.prepare_multiwoz_data \
  --input /path/to/MultiWOZ_2.1/data.json \
  --out-turns /tmp/p2_turns.json \
  --out-segments /tmp/p2_segments.json \
  --max-dialogues 100 \
  --min-user-turns 3 \
  --seed 42
```

The raw MultiWOZ file is **not redistributed** in this repository. The exact processed E3 input is redistributed here for result verification, subject to confirmation that this redistribution is compatible with the source dataset terms. Users must obtain the raw source themselves for end-to-end rebuilding.

## Portable verification

Run from the repository root. Using `git show` verifies the canonical LF Git-blob bytes and is portable across checkouts with different line-ending settings:

```bash
set -euo pipefail
BASE=experiments/sci34_supplement/results/e3_exact_rescue

test "$(git show "HEAD:$BASE/p2_turns.json" | sha256sum | cut -d' ' -f1)" = \
  a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c

test "$(git show "HEAD:$BASE/e3_manifest.json" | sha256sum | cut -d' ' -f1)" = \
  7690f1003109a37c6f216b674ff6df2b71a4bfac98f6992c1eb23b37f98967a4

uv run --frozen python - <<'PY'
import hashlib, json, subprocess
base = "experiments/sci34_supplement/results/e3_exact_rescue"

def blob(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"])

turns_bytes = blob(f"{base}/p2_turns.json")
manifest = json.loads(blob(f"{base}/e3_manifest.json"))
turns = json.loads(turns_bytes)
actual = hashlib.sha256(turns_bytes).hexdigest()
assert actual == "a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c"
assert manifest["run_id"] == "sci34_f11ccba_20260901_e3"
assert manifest["input"]["sha256"] == actual
assert manifest["input"]["sample_count"] == len(turns) == 100
assert manifest["input"]["sample_ids"] == [row["id"] for row in turns]
print("E3 exact-input rescue verification: PASS")
PY
```

On Linux or a checkout configured not to translate line endings, `sha256sum "$BASE/p2_turns.json"` also yields the canonical hash. On Windows with `core.autocrlf=true`, the worktree hash can differ while the Git blob remains correct; use the commands above.

## Models and third-party terms

Model weights are **not redistributed**. The E3 run used a local Qwen2-7B-Instruct snapshot; `e3_model_identity.json` provides metadata/inventory hashes and strong content identity `209f3a9cbccde56fb9ed39fca06a86b11c1aa0ebf721d897baab708d6cab2133` so a separately obtained snapshot can be checked.

MultiWOZ, Qwen2-7B-Instruct, and all other third-party assets retain their own licenses and terms. This README is a technical provenance statement, not a legal determination. Before public release, the author/institution must confirm that distributing `p2_turns.json` is permitted and provide appropriate notices/attribution. The repository itself currently has no confirmed root license; see [`../../../../REPRODUCIBILITY.md`](../../../../REPRODUCIBILITY.md) and [`../../../../paper2/declarations.md`](../../../../paper2/declarations.md).
