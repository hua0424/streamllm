"""Protocol constants are frozen before either engineering pilot or formal run."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments/sci34_supplement/results"
C2 = RESULTS / "c2_crop_integrity/c2crop_82103004_20260903T080512Z"
E3 = RESULTS / "e3/sci34_f11ccba_20260901_e3"
# Non-Cartesian grid: ordinary across lengths plus short boundary cases and
# medium repeated interruption. IDs refer to existing immutable C2 fixtures.
CASE_INDICES = (0, 1, 2, 3, 4, 5, 9, 15, 17)
PROTOCOL = {
    "version": 1, "seed": 20260906, "sessions": 4, "pairs_per_case": 4,
    "warmup_pairs_per_case": 1, "continuation_cap": 8,
    "case_indices": list(CASE_INDICES), "arms": ["crop", "rebuild"],
    "pilot_case_indices": [1, 15, 17], "pilot_sessions": 1, "pilot_pairs": 1,
    "bootstrap_draws": 10000,
    "estimand": "per fixed case/event mean paired rebuild-minus-crop; session-cluster bootstrap",
    "numerical_gate": "finite only; no cross-topology tolerance or equivalence test",
    "timing": "synchronized wall time; start through USER_OPEN then suffix/header then first generator yield (including token KV commit)",
    "scope": "controlled text fixtures; software boundaries; single GPU; no acoustic evidence",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cases(pilot=False):
    from experiments.sci34_supplement.c2_crop_integrity.protocol import load_cases
    source = load_cases(BASE.parent / "c2_crop_integrity/cases.json", formal=False)
    indices = PROTOCOL["pilot_case_indices"] if pilot else CASE_INDICES
    return [replace(source[i], next_user=source[i].next_user or "State the next confirmed step briefly.") for i in indices]


def order(session, case_index, repeat):
    return ("crop", "rebuild") if (session + case_index + repeat) % 2 == 0 else ("rebuild", "crop")


def expected_grid(pilot=False):
    sessions = 1 if pilot else PROTOCOL["sessions"]
    repeats = 1 if pilot else PROTOCOL["pairs_per_case"]
    return [(s, c.id, r, event, arm) for s in range(sessions) for c in cases(pilot)
            for r in range(repeats) for event in range(2 if c.second_crop_fraction is not None else 1)
            for arm in ("crop", "rebuild")]
