"""Typesetting-only extraction of accepted JSON; no model or experiment execution.
Usage: uv run --no-sync python assets/generate_tables.py --repo REPOSITORY_ROOT
The checked-in tables suffice for standalone LaTeX compilation without the repo.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
args = parser.parse_args()
root = args.repo.resolve()
out = Path(__file__).resolve().parents[1] / 'tables'
out.mkdir(exist_ok=True)
base = root / 'experiments/sci34_supplement/results'
r = base / 'recovery_boundary/rb_formal_20260908T023934Z_f69e6478'
a = json.loads((r / 'analysis.json').read_text(encoding='utf-8'))
m = json.loads((r / 'manifest.json').read_text(encoding='utf-8'))
rows = [json.loads(line) for p in sorted(r.glob('session_*/records.jsonl'))
        for line in p.read_text(encoding='utf-8').splitlines()]
assert len(rows) == 320 and len(a['cells']) == 10
pairs = {}
for row in rows:
    assert row['ready_total_ns'] == row['recovery_ns'] + row['suffix_ready_ns']
    assert row['first_deliverable_ns'] == row['ready_total_ns'] + row['decode_first_ns']
    assert row['delivered']
    st = row['ready_state']
    n = len(st['token_ids'])
    assert st['token_ids'] == row['ready_ids_expected']
    assert st['seq_length'] == st['kv_length'] == n
    assert st['mask'] == [[1] * n]
    assert st['role_phase'] == 'assistant_open' and st['end_reason'] == 'none'
    key = (row['session'], row['case_id'], row['repeat'], row['event'])
    assert row['arm'] not in pairs.setdefault(key, {})
    pairs[key][row['arm']] = row
assert len(pairs) == 160
for pair in pairs.values():
    assert set(pair) == {'crop', 'rebuild'}
    for field in ('pre_ids', 'pre_state', 'keep', 'recovered_ids', 'suffix', 'ready_state'):
        assert pair['crop'][field] == pair['rebuild'][field]
# Recompute the frozen process-cluster algorithm from raw interval differences.
rng = random.Random(20260906)
short = []
full = []
labels = {
    '01': 'Zero playback (512)', '02': 'Fragment boundary (512)',
    '03': 'Mid-fragment (512)', '04': 'Reply-tail no-op (512)',
    '05': 'Pending EOT (512)', '06': 'Full invalidation (512)',
    '10': 'Fragment boundary (2048)', '16': 'Repeated interruption (2048)',
    '18': 'Fragment boundary (8192)',
}
for cell in a['cells']:
    case, event = cell['case_id'], cell['event']
    cid = case.split('_')[1]
    selected = [row for row in rows if row['case_id'] == case and row['event'] == event]
    for metric, endpoint in [('recovery_ns', 'Rec'), ('ready_total_ns', 'Ready'),
                             ('first_deliverable_ns', 'First')]:
        means = []
        for session in range(4):
            ds = [(pairs[(session, case, rep, event)]['rebuild'][metric] -
                   pairs[(session, case, rep, event)]['crop'][metric]) / 1e6
                  for rep in range(4)]
            means.append(statistics.mean(ds))
        draws = sorted(statistics.mean(rng.choices(means, k=4)) for _ in range(10000))
        actual = cell['metrics'][metric]
        assert means == actual['session_mean_differences_ms']
        assert statistics.mean(means) == actual['mean_rebuild_minus_crop_ms']
        assert [draws[249], draws[9749]] == actual['session_cluster_percentile_ci95_ms']
        mean = actual['mean_rebuild_minus_crop_ms']
        low, high = actual['session_cluster_percentile_ci95_ms']
        cm, rm = [statistics.mean(row[metric] / 1e6 for row in selected if row['arm'] == arm)
                  for arm in ('crop', 'rebuild')]
        full.append(f'{cid}/{event} & {endpoint} & {cm:.3f} & {rm:.3f} & {mean:.3f} [{low:.3f}, {high:.3f}] \\\\')
        if metric == 'first_deliverable_ns':
            short.append(f'{cid}/{event} & {labels[cid]} & {mean:.3f} & [{low:.3f}, {high:.3f}] \\\\')
(out / 'r_first_deliverable_rows.tex').write_text('\n'.join(short) + '\n', encoding='utf-8')
(out / 'r_all_endpoints_rows.tex').write_text('\n'.join(full) + '\n', encoding='utf-8')
e1file = base / 'e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z/analysis_v2.json'
e1 = json.loads(e1file.read_text(encoding='utf-8'))
cs = []
for condition, c in e1['condition_summaries'].items():
    label = ('A full prefill' if condition == 'system_a_full_prefill' else
             'B never trigger' if condition == 'b_never_speculate' else
             'B@' + condition.removeprefix('b_threshold_'))
    ready = c['candidate_selection_compute_readiness_ms']['summary']['mean']
    oracle = c['ttft_eff_ms_oracle_latency_lower_bound']['mean']
    cs.append(f'{label} & {ready:.3f} & {oracle:.3f} & {100*c["survival_rate"]:.1f} & {100*c["pooled_token_waste_ratio"]:.3f} \\\\')
assert len(cs) == 10
(out / 'candidate_grid_rows.tex').write_text('\n'.join(cs) + '\n', encoding='utf-8')
e3file = base / 'e3/sci34_f11ccba_20260901_e3/analysis_weighting_dedup_v2.json'
e3 = json.loads(e3file.read_text(encoding='utf-8'))
es = []
for target, t in e3['targets'].items():
    for detector, d in t['metrics'].items():
        for key, abbr in [('label_weighted', 'L'), ('dialogue_weighted', 'D'),
                          ('unique_semantic_group_weighted', 'K'), ('unique_dialogue_weighted', 'KD')]:
            e = d['estimands'][key]
            mean = 100 * e['generation_minus_playback']
            low, high = [100*x for x in e['difference_95_ci']]
            assert low < 0 < high
            es.append(f'{target.title()} & {"Lexical" if detector == "rule" else "Judge"} & {abbr} & ${mean:.2f}$ [${low:.2f}$, ${high:.2f}$] \\\\')
(out / 'e3_sensitivity_rows.tex').write_text('\n'.join(es) + '\n', encoding='utf-8')
b = json.loads((r / 'boundary.json').read_text(encoding='utf-8'))
assert b['ok'] and b['trajectories'] == 100 and b['stored_records_checked'] == 800
assert len(b['boundary_checks']) == 1118 and len(b['c2_closure_checks']) == 27
diag = a['diagnostics']
assert sum(x['top1_same'] for x in diag) == 160
assert sum(x['continuation_same'] for x in diag) == 144
assert max(x['max_abs_logit_difference'] for x in diag) == 0.59375
assert {(x['key'][1], x['key'][3]) for x in diag if not x['continuation_same']} == {('c2_16_second_medium_eos', 1)}
assert sum(p['crop']['continuation_ids'] == p['rebuild']['continuation_ids'] for p in pairs.values()) == 144
sources = [r/'analysis.json', r/'manifest.json', r/'boundary.json', r/'validation.json', e1file, e3file]
report = {
    'scope': 'typesetting extraction and raw interval/structural cross-check; no GPU, no new estimates',
    'r_records': len(rows), 'r_pairs': len(pairs), 'r_cells': len(a['cells']),
    'raw_interval_identity_checks': 320, 'raw_matched_state_pairs': 160,
    'recomputed_metrics_and_bootstrap_intervals': 30,
    'source_files': len(m['source']['files']), 'source_commit': m['source']['commit'],
    'top1_same_stored_diagnostics': 160, 'continuation_same_recomputed': 144,
    'boundary_cursor_checks_stored': 1118, 'closure_checks_stored': 27,
    'npy_status': 'prior-turn complete-tar verification accepted; not reopened or reverified in this revision',
    'sources_sha256_worktree': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
}
(Path(__file__).resolve().parent / 'numeric_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
