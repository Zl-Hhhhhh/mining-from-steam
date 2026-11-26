"""Pinned RQ3 inputs and transparent measurement sensitivities.

K8 overlap projection follows the outcome-blind algorithm in rq24_inputs.py,
Lin commit ef2f958; implemented here so RQ3 does not depend on an unmerged branch.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import analysis_common as common
import structured_rq_weights as structured

ROOT = Path(__file__).resolve().parents[1]
TOPICS = tuple(f'T{i:02d}' for i in range(1, 9))
LANGUAGES = ('english', 'schinese')


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def checked_weights(rows, scope, ids):
    key = f'topic_weights_{scope}_units'
    index = [str(r['review_id']) for r in rows]
    if len(set(index)) != len(index) or set(index) != set(ids):
        raise ValueError('Topic review IDs must be unique and exactly match the master')
    if any(set(r[key]) - set(TOPICS) for r in rows):
        raise ValueError('Unexpected primary topic')
    w = pd.DataFrame([{t: r[key].get(t, 0.) for t in TOPICS} for r in rows], index=index).loc[ids]
    x = w.to_numpy(dtype=float); totals = x.sum(axis=1)
    if not np.isfinite(x).all() or (x < 0).any() or not ((totals == 0) | np.isclose(totals, 1, atol=5e-6, rtol=0)).all():
        raise ValueError('Invalid or unnormalized review topic weights')
    return w.div(w.sum(axis=1).replace(0, 1), axis=0)


def aggregate_units(labels, review_map, *, exclude_routes=()):
    """Drop units, not entire reviews; then renormalize each review's evidence."""
    by_id = {r['unit_id']: r for r in labels}
    rows = []
    for rid, uids in review_map.items():
        all_counts, fit_counts = Counter(), Counter()
        for uid in uids:
            row = by_id[uid]
            if set(exclude_routes).intersection(row['auxiliary_routes']):
                continue
            for t in row['topic_ids']:
                all_counts[t] += 1 / len(row['topic_ids'])
                if row['assignment_scope'] == 'fit_candidate':
                    fit_counts[t] += 1 / len(row['topic_ids'])
        def norm(c):
            total = sum(c.values())
            return {k: round(v / total, 6) for k, v in c.items()} if total else {}
        rows.append({'review_id': rid, 'topic_weights_fit_units': norm(fit_counts),
                     'topic_weights_all_units': norm(all_counts)})
    return rows


def script_flags(text, language):
    """Script diagnostics only: not a validated language identifier."""
    han = sum('\u3400' <= c <= '\u9fff' for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    other = sum(('\u0400' <= c <= '\u052f') or ('\u3040' <= c <= '\u30ff') or ('\uac00' <= c <= '\ud7af') for c in text)
    mixed = han >= 4 and latin >= 12 and min(han, latin) / max(han + latin, 1) >= .2
    mismatch = ((language == 'schinese' and han == 0 and latin >= 8)
                or (language == 'english' and han >= 4 and han / max(han + latin, 1) >= .5)
                or (other >= 4 and other / max(han + latin + other, 1) >= .3))
    return mismatch, mixed


def common_window_mask(frame):
    start = pd.to_datetime(frame.common_window_start_utc, utc=True, errors='coerce')
    end = pd.to_datetime(frame.common_window_end_utc, utc=True, errors='coerce')
    dates = pd.to_datetime(frame.timestamp_created, unit='s', utc=True).dt.normalize()
    return dates.ge(start) & dates.le(end)


def k8_policy(primary, alternative, config):
    overlaps = defaultdict(Counter)
    for a, b in zip(primary, alternative, strict=True):
        if a['unit_id'] != b['unit_id'] or a['assignment_scope'] != b['assignment_scope']:
            raise ValueError('K8 and K10 populations differ')
        if b['assignment_scope'] == 'fit_candidate':
            overlaps[str(b['cluster'])][str(a['cluster'])] += 1
    policy, audit = {}, {}
    for cluster, counts in sorted(overlaps.items()):
        parents = [k for k in sorted(counts) if counts[k] / sum(counts.values()) >= .20]
        if not parents:
            parents = [max(sorted(counts), key=counts.get)]
        p = config['cluster_topic_map'][parents[0]] if len(parents) == 1 else {
            'topic_ids': sorted({t for k in parents for t in config['cluster_topic_map'][k]['topic_ids']}),
            'confidence': 'low', 'route': 'multi_candidate'}
        policy[cluster] = p
        audit[cluster] = {'fit_overlaps': dict(counts), 'parents_used': parents, 'policy': p}
    return policy, audit


def load(config_path, root=None):
    root = Path(root or ROOT); config = common.read_json(config_path)
    paths, sources = {}, {}
    for name, record in config['inputs'].items():
        path = root / record['path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Input escapes repository')
        sources[name] = {'path': record['path'], 'sha256': common.sha256(path),
                         'expected_sha256': record['sha256'], 'match': common.hash_status(path, record['sha256'])}
        paths[name] = path
    master = pd.DataFrame(jsonl(paths['master'])).rename(columns={'recommendationid': 'review_id'})
    master['review_id'] = master.review_id.astype(str)
    if master.review_id.duplicated().any():
        raise ValueError('Duplicate clean review IDs')
    master = master.set_index('review_id', verify_integrity=True)
    master['appid'] = master.appid.astype(str); master['steamid'] = master.steamid.astype(str)
    if not master.voted_up.map(lambda x: isinstance(x, bool)).all():
        raise ValueError('Recommendation labels must be booleans')
    if set(master.language) - set(LANGUAGES):
        raise ValueError('Unexpected source-language label')
    for field in ['association_eligible', 'in_common_language_window', 'is_text_canonical', 'is_short_text', 'playtime_analysis_eligible']:
        if not master[field].map(lambda x: isinstance(x, bool)).all():
            raise ValueError('Invalid boolean: ' + field)
    meta = pd.read_csv(paths['metadata'], dtype={'appid': str}).set_index('appid', verify_integrity=True)
    if not set(master.appid) <= set(meta.index):
        raise ValueError('Missing game metadata')
    master['complete_game'] = master.appid.map(meta.sample_status).eq('complete')
    master['recommend'] = master.voted_up.astype(int)
    master['log_hours'] = np.log1p(pd.to_numeric(master.playtime_at_review, errors='coerce').clip(lower=0) / 60)
    # Select one review per author across games deterministically for sensitivity.
    ordered = master.loc[master.association_eligible].reset_index().sort_values(['timestamp_created', 'review_id'], kind='stable')
    first = set(ordered.loc[~ordered.steamid.duplicated(), 'review_id'])
    master['one_per_author'] = master.index.isin(first)
    flags = [script_flags(str(text), str(lang)) for text, lang in zip(master.review_clean, master.language, strict=True)]
    master['script_mismatch'] = [v[0] for v in flags]; master['mixed_script'] = [v[1] for v in flags]
    if not np.isfinite(master.loc[master.playtime_analysis_eligible, 'log_hours']).all():
        raise ValueError('Nonfinite eligible playtime')
    # Validate the published overlap-window flags against the recorded UTC bounds.
    derived_window = common_window_mask(master)
    if not np.array_equal(derived_window.to_numpy(), master.in_common_language_window.to_numpy()):
        raise ValueError('Common date-window flags and UTC bounds disagree')
    if config['topic_version'] == structured.TOPIC_VERSION:
        reviews, _, variants, _, _, diagnostics = structured.load(paths, master.index, k8_policy)
        weights = {scope: checked_weights(reviews, scope, master.index) for scope in ('fit', 'all')}
        weights.update({name: checked_weights(rows, config['primary_scope'], master.index)
                        for name, rows in variants.items()})
        by_review = {row['review_id']: row for row in reviews}
        for field, key in (('confidence', 'confidence'), ('fit_units', 'fit_unit_count'), ('all_units', 'unit_count')):
            master[field] = [by_review[rid][key] for rid in master.index]
        diagnostics['script_diagnostic_is_language_identification'] = False
        return config, master, weights, sources, diagnostics

    topic = common.read_json(paths['topic_manifest']); contract = common.read_json(paths['contract'])
    if topic['topic_version'] != config['topic_version'] or contract['topic_version'] != config['topic_version']:
        raise ValueError('Topic version mismatch')
    if contract['recommendation_label_used_for_topic_generation'] or contract['playtime_used_for_topic_generation']:
        raise ValueError('Outcome labels entered topic generation')
    reviews = jsonl(paths['reviews']); labels = jsonl(paths['labels']); units = jsonl(paths['units'])
    review_map = common.read_json(paths['review_map'])
    if set(review_map) != set(master.index) or [x['unit_id'] for x in units] != [x['unit_id'] for x in labels]:
        raise ValueError('Unit or review ID population mismatch')
    if len({x['unit_id'] for x in labels}) != len(labels):
        raise ValueError('Duplicate unit IDs')
    expected_pairs = {(rid, u['unit_id']) for u in units for rid in u['review_ids']}
    if expected_pairs != {(rid, uid) for rid, uids in review_map.items() for uid in uids}:
        raise ValueError('Invalid inverse review/unit mapping')
    rebuilt = aggregate_units(labels, review_map)
    weights = {scope: checked_weights(reviews, scope, master.index) for scope in ['fit', 'all']}
    for scope in ['fit', 'all']:
        if not np.allclose(weights[scope], checked_weights(rebuilt, scope, master.index), atol=1e-12, rtol=0):
            raise ValueError('Published weights differ from unit aggregation')
    by_review = {str(r['review_id']): r for r in reviews}
    master['confidence'] = [by_review[r]['confidence'] for r in master.index]
    master['fit_units'] = [by_review[r]['fit_unit_count'] for r in master.index]
    master['all_units'] = [by_review[r]['unit_count'] for r in master.index]
    weights['no_auxiliary'] = checked_weights(aggregate_units(labels, review_map, exclude_routes=('A01', 'A02')), 'fit', master.index)
    import assign_topics_v1_cluster_derived as assignment
    lookup = {u['unit_id']: i for i, u in enumerate(units)}
    vector_ids = common.read_json(paths['vector_ids']); vectors = np.load(paths['vectors'], mmap_mode='r', allow_pickle=False)
    if vector_ids != list(lookup) or vectors.shape != (len(units), 1024):
        raise ValueError('Vector shape/order mismatch')
    topic_config = common.read_json(paths['topic_config']); seeds = common.read_json(paths['seeds'])
    primary = assignment.load_assignments(paths['clusters'], units)
    prototypes = assignment.build_prototypes(seeds, units, lookup, vectors)
    alternative = assignment.load_assignments(paths['k8_clusters'], units)
    policy, audit = k8_policy(primary, alternative, topic_config)
    k8 = assignment.assign_units({**topic_config, 'cluster_topic_map': policy}, units, alternative, vectors, prototypes)
    weights['k8_overlap_proxy'] = checked_weights(aggregate_units(k8, review_map), 'fit', master.index)
    unrestricted = {k: {**p, 'topic_ids': list(TOPICS), 'route': 'multi_candidate'} for k, p in topic_config['cluster_topic_map'].items()}
    global_labels = assignment.assign_units({**topic_config, 'cluster_topic_map': unrestricted}, units, primary, vectors, prototypes)
    weights['unrestricted_seeds'] = checked_weights(aggregate_units(global_labels, review_map), 'fit', master.index)
    diagnostics = {'review_weights_reaggregated': True, 'master_reviews': len(master),
                   'fit_evidence_reviews': int(weights['fit'].sum(axis=1).gt(0).sum()),
                   'all_evidence_reviews': int(weights['all'].sum(axis=1).gt(0).sum()),
                   'unit_confidence_counts': dict(Counter(r['confidence'] for r in labels)),
                   'high_confidence_topics': dict(Counter(t for r in labels if r['confidence'] == 'high' for t in r['topic_ids'])),
                   'k8_proxy_mapping': audit, 'script_diagnostic_is_language_identification': False,
                   'alternative_mapping_uses_outcomes': False}
    return config, master, weights, sources, diagnostics
