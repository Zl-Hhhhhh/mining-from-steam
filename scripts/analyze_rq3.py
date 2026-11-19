"""Reproducible exploratory RQ3, independently of other members' RQ branches."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, platform, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import analysis_common as common
from rq3_inputs import ROOT, TOPICS, LANGUAGES, load
from rq3_statistics import paired_estimate, holm, fit_interaction


def digest_ids(ids):
    return hashlib.sha256(('\n'.join(sorted(map(str, ids))) + '\n').encode()).hexdigest()


def variants(frame, weights, config):
    base = frame.association_eligible.copy()
    common = frame.loc[base & frame.in_common_language_window]
    n = common.groupby(['appid','language']).size().unstack(fill_value=0).reindex(columns=LANGUAGES, fill_value=0)
    evidence = common.loc[weights['fit'].loc[common.index].sum(axis=1).gt(0)]
    e = evidence.groupby(['appid','language']).size().unstack(fill_value=0).reindex(index=n.index, columns=LANGUAGES, fill_value=0)
    fixed_games = n.index[n.ge(config['minimum_reviews_per_language']).all(axis=1)
                         & e.ge(config['minimum_topic_reviews_per_language']).all(axis=1)]
    return {
        'primary_fit': (base, weights['fit'], 20, TOPICS),
        'all_units': (base, weights['all'], 20, TOPICS),
        'all_same_fit_population': (base & weights['fit'].sum(axis=1).gt(0), weights['all'], 20, TOPICS),
        'common_window': (base & frame.in_common_language_window, weights['fit'], 20, TOPICS),
        'original_same_common_games': (base & frame.appid.isin(fixed_games), weights['fit'], 20, TOPICS),
        'unique_text': (base & frame.is_text_canonical, weights['fit'], 20, TOPICS),
        'one_per_author': (base & frame.one_per_author, weights['fit'], 20, TOPICS),
        'complete_games': (base & frame.complete_game, weights['fit'], 20, TOPICS),
        'exclude_short': (base & ~frame.is_short_text, weights['fit'], 20, TOPICS),
        'script_consistent': (base & ~frame.script_mismatch & ~frame.mixed_script, weights['fit'], 20, TOPICS),
        'no_auxiliary': (base, weights['no_auxiliary'], 20, TOPICS),
        'high_confidence_T02_T06': (base & frame.confidence.eq('high'), weights['fit'], 20, ('T02', 'T06')),
        'support_30': (base, weights['fit'], config['strict_topic_minimum'], TOPICS),
        'unrestricted_seeds': (base, weights['unrestricted_seeds'], 20, TOPICS),
        'k8_overlap_proxy': (base, weights['k8_overlap_proxy'], 20, TOPICS),
        'raw_master_denominator': (pd.Series(True, index=frame.index), weights['fit'], 20, TOPICS),
    }


def describe_scenario(name, frame, weights, minimum, active_topics, config):
    support, game_rows = [], []
    for appid, g in frame.groupby('appid', sort=True):
        record = {'scenario': name, 'appid': str(appid), 'game': str(g.game.iloc[0]), 'topic_minimum_per_language': minimum}
        lang_rows = {}
        for lang in LANGUAGES:
            sub = g.loc[g.language.eq(lang)]
            evidence = weights.loc[sub.index].sum(axis=1).gt(0)
            selected = sub.loc[evidence]
            lang_rows[lang] = (sub, selected)
            record[lang + '_n'] = len(sub); record[lang + '_evidence_n'] = len(selected)
        paired = all(record[lang + '_n'] >= config['minimum_reviews_per_language'] for lang in LANGUAGES)
        theme_paired = paired and all(record[lang + '_evidence_n'] >= minimum for lang in LANGUAGES)
        record.update(paired_for_coverage=paired, paired_for_theme=theme_paired,
                      coverage_exclusion='' if paired else 'insufficient_base_language_support',
                      theme_exclusion='' if theme_paired else ('insufficient_base_language_support' if not paired else 'insufficient_topic_language_support'))
        support.append(record)
        for measure in ('coverage', 'recommendation', *TOPICS):
            if measure in TOPICS and (not theme_paired or measure not in active_topics):
                continue
            if measure not in TOPICS and not paired:
                continue
            row = {'scenario': name, 'appid': str(appid), 'game': record['game'], 'measure': measure}
            for lang in LANGUAGES:
                sub, selected = lang_rows[lang]
                if measure == 'coverage':
                    value = len(selected) / len(sub); n = len(sub)
                elif measure == 'recommendation':
                    value = sub.recommend.mean(); n = len(sub)
                else:
                    value = weights.loc[selected.index, measure].mean(); n = len(selected)
                row[lang + '_mean'] = float(value); row[lang + '_n'] = int(n)
            row['difference_pp'] = (row['schinese_mean'] - row['english_mean']) * 100
            game_rows.append(row)
    support_frame = pd.DataFrame(support)
    games = pd.DataFrame(game_rows, columns=['scenario', 'appid', 'game', 'measure', 'english_mean', 'english_n', 'schinese_mean', 'schinese_n', 'difference_pp'])
    estimates = []
    for measure in ('coverage', 'recommendation', *TOPICS):
        rows = games.loc[games.measure.eq(measure)]
        effect = paired_estimate(rows, seed=config['seed'], reps=config['bootstrap_replicates'])
        if measure in TOPICS and measure not in active_topics:
            effect = {'status': 'not_supported_by_high_confidence_selection', 'games': 0}
        estimates.append({'scenario': name, 'measure': measure, **effect})
    holm([r for r in estimates if r['measure'] in TOPICS])
    holm([r for r in estimates if r['measure'] not in TOPICS])
    if support_frame.empty:
        coverage_games, theme_games = set(), set()
    else:
        coverage_games = set(support_frame.loc[support_frame.paired_for_coverage, 'appid'])
        theme_games = set(support_frame.loc[support_frame.paired_for_theme, 'appid'])
    coverage_ids = frame.index[frame.appid.isin(coverage_games)]
    theme_ids = frame.index[frame.appid.isin(theme_games) & weights.loc[frame.index].sum(axis=1).gt(0)]
    flow = []
    for lang in LANGUAGES:
        for stage, idx in [('after_scenario_filters', frame.index), ('paired_base_sample', coverage_ids), ('paired_topic_sample', theme_ids)]:
            selected = frame.loc[idx]; selected = selected.loc[selected.language.eq(lang)]
            flow.append({'scenario': name, 'stage': stage, 'language': lang, 'n': len(selected),
                         'games': selected.appid.nunique(), 'review_ids_sha256': digest_ids(selected.index)})
    return estimates, games, support_frame, flow, coverage_ids, theme_ids


def verify_output(config_path, out):
    cfg = common.read_json(config_path); manifest = common.read_json(out / 'manifest.json')
    if manifest['config_sha256'] != common.sha256(config_path):
        raise ValueError('Output configuration differs')
    for entry in cfg['inputs'].values():
        common.hash_status(ROOT / entry['path'], entry['sha256'])
    for entry in manifest['generators']:
        if common.sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Generator changed: ' + entry['path'])
    for name, sha in manifest['outputs_sha256'].items():
        if common.sha256(out / name) != sha:
            raise ValueError('Output changed: ' + name)
    print(json.dumps({'status': 'passed', 'phase': 'verify_output', 'output_files': len(manifest['outputs_sha256']), 'topic_version': cfg['topic_version']}), flush=True)
    return manifest


def run(config_path, output=None):
    config_path = Path(config_path).resolve(); cfg = common.read_json(config_path)
    out = Path(output).resolve() if output else ROOT / cfg['output']
    if out.exists():
        raise FileExistsError('Preserve existing analysis; use --verify-output or a new --output directory')
    if cfg['confidence_level'] != .95 or cfg['primary_scope'] != 'fit' or cfg['topic_reference'] != 'T01':
        raise ValueError('Unsupported analysis protocol')
    print(json.dumps({'phase': 'loading_verified_inputs'}), flush=True)
    config, frame, weight_variants, sources, diagnostics = load(config_path)
    out.mkdir(parents=True)
    all_estimates, all_games, all_support, flows, interaction_rows, interaction_diag = [], [], [], [], [], []
    audit_cols = ['appid', 'language', 'association_eligible', 'in_common_language_window', 'complete_game',
                  'is_text_canonical', 'one_per_author', 'is_short_text', 'script_mismatch', 'mixed_script', 'confidence']
    inclusion = frame[audit_cols].copy(); inclusion['fit_evidence'] = weight_variants['fit'].sum(axis=1).gt(0)
    inclusion['all_evidence'] = weight_variants['all'].sum(axis=1).gt(0)
    interaction_scenarios = {'primary_fit', 'all_units', 'common_window', 'unique_text', 'complete_games', 'script_consistent', 'no_auxiliary'}
    sample_stats = {}
    for name, (mask, w, minimum, active) in variants(frame, weight_variants, config).items():
        if minimum == 20:
            minimum = config['minimum_topic_reviews_per_language']
        subset = frame.loc[mask]
        estimates, games, support, flow, cov_ids, topic_ids = describe_scenario(name, subset, w, minimum, active, config)
        all_estimates.extend(estimates); all_games.append(games); all_support.append(support); flows.extend(flow)
        inclusion[name + '__coverage'] = inclusion.index.isin(cov_ids)
        inclusion[name + '__topic'] = inclusion.index.isin(topic_ids)
        sample_stats[name] = {'after_filters': len(subset), 'paired_base_reviews': len(cov_ids), 'paired_topic_reviews': len(topic_ids),
                              'paired_base_ids_sha256': digest_ids(cov_ids), 'paired_topic_ids_sha256': digest_ids(topic_ids)}
        print(json.dumps({'phase': 'descriptive_complete', 'scenario': name, **sample_stats[name]}), flush=True)
        if name in interaction_scenarios:
            print(json.dumps({'phase': 'interaction_start', 'scenario': name}), flush=True)
            rows, diag = fit_interaction(subset.loc[topic_ids], w, config)
            for row in rows: row['scenario'] = name
            diag['scenario'] = name
            interaction_rows.extend(rows); interaction_diag.append(diag)
            print(json.dumps({'phase': 'interaction_complete', 'scenario': name, 'status': diag['status'], 'n': diag.get('n_used')}), flush=True)
    common.write_csv(out / 'paired_estimates.csv', all_estimates)
    common.write_csv(out / 'game_language_differences.csv', pd.concat(all_games, ignore_index=True))
    common.write_csv(out / 'game_support.csv', pd.concat(all_support, ignore_index=True))
    common.write_csv(out / 'sample_flow.csv', flows)
    common.write_csv(out / 'review_inclusion.csv', inclusion.reset_index())
    common.write_csv(out / 'interaction_effects.csv', interaction_rows)
    common.write_csv(out / 'model_diagnostics.csv', [{k:d.get(k) for k in ['scenario','status','n_input','n_used','games_used','parameters','rank','separation','converged']} for d in interaction_diag])
    # Full source-sample diagnostics retain the denominator before eligibility filtering.
    coverage = []
    for lang, g in frame.groupby('language', sort=True):
        eligible = g.loc[g.association_eligible]
        for pop, sub in [('master', g), ('association_eligible', eligible)]:
            coverage.append({'population': pop, 'language': lang, 'n': len(sub),
                             'fit_evidence': int(weight_variants['fit'].loc[sub.index].sum(axis=1).gt(0).sum()),
                             'all_evidence': int(weight_variants['all'].loc[sub.index].sum(axis=1).gt(0).sum()),
                             'source_script_mismatch': int(sub.script_mismatch.sum()), 'mixed_script': int(sub.mixed_script.sum())})
    common.write_csv(out / 'source_coverage.csv', coverage)
    diagnostics.update(samples=sample_stats, interaction_models=interaction_diag,
                       bootstrap={'unit': 'paired game means', 'reps': config['bootstrap_replicates'], 'seed': config['seed'],
                                  'interval': 'pointwise percentile 95%', 'does_not_model_topic_measurement_error': True},
                       p_values='Paired-game t tests; Holm within eight themes per scenario and separately two diagnostic outcomes. Exploratory, not probability-sampled population inference.')
    common.save_json(out / 'diagnostics.json', diagnostics)
    from rq3_report import render
    render(out, config, frame, pd.DataFrame(all_estimates), pd.concat(all_games, ignore_index=True),
           pd.DataFrame(interaction_rows), diagnostics)
    generators = [Path(__file__), ROOT / 'scripts/rq3_inputs.py', ROOT / 'scripts/rq3_statistics.py', ROOT / 'scripts/rq3_report.py',
                  ROOT / 'scripts/analysis_common.py', ROOT / 'scripts/assign_topics_v1_cluster_derived.py']
    manifest = {'analysis_version': config['analysis_version'], 'topic_version': config['topic_version'],
                'source_commit': config['source_commit'], 'config_sha256': common.sha256(config_path),
                'inputs': sources, 'generators': [{'path': common.relative(p), 'sha256': common.sha256(p)} for p in generators],
                'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'scikit-learn', 'threadpoolctl']}},
                'output_scope': 'Exploratory RQ3; source language is not nationality, theme weights are not probabilities.',
                'outputs_sha256': {p.relative_to(out).as_posix(): common.sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    common.save_json(out / 'manifest.json', manifest)
    print(json.dumps({'phase': 'complete', 'output': str(out), 'scenarios': len(sample_stats),
                      'interaction_models': len(interaction_diag)}), flush=True)
    return verify_output(config_path, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/rq3_exploratory.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-output', action='store_true')
    args = parser.parse_args()
    cfg = common.read_json(args.config)
    out = args.output or ROOT / cfg['output']
    with threadpool_limits(limits=2):
        if args.verify_output: verify_output(args.config, out)
        else: run(args.config, args.output)


if __name__ == '__main__': main()
