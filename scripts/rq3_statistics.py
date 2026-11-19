"""Paired game summaries and explicitly diagnostic interaction models.

GLM diagnostics/covariance follow the reviewed RQ2/RQ4 approach from Lin
ef2f958; RQ3 contrasts, support checks and tests are independent of that branch.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, linprog
from scipy.stats import f, t
from statsmodels.api import GLM, families
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from rq3_inputs import TOPICS


def holm(rows, key='p_game_t'):
    valid = [r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    if valid:
        values = multipletests([r[key] for r in valid], method='holm')[1]
        for r, value in zip(valid, values, strict=True):
            r[key + '_holm'] = float(value)


def paired_estimate(game_rows, *, seed, reps):
    """Resample paired game means, preserving both language observations."""
    rows = game_rows.sort_values('appid', kind='stable')
    en = rows.english_mean.to_numpy(float); zh = rows.schinese_mean.to_numpy(float)
    d = zh - en; n = len(d)
    if n == 0:
        return {'status': 'no_paired_games', 'games': 0}
    out = {'status': 'ok' if n >= 2 else 'one_game_only', 'games': n,
           'english_mean': float(en.mean()), 'schinese_mean': float(zh.mean()),
           'difference_pp': float(d.mean() * 100),
           'english_n': int(rows.english_n.sum()), 'schinese_n': int(rows.schinese_n.sum()),
           'games_positive': int((d > 0).sum()), 'games_negative': int((d < 0).sum())}
    out['comment_weighted_difference_pp'] = float((np.average(zh, weights=rows.schinese_n) - np.average(en, weights=rows.english_n)) * 100)
    if n >= 2:
        rng = np.random.default_rng(seed)
        boot = d[rng.integers(0, n, size=(reps, n))].mean(axis=1) * 100
        out['ci_low_pp'], out['ci_high_pp'] = [float(v) for v in np.quantile(boot, [.025, .975])]
        se = d.std(ddof=1) / np.sqrt(n)
        out['p_game_t'] = float(2 * t.sf(abs(d.mean() / se), n - 1)) if se > 0 else (1. if np.all(d == 0) else None)
    return out


def separation(x, y):
    z = (2 * y[:, None] - 1) * x
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected', category=OptimizeWarning)
        a = linprog(-z.mean(axis=0), A_ub=-z, b_ub=np.zeros(len(z)), bounds=[(-1, 1)] * x.shape[1],
                    method='highs', options={'threads': 1})
        if not a.success:
            return 'solver_failed'
        if -a.fun <= 1e-8:
            return 'none'
        b = linprog(np.zeros(x.shape[1]), A_ub=-z, b_ub=-np.ones(len(z)), bounds=[(None, None)] * x.shape[1],
                    method='highs', options={'threads': 1})
    return 'complete' if b.success else 'quasi' if b.status == 2 else 'solver_failed'


def fit_interaction(frame, weights, config):
    """Weight effects per 10 percentage-point reallocation from fixed T01."""
    counts = frame.groupby('appid').recommend.nunique()
    excluded = sorted(counts.index[counts < 2].astype(str))
    use = frame.loc[~frame.appid.isin(excluded) & frame.playtime_analysis_eligible].copy()
    w = weights.loc[use.index]
    diag = {'status': 'pending', 'n_input': len(frame), 'n_used': len(use), 'games_used': int(use.appid.nunique()),
            'excluded_single_outcome_games': excluded, 'reference': config['topic_reference'],
            'weight_step': .10, 'confidence_interval': 'game-clustered t, df=G-1, finite-sample correction',
            'formula': 'recommend ~ game + language + log1p(hours) + seven topic weights + language:seven topic weights'}
    if len(use) < 100 or use.language.nunique() < 2 or use.appid.nunique() < 3:
        return [], diag | {'status': 'insufficient_sample'}
    # Remove games made homogeneous by missing/invalid time filtering as well.
    varied = use.groupby('appid').recommend.nunique()
    more = sorted(varied.index[varied < 2].astype(str))
    use = use.loc[~use.appid.isin(more)]; w = w.loc[use.index]
    diag.update(n_used=len(use), games_used=int(use.appid.nunique()),
                excluded_single_outcome_games=sorted(set(excluded + more)))
    if len(use) < 100 or use.language.nunique() < 2 or use.appid.nunique() < 3:
        return [], diag | {'status': 'insufficient_sample_after_time_filter'}
    support = {}
    for topic in TOPICS:
        support[topic] = {}
        for lang in ['english', 'schinese']:
            mask = use.language.eq(lang) & w[topic].gt(0)
            n = int(mask.sum()); positives = int(use.loc[mask, 'recommend'].sum())
            support[topic][lang] = {'n': n, 'recommended': positives, 'not_recommended': n - positives}
    diag['topic_support'] = support
    unsupported = [topic for topic in TOPICS if any(
        v['n'] < config['interaction_minimum_topic_reviews_per_language']
        or min(v['recommended'], v['not_recommended']) < config['interaction_minimum_outcomes_per_language']
        for v in support[topic].values())]
    diag['unsupported_report_contrasts'] = unsupported
    lang = use.language.eq('schinese').astype(float)
    x = pd.DataFrame({'const': 1., 'language_schinese': lang, 'log1p_hours': use.log_hours}, index=use.index)
    x = pd.concat([x, pd.get_dummies(use.appid, prefix='game', drop_first=True, dtype=float)], axis=1)
    for topic in TOPICS:
        if topic != config['topic_reference']:
            x[topic] = w[topic] * 10
            x['zh_' + topic] = w[topic] * lang * 10
    values = x.to_numpy(float); y = use.recommend.to_numpy(float)
    diag.update(parameters=x.shape[1], rank=int(np.linalg.matrix_rank(values)), recommended=int(y.sum()),
                not_recommended=int(len(y) - y.sum()))
    if diag['rank'] != x.shape[1] or len(y) <= x.shape[1]:
        return [], diag | {'status': 'rank_deficient'}
    diag['separation'] = separation(values, y)
    if diag['separation'] != 'none':
        return [], diag | {'status': 'separated_or_solver_failed'}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = GLM(y, x, family=families.Binomial()).fit(cov_type='cluster',
                    cov_kwds={'groups': use.appid.to_numpy(), 'use_correction': True, 'df_correction': True}, use_t=True)
    diag['warnings'] = sorted({v.category.__name__ for v in caught})
    diag['converged'] = bool(result.converged)
    if not result.converged or any(issubclass(v.category, PerfectSeparationWarning) for v in caught):
        return [], diag | {'status': 'invalid_fit'}
    if not np.isfinite(result.params).all() or not np.isfinite(result.cov_params()).all().all():
        return [], diag | {'status': 'nonfinite_fit'}
    diag.update(status='ok', iterations=int(result.fit_history['iteration']),
                max_absolute_coefficient=float(abs(result.params).max()),
                fitted_probability_min=float(min(result.fittedvalues)), fitted_probability_max=float(max(result.fittedvalues)))
    df = int(use.appid.nunique()) - 1
    rows = []
    for topic in TOPICS:
        if topic == config['topic_reference']:
            continue
        for role, contrast in [('english', {topic: 1}), ('schinese', {topic: 1, 'zh_' + topic: 1}), ('language_interaction', {'zh_' + topic: 1})]:
            row = {'topic': topic, 'role': role, 'reference': config['topic_reference'], 'n': len(use),
                   'games': int(use.appid.nunique()), 'step': .10, 'status': 'ok'}
            if topic in unsupported or config['topic_reference'] in unsupported:
                rows.append(row | {'status': 'insufficient_topic_outcome_support'}); continue
            names = list(contrast); c = np.array(list(contrast.values()), dtype=float)
            beta = float(c @ result.params.loc[names]); variance = float(c @ result.cov_params().loc[names, names].to_numpy() @ c)
            if not np.isfinite(variance) or variance <= 0:
                rows.append(row | {'status': 'nonpositive_variance'}); continue
            se = np.sqrt(variance); delta = t.ppf(.975, df) * se
            rows.append(row | {'log_or': beta, 'cluster_se': float(se), 'df': df, 'odds_ratio': float(np.exp(beta)),
                               'ci_low': float(np.exp(beta - delta)), 'ci_high': float(np.exp(beta + delta)),
                               'p': float(2 * t.sf(abs(beta / se), df))})
    holm([r for r in rows if r['role'] == 'language_interaction'], key='p')
    names = ['zh_' + v for v in TOPICS if v != config['topic_reference']]
    covariance = result.cov_params().loc[names, names].to_numpy(); b = result.params.loc[names].to_numpy()
    if np.linalg.matrix_rank(covariance) == len(names) and np.linalg.eigvalsh(covariance).min() > 0:
        statistic = float(b @ np.linalg.solve(covariance, b) / len(names))
        diag['joint_interaction_test'] = {'F': statistic, 'df_num': len(names), 'df_den': df,
                                           'p': float(f.sf(statistic, len(names), df))}
    else:
        diag['joint_interaction_test'] = {'status': 'singular_covariance'}
    return rows, diag
