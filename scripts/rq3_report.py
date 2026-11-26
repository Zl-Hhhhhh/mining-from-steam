"""Standalone scientific figures and a self-contained RQ3 browser report."""
from __future__ import annotations
import base64, html, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from rq3_inputs import TOPICS

NAMES = dict(zip(TOPICS, ['Gameplay', 'Controls & difficulty', 'Performance & bugs', 'Story & characters',
                          'Audiovisual', 'Multiplayer & community', 'Price & monetization', 'Content & support']))


def render(out, config, frame, estimates, games, interactions, diagnostics, config_path):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figures = out / 'figures'; figures.mkdir()
    scope = config['primary_scope']
    primary = f'primary_{scope}'
    secondary = 'fit_units' if scope == 'all' else 'all_units'
    main = estimates.loc[estimates.scenario.eq(primary)].set_index('measure')
    themes = main.loc[list(TOPICS)]
    core = {primary,secondary,'all_same_fit_population','common_window','original_same_common_games','unique_text',
            'one_per_author','complete_games','exclude_short','script_consistent','support_30'}
    robustness = []
    for topic in TOPICS:
        for family, scenarios in [('core_sample_and_scope',core),
                                  ('mapping_diagnostics',{'no_auxiliary','unrestricted_seeds','k8_overlap_proxy'})]:
            selected = estimates.loc[estimates.measure.eq(topic) & estimates.scenario.isin(scenarios) & estimates.status.eq('ok')]
            values = selected.difference_pp.astype(float)
            robustness.append({'topic':topic,'family':family,'estimable_scenarios':len(selected),
                               'minimum_difference_pp':float(values.min()) if len(values) else np.nan,
                               'maximum_difference_pp':float(values.max()) if len(values) else np.nan,
                               'same_direction_as_primary':bool((np.sign(values)==np.sign(main.loc[topic,'difference_pp'])).all()) if len(values) else False,
                               'pointwise_intervals_excluding_zero':int(((selected.ci_low_pp>0)|(selected.ci_high_pp<0)).sum()),
                               'holm_p_below_005':int(selected.p_game_t_holm.lt(.05).sum())})
    pd.DataFrame(robustness).to_csv(out/'robustness_summary.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    fig, ax = plt.subplots(figsize=(9, 5.8), layout='constrained')
    y = np.arange(8); point = themes.difference_pp.to_numpy(float)
    lo = themes.ci_low_pp.to_numpy(float); hi = themes.ci_high_pp.to_numpy(float)
    ax.errorbar(point, y, xerr=np.vstack([point-lo, hi-point]), fmt='o', color='#245fa6', capsize=4)
    ax.axvline(0, color='#666', lw=1); ax.set_yticks(y, [t + '  ' + NAMES[t] for t in TOPICS]); ax.invert_yaxis()
    ax.set_xlabel('Chinese minus English mean topic weight (percentage points)')
    ax.set_title('RQ3: topic composition among detected-evidence reviews\n'
                 f'{int(themes.games.iloc[0])} paired games; equal weights; pointwise 95% paired-game bootstrap intervals', fontsize=11)
    ax.grid(axis='x', alpha=.2)
    fig.savefig(figures / 'topic_language_differences.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    coverage = games.loc[games.scenario.eq(primary) & games.measure.eq('coverage')]
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    ax.scatter(coverage.english_mean*100, coverage.schinese_mean*100, color='#245fa6', alpha=.8)
    bound = max(60, np.ceil(max(coverage.english_mean.max(), coverage.schinese_mean.max())*100/10)*10)
    ax.plot([0, bound], [0, bound], '--', color='#888', lw=1)
    ax.set(xlim=(0,bound), ylim=(0,bound), xlabel='English reviews with topic evidence (%)',
           ylabel='Chinese reviews with topic evidence (%)', title='RQ3: detection coverage within each game')
    ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.savefig(figures / 'coverage_by_game.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    g = games.loc[games.scenario.eq(primary) & games.measure.isin(TOPICS)]
    heat = g.pivot(index='game', columns='measure', values='difference_pp').reindex(columns=TOPICS).sort_index()
    limit = max(1., float(abs(heat).max().max()))
    fig, ax = plt.subplots(figsize=(10, max(7, .25*len(heat)+2)), layout='constrained')
    image = ax.imshow(heat.to_numpy(dtype=float), aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(8), TOPICS); ax.set_yticks(range(len(heat)), heat.index, fontsize=8)
    ax.set_title('Within-game topic differences: Chinese minus English')
    fig.colorbar(image, ax=ax, label='Percentage points', shrink=.7)
    fig.savefig(figures / 'game_topic_difference_heatmap.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    sensitivity = estimates.loc[estimates.measure.isin(TOPICS)].pivot(index='scenario', columns='measure', values='difference_pp').reindex(columns=TOPICS)
    limit = max(1., float(abs(sensitivity).max().max()))
    fig, ax = plt.subplots(figsize=(11, 7.5), layout='constrained')
    image = ax.imshow(sensitivity.to_numpy(dtype=float), aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(8), TOPICS); ax.set_yticks(range(len(sensitivity)), sensitivity.index, fontsize=8)
    ax.set_title('Sensitivity: equal-game conditional topic differences')
    fig.colorbar(image, ax=ax, label='Chinese minus English (percentage points)')
    fig.savefig(figures / 'sensitivity_comparison.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    if not interactions.empty:
        z = interactions.loc[interactions.scenario.eq(primary) & interactions.role.eq('language_interaction') & interactions.status.eq('ok')]
        if not z.empty:
            fig, ax = plt.subplots(figsize=(9, 5.4), layout='constrained'); y = np.arange(len(z))
            v = z.odds_ratio.to_numpy(float); lo = z.ci_low.to_numpy(float); hi = z.ci_high.to_numpy(float)
            ax.errorbar(v, y, xerr=np.vstack([v-lo,hi-v]), fmt='o', capsize=4, color='#945b2b')
            ax.set_xscale('log'); ax.axvline(1, color='#888', lw=1)
            ax.set_yticks(y, [t+' vs T01' for t in z.topic]); ax.invert_yaxis()
            ax.set_xlabel('Ratio of recommendation ORs: Chinese / English\nPer 10-point weight reallocation from T01')
            ax.set_title('Theme × language interactions, game fixed effects')
            fig.savefig(figures / 'topic_language_interactions.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    def fmt(v):
        return '—' if pd.isna(v) else f'{v:.3f}'
    lines = ['# RQ3 中英文来源评论差异：探索性结果', '',
             f"主题版本：`{config['topic_version']}`；原始输入提交：`{config['source_commit']}`。", '',
             f"主样本按 association_eligible 筛选。每款游戏的两种来源语言各需至少 {config['minimum_reviews_per_language']} 条合格评论。"
             f"比较主题构成时，每种语言还需至少 {config['minimum_topic_reviews_per_language']} 条有 `{scope}` 主题证据的评论。",
             '语言指采集来源标签，不代表国籍。主题权重是评论内证据构成，不是概率；无证据不等于没有主题。', '',
             '## 主口径', '',
             '| 结果 | 配对游戏数 | 英文评论数 | 中文评论数 | 中文−英文（百分点） | 95%区间 |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for key, label in [('coverage','主题检出率'),('recommendation','推荐率')]:
        x = main.loc[key]
        lines.append(f"| {label} | {int(x.games)} | {int(x.english_n)} | {int(x.schinese_n)} | {x.difference_pp:.2f} | [{x.ci_low_pp:.2f}, {x.ci_high_pp:.2f}] |")
    lines += ['', '覆盖率／推荐率和主题构成各自按纳入规则选择游戏；其分母不同，不能混成同一个比例。', '',
              '| 主题 | 英文平均权重% | 中文平均权重% | 差异百分点 | 95%区间 | Holm p |', '| --- | ---: | ---: | ---: | --- | ---: |']
    for topic in TOPICS:
        x = themes.loc[topic]
        lines.append(f"| {topic} {NAMES[topic]} | {x.english_mean*100:.2f} | {x.schinese_mean*100:.2f} | {x.difference_pp:.2f} | [{x.ci_low_pp:.2f}, {x.ci_high_pp:.2f}] | {fmt(x.get('p_game_t_holm'))} |")
    lines += ['', '区间为按游戏成对重采样的逐项95%百分位区间；p为配对游戏差异的t检验，八主题在同一设定内作Holm校正。区间本身没有作同时覆盖校正。重采样不包含主题误分的不确定性，也不代表对全Steam随机抽样的推断。', '',
              '## 差异在不同设定下是否保留', '',
              '| 主题 | 核心设定数 | 核心差异范围（百分点） | 方向一致 | 核心Holm p<0.05次数 | 映射诊断差异范围 |',
              '| --- | ---: | --- | --- | ---: | --- |']
    for topic in TOPICS:
        a=next(r for r in robustness if r['topic']==topic and r['family']=='core_sample_and_scope')
        b=next(r for r in robustness if r['topic']==topic and r['family']=='mapping_diagnostics')
        lines.append(f"| {topic} | {a['estimable_scenarios']} | [{a['minimum_difference_pp']:.2f}, {a['maximum_difference_pp']:.2f}] | {'是' if a['same_direction_as_primary'] else '否'} | {a['holm_p_below_005']} | [{b['minimum_difference_pp']:.2f}, {b['maximum_difference_pp']:.2f}] |")
    lines += ['', '核心设定包括 fit、all、固定 fit 样本的 all、双语共同窗口、固定共同窗口游戏后使用原窗口、去重、每作者一条、完整游戏、排除短评、书写系统诊断和更高的样本门槛。映射对照包括排除辅助路由、开放全部候选主题和 K8 重叠代理。', '',
              '## 交互模型', '',
              '控制游戏、语言和log1p发表时游玩小时数；以T01为参照，比较其权重减少10个百分点、对应主题增加10个百分点时的推荐优势变化。语言交互是两种语言OR之比，不是两个分组p值的比较。英文／中文各自OR仅为辅助解释，主对比为七项交互并作Holm校正。']
    for diag in diagnostics['interaction_models']:
        joint = diag.get('joint_interaction_test', {})
        lines.append(f"- `{diag['scenario']}`：{diag['status']}，N={diag.get('n_used',0)}，游戏={diag.get('games_used',0)}；联合交互p={fmt(joint.get('p',np.nan))}。")
    if not interactions.empty:
        z = interactions.loc[interactions.scenario.eq(primary) & interactions.role.eq('language_interaction')]
        lines += ['', '| 主题对照 | 中文/英文 OR之比 | 95%区间 | Holm p | 状态 |', '| --- | ---: | --- | ---: | --- |']
        for _, x in z.iterrows():
            lines.append(f"| {x.topic} vs T01 | {fmt(x.get('odds_ratio'))} | [{fmt(x.get('ci_low'))}, {fmt(x.get('ci_high'))}] | {fmt(x.get('p_holm'))} | {x.status} |")
    lines += ['', '## 敏感性及限制', '',
              '- 所有设定的样本量、游戏支持及区间见 paired_estimates.csv、game_support.csv、sample_flow.csv；review_inclusion.csv 保留逐评论纳入状态，不输出作者标识。',
              '- common_window 采用源数据中的同游戏双语UTC共同日期范围，并重新检查每种语言的样本支持。',
              '- original_same_common_games 在原日期范围中只保留共同窗口满足主题比较要求的游戏；与 common_window 比较时保持游戏组相同。',
              ('- all_same_fit_population 对有 fit 证据的评论改用 all 权重，维持 fit_units 的总体分母、配对游戏和主题评论。'
               if scope == 'all' else
               '- all_same_fit_population 在有 fit 证据的评论中使用 all 权重，其覆盖率只描述该条件样本。'),
              '- no_auxiliary 在单元级排除A01/A02后重新汇总评论权重。',
              '- 高置信度设定只估计有证据且达到配对门槛的主题。',
              '- k8_overlap_proxy 按共享拟合单元的重叠比例继承 K10 候选主题，门槛为 20%，多个父簇取候选并集。',
              '- unrestricted_seeds 使用原种子和阈值，允许全部八个主题参加相似度比较。',
              *(['- 上述两种替代映射仅重判新版保留的历史片段，并沿用新版的评论关联与新增片段标签。']
                if 'alternative_mapping_scope' in diagnostics else []),
              '- script_consistent 只按字符规则排除明显不符／混合书写文本，不是可靠的实际语言识别，尤其不能排除拉丁字母的其他语言。',
              '- one_per_author 在合格评论中保留每位作者最早一条，检查跨游戏重复作者影响；unique_text 使用既定同游戏同语言规范化文本去重。',
              '- 目前主题语义仍有误分，来源语言间覆盖率不同，不能将主题构成差异直接称为玩家真实偏好差异。', '',
              '## 复现', '',
              '在仓库根目录、安装项目 requirements-analysis.txt 的环境中运行：', '',
              '```powershell', f'python -B scripts/analyze_rq3.py --config {config_path} --verify-output',
              f'python -B scripts/analyze_rq3.py --config {config_path} --output output/rq3/reproduced', '```', '',
              'manifest.json 记录输入、生成脚本及输出的 SHA-256。']
    (out / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    e = lambda x: html.escape(str(x))
    image_blocks = []
    for p in sorted(figures.glob('*.png')):
        image_blocks.append('<figure><img alt="'+e(p.stem)+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+e(p.name)+'</figcaption></figure>')
    table = themes.reset_index()[['measure','games','english_n','schinese_n','difference_pp','ci_low_pp','ci_high_pp','p_game_t_holm']].to_html(index=False, float_format=lambda v:f'{v:.3f}', border=0)
    interaction_table = interactions.loc[interactions.role.eq('language_interaction')].to_html(index=False, float_format=lambda v:f'{v:.4f}', border=0) if not interactions.empty else '<p>没有可估计的交互模型。</p>'
    headline = f"覆盖率差异 {main.loc['coverage','difference_pp']:.2f} 个百分点；推荐率差异 {main.loc['recommendation','difference_pp']:.2f} 个百分点（中文−英文，游戏等权）。"
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>RQ3 中英文差异探索性分析</title><style>body{margin:0;background:#f2f5f8;color:#182d42;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:24px}section,figure{background:white;border:1px solid #d8e1e8;border-radius:8px;padding:20px;margin:18px 0}img{max-width:100%;height:auto}figure{margin:18px 0}figcaption{color:#53667a;font-size:13px}.note{background:#fff3d0;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}th,td{padding:8px;border-bottom:1px solid #d8e1e8;text-align:right}th:first-child,td:first-child{text-align:left}summary{cursor:pointer}a{color:#195aa5}@media(max-width:650px){main{padding:10px}section,figure{padding:12px}h1{font-size:25px}}</style><main><h1>RQ3：中英文来源评论差异</h1><p>'''+e(config['topic_version'])+'</p><p>'+e(headline)+'''</p><p class="note">探索性结果。语言不是国籍；主题未检出不等于主题不存在。主题构成只在有证据样本中比较。区间没有包含主题误分的不确定性。请同时查看覆盖率和敏感性结果。</p><section><h2>主主题比较</h2><p>中文−英文，游戏等权。区间为逐项95%按游戏成对bootstrap；p值按八主题Holm校正。</p><div class="scroll">'''+table+'</div></section>'+''.join(image_blocks)+'<section><details><summary>查看交互模型明细</summary><div class="scroll">'+interaction_table+'</div></details></section><section><h2>说明与原始表</h2><ul>'+''.join('<li><a href="'+p+'">'+p+'</a></li>' for p in ['README.md','paired_estimates.csv','robustness_summary.csv','source_coverage.csv','game_language_differences.csv','game_support.csv','sample_flow.csv','interaction_effects.csv','model_diagnostics.csv','diagnostics.json','manifest.json'])+'</ul></section></main></html>'
    (out / 'report.html').write_text(page, encoding='utf-8')
