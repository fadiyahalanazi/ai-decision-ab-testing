"""AI recommendations A/B experiment: design, simulation, analysis and decision.
Run: python main.py
All data are synthetic and do not represent real users.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
CONFIG = {'project': 'AI Recommendations for Business Decision Accuracy', 'baseline_conversion': 0.6, 'mde_abs': 0.05, 'alpha': 0.05, 'power': 0.8, 'daily_new_users': 200, 'treatment_share': 0.5, 'true_lift_abs': 0.05, 'order_value_mean': 120, 'order_value_sd': 50, 'order_value_effect': 0, 'margin': 0.25, 'extra_cost_per_order_b': 0, 'cancel_rate': 0.05, 'cancel_effect': 0.0, 'support_rate': 0.03, 'support_effect': 0.0, 'activity_strength': 0.8, 'novelty_boost': 0.0, 'srm_bug_share': 0.0, 'seed': 2026}

DESIGN = {'title_ar': 'أثر توصيات الذكاء الاصطناعي على دقة قرارات الأعمال', 'question': 'هل تؤدي توصيات الذكاء الاصطناعي إلى تحسين دقة قرارات الأعمال؟', 'hypothesis': 'نتوقع تحسن دقة القرارات عند عرض توصيات AI بجانب لوحة البيانات.', 'population': 'مستخدمو منصة افتراضية لدعم قرارات الأعمال', 'randomization_unit': 'user', 'allocation': {'A': 0.5, 'B': 0.5}, 'control': 'لوحة بيانات دون توصيات AI', 'treatment': 'لوحة بيانات مع توصيات AI', 'primary_metric': 'correct_decision', 'secondary_metrics': ['decision_time', 'satisfaction'], 'guardrails': ['high_impact_error'], 'data_source': 'synthetic simulation; no real users'}

def stage_1():
    out = ROOT / 'outputs' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'experiment_config.json').write_text(json.dumps({'config': CONFIG, 'design': DESIGN}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('01 | Experiment design saved')

def n_per_group_proportions(p_control, p_treatment, alpha=0.05, power=0.8):
    z_a, z_b = (stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power))
    var = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p_treatment - p_control) ** 2))

def mde_proportions(p_control, n_per_group, alpha=0.05, power=0.8):
    z_a, z_b = (stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power))
    return (z_a + z_b) * np.sqrt(2 * p_control * (1 - p_control) / n_per_group)

def duration_days(n_per_group, daily_new_users, groups=2, round_to_weeks=True):
    days = int(np.ceil(n_per_group * groups / daily_new_users))
    return int(np.ceil(days / 7) * 7) if round_to_weeks else days

def stage_2():
    path = ROOT / 'outputs' / 'results'
    cfg = json.loads((path / 'experiment_config.json').read_text(encoding='utf-8'))['config']
    per_group = n_per_group_proportions(cfg['baseline_conversion'], cfg['baseline_conversion'] + cfg['mde_abs'], cfg['alpha'], cfg['power'])
    days = duration_days(per_group, cfg['daily_new_users'])
    planned = days * cfg['daily_new_users']
    results = {'n_per_group_required': per_group, 'n_total_required': 2 * per_group, 'duration_days': days, 'planned_users': planned, 'planned_per_group': planned // 2, 'mde_planned': float(mde_proportions(cfg['baseline_conversion'], planned // 2, cfg['alpha'], cfg['power']))}
    (path / 'sample_size.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('02 | Sample size and duration:', results)

matplotlib.use('Agg')

ANALYSIS_PLAN = {'srm': 'Chi-square goodness of fit, p < 0.001 flags SRM', 'primary': 'B minus A difference, normal approximation 95% CI, two-sided p-value', 'cuped': 'Pre-period decisions covariate, variance reduction', 'secondary': 'Two-sided tests with Benjamini-Hochberg correction', 'guardrail': 'High-impact error; flag statistically significant increase', 'novelty': 'Plot daily difference without interim decision-making', 'analysis_timing': 'After the planned 21 days'}

def diff_ci(a, b, alpha=0.05):
    a, b = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return {'الفرق': float(diff), 'أدنى 95%': float(diff - z * se), 'أعلى 95%': float(diff + z * se), 'SE': float(se), 'p': float(2 * stats.norm.sf(abs(diff / se)))}

def simulate_experiment(cfg, duration_days):
    """محاكاة أثر توصيات AI على دقة قرارات الأعمال."""
    r = np.random.default_rng(cfg['seed'])
    n = duration_days * cfg['daily_new_users']
    is_b = (r.random(n) < cfg['treatment_share']).astype(int)
    day = r.integers(1, duration_days + 1, n)
    experience = r.normal(0, 1, n)
    base = cfg['baseline_conversion']
    logit = np.log(base / (1 - base)) + cfg['activity_strength'] * experience
    p0 = 1 / (1 + np.exp(-logit))
    p0 = np.clip(p0 * base / p0.mean(), 0, 1)
    pre_decisions_90d = r.poisson(5 * p0 / base)
    lift = cfg['true_lift_abs'] * (1 + cfg['novelty_boost'] * np.exp(-(day - 1) / 3))
    correct_decision = r.binomial(1, np.clip(p0 + lift * is_b, 0, 1))
    decision_time = np.maximum(10, r.normal(90 - 15 * is_b, 20, n))
    satisfaction = np.clip(np.rint(r.normal(3.5 + 0.3 * is_b, 0.8, n)), 1, 5).astype(int)
    high_impact_error = ((correct_decision == 0) & (r.random(n) < 0.2)).astype(int)
    df = pd.DataFrame({'variant': np.where(is_b == 1, 'B', 'A'), 'day': day, 'pre_decisions_90d': pre_decisions_90d, 'correct_decision': correct_decision, 'decision_time': decision_time, 'satisfaction': satisfaction, 'high_impact_error': high_impact_error})
    if cfg['srm_bug_share'] > 0:
        lost = (df['variant'] == 'B') & (experience < 0) & (r.random(n) < cfg['srm_bug_share'] * 2)
        df = df[~lost]
    return df.reset_index(drop=True)

def stage_3():
    out = ROOT / 'outputs' / 'results'
    charts = ROOT / 'outputs' / 'charts'
    charts.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((out / 'experiment_config.json').read_text(encoding='utf-8'))['config']
    duration = json.loads((out / 'sample_size.json').read_text(encoding='utf-8'))['duration_days']
    data = simulate_experiment(cfg, duration)
    (ROOT / 'data').mkdir(exist_ok=True)
    data.to_csv(ROOT / 'data' / 'experiment_data.csv', index=False)
    counts = data['variant'].value_counts()
    na, nb = (int(counts.get('A', 0)), int(counts.get('B', 0)))
    _, srm_p = stats.chisquare([na, nb], f_exp=[(na + nb) * (1 - cfg['treatment_share']), (na + nb) * cfg['treatment_share']])
    A, B = (data[data.variant == 'A'], data[data.variant == 'B'])
    primary = diff_ci(A['correct_decision'], B['correct_decision'])
    theta = np.cov(data['correct_decision'], data['pre_decisions_90d'])[0, 1] / data['pre_decisions_90d'].var(ddof=1)
    adjusted = data['correct_decision'] - theta * (data['pre_decisions_90d'] - data['pre_decisions_90d'].mean())
    cuped = diff_ci(adjusted[data.variant == 'A'], adjusted[data.variant == 'B'])
    secondary = {name: diff_ci(A[col], B[col]) for name, col in [('decision_time', 'decision_time'), ('satisfaction', 'satisfaction')]}
    bh = multipletests([secondary[k]['p'] for k in secondary], alpha=cfg['alpha'], method='fdr_bh')[0]
    for k, flag in zip(secondary, bh):
        secondary[k]['significant_after_bh'] = bool(flag)
    guard = diff_ci(A['high_impact_error'], B['high_impact_error'])
    guard['status'] = 'significant_harm' if guard['أدنى 95%'] > 0 else 'no_significant_harm_detected'
    daily = data.groupby(['day', 'variant'])['correct_decision'].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(daily.index, (daily['B'] - daily['A']) * 100, marker='o', color='#0E7490')
    ax.axhline(primary['الفرق'] * 100, linestyle='--', color='#E58A1F')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set(xlabel='Day', ylabel='Effect (percentage points)', title='Daily treatment effect (novelty check)')
    fig.tight_layout()
    fig.savefig(charts / 'novelty_effect.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(['A', 'B'], [A['correct_decision'].mean() * 100, B['correct_decision'].mean() * 100], color=['#0E7490', '#E58A1F'])
    ax.set(ylabel='Correct decisions (%)', title='Decision accuracy by group')
    fig.tight_layout()
    fig.savefig(charts / 'decision_accuracy.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, res in enumerate([primary, cuped]):
        ax.errorbar(res['الفرق'] * 100, i, xerr=[[(res['الفرق'] - res['أدنى 95%']) * 100], [(res['أعلى 95%'] - res['الفرق']) * 100]], fmt='o', capsize=5)
    ax.set(yticks=[0, 1], yticklabels=['Raw', 'CUPED'], xlabel='Effect (percentage points)', title='95% confidence intervals')
    ax.axvline(0, color='gray', linestyle='--')
    fig.tight_layout()
    fig.savefig(charts / 'confidence_intervals.png', dpi=160)
    plt.close(fig)
    results = {'analysis_plan': ANALYSIS_PLAN, 'n': len(data), 'counts': {'A': na, 'B': nb}, 'srm_p': float(srm_p), 'srm_ok': bool(srm_p >= 0.001), 'duplicates': int(data.duplicated().sum()), 'missing_values': {k: int(v) for k, v in data.isna().sum().items()}, 'accuracy': {'A': float(A['correct_decision'].mean()), 'B': float(B['correct_decision'].mean())}, 'primary': primary, 'cuped': cuped, 'variance_reduction': float(1 - (cuped['SE'] / primary['SE']) ** 2), 'secondary': secondary, 'guardrail': {'A': float(A['high_impact_error'].mean()), 'B': float(B['high_impact_error'].mean()), **guard}}
    (out / 'analysis_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('03 | Statistical analysis complete', 'A', f"{results['accuracy']['A']:.2%}", 'B', f"{results['accuracy']['B']:.2%}", 'p', f"{primary['p']:.4f}")

def decide(ci_low, ci_high, mde, guardrails_ok=True):
    if ci_high < 0:
        return 'لا نطلق، لأن النسخة الجديدة أضرت بالمقياس الرئيسي.'
    if ci_low > 0 and (not guardrails_ok):
        return 'لا نطلق الآن، لأن المقياس الرئيسي تحسن لكن أحد مقاييس الحماية ساء.'
    if ci_low > 0:
        return 'نطلق، لأن الأثر إيجابي بثقة ومقاييس الحماية سليمة.'
    if ci_high < mde:
        return 'لا نطلق، لأننا استبعدنا وجود أثر بالحجم الذي يهمنا.'
    return 'النتيجة غير حاسمة؛ نعيد تصميم التجربة أو نجمع أدلة إضافية.'

def stage_4():
    out = ROOT / "outputs" / "results"
    cfg = json.loads((out / "experiment_config.json").read_text(encoding="utf-8"))["config"]
    results = json.loads((out / "analysis_results.json").read_text(encoding="utf-8"))
    primary = results["primary"]
    guardrails_ok = results["guardrail"]["status"] != "significant_harm"
    decision = decide(primary["أدنى 95%"], primary["أعلى 95%"], cfg["mde_abs"], guardrails_ok)
    (out / "final_decision.json").write_text(
        json.dumps({"decision": decision, "note": "Synthetic simulation only; not real user evidence"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("04 | Final decision saved")


def main():
    stage_1()
    stage_2()
    stage_3()
    stage_4()
    print("All four stages completed successfully")


if __name__ == "__main__":
    main()
