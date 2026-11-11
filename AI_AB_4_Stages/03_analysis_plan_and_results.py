from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

ANALYSIS_PLAN = {
    "srm": "Chi-square goodness of fit, p < 0.001 flags SRM",
    "primary": "B minus A difference, normal approximation 95% CI, two-sided p-value",
    "cuped": "Pre-period decisions covariate, variance reduction",
    "secondary": "Two-sided tests with Benjamini-Hochberg correction",
    "guardrail": "High-impact error; flag statistically significant increase",
    "novelty": "Plot daily difference without interim decision-making",
    "analysis_timing": "After the planned 21 days",
}

def diff_ci(a, b, alpha=0.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"الفرق": float(diff), "أدنى 95%": float(diff - z * se), "أعلى 95%": float(diff + z * se), "SE": float(se), "p": float(2 * stats.norm.sf(abs(diff / se)))}

def simulate_experiment(cfg, duration_days):
    """محاكاة أثر توصيات AI على دقة قرارات الأعمال."""

    r = np.random.default_rng(cfg["seed"])
    n = duration_days * cfg["daily_new_users"]

    # توزيع المستخدمين بين A و B
    is_b = (
        r.random(n) < cfg["treatment_share"]
    ).astype(int)

    day = r.integers(1, duration_days + 1, n)

    # اختلاف مستوى خبرة المستخدمين
    experience = r.normal(0, 1, n)

    base = cfg["baseline_conversion"]
    logit = (
        np.log(base / (1 - base))
        + cfg["activity_strength"] * experience
    )

    p0 = 1 / (1 + np.exp(-logit))
    p0 = np.clip(p0 * base / p0.mean(), 0, 1)

    # عدد القرارات السابقة كمؤشر للخبرة
    pre_decisions_90d = r.poisson(
        5 * p0 / base
    )

    # أثر توصيات الذكاء الاصطناعي
    lift = cfg["true_lift_abs"] * (
        1 + cfg["novelty_boost"]
        * np.exp(-(day - 1) / 3)
    )

    # هل القرار صحيح؟
    correct_decision = r.binomial(
        1, np.clip(p0 + lift * is_b, 0, 1)
    )

    # زمن اتخاذ القرار بالثواني
    decision_time = np.maximum(
        10, r.normal(90 - 15 * is_b, 20, n)
    )

    # رضا المستخدم: من 1 إلى 5
    satisfaction = np.clip(
        np.rint(r.normal(
            3.5 + 0.3 * is_b, 0.8, n
        )),
        1, 5
    ).astype(int)

    # قرارات خاطئة عالية الأثر
    high_impact_error = (
        (correct_decision == 0)
        & (r.random(n) < 0.20)
    ).astype(int)

    df = pd.DataFrame({
        "variant": np.where(is_b == 1, "B", "A"),
        "day": day,
        "pre_decisions_90d": pre_decisions_90d,
        "correct_decision": correct_decision,
        "decision_time": decision_time,
        "satisfaction": satisfaction,
        "high_impact_error": high_impact_error
    })

    # محاكاة خلل محتمل في توزيع العينة
    if cfg["srm_bug_share"] > 0:
        lost = (
            (df["variant"] == "B")
            & (experience < 0)
            & (r.random(n) < cfg["srm_bug_share"] * 2)
        )
        df = df[~lost]

    return df.reset_index(drop=True)



def main():
    out = ROOT / "outputs" / "results"
    charts = ROOT / "outputs" / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((out / "experiment_config.json").read_text(encoding="utf-8"))["config"]
    duration = json.loads((out / "sample_size.json").read_text(encoding="utf-8"))["duration_days"]
    data = simulate_experiment(cfg, duration)
    (ROOT / "data").mkdir(exist_ok=True)
    data.to_csv(ROOT / "data" / "experiment_data.csv", index=False)
    counts = data["variant"].value_counts()
    na, nb = int(counts.get("A", 0)), int(counts.get("B", 0))
    _, srm_p = stats.chisquare([na, nb], f_exp=[(na + nb) * (1 - cfg["treatment_share"]), (na + nb) * cfg["treatment_share"]])
    A, B = data[data.variant == "A"], data[data.variant == "B"]
    primary = diff_ci(A["correct_decision"], B["correct_decision"])
    theta = np.cov(data["correct_decision"], data["pre_decisions_90d"])[0, 1] / data["pre_decisions_90d"].var(ddof=1)
    adjusted = data["correct_decision"] - theta * (data["pre_decisions_90d"] - data["pre_decisions_90d"].mean())
    cuped = diff_ci(adjusted[data.variant == "A"], adjusted[data.variant == "B"])
    secondary = {name: diff_ci(A[col], B[col]) for name, col in [("decision_time", "decision_time"), ("satisfaction", "satisfaction")]}
    bh = multipletests([secondary[k]["p"] for k in secondary], alpha=cfg["alpha"], method="fdr_bh")[0]
    for k, flag in zip(secondary, bh): secondary[k]["significant_after_bh"] = bool(flag)
    guard = diff_ci(A["high_impact_error"], B["high_impact_error"])
    guard["status"] = "significant_harm" if guard["أدنى 95%"] > 0 else "no_significant_harm_detected"
    daily = data.groupby(["day", "variant"])["correct_decision"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(daily.index, (daily["B"] - daily["A"]) * 100, marker="o", color="#0E7490")
    ax.axhline(primary["الفرق"] * 100, linestyle="--", color="#E58A1F")
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Day", ylabel="Effect (percentage points)", title="Daily treatment effect (novelty check)")
    fig.tight_layout(); fig.savefig(charts / "novelty_effect.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["A", "B"], [A["correct_decision"].mean() * 100, B["correct_decision"].mean() * 100], color=["#0E7490", "#E58A1F"])
    ax.set(ylabel="Correct decisions (%)", title="Decision accuracy by group")
    fig.tight_layout(); fig.savefig(charts / "decision_accuracy.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, res in enumerate([primary, cuped]):
        ax.errorbar(res["الفرق"] * 100, i, xerr=[[ (res["الفرق"] - res["أدنى 95%"]) * 100 ],[(res["أعلى 95%"] - res["الفرق"]) * 100]], fmt="o", capsize=5)
    ax.set(yticks=[0, 1], yticklabels=["Raw", "CUPED"], xlabel="Effect (percentage points)", title="95% confidence intervals")
    ax.axvline(0, color="gray", linestyle="--"); fig.tight_layout(); fig.savefig(charts / "confidence_intervals.png", dpi=160); plt.close(fig)
    results = {"analysis_plan": ANALYSIS_PLAN, "n": len(data), "counts": {"A": na, "B": nb}, "srm_p": float(srm_p), "srm_ok": bool(srm_p >= 0.001), "duplicates": int(data.duplicated().sum()), "missing_values": {k: int(v) for k, v in data.isna().sum().items()}, "accuracy": {"A": float(A["correct_decision"].mean()), "B": float(B["correct_decision"].mean())}, "primary": primary, "cuped": cuped, "variance_reduction": float(1 - (cuped["SE"] / primary["SE"]) ** 2), "secondary": secondary, "guardrail": {"A": float(A["high_impact_error"].mean()), "B": float(B["high_impact_error"].mean()), **guard}}
    (out / "analysis_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("03 | Statistical analysis complete", "A", f'{results["accuracy"]["A"]:.2%}', "B", f'{results["accuracy"]["B"]:.2%}', "p", f'{primary["p"]:.4f}')

if __name__ == "__main__":
    main()
