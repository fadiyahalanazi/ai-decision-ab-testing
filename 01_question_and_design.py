from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
CONFIG = {
    "project": "AI Recommendations for Business Decision Accuracy",

    "baseline_conversion": 0.60,
    "mde_abs": 0.05,
    "alpha": 0.05,
    "power": 0.80,
    "daily_new_users": 200,
    "treatment_share": 0.5,

    "true_lift_abs": 0.05,

    "order_value_mean": 120,
    "order_value_sd": 50,
    "order_value_effect": 0,
    "margin": 0.25,
    "extra_cost_per_order_b": 0,

    "cancel_rate": 0.05,
    "cancel_effect": 0.000,
    "support_rate": 0.03,
    "support_effect": 0.000,

    "activity_strength": 0.8,
    "novelty_boost": 0.0,
    "srm_bug_share": 0.0,
    "seed": 2026,
}

DESIGN = {
    "title_ar": "أثر توصيات الذكاء الاصطناعي على دقة قرارات الأعمال",
    "question": "هل تؤدي توصيات الذكاء الاصطناعي إلى تحسين دقة قرارات الأعمال؟",
    "hypothesis": "نتوقع تحسن دقة القرارات عند عرض توصيات AI بجانب لوحة البيانات.",
    "population": "مستخدمو منصة افتراضية لدعم قرارات الأعمال",
    "randomization_unit": "user",
    "allocation": {"A": 0.5, "B": 0.5},
    "control": "لوحة بيانات دون توصيات AI",
    "treatment": "لوحة بيانات مع توصيات AI",
    "primary_metric": "correct_decision",
    "secondary_metrics": ["decision_time", "satisfaction"],
    "guardrails": ["high_impact_error"],
    "data_source": "synthetic simulation; no real users",
}

def main():
    out = ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_config.json").write_text(json.dumps({"config": CONFIG, "design": DESIGN}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("01 | Experiment design saved")

if __name__ == "__main__":
    main()
