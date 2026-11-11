from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
import math
from scipy import stats
import numpy as np

def n_per_group_proportions(p_control, p_treatment, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p_treatment - p_control) ** 2))

def mde_proportions(p_control, n_per_group, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (z_a + z_b) * np.sqrt(2 * p_control * (1 - p_control) / n_per_group)

def duration_days(n_per_group, daily_new_users, groups=2, round_to_weeks=True):
    days = int(np.ceil(n_per_group * groups / daily_new_users))
    return int(np.ceil(days / 7) * 7) if round_to_weeks else days

def main():
    path = ROOT / "outputs" / "results"
    cfg = json.loads((path / "experiment_config.json").read_text(encoding="utf-8"))["config"]
    per_group = n_per_group_proportions(cfg["baseline_conversion"], cfg["baseline_conversion"] + cfg["mde_abs"], cfg["alpha"], cfg["power"])
    days = duration_days(per_group, cfg["daily_new_users"])
    planned = days * cfg["daily_new_users"]
    results = {"n_per_group_required": per_group, "n_total_required": 2 * per_group, "duration_days": days, "planned_users": planned, "planned_per_group": planned // 2, "mde_planned": float(mde_proportions(cfg["baseline_conversion"], planned // 2, cfg["alpha"], cfg["power"]))}
    (path / "sample_size.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("02 | Sample size and duration:", results)

if __name__ == "__main__":
    main()
