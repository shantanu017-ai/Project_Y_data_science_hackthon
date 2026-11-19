"""Retention exposure analysis: does a bad first 30 days predict churn?

Uses the rider feature table built by retention_model.py. To limit reverse
causality (early quitters necessarily have few attempts), the main model is
restricted to riders with >= 8 attempts in their first 30 days.
"""
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "S:/project_Y/outputs"
df = pd.read_parquet("S:/project_Y/data/clean/rider_features.parquet")
df["cohort_year"] = df.cohort_month.str[:4]
df["fail_pct"] = 100 * df.fail30_rate

bins = [-0.01, 0.0001, 2, 5, 10, 100]
labels = ["0%", "0-2%", "2-5%", "5-10%", ">10%"]
df["fail_bucket"] = pd.cut(df.fail_pct, bins=bins, labels=labels)

rows = []
for min_att in (1, 5, 8, 15):
    sub = df[(df.cohort_month <= "2025-03") & (df.att30 >= min_att)]
    for bucket, g in sub.groupby("fail_bucket", observed=True):
        rows.append({
            "min_attempts_30d": min_att,
            "fail_bucket": bucket,
            "riders": len(g),
            "pct_riders": round(100 * len(g) / len(sub), 1),
            "mean_fail_pct": round(g.fail_pct.mean(), 1),
            "d60_retention_pct": round(100 * g.ret60.mean(), 1),
            "d90_retention_pct": round(100 * g.ret90.mean(), 1),
        })
exposure = pd.DataFrame(rows)
exposure.to_csv(f"{OUT}/retention_exposure.csv", index=False)

# split by year to show the pattern repeats
rows2 = []
for (year, bucket), g in df[(df.att30 >= 8) & (df.cohort_month <= "2025-03")].groupby(["cohort_year", "fail_bucket"], observed=True):
    rows2.append({"cohort_year": year, "fail_bucket": bucket, "riders": len(g),
                  "mean_fail_pct": round(g.fail_pct.mean(), 1),
                  "d60_retention_pct": round(100 * g.ret60.mean(), 1)})
pd.DataFrame(rows2).to_csv(f"{OUT}/retention_exposure_by_year.csv", index=False)

# monthly cohort retention
cohort = (df[df.att30 >= 1].groupby("cohort_month")
          .agg(riders=("rider_id", "size"), d30=("ret60", "size")))
cohort = (df.groupby("cohort_month").agg(riders=("rider_id", "size"),
                                         d60=("ret60", "mean"), d90=("ret90", "mean")))
cohort["d60_pct"] = (100 * cohort.d60).round(1)
cohort["d90_pct"] = (100 * cohort.d90).round(1)
cohort[["riders", "d60_pct", "d90_pct"]].to_csv(f"{OUT}/retention_cohorts.csv")

# segment retention (riders with >= 8 attempts)
sub = df[(df.att30 >= 8) & (df.cohort_month <= "2025-03")]
seg = (sub.groupby(["vehicle_class", "is_partner"])
       .agg(riders=("rider_id", "size"), d60=("ret60", "mean"), d90=("ret90", "mean")))
seg["d60_pct"] = (100 * seg.d60).round(1)
seg["d90_pct"] = (100 * seg.d90).round(1)
seg.reset_index().to_csv(f"{OUT}/retention_segments.csv", index=False)

# experience-only model, avoiding raw usage counts as predictors
m = sub.copy()
m["log_wait"] = np.log1p(m.wait30)
m["fail_pp"] = m.fail_pct
m["nostock_pp"] = 100 * m.nostock30_rate
m["abandon_pp"] = 100 * m.abandon30_rate
m["syserr_pp"] = 100 * m.syserr30_rate
core = ["fail_pp", "log_wait", "soh30", "peakshare30", "is_3w", "is_partner",
        "kyc_verified", "signup_to_first", "plan_type", "signup_channel",
        "home_city", "first_type"]
X = pd.get_dummies(m[core], columns=["plan_type", "signup_channel", "home_city", "first_type"],
                   drop_first=True).astype(float)
y = m.ret60.astype(int)
pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                     LogisticRegression(max_iter=3000, C=0.5))
pipe.fit(X, y)
clf = pipe.named_steps["logisticregression"]
auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
i = list(X.columns).index("fail_pp")
coefs = pd.DataFrame({"feature": X.columns, "coef": clf.coef_[0],
                      "odds_ratio_per_sd": np.exp(clf.coef_[0])})
coefs["odds_ratio_per_10pp"] = np.where(coefs.feature == "fail_pp",
                                        np.exp(clf.coef_[0] * 10), np.nan)

# bootstrap the per-10pp odds ratio
rng = np.random.default_rng(42)
ors = []
for _ in range(100):
    idx = rng.integers(0, len(X), len(X))
    p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=3000, C=0.5))
    p.fit(X.iloc[idx], y.iloc[idx])
    b = p.named_steps["logisticregression"].coef_[0]
    j = list(X.columns).index("fail_pp")
    ors.append(np.exp(b[j] * 10))
ci_lo, ci_hi = np.percentile(ors, [2.5, 97.5])
coefs.sort_values("coef", key=abs, ascending=False).to_csv(f"{OUT}/retention_model_coefficients.csv", index=False)

summary = {
    "sample_riders": int(len(m)),
    "cohorts": "signup through 2025-03",
    "restriction": ">=8 attempts in first 30 days",
    "auc_d60": round(float(auc), 3),
    "base_d60_pct": round(100 * float(y.mean()), 1),
    "fail10pp_odds_ratio": round(float(np.exp(clf.coef_[0][i] * 10)), 3),
    "fail10pp_or_ci95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
    "exposure_table": exposure[exposure.min_attempts_30d == 8].to_dict(orient="records"),
    "segments": seg.reset_index().round(3).to_dict(orient="records"),
}
with open(f"{OUT}/retention_model.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(exposure[exposure.min_attempts_30d == 8].to_string(index=False))
print("\nexperience-only model AUC:", round(auc, 3), "| n:", len(m))
print(coefs.head(8).to_string(index=False))
print("\nfail_pp OR per +10pp:", round(float(np.exp(clf.coef_[0][i] * 10)), 3), "CI", round(float(ci_lo), 3), round(float(ci_hi), 3))
print("\nsegments:\n", seg.reset_index().to_string(index=False))
print("\ncohorts tail:\n", cohort.tail(8).to_string())
