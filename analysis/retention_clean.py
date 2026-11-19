"""Clean retention analysis with an unambiguous time order.

failure exposure  : days 0-21 after first event
still active      : completed swap in days 22-30 (sample condition)
retention (D60)   : completed swap in days 31-60

Conditioning on late-month activity removes the mechanical link between
early usage volume and the retention window.
"""
import json
import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "S:/project_Y/outputs"
P = "S:/project_Y/data/clean/events.parquet"
R = "S:/project_Y/data/clean/riders.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='1.4GB'; SET threads=2; SET temp_directory='S:/project_Y/tmp';")

sql = f"""
WITH fe AS (
  SELECT rider_id, min(ts_corrected) first_ts
  FROM read_parquet('{P}') WHERE NOT is_test_station GROUP BY 1
),
early AS (
  SELECT e.rider_id,
    count(*) att21,
    sum(CASE WHEN NOT e.is_completed THEN 1 ELSE 0 END) fail21,
    sum(CASE WHEN e.event_type='failed_no_charged_battery' THEN 1 ELSE 0 END) nostock21,
    sum(CASE WHEN e.event_type='abandoned_queue' THEN 1 ELSE 0 END) abandon21,
    avg(CASE WHEN e.is_completed THEN e.queue_wait_sec END) wait21,
    avg(e.soh_out_pct) soh21,
    avg(CASE WHEN e.tariff_code='PEAK' THEN 1.0 ELSE 0 END) peak21
  FROM read_parquet('{P}') e JOIN fe USING(rider_id)
  WHERE NOT e.is_test_station
    AND e.ts_corrected > fe.first_ts
    AND e.ts_corrected <= fe.first_ts + INTERVAL 21 DAY
  GROUP BY 1
),
active AS (
  SELECT fe.rider_id,
    max(CASE WHEN e.is_completed AND e.ts_corrected > fe.first_ts + INTERVAL 21 DAY
             AND e.ts_corrected <= fe.first_ts + INTERVAL 30 DAY THEN 1 ELSE 0 END) active_22_30
  FROM fe JOIN read_parquet('{P}') e ON e.rider_id=fe.rider_id
  WHERE NOT e.is_test_station GROUP BY 1
),
ret AS (
  SELECT fe.rider_id,
    max(CASE WHEN e.is_completed AND e.ts_corrected > fe.first_ts + INTERVAL 30 DAY
             AND e.ts_corrected <= fe.first_ts + INTERVAL 60 DAY THEN 1 ELSE 0 END) ret60,
    max(CASE WHEN e.is_completed AND e.ts_corrected > fe.first_ts + INTERVAL 30 DAY
             AND e.ts_corrected <= fe.first_ts + INTERVAL 90 DAY THEN 1 ELSE 0 END) ret90
  FROM fe JOIN read_parquet('{P}') e ON e.rider_id=fe.rider_id
  WHERE NOT is_test_station GROUP BY 1
),
firstev AS (
  SELECT e.rider_id, e.event_type first_type, e.city first_city
  FROM read_parquet('{P}') e JOIN fe USING(rider_id)
  WHERE NOT e.is_test_station AND e.ts_corrected=fe.first_ts
  QUALIFY row_number() OVER (PARTITION BY e.rider_id ORDER BY e.event_id)=1
)
SELECT r.rider_id, r.vehicle_class, r.plan_type, r.signup_channel, r.kyc_verified,
       r.home_city_norm home_city, r.partner_id IS NOT NULL is_partner,
       date_diff('day', r.signup_date, fe.first_ts) signup_to_first,
       strftime(fe.first_ts,'%Y-%m') cohort_month, fe.first_ts,
       fv.first_type, fv.first_city,
       e.*, a.active_22_30, t.ret60, t.ret90
FROM read_parquet('{R}') r
JOIN fe USING(rider_id) JOIN early e USING(rider_id)
JOIN active a USING(rider_id) JOIN ret t USING(rider_id)
JOIN firstev fv USING(rider_id)
WHERE fe.first_ts <= DATE '2025-03-31'
"""
df = con.execute(sql).fetchdf()
df["fail_pct"] = 100 * df.fail21 / df.att21
df["is_3w"] = (df.vehicle_class == "3W").astype(int)
df["log_att"] = np.log1p(df.att21)
df["log_wait"] = np.log1p(df.wait21)
df["cohort_year"] = df.cohort_month.str[:4]

print("all riders with >=1 early attempt:", len(df))
print("active in days 22-30:", int(df.active_22_30.sum()),
      f"({100*df.active_22_30.mean():.1f}%)")
s = df[df.active_22_30 == 1].copy()
print("D60 retention in sample:", round(100 * s.ret60.mean(), 1), "| D90:", round(100 * s.ret90.mean(), 1))

bins = [-0.01, 0.0001, 5, 10, 20, 100]
labels = ["0%", "0-5%", "5-10%", "10-20%", ">20%"]
s["fail_bucket"] = pd.cut(s.fail_pct, bins=bins, labels=labels)
tab = (s.groupby("fail_bucket", observed=True)
       .agg(riders=("rider_id", "size"), mean_fail_pct=("fail_pct", "mean"),
            d60=("ret60", "mean"), d90=("ret90", "mean")))
tab["share_pct"] = (100 * tab.riders / len(s)).round(1)
tab["mean_fail_pct"] = tab.mean_fail_pct.round(1)
tab["d60_pct"] = (100 * tab.d60).round(1)
tab["d90_pct"] = (100 * tab.d90).round(1)
tab[["riders", "share_pct", "mean_fail_pct", "d60_pct", "d90_pct"]].to_csv(f"{OUT}/retention_exposure.csv")
print("\nexposure table (clean):\n", tab[["riders","share_pct","mean_fail_pct","d60_pct","d90_pct"]].to_string())

rows = []
for (year, bucket), g in s.groupby(["cohort_year", "fail_bucket"], observed=True):
    rows.append({"cohort_year": year, "fail_bucket": bucket, "riders": len(g),
                 "d60_retention_pct": round(100 * g.ret60.mean(), 1)})
pd.DataFrame(rows).to_csv(f"{OUT}/retention_exposure_by_year.csv", index=False)

# by usage band (show the effect is not a volume artifact)
band = pd.cut(s.att21, [0, 9, 19, 39, 10000], labels=["4-9", "10-19", "20-39", "40+"])
piv = (s.groupby([band, "fail_bucket"], observed=True)
       .agg(n=("rider_id", "size"), d60=("ret60", "mean")))
piv["d60_pct"] = (100 * piv.d60).round(1)
piv[["n", "d60_pct"]].to_csv(f"{OUT}/retention_exposure_by_usage.csv")
print("\nby usage band:\n", piv[["n","d60_pct"]].to_string())

# cohort trend (all riders, not only active)
cohort = df.groupby("cohort_month").agg(riders=("rider_id","size"),
                                        d60=("ret60","mean"), d90=("ret90","mean"))
cohort["d60_pct"] = (100 * cohort.d60).round(1)
cohort["d90_pct"] = (100 * cohort.d90).round(1)
cohort[["riders","d60_pct","d90_pct"]].to_csv(f"{OUT}/retention_cohorts.csv")

seg = (s.groupby(["vehicle_class", "is_partner"])
       .agg(riders=("rider_id","size"), d60=("ret60","mean")))
seg["d60_pct"] = (100 * seg.d60).round(1)
seg.reset_index()[["vehicle_class","is_partner","riders","d60_pct"]].to_csv(f"{OUT}/retention_segments.csv", index=False)

# model: failure exposure in days 1-21 -> D60, controlling for usage volume
core = ["fail_pct", "log_att", "log_wait", "soh21", "peak21", "is_3w", "is_partner",
        "kyc_verified", "signup_to_first", "plan_type", "signup_channel",
        "home_city", "first_type"]
X = pd.get_dummies(s[core], columns=["plan_type","signup_channel","home_city","first_type"],
                   drop_first=True).astype(float)
y = s.ret60.astype(int)
pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                     LogisticRegression(max_iter=3000, C=0.5))
pipe.fit(X, y)
clf = pipe.named_steps["logisticregression"]
auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
i = list(X.columns).index("fail_pct")
scale = pipe.named_steps["standardscaler"].scale_[i]
or10 = float(np.exp(clf.coef_[0][i] * 10 / scale))

rng = np.random.default_rng(42)
ors = []
for _ in range(100):
    idx = rng.integers(0, len(X), len(X))
    p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=3000, C=0.5))
    p.fit(X.iloc[idx], y.iloc[idx])
    b = p.named_steps["logisticregression"].coef_[0]
    sc = p.named_steps["standardscaler"].scale_[i]
    ors.append(np.exp(b[i] * 10 / sc))
ci_lo, ci_hi = np.percentile(ors, [2.5, 97.5])

coefs = pd.DataFrame({"feature": X.columns, "coef_std": clf.coef_[0],
                      "odds_ratio_per_sd": np.exp(clf.coef_[0])})
coefs.sort_values("coef_std", key=abs, ascending=False).to_csv(f"{OUT}/retention_model_coefficients.csv", index=False)

summary = {
    "riders_with_early_attempts": int(len(df)),
    "share_active_days_22_30_pct": round(100 * float(df.active_22_30.mean()), 1),
    "model_sample": int(len(s)),
    "base_d60_pct": round(100 * float(y.mean()), 1),
    "auc_d60": round(float(auc), 3),
    "fail10pp_odds_ratio": round(or10, 3),
    "fail10pp_or_ci95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
    "exposure_table": tab[["riders","share_pct","mean_fail_pct","d60_pct","d90_pct"]].reset_index().rename(columns={"fail_bucket":"bucket"}).to_dict(orient="records"),
}
with open(f"{OUT}/retention_model.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nmodel AUC:", round(auc, 3), "| OR per +10pp failure:", round(or10, 3),
      "CI", round(float(ci_lo), 3), round(float(ci_hi), 3))
print(coefs.head(10).to_string(index=False))
print("\ncohorts tail:\n", cohort.tail(9).to_string())
print("\nsegments:\n", seg.reset_index().to_string(index=False))
print("\nsaved clean retention outputs")
