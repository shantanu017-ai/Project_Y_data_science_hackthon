"""Rider churn model: first-30-day experience -> still active in days 31-90."""
import duckdb, numpy as np, pandas as pd, json, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

con = duckdb.connect()
con.execute("SET memory_limit='1.4GB'; SET threads=2; SET temp_directory='S:/project_Y/tmp';")
P = "S:/project_Y/data/clean/events.parquet"
R = "S:/project_Y/data/clean/riders.parquet"
OUT = "S:/project_Y/outputs"

sql = f"""
WITH fe AS (
  SELECT rider_id, min(ts_corrected) AS first_ts
  FROM read_parquet('{P}') WHERE NOT is_test_station GROUP BY 1
),
w30 AS (
  SELECT e.rider_id,
    count(*) AS att30,
    sum(CASE WHEN e.is_completed THEN 0 ELSE 1 END) AS fail30,
    sum(CASE WHEN e.event_type='abandoned_queue' THEN 1 ELSE 0 END) AS abandon30,
    sum(CASE WHEN e.event_type='failed_system_error' THEN 1 ELSE 0 END) AS syserr30,
    sum(CASE WHEN e.event_type='failed_no_charged_battery' THEN 1 ELSE 0 END) AS nostock30,
    avg(CASE WHEN e.is_completed THEN e.queue_wait_sec END) AS wait30,
    quantile_cont(CASE WHEN e.is_completed THEN e.queue_wait_sec END, 0.9) AS wait90_30,
    count(DISTINCT e.station_id) AS stations30,
    avg(e.soh_out_pct) AS soh30,
    avg(CASE WHEN e.tariff_code='PEAK' THEN 1.0 ELSE 0 END) AS peakshare30,
    avg(CASE WHEN e.tariff_code='OFFPEAK' THEN 1.0 ELSE 0 END) AS offpeakshare30,
    min(e.ts_corrected) AS f2
  FROM read_parquet('{P}') e JOIN fe USING(rider_id)
  WHERE NOT e.is_test_station AND e.ts_corrected <= fe.first_ts + INTERVAL 30 DAY
  GROUP BY 1
),
ret60 AS (
  SELECT fe.rider_id,
    max(CASE WHEN e.is_completed AND e.ts_corrected > fe.first_ts + INTERVAL 30 DAY
                 AND e.ts_corrected <= fe.first_ts + INTERVAL 60 DAY THEN 1 ELSE 0 END) AS ret60
  FROM fe JOIN read_parquet('{P}') e ON e.rider_id=fe.rider_id
  WHERE NOT e.is_test_station GROUP BY 1
),
ret90 AS (
  SELECT fe.rider_id,
    max(CASE WHEN e.is_completed AND e.ts_corrected > fe.first_ts + INTERVAL 30 DAY
                 AND e.ts_corrected <= fe.first_ts + INTERVAL 90 DAY THEN 1 ELSE 0 END) AS ret90
  FROM fe JOIN read_parquet('{P}') e ON e.rider_id=fe.rider_id
  WHERE NOT e.is_test_station GROUP BY 1
),
firstev AS (
  SELECT e.rider_id, e.event_type AS first_type, e.queue_wait_sec AS first_wait,
         e.city AS first_city, e.station_id AS first_station, e.is_completed AS first_completed
  FROM read_parquet('{P}') e JOIN fe USING(rider_id)
  WHERE NOT e.is_test_station AND e.ts_corrected=fe.first_ts
  QUALIFY row_number() OVER (PARTITION BY e.rider_id ORDER BY e.event_id)=1
)
SELECT r.rider_id, r.plan_type, r.signup_channel, r.vehicle_class, r.kyc_verified,
       r.home_city_norm AS home_city, r.partner_id IS NOT NULL AS is_partner,
       date_diff('day', r.signup_date, fe.first_ts) AS signup_to_first,
       fe.first_ts AS first_ts, fv.first_type, fv.first_wait, fv.first_city, fv.first_station,
       w.*, ret60.ret60, ret90.ret90
FROM read_parquet('{R}') r
JOIN fe USING(rider_id)
JOIN w30 w USING(rider_id)
JOIN firstev fv USING(rider_id)
JOIN ret60 USING(rider_id) JOIN ret90 USING(rider_id)
"""
df = con.execute(sql).fetchdf()
df["fail30_rate"] = df.fail30 / df.att30
df["nostock30_rate"] = df.nostock30 / df.att30
df["abandon30_rate"] = df.abandon30 / df.att30
df["syserr30_rate"] = df.syserr30 / df.att30
df["log_att"] = np.log1p(df.att30)
df["first_month"] = pd.to_datetime(df.first_ts).dt.strftime("%Y-%m")
df["cohort_month"] = pd.to_datetime(df.first_ts).dt.to_period("M").astype(str)
df["is_3w"] = (df.vehicle_class == "3W").astype(int)
df["first_failed"] = (df.first_type != "swap_completed").astype(int)

# modelling window: cohorts with full 90-day observation
m = df[(df.cohort_month <= "2025-03")].copy()
m["log_wait"] = np.log1p(m.wait30)
m["log_stations"] = np.log1p(m.stations30)

def fit(target, feats, data):
    X = pd.get_dummies(data[feats], columns=["plan_type","signup_channel","home_city","first_city","first_type"],
                       drop_first=True, dtype=float)
    y = data[target].astype(int)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(X, y)
    clf = pipe.named_steps["logisticregression"]
    auc = roc_auc_score(y, pipe.predict_proba(X)[:,1])
    coefs = pd.DataFrame({"feature": X.columns, "coef": clf.coef_[0], "odds_ratio": np.exp(clf.coef_[0])})
    coefs = coefs.sort_values("coef", key=abs, ascending=False)
    return auc, coefs, X, y

core = ["fail30_rate","log_att","log_wait","log_stations",
        "soh30","peakshare30","offpeakshare30","is_3w","is_partner","kyc_verified",
        "signup_to_first","plan_type","signup_channel","home_city","first_city","first_type"]
features = [f for f in core if f in m.columns]

print("sample:", len(m), "ret90 rate:", round(100*m.ret90.mean(),2), "ret60:", round(100*m.ret60.mean(),2))
auc90, coef90, X, y = fit("ret90", features, m)
print("AUC ret90:", round(auc90,3))
print(coef90.head(20).to_string(index=False))

# univariate deciles for the story
cuts = pd.qcut(m.fail30_rate, 5, duplicates="drop")
print("\nret90 by first-30d failure rate quintile:")
print(m.groupby(cuts, observed=True).agg(n=("rider_id","size"), fail_rate=("fail30_rate","mean"), ret90=("ret90","mean"), ret60=("ret60","mean")).assign(ret90=lambda x:(100*x.ret90).round(1), ret60=lambda x:(100*x.ret60).round(1), fail_rate=lambda x:(100*x.fail_rate).round(1)).to_string())
print("\nret90 by first-30d wait quintile:")
cuts_w = pd.qcut(m.wait30, 5, duplicates="drop")
print(m.groupby(cuts_w, observed=True).agg(n=("rider_id","size"), wait=("wait30","mean"), ret90=("ret90","mean")).assign(ret90=lambda x:(100*x.ret90).round(1)).to_string())
print("\nret90 by vehicle:")
print(m.groupby("vehicle_class").agg(n=("rider_id","size"), ret90=("ret90","mean")).assign(ret90=lambda x:(100*x.ret90).round(1)).to_string())
print("\nret90 by first month:")
print(m.groupby("cohort_month").agg(n=("rider_id","size"), ret60=("ret60","mean"), ret90=("ret90","mean")).assign(ret60=lambda x:(100*x.ret60).round(1), ret90=lambda x:(100*x.ret90).round(1)).to_string())

coef90.to_csv(os.path.join(OUT, "churn_model_coefficients.csv"), index=False)
df.to_parquet("S:/project_Y/data/clean/rider_features.parquet", index=False)
summary = {
    "model_sample": int(len(m)), "auc_ret90": round(float(auc90), 3),
    "ret90_rate": round(float(m.ret90.mean()), 4), "ret60_rate": round(float(m.ret60.mean()), 4),
    "top_odds": coef90.head(12).to_dict(orient="records"),
}
with open(os.path.join(OUT, "churn_model.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nsaved churn model outputs")
