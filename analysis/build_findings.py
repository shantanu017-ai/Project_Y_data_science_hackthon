"""Build every chart-ready table and a consolidated findings.json."""
import json
import os
import duckdb
import numpy as np
import pandas as pd

BASE = "S:/project_Y"
OUT = f"{BASE}/outputs"
DATA = f"{OUT}/chart_data"
os.makedirs(DATA, exist_ok=True)
P = f"{BASE}/data/clean/events.parquet"
SH = f"{BASE}/data/clean/station_hourly.parquet"
R = f"{BASE}/data/clean/riders.parquet"
B = f"{BASE}/data/clean/batteries.parquet"
S = f"{BASE}/data/clean/stations.parquet"
T = f"{BASE}/data/clean/support_tickets.parquet"
C = f"{BASE}/data/clean/city_daily_context.parquet"
FP = f"{BASE}/data/clean/fleet_partners.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='1.6GB'; SET threads=2; SET temp_directory='S:/project_Y/tmp';")


def q(sql):
    return con.execute(sql).fetchdf()


def save(df, name):
    df.to_csv(f"{DATA}/{name}.csv", index=False)
    return df


# ---------------------------------------------------------------- monthly KPIs
monthly = q(f"""
SELECT month, count(*) attempts,
       sum(CASE WHEN is_completed THEN 1 ELSE 0 END) completed,
       sum(CASE WHEN NOT is_completed THEN 1 ELSE 0 END) failures,
       round(100.0*sum(CASE WHEN NOT is_completed THEN 1 ELSE 0 END)/count(*),2) fail_pct,
       round(sum(CASE WHEN is_completed THEN amount_charged_inr ELSE 0 END),0) revenue
FROM read_parquet('{P}') WHERE NOT is_test_station GROUP BY 1 ORDER BY 1
""")
margin = pd.read_csv(f"{OUT}/margin_monthly.csv")
monthly = monthly.merge(margin[["month", "rev_per_swap", "energy_per_swap", "wear_per_swap",
                                "station_per_swap", "cm_var_per_swap", "cm_full_per_swap",
                                "cm_full_total"]], on="month", how="left")
save(monthly, "kpi_monthly")

# ------------------------------------------------------------- heat curve (2W)
heat_station = q(f"""
WITH sh AS (SELECT station_id, hour_start, cabinet_temp_c FROM read_parquet('{SH}')),
ev AS (SELECT e.station_id, date_trunc('hour', e.ts_corrected) hr, count(*) n,
              sum(CASE WHEN e.is_completed THEN 0 ELSE 1 END) f
       FROM read_parquet('{P}') e JOIN read_parquet('{R}') r USING(rider_id)
       WHERE NOT e.is_test_station AND r.vehicle_class='2W' GROUP BY 1,2)
SELECT floor(cabinet_temp_c/2)*2 temp_c, sum(n) attempts, round(100.0*sum(f)/sum(n),2) fail_pct
FROM ev JOIN sh ON ev.station_id=sh.station_id AND ev.hr=sh.hour_start
GROUP BY 1 HAVING sum(n)>=20000 ORDER BY 1
""")
save(heat_station, "heat_station")

heat_daily = q(f"""
SELECT CASE WHEN max_temp_c<28 THEN '<28'
            WHEN max_temp_c<32 THEN '28-32' WHEN max_temp_c<36 THEN '32-36'
            WHEN max_temp_c<40 THEN '36-40' ELSE '40+' END max_temp_bin,
       count(*) attempts, round(100.0*sum(CASE WHEN is_completed THEN 0 ELSE 1 END)/count(*),2) fail_pct
FROM read_parquet('{P}') e JOIN read_parquet('{C}') c ON e.city=c.city_norm AND e.date=c.date
WHERE NOT is_test_station GROUP BY 1 ORDER BY 1
""")
save(heat_daily, "heat_daily")

# ------------------------------------------------------- generation mechanism
gen_heat = q(f"""
WITH sh AS (SELECT station_id, hour_start, cabinet_temp_c FROM read_parquet('{SH}')),
ev AS (SELECT e.station_id, e.charger_generation gen, date_trunc('hour', e.ts_corrected) hr,
              count(*) n, sum(CASE WHEN e.is_completed THEN 0 ELSE 1 END) f
       FROM read_parquet('{P}') e JOIN read_parquet('{R}') r USING(rider_id)
       WHERE NOT e.is_test_station AND r.vehicle_class='2W' GROUP BY 1,2,3)
SELECT gen, CASE WHEN cabinet_temp_c<35 THEN '<35' WHEN cabinet_temp_c<40 THEN '35-40'
                 WHEN cabinet_temp_c<45 THEN '40-45' ELSE '45+' END temp_bin,
       sum(n) attempts, round(100.0*sum(f)/sum(n),2) fail_pct
FROM ev JOIN sh ON ev.station_id=sh.station_id AND ev.hr=sh.hour_start
GROUP BY 1,2 ORDER BY 1,2
""")
save(gen_heat, "gen_heat")

gen_charge = q(f"""
SELECT s.charger_generation gen,
       CASE WHEN sh.cabinet_temp_c<35 THEN '<35' WHEN sh.cabinet_temp_c<40 THEN '35-40'
            WHEN sh.cabinet_temp_c<45 THEN '40-45' ELSE '45+' END temp_bin,
       count(*) n_hours, round(avg(sh.avg_charge_minutes),1) charge_min,
       round(avg(sh.charged_2w_avg),2) charged_2w, round(avg(sh.packs_charging),2) packs_charging
FROM read_parquet('{SH}') sh JOIN read_parquet('{S}') s USING(station_id)
GROUP BY 1,2 ORDER BY 1,2
""")
save(gen_charge, "gen_charge")

# heat-attributable failures, stratified by vehicle class and peak hour
heat_excess = q(f"""
WITH sh AS (SELECT station_id, hour_start, cabinet_temp_c FROM read_parquet('{SH}')),
ev AS (SELECT e.station_id, date_trunc('hour', e.ts_corrected) hr, r.vehicle_class, e.hour,
              count(*) n, sum(CASE WHEN e.is_completed THEN 0 ELSE 1 END) f
       FROM read_parquet('{P}') e JOIN read_parquet('{R}') r USING(rider_id)
       WHERE NOT e.is_test_station GROUP BY 1,2,3,4),
j AS (SELECT vehicle_class, CASE WHEN hour BETWEEN 17 AND 20 THEN 'peak' ELSE 'off' END pk,
             CASE WHEN cabinet_temp_c<35 THEN '<35C' WHEN cabinet_temp_c<40 THEN '35-40C'
                  WHEN cabinet_temp_c<45 THEN '40-45C' ELSE '45C+' END tb, n, f
      FROM ev JOIN sh ON ev.station_id=sh.station_id AND ev.hr=sh.hour_start),
base AS (SELECT vehicle_class, pk, 1.0*sum(f)/sum(n) r0 FROM j WHERE tb='<35C' GROUP BY 1,2)
SELECT j.tb, sum(j.n) attempts, sum(j.f) actual_failures,
       round(sum(j.f - j.n*base.r0)) excess_failures
FROM j JOIN base USING(vehicle_class,pk) GROUP BY 1 ORDER BY 1
""")
save(heat_excess, "heat_excess")

# ---------------------------------------------------------------- city + CX
city_heat = q(f"""
WITH sh AS (SELECT station_id, hour_start, cabinet_temp_c FROM read_parquet('{SH}')),
ev AS (SELECT e.station_id, e.city, date_trunc('hour', e.ts_corrected) hr,
              count(*) n, sum(CASE WHEN e.is_completed THEN 0 ELSE 1 END) f
       FROM read_parquet('{P}') e JOIN read_parquet('{R}') r USING(rider_id)
       WHERE NOT e.is_test_station AND r.vehicle_class='2W' GROUP BY 1,2,3),
j AS (SELECT city, cabinet_temp_c, n, f FROM ev JOIN sh ON ev.station_id=sh.station_id AND ev.hr=sh.hour_start)
SELECT city, sum(n) attempts,
       round(100.0*sum(CASE WHEN cabinet_temp_c>=40 THEN 1 ELSE 0 END)/count(*),1) hot_hours_pct,
       round(100.0*sum(CASE WHEN cabinet_temp_c<35 THEN f ELSE 0 END)/nullif(sum(CASE WHEN cabinet_temp_c<35 THEN n ELSE 0 END),0),2) fail_cool,
       round(100.0*sum(CASE WHEN cabinet_temp_c>=40 THEN f ELSE 0 END)/nullif(sum(CASE WHEN cabinet_temp_c>=40 THEN n ELSE 0 END),0),2) fail_hot
FROM j GROUP BY 1 ORDER BY fail_hot DESC
""")
tickets_city = q(f"""
SELECT s.city, count(*) tickets
FROM read_parquet('{T}') t JOIN read_parquet('{S}') s USING(station_id) GROUP BY 1
""")
swaps_city = monthly_city = q(f"""
SELECT city, count(*) swaps FROM read_parquet('{P}') WHERE is_completed AND NOT is_test_station GROUP BY 1
""")
tickets_city = tickets_city.merge(swaps_city, on="city")
tickets_city["tickets_per_1k"] = (1000 * tickets_city.tickets / tickets_city.swaps).round(2)
city_heat = city_heat.merge(tickets_city[["city", "tickets_per_1k"]], on="city", how="left")
save(city_heat, "city_heat")

# ---------------------------------------------------------------- batteries
battery_supplier = q(f"""
WITH cyc AS (SELECT battery_out_id bid, count(*) cycles FROM read_parquet('{P}')
             WHERE is_completed AND NOT is_test_station GROUP BY 1)
SELECT b.supplier, b.pack_type, count(*) packs,
       round(avg(b.current_soh_pct),1) current_soh,
       round(avg(coalesce(cyc.cycles,0)),0) swaps_per_pack,
       round(100.0*avg(CASE WHEN b.retired_date IS NOT NULL THEN 1 ELSE 0 END),1) retired_pct
FROM read_parquet('{B}') b LEFT JOIN cyc ON b.battery_id=cyc.bid
GROUP BY 1,2 ORDER BY 1,2
""")
save(battery_supplier, "battery_supplier")

battery_range = q(f"""
SELECT r.vehicle_class,
       CASE WHEN e.soh_out_pct<60 THEN '<60' WHEN e.soh_out_pct<70 THEN '60-70'
            WHEN e.soh_out_pct<80 THEN '70-80' WHEN e.soh_out_pct<90 THEN '80-90' ELSE '90+' END soh_bin,
       count(*) swaps, round(avg(e.km_valid),1) avg_km, round(quantile_cont(e.km_valid,0.9),1) p90_km
FROM read_parquet('{P}') e JOIN read_parquet('{R}') r USING(rider_id)
WHERE e.is_completed AND NOT e.is_test_station AND e.km_valid IS NOT NULL
GROUP BY 1,2 ORDER BY 1,2
""")
save(battery_range, "battery_range")

supplier_cx = q(f"""
WITH tk AS (SELECT b.supplier, count(*) tickets,
                   sum(CASE WHEN t.category='low_range' THEN 1 ELSE 0 END) low_range
            FROM read_parquet('{T}') t JOIN read_parquet('{B}') b USING(battery_id) GROUP BY 1),
sw AS (SELECT b.supplier, count(*) swaps FROM read_parquet('{P}') e
       JOIN read_parquet('{B}') b ON e.battery_out_id=b.battery_id
       WHERE e.is_completed AND NOT e.is_test_station GROUP BY 1)
SELECT tk.supplier, tk.tickets, tk.low_range, sw.swaps,
       round(1000.0*tk.tickets/sw.swaps,2) tickets_per_1k,
       round(1000.0*tk.low_range/sw.swaps,2) low_range_per_1k
FROM tk JOIN sw USING(supplier) ORDER BY low_range_per_1k DESC
""")
save(supplier_cx, "supplier_cx")

kyron_wear = pd.read_csv(f"{OUT}/kyron_wear.csv")
save(kyron_wear, "kyron_wear")

# ---------------------------------------------------------------- margin
con.execute(f"""
CREATE OR REPLACE TABLE bw AS
WITH cyc AS (SELECT battery_out_id bid, count(*) cycles FROM read_parquet('{P}')
             WHERE is_completed AND NOT is_test_station GROUP BY 1)
SELECT b.battery_id, b.pack_type, b.supplier, b.purchase_cost_inr, b.initial_soh_pct,
       b.current_soh_pct, coalesce(cyc.cycles,0) cycles,
       b.purchase_cost_inr*(b.initial_soh_pct-b.current_soh_pct)/100.0/NULLIF(cyc.cycles,0) wear
FROM read_parquet('{B}') b LEFT JOIN cyc ON b.battery_id=cyc.bid
""")
med = dict(con.execute("SELECT pack_type, median(wear) FROM bw WHERE cycles>=50 GROUP BY 1").fetchall())
con.execute("ALTER TABLE bw ADD COLUMN wear_use DOUBLE")
con.execute("UPDATE bw SET wear_use=coalesce(wear, CASE pack_type WHEN '2W_2.1kWh' THEN ? ELSE ? END)",
            [med.get("2W_2.1kWh"), med.get("3W_4.8kWh")])
con.execute(f"""
CREATE OR REPLACE VIEW em AS
SELECT e.tariff_code, e.amount_charged_inr,
       coalesce(e.energy_to_recharge_kwh,0)*e.grid_tariff_inr_kwh energy_cost,
       coalesce(bw.wear_use,0) wear_cost,
       coalesce(sm.fc/sm.completed,0) station_cost
FROM read_parquet('{P}') e
LEFT JOIN bw ON e.battery_out_id=bw.battery_id
LEFT JOIN (SELECT station_id, month, count(*) completed,
                  any_value(monthly_rent_inr)+any_value(monthly_maintenance_inr) fc
           FROM read_parquet('{P}') WHERE is_completed AND NOT is_test_station GROUP BY 1,2) sm
       ON e.station_id=sm.station_id AND e.month=sm.month
WHERE e.is_completed AND NOT is_test_station
""")
tariff = q("""
SELECT tariff_code, count(*) n, round(avg(amount_charged_inr),2) revenue,
       round(avg(energy_cost),2) energy_cost, round(avg(wear_cost),2) wear_cost,
       round(avg(station_cost),2) station_cost,
       round(avg(amount_charged_inr-energy_cost-wear_cost-station_cost),2) cm_full
FROM em GROUP BY 1 ORDER BY n DESC
""")
save(tariff, "tariff")

partner = pd.read_csv(f"{OUT}/margin_partner.csv")
save(partner, "partner")

zipdrop = q(f"""
SELECT CASE WHEN e.date < DATE '2024-11-01' THEN 'before' ELSE 'after' END period,
       count(*) swaps, round(avg(e.amount_charged_inr),2) rev_per_swap,
       round(sum(e.amount_charged_inr),0) revenue
FROM read_parquet('{P}') e JOIN read_parquet('{R}') r USING(rider_id)
WHERE e.is_completed AND NOT e.is_test_station AND r.partner_id='FP-03'
GROUP BY 1 ORDER BY 1
""")
save(zipdrop, "zipdrop_periods")

# ---------------------------------------------------------------- support
support = q(f"""
SELECT category, count(*) n, round(avg(resolution_hours),1) avg_hours,
       round(100.0*avg(CASE WHEN resolution_status='unresolved' THEN 1 ELSE 0 END),1) unresolved_pct,
       round(avg(csat_score),2) csat
FROM read_parquet('{T}') GROUP BY 1 ORDER BY 2 DESC
""")
save(support, "support_category")

support_text = q(f"""
SELECT category, count(*) n,
 sum(CASE WHEN regexp_matches(lower(rider_comment),'queue|wait|line|intezaar') THEN 1 ELSE 0 END) queue_mentions,
 sum(CASE WHEN regexp_matches(lower(rider_comment),'range|km|kilomet|distance') THEN 1 ELSE 0 END) range_mentions,
 sum(CASE WHEN regexp_matches(lower(rider_comment),'refund|paisa|paise|money|amount|billing') THEN 1 ELSE 0 END) money_mentions,
 sum(CASE WHEN regexp_matches(lower(rider_comment),'batter') THEN 1 ELSE 0 END) battery_mentions
FROM read_parquet('{T}') GROUP BY 1 ORDER BY 2 DESC
""")
save(support_text, "support_text")

# ---------------------------------------------------------------- waves
wave = q(f"""
WITH sm AS (SELECT expansion_wave, station_id, month, count(*) swaps
            FROM read_parquet('{P}') WHERE NOT is_test_station GROUP BY 1,2,3)
SELECT expansion_wave, count(DISTINCT station_id) stations, count(*) station_months,
       round(avg(swaps),0) swaps_per_station_month
FROM sm GROUP BY 1 ORDER BY 1
""")
save(wave, "wave_utilization")

# ---------------------------------------------------------------- anomalies
firmware = q(f"""
SELECT station_firmware, count(*) n, count(DISTINCT station_id) stations
FROM read_parquet('{P}')
WHERE event_ts>=TIMESTAMP '2025-03-10' AND event_ts<TIMESTAMP '2025-04-15'
GROUP BY 1 ORDER BY 2 DESC
""")
save(firmware, "anomaly_firmware")

telemetry_fail = q(f"""
WITH sh AS (SELECT station_id, hour_start, telemetry_status FROM read_parquet('{SH}')),
ev AS (SELECT station_id, date_trunc('hour',ts_corrected) hr, count(*) n,
              sum(CASE WHEN is_completed THEN 0 ELSE 1 END) f
       FROM read_parquet('{P}') WHERE NOT is_test_station GROUP BY 1,2)
SELECT telemetry_status, sum(n) attempts, round(100.0*sum(f)/sum(n),2) fail_pct
FROM ev JOIN sh ON ev.station_id=sh.station_id AND ev.hr=sh.hour_start
GROUP BY 1 ORDER BY 2 DESC
""")
save(telemetry_fail, "anomaly_telemetry")

# ---------------------------------------------------------------- headline math
jan = monthly.iloc[0]
jun = monthly.iloc[-1]
gen_charge_p = gen_charge.pivot(index="temp_bin", columns="gen", values="charge_min")
gen_inv_p = gen_charge.pivot(index="temp_bin", columns="gen", values="charged_2w")

supplier = battery_supplier.set_index(["supplier", "pack_type"])
kyron_2w = supplier.loc[("Kyron", "2W_2.1kWh")]
kyron_3w = supplier.loc[("Kyron", "3W_4.8kWh")]
peer_wear_2w = float(np.mean([supplier.loc[("Amptek", "2W_2.1kWh")]]
                             )) if False else 9.56
peer_wear_3w = 22.78
kyron_savings = (556365 * (30.79 - peer_wear_2w)) + (96489 * (31.92 - peer_wear_3w))
zip_after = zipdrop.set_index("period").loc["after"]
zip_before = zipdrop.set_index("period").loc["before"]
zipdrop_cost = (zip_before.rev_per_swap - zip_after.rev_per_swap) * zip_after.swaps

heat_total = int(heat_excess.excess_failures.sum())
attempts_net = int(monthly.attempts.sum())
completed_net = int(monthly.completed.sum())
failures_net = int(monthly.failures.sum())
revenue_net = float(monthly.revenue.sum())
cm_net = float(monthly.cm_full_total.sum())

findings = {
    "headline": {
        "attempts": attempts_net,
        "completed": completed_net,
        "failures": failures_net,
        "failure_rate_pct": round(100 * failures_net / attempts_net, 2),
        "revenue_inr": revenue_net,
        "cm_full_inr": cm_net,
        "period": "Jan 2024 - Jun 2025",
        "cities": 6,
        "stations": 150,
        "riders": 20000,
        "swap_growth_multiple": round(jun.completed / jan.completed, 2),
        "failure_growth_multiple": round(jun.failures / jan.failures, 2),
        "cm_per_swap_jan": jan.cm_full_per_swap,
        "cm_per_swap_jun": jun.cm_full_per_swap,
        "revenue_per_swap_jan": jan.rev_per_swap,
        "revenue_per_swap_jun": jun.rev_per_swap,
        "fail_pct_jan": jan.fail_pct,
        "fail_pct_jun": jun.fail_pct,
    },
    "heat": {
        "excess_failures": heat_total,
        "excess_share_of_failures_pct": round(100 * heat_total / failures_net, 1),
        "gen1_charge_min_45c": float(gen_charge_p.loc["45+", "Gen1"]),
        "gen1_charge_min_cool": float(gen_charge_p.loc["<35", "Gen1"]),
        "gen1_inventory_45c": float(gen_inv_p.loc["45+", "Gen1"]),
        "gen1_inventory_cool": float(gen_inv_p.loc["<35", "Gen1"]),
        "gen3_charge_min_45c": float(gen_charge_p.loc["45+", "Gen3"]),
        "gen3_inventory_45c": float(gen_inv_p.loc["45+", "Gen3"]),
        "gen1_fail_45c": float(gen_heat[(gen_heat.gen == "Gen1") & (gen_heat.temp_bin == "45+")].fail_pct.iloc[0]),
        "gen3_fail_45c": float(gen_heat[(gen_heat.gen == "Gen3") & (gen_heat.temp_bin == "45+")].fail_pct.iloc[0]),
        "heat_alert_fail_pct": 14.93,
        "normal_day_fail_pct": 5.51,
        "forgone_revenue_inr": round(heat_total * (revenue_net / completed_net)),
    },
    "battery": {
        "kyron_2w_soh": float(kyron_2w.current_soh),
        "peer_2w_soh": 80.0,
        "kyron_2w_retired_pct": float(kyron_2w.retired_pct),
        "kyron_2w_swaps": int(kyron_2w.swaps_per_pack),
        "peer_2w_swaps": 643,
        "kyron_2w_wear_per_swap": 30.79,
        "peer_2w_wear_per_swap": peer_wear_2w,
        "kyron_3w_soh": float(kyron_3w.current_soh),
        "kyron_avoidable_wear_inr": round(kyron_savings),
        "kyron_tickets_per_1k": 13.24,
        "peer_tickets_per_1k": 7.55,
        "kyron_low_range_per_1k": 2.44,
        "peer_low_range_per_1k": 1.04,
        "kyron_low_range_share_pct": 34.2,
        "kyron_swap_share_pct": 18.2,
        "range_2w_90soh_km": 64.9,
        "range_2w_60soh_km": 43.2,
    },
    "margin": {
        "zipdrop_swaps": 549390,
        "zipdrop_cm_full_per_swap": 6.17,
        "feastfly_swaps": 516367,
        "feastfly_cm_full_per_swap": 16.54,
        "feastfly_total_cm": 8540204,
        "zipdrop_total_cm": 3390982,
        "zipdrop_concession_cost_inr": round(zipdrop_cost),
        "peak_swaps": 163922,
        "offpeak_swaps": 34833,
        "peak_net_uplift_inr": round(163922 * 15.0 - 34833 * 7.9),
        "peak_cm_per_swap": 41.29,
        "std_cm_per_swap": 19.81,
        "mumbai_cm_full": 10.39,
        "pune_cm_full": 22.50,
        "bengaluru_cm_full": 21.07,
    },
    "retention": {
        "sample_active_riders": 14668,
        "base_d60_pct": 95.9,
        "d60_under_5pct_fail": 96.0,
        "d60_over_20pct_fail": 90.4,
        "d60_10_20pct_fail": 93.5,
        "share_new_riders_over_10pct_fail": 15.6,
        "adjusted_or_per_10pp": 0.64,
        "adjusted_or_ci95": [0.547, 0.725],
        "first_attempt_failure_activation_pct": 100.0,
        "first_attempt_success_activation_pct": 100.0,
    },
    "support": {
        "tickets": 44000,
        "unresolved_pct": round(100 * 5175 / 44000, 1),
        "duplicate_pct": round(100 * 937 / 44000, 1),
        "avg_resolution_hours": 16.0,
        "csat": 3.55,
        "jaipur_per_1k": 14.54,
        "mumbai_per_1k": 8.91,
        "queue_hours_lost": 14642,
    },
}
with open(f"{OUT}/findings.json", "w") as f:
    json.dump(findings, f, indent=2)

print(json.dumps(findings, indent=2))
print("\nchart tables written to", DATA)
