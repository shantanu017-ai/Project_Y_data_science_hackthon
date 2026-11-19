"""Contribution margin model using views (memory-safe)."""
import duckdb, pandas as pd, json, os
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120)

con = duckdb.connect()
con.execute("SET memory_limit='1.6GB'; SET threads=2; SET temp_directory='S:/project_Y/tmp';")
P="S:/project_Y/data/clean/events.parquet"; B="S:/project_Y/data/clean/batteries.parquet"
R="S:/project_Y/data/clean/riders.parquet"; FP="S:/project_Y/data/clean/fleet_partners.parquet"
OUT="S:/project_Y/outputs"

con.execute(f"""
CREATE OR REPLACE TABLE bw AS
WITH cyc AS (SELECT battery_out_id bid, count(*) cycles FROM read_parquet('{P}')
             WHERE is_completed AND NOT is_test_station GROUP BY 1)
SELECT b.battery_id, b.pack_type, b.supplier, b.purchase_cost_inr, b.initial_soh_pct, b.current_soh_pct,
       coalesce(cyc.cycles,0) cycles,
       b.purchase_cost_inr*(b.initial_soh_pct-b.current_soh_pct)/100.0/NULLIF(cyc.cycles,0) wear
FROM read_parquet('{B}') b LEFT JOIN cyc ON b.battery_id=cyc.bid
""")
med = dict(con.execute("SELECT pack_type, median(wear) FROM bw WHERE cycles>=50 GROUP BY 1").fetchall())
con.execute("ALTER TABLE bw ADD COLUMN wear_use DOUBLE")
con.execute("UPDATE bw SET wear_use = coalesce(wear, CASE pack_type WHEN '2W_2.1kWh' THEN ? ELSE ? END)",
            [med.get("2W_2.1kWh"), med.get("3W_4.8kWh")])

con.execute(f"""
CREATE OR REPLACE VIEW ev_margin AS
SELECT e.station_id, e.rider_id, e.month, e.city, e.tariff_code, e.battery_out_id,
       e.amount_charged_inr,
       coalesce(e.energy_to_recharge_kwh,0)*e.grid_tariff_inr_kwh AS energy_cost,
       coalesce(bw.wear_use,0) AS wear_cost,
       coalesce(sm.fixed_cost/sm.completed,0) AS station_cost,
       e.amount_charged_inr - coalesce(e.energy_to_recharge_kwh,0)*e.grid_tariff_inr_kwh
         - coalesce(bw.wear_use,0) AS cm_var,
       e.amount_charged_inr - coalesce(e.energy_to_recharge_kwh,0)*e.grid_tariff_inr_kwh
         - coalesce(bw.wear_use,0) - coalesce(sm.fixed_cost/sm.completed,0) AS cm_full
FROM read_parquet('{P}') e
LEFT JOIN bw ON e.battery_out_id=bw.battery_id
LEFT JOIN (
  SELECT station_id, month, count(*) completed, any_value(monthly_rent_inr)+any_value(monthly_maintenance_inr) AS fixed_cost
  FROM read_parquet('{P}') WHERE is_completed AND NOT is_test_station GROUP BY 1,2
) sm ON e.station_id=sm.station_id AND e.month=sm.month
WHERE e.is_completed AND NOT e.is_test_station
""")

def q(s): return con.execute(s).fetchdf()

monthly = q("""
SELECT month, count(*) completed, round(sum(amount_charged_inr),0) revenue,
 round(avg(amount_charged_inr),2) rev_per_swap,
 round(avg(energy_cost),2) energy_per_swap, round(avg(wear_cost),2) wear_per_swap,
 round(avg(station_cost),2) station_per_swap,
 round(avg(cm_var),2) cm_var_per_swap, round(avg(cm_full),2) cm_full_per_swap,
 round(sum(cm_var),0) cm_var_total, round(sum(cm_full),0) cm_full_total
FROM ev_margin GROUP BY 1 ORDER BY 1
""")
print("\nMONTHLY MARGIN:\n", monthly.to_string(index=False))
monthly.to_csv(os.path.join(OUT,"margin_monthly.csv"), index=False)

by_city = q("""
SELECT city, count(*) completed, round(avg(amount_charged_inr),2) rev, round(avg(cm_var),2) cm_var,
 round(avg(cm_full),2) cm_full, round(sum(cm_full),0) cm_full_total
FROM ev_margin GROUP BY 1 ORDER BY 5
""")
print("\nBY CITY:\n", by_city.to_string(index=False))
by_city.to_csv(os.path.join(OUT,"margin_city.csv"), index=False)

by_partner = q(f"""
SELECT r.partner_id, fp.partner_name, fp.partner_segment, fp.discount_pct, fp.discount_pct_after_amendment,
 count(*) completed, round(avg(e.amount_charged_inr),2) rev, round(avg(e.cm_var),2) cm_var, round(avg(e.cm_full),2) cm_full,
 round(sum(e.cm_full),0) total_cm_full, round(avg(e.wear_cost),2) wear
FROM ev_margin e JOIN read_parquet('{R}') r USING(rider_id) JOIN read_parquet('{FP}') fp ON r.partner_id=fp.partner_id
GROUP BY 1,2,3,4,5 ORDER BY 10
""")
print("\nBY PARTNER:\n", by_partner.to_string(index=False))
by_partner.to_csv(os.path.join(OUT,"margin_partner.csv"), index=False)

by_sup = q("""
SELECT bw.supplier, bw.pack_type, count(*) completed, round(avg(m.amount_charged_inr),2) rev,
 round(avg(m.energy_cost),2) energy, round(avg(m.wear_cost),2) wear, round(avg(m.cm_var),2) cm_var,
 round(sum(m.wear_cost),0) wear_total
FROM ev_margin m JOIN bw ON m.battery_out_id=bw.battery_id GROUP BY 1,2 ORDER BY 1,2
""")
print("\nBY BATTERY SUPPLIER:\n", by_sup.to_string(index=False))
by_sup.to_csv(os.path.join(OUT,"margin_supplier.csv"), index=False)

by_tariff = q("""
SELECT tariff_code, count(*) n, round(avg(amount_charged_inr),2) rev, round(avg(cm_var),2) cm_var, round(avg(cm_full),2) cm_full
FROM ev_margin GROUP BY 1 ORDER BY 2 DESC
""")
print("\nBY TARIFF:\n", by_tariff.to_string(index=False))

ky = q("""
SELECT m.month, sum(CASE WHEN bw.supplier='Kyron' THEN 1 ELSE 0 END) kyron_swaps,
 round(sum(CASE WHEN bw.supplier='Kyron' THEN m.wear_cost ELSE 0 END),0) kyron_wear,
 round(sum(m.wear_cost),0) total_wear
FROM ev_margin m LEFT JOIN bw ON m.battery_out_id=bw.battery_id GROUP BY 1 ORDER BY 1
""")
print("\nKYRON WEAR:\n", ky.to_string(index=False))
ky.to_csv(os.path.join(OUT,"kyron_wear.csv"), index=False)

summary = {"monthly": monthly.to_dict(orient="records"), "city": by_city.to_dict(orient="records"),
           "partner": by_partner.to_dict(orient="records"), "supplier": by_sup.to_dict(orient="records")}
with open(os.path.join(OUT,"margin.json"),"w") as f: json.dump(summary,f,indent=2,default=str)
print("\nsaved margin outputs")
