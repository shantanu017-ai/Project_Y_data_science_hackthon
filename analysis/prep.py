"""VoltRelay hackathon - data prep. Produces cleaned parquet files + QC report."""
import os, json
import numpy as np
import pandas as pd

D = r"S:\project_Y\data\Hackathon Data Set _ Gradient"
CLEAN = r"S:\project_Y\data\clean"
OUT = r"S:\project_Y\outputs"
os.makedirs(CLEAN, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

CITY_MAP = {
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
    "delhi ncr": "Delhi NCR", "delhi": "Delhi NCR", "new delhi": "Delhi NCR",
    "gurgaon": "Delhi NCR",
    "mumbai": "Mumbai", "bombay": "Mumbai", "mum": "Mumbai",
    "pune": "Pune", "pun": "Pune",
    "hyderabad": "Hyderabad", "hyd": "Hyderabad",
    "jaipur": "Jaipur", "jai": "Jaipur",
}

def norm_city(s):
    return s.astype(str).str.strip().str.lower().map(CITY_MAP).fillna(s)

def main():
    qc = {}

    stations = pd.read_csv(os.path.join(D, "stations.csv"),
                           parse_dates=["commissioned_date", "decommissioned_date",
                                        "firmware_updated_date", "competitor_within_1_5km_since"])
    stations["is_test"] = stations.station_id.str.startswith("STN-TST")
    stations["city"] = norm_city(stations["city"])
    stations.to_parquet(os.path.join(CLEAN, "stations.parquet"), index=False)
    qc["stations"] = len(stations)

    riders = pd.read_csv(os.path.join(D, "riders.csv"), parse_dates=["signup_date"])
    riders["home_city_raw"] = riders["home_city"]
    riders["home_city"] = norm_city(riders["home_city"])
    qc["riders_city_variants_collapsed"] = int(riders.home_city_raw.str.lower().str.strip().nunique()
                                               - riders.home_city.nunique())
    riders.to_parquet(os.path.join(CLEAN, "riders.parquet"), index=False)

    batteries = pd.read_csv(os.path.join(D, "batteries.csv"),
                            parse_dates=["manufacture_date", "commission_date", "retired_date"])
    batteries.to_parquet(os.path.join(CLEAN, "batteries.parquet"), index=False)

    partners = pd.read_csv(os.path.join(D, "fleet_partners.csv"), parse_dates=["contract_start_date", "amendment_date"])
    partners.to_parquet(os.path.join(CLEAN, "fleet_partners.parquet"), index=False)

    city_ctx = pd.read_csv(os.path.join(D, "city_daily_context.csv"), parse_dates=["date"])
    city_ctx["city"] = norm_city(city_ctx["city"])
    city_ctx.to_parquet(os.path.join(CLEAN, "city_daily_context.parquet"), index=False)

    tickets = pd.read_csv(os.path.join(D, "support_tickets.csv"), parse_dates=["created_ts"])
    tickets["station_id"] = tickets["station_id"].fillna("")
    tickets.to_parquet(os.path.join(CLEAN, "support_tickets.parquet"), index=False)
    qc["tickets"] = len(tickets)

    usecols = ["event_id", "rider_id", "station_id", "event_ts", "event_type", "queue_wait_sec",
               "battery_in_id", "battery_out_id", "soc_in_pct", "soh_in_pct", "soc_out_pct",
               "soh_out_pct", "km_since_last_swap", "tariff_code", "list_price_inr", "discount_inr",
               "amount_charged_inr", "payment_mode", "energy_to_recharge_kwh", "station_firmware",
               "sync_mode"]
    ev = pd.read_csv(os.path.join(D, "swap_events.csv"), usecols=usecols, parse_dates=["event_ts"])
    qc["swap_rows_raw"] = len(ev)

    for c in ["rider_id", "station_id", "event_type", "tariff_code", "payment_mode",
              "station_firmware", "sync_mode"]:
        ev[c] = ev[c].astype("category")

    ev = ev.sort_values(["rider_id", "station_id", "event_ts"], kind="mergesort")
    grp = ev.groupby(["rider_id", "station_id"], observed=True)
    dt = grp["event_ts"].diff().dt.total_seconds()
    same_batt = (ev["battery_in_id"].fillna("NA") == ev.groupby(["rider_id", "station_id"], observed=True)["battery_in_id"].shift().fillna("NA"))
    offline = (ev["sync_mode"] == "offline_batch") | (ev.groupby(["rider_id", "station_id"], observed=True)["sync_mode"].shift() == "offline_batch")
    dup = (dt >= 0) & (dt <= 180) & same_batt & offline
    qc["near_duplicates_removed"] = int(dup.sum())
    ev = ev.loc[~dup].copy()
    qc["swap_rows_clean"] = len(ev)

    ev["is_test_station"] = ev["station_id"].astype(str).str.startswith("STN-TST")
    qc["test_station_events"] = int(ev.is_test_station.sum())

    ev = ev.merge(stations[["station_id", "city", "zone", "location_type", "host_type",
                            "expansion_wave", "charger_generation", "slots_2w", "slots_3w",
                            "monthly_rent_inr", "monthly_maintenance_inr", "grid_tariff_inr_kwh",
                            "connectivity_tier", "firmware_version", "commissioned_date"]],
                  on="station_id", how="left")

    bug = (ev["station_firmware"].astype(str) == "v3.2.0") & \
          (ev["event_ts"] >= "2025-03-10") & (ev["event_ts"] < "2025-04-15")
    qc["tz_bug_events_corrected"] = int(bug.sum())
    ev["ts_corrected"] = ev["event_ts"] + pd.to_timedelta(np.where(bug, 330, 0), unit="m")
    ev["date"] = ev["ts_corrected"].dt.date
    ev["hour"] = ev["ts_corrected"].dt.hour
    ev["month"] = ev["ts_corrected"].dt.to_period("M").astype(str)
    ev["week"] = ev["ts_corrected"].dt.to_period("W").astype(str)
    ev["dow"] = ev["ts_corrected"].dt.day_name()
    ev["is_completed"] = ev["event_type"] == "swap_completed"

    km = ev["km_since_last_swap"]
    ev["km_valid"] = km.where((km > 0) & (km <= 300))
    qc["km_invalid"] = int((km.notna() & ev["km_valid"].isna()).sum())
    qc["soc_over_100"] = int((ev["soc_out_pct"] > 100).sum())

    ev.to_parquet(os.path.join(CLEAN, "swap_events.parquet"), index=False)
    qc["date_range_corrected"] = [str(ev.ts_corrected.min()), str(ev.ts_corrected.max())]

    hs = pd.read_csv(os.path.join(D, "station_hourly_status.csv"), parse_dates=["hour_start"])
    hs["station_id"] = hs["station_id"].astype("category")
    hs.to_parquet(os.path.join(CLEAN, "station_hourly.parquet"), index=False)
    qc["hourly_rows"] = len(hs)

    with open(os.path.join(OUT, "qc.json"), "w") as f:
        json.dump(qc, f, indent=2)
    print(json.dumps(qc, indent=2))

if __name__ == "__main__":
    main()
