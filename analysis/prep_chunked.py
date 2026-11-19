"""Memory-safe prep: stream swap_events in chunks, dedupe, write parquet."""
import os, json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

D = r"S:\project_Y\data\Hackathon Data Set _ Gradient"
CLEAN = r"S:\project_Y\data\clean"
OUT = r"S:\project_Y\outputs"
os.makedirs(CLEAN, exist_ok=True); os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

CITY_MAP = {
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
    "delhi ncr": "Delhi NCR", "delhi": "Delhi NCR", "new delhi": "Delhi NCR", "gurgaon": "Delhi NCR",
    "mumbai": "Mumbai", "bombay": "Mumbai", "mum": "Mumbai",
    "pune": "Pune", "pun": "Pune",
    "hyderabad": "Hyderabad", "hyd": "Hyderabad",
    "jaipur": "Jaipur", "jai": "Jaipur",
}
def norm_city(s):
    return s.astype(str).str.strip().str.lower().map(CITY_MAP).fillna(s)

DT = {
    "event_id": "string", "rider_id": "string", "station_id": "string", "event_type": "string",
    "attempt_seq": "int64", "queue_wait_sec": "int64", "battery_in_id": "string",
    "battery_out_id": "string", "soc_in_pct": "float64", "soh_in_pct": "float64",
    "soc_out_pct": "float64", "soh_out_pct": "float64", "km_since_last_swap": "float64",
    "tariff_code": "string", "list_price_inr": "float64", "discount_inr": "float64",
    "amount_charged_inr": "float64", "payment_mode": "string", "energy_to_recharge_kwh": "float64",
    "station_firmware": "string", "sync_mode": "string",
}

def main():
    qc = {}
    stations = pd.read_csv(os.path.join(D, "stations.csv"),
                           parse_dates=["commissioned_date", "decommissioned_date",
                                        "firmware_updated_date", "competitor_within_1_5km_since"])
    stations["is_test"] = stations.station_id.str.startswith("STN-TST")
    stations["city_norm"] = norm_city(stations["city"])
    stations.to_parquet(os.path.join(CLEAN, "stations.parquet"), index=False)

    station_cols = ["station_id", "city_norm", "zone", "location_type", "host_type", "expansion_wave",
                    "charger_generation", "slots_2w", "slots_3w", "inventory_target_2w",
                    "inventory_target_3w", "monthly_rent_inr", "monthly_maintenance_inr",
                    "grid_tariff_inr_kwh", "connectivity_tier", "firmware_version",
                    "commissioned_date", "is_test"]
    sjoin = stations[station_cols].copy()
    sjoin["commissioned_date"] = pd.to_datetime(sjoin["commissioned_date"])
    sjoin["key"] = sjoin["station_id"]

    riders = pd.read_csv(os.path.join(D, "riders.csv"), parse_dates=["signup_date"])
    riders["home_city_raw"] = riders["home_city"]
    riders["home_city_norm"] = norm_city(riders["home_city"])
    riders.to_parquet(os.path.join(CLEAN, "riders.parquet"), index=False)

    batteries = pd.read_csv(os.path.join(D, "batteries.csv"),
                            parse_dates=["manufacture_date", "commission_date", "retired_date"])
    batteries.to_parquet(os.path.join(CLEAN, "batteries.parquet"), index=False)

    partners = pd.read_csv(os.path.join(D, "fleet_partners.csv"),
                           parse_dates=["contract_start_date", "amendment_date"])
    partners.to_parquet(os.path.join(CLEAN, "fleet_partners.parquet"), index=False)

    city_ctx = pd.read_csv(os.path.join(D, "city_daily_context.csv"), parse_dates=["date"])
    city_ctx["city_norm"] = norm_city(city_ctx["city"])
    city_ctx.to_parquet(os.path.join(CLEAN, "city_daily_context.parquet"), index=False)

    tickets = pd.read_csv(os.path.join(D, "support_tickets.csv"), parse_dates=["created_ts"])
    tickets.to_parquet(os.path.join(CLEAN, "support_tickets.parquet"), index=False)

    # hourly telemetry in chunks
    hs_path = os.path.join(D, "station_hourly_status.csv")
    first = True
    n = 0
    writer = None
    for ch in pd.read_csv(hs_path, chunksize=250_000, parse_dates=["hour_start"],
                          dtype={"station_id": "string"}):
        tbl = pa.Table.from_pandas(ch, preserve_index=False)
        if first:
            schema = tbl.schema
            writer = pq.ParquetWriter(os.path.join(CLEAN, "station_hourly.parquet"), schema)
            first = False
        writer.write_table(tbl.cast(schema))
        n += len(ch)
    writer.close()
    qc["hourly_rows"] = n

    # swap events stream
    path = os.path.join(D, "swap_events.csv")
    reader = pd.read_csv(path, chunksize=200_000, dtype=DT,
                         parse_dates=["event_ts"], date_format="%Y-%m-%d %H:%M:%S",
                         keep_default_na=True)
    last_seen = {}
    dup_n = 0; raw_n = 0; test_n = 0; bug_n = 0; bad_km = 0; soc100 = 0
    writer = None; schema = None; first = True
    tprev_max = pd.Timestamp.min
    monotonic = True

    for ch in reader:
        raw_n += len(ch)
        ts = ch["event_ts"]
        if ts.iloc[0] < tprev_max:
            monotonic = False
        tprev_max = max(tprev_max, ts.max())

        keep = np.ones(len(ch), dtype=bool)
        r = ch["rider_id"].to_numpy(); s = ch["station_id"].to_numpy()
        b = ch["battery_in_id"].fillna("NA").to_numpy()
        sm = (ch["sync_mode"] == "offline_batch").to_numpy()
        tsn = ts.to_numpy()
        for i in range(len(ch)):
            key = (r[i], s[i])
            prev = last_seen.get(key)
            if prev is not None:
                pts, pb, psm = prev
                dt = (tsn[i] - pts) / np.timedelta64(1, "s")
                if 0 <= dt <= 180 and b[i] == pb and (sm[i] or psm):
                    keep[i] = False
                    dup_n += 1
                    continue
            last_seen[key] = (tsn[i], b[i], sm[i])

        ch = ch.loc[keep].copy()
        bug = (ch["station_firmware"] == "v3.2.0") & (ch["event_ts"] >= "2025-03-10") & (ch["event_ts"] < "2025-04-15")
        bug_n += int(bug.sum())
        ch["ts_corrected"] = ch["event_ts"] + pd.to_timedelta(np.where(bug, 330, 0), unit="m")
        ch["is_test_station"] = ch["station_id"].str.startswith("STN-TST")
        test_n += int(ch["is_test_station"].sum())

        ch["is_completed"] = ch["event_type"] == "swap_completed"
        km = ch["km_since_last_swap"]
        ch["km_valid"] = km.where((km > 0) & (km <= 300))
        bad_km += int((km.notna() & ch["km_valid"].isna()).sum())
        soc100 += int((ch["soc_out_pct"] > 100).sum())

        ch["date"] = ch["ts_corrected"].dt.date
        ch["hour"] = ch["ts_corrected"].dt.hour.astype("int32")
        ch["month"] = ch["ts_corrected"].dt.strftime("%Y-%m")
        ch["week"] = ch["ts_corrected"].dt.strftime("%Y-%W")
        ch["dow"] = ch["ts_corrected"].dt.dayofweek.astype("int32")

        ch = ch.merge(sjoin.drop(columns=["station_id"]).rename(columns={"city_norm": "city"}),
                      left_on="station_id", right_on="key", how="left").drop(columns=["key"])
        ch = ch.drop(columns=["is_test_station"]).rename(columns={"is_test": "is_test_station"})

        if first:
            schema = pa.Table.from_pandas(ch, preserve_index=False).schema
            writer = pq.ParquetWriter(os.path.join(CLEAN, "events.parquet"), schema)
            first = False
        writer.write_table(pa.Table.from_pandas(ch, preserve_index=False).cast(schema))

    writer.close()
    qc.update({
        "swap_rows_raw": raw_n,
        "near_duplicates_removed": dup_n,
        "swap_rows_clean": raw_n - dup_n,
        "file_sorted_by_ts": bool(monotonic),
        "test_station_events": test_n,
        "tz_bug_events_corrected": bug_n,
        "km_invalid": bad_km,
        "soc_out_over_100": soc100,
    })
    ev = pd.read_parquet(os.path.join(CLEAN, "events.parquet"), columns=["ts_corrected", "event_type", "tariff_code"])
    qc["date_range_corrected"] = [str(ev.ts_corrected.min()), str(ev.ts_corrected.max())]
    qc["event_type_counts"] = ev.event_type.value_counts().to_dict()
    qc["tariff_counts"] = ev.tariff_code.value_counts().to_dict()
    qc["riders"] = len(riders)
    qc["tickets"] = len(tickets)
    qc["home_city_variants"] = [int(riders.home_city_raw.str.lower().str.strip().nunique()),
                                int(riders.home_city_norm.nunique())]
    with open(os.path.join(OUT, "qc.json"), "w") as f:
        json.dump(qc, f, indent=2, default=str)
    print(json.dumps(qc, indent=2, default=str))

if __name__ == "__main__":
    main()
