"""VoltRelay data prep with DuckDB (memory-safe). Builds data/voltrelay.duckdb."""
import os, json
import duckdb

D = r"S:\project_Y\data\Hackathon Data Set _ Gradient"
DB = r"S:\project_Y\data\voltrelay.duckdb"
OUT = r"S:\project_Y\outputs"
os.makedirs(OUT, exist_ok=True)

CITY_CASE = """
CASE lower(trim({col}))
    WHEN 'bengaluru' THEN 'Bengaluru' WHEN 'bangalore' THEN 'Bengaluru' WHEN 'blr' THEN 'Bengaluru'
    WHEN 'delhi ncr' THEN 'Delhi NCR' WHEN 'delhi' THEN 'Delhi NCR' WHEN 'new delhi' THEN 'Delhi NCR'
        WHEN 'gurgaon' THEN 'Delhi NCR'
    WHEN 'mumbai' THEN 'Mumbai' WHEN 'bombay' THEN 'Mumbai' WHEN 'mum' THEN 'Mumbai'
    WHEN 'pune' THEN 'Pune' WHEN 'pun' THEN 'Pune'
    WHEN 'hyderabad' THEN 'Hyderabad' WHEN 'hyd' THEN 'Hyderabad'
    WHEN 'jaipur' THEN 'Jaipur' WHEN 'jai' THEN 'Jaipur'
    ELSE trim({col}) END
"""

SWAP_COLS = {
    "event_id": "VARCHAR", "rider_id": "VARCHAR", "station_id": "VARCHAR", "event_ts": "TIMESTAMP",
    "event_type": "VARCHAR", "attempt_seq": "INTEGER", "queue_wait_sec": "INTEGER",
    "battery_in_id": "VARCHAR", "battery_out_id": "VARCHAR", "soc_in_pct": "DOUBLE",
    "soh_in_pct": "DOUBLE", "soc_out_pct": "DOUBLE", "soh_out_pct": "DOUBLE",
    "km_since_last_swap": "DOUBLE", "tariff_code": "VARCHAR", "list_price_inr": "DOUBLE",
    "discount_inr": "DOUBLE", "amount_charged_inr": "DOUBLE", "payment_mode": "VARCHAR",
    "energy_to_recharge_kwh": "DOUBLE", "station_firmware": "VARCHAR", "sync_mode": "VARCHAR",
}

def swap_reader(path):
    import json
    return (f"read_csv('{path}', columns={json.dumps(SWAP_COLS)}, header=true, delim=',', "
            f"quote='\"', escape='\"', strict_mode=false, max_line_size=10000000)")

def main():
    con = duckdb.connect(DB)
    con.execute("SET memory_limit='2GB'; SET threads=4; SET preserve_insertion_order=false;")

    p = lambda f: os.path.join(D, f).replace("\\", "/")

    con.execute(f"""
        CREATE OR REPLACE TABLE stations AS
        SELECT *, station_id LIKE 'STN-TST%' AS is_test,
               {CITY_CASE.format(col='city')} AS city_norm
        FROM read_csv_auto('{p("stations.csv")}', header=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE riders AS
        SELECT *, home_city AS home_city_raw,
               {CITY_CASE.format(col='home_city')} AS home_city_norm
        FROM read_csv_auto('{p("riders.csv")}', header=true)
    """)
    con.execute(f"CREATE OR REPLACE TABLE batteries AS SELECT * FROM read_csv_auto('{p('batteries.csv')}', header=true)")
    con.execute(f"CREATE OR REPLACE TABLE partners AS SELECT * FROM read_csv_auto('{p('fleet_partners.csv')}', header=true)")
    con.execute(f"""
        CREATE OR REPLACE TABLE city_ctx AS
        SELECT *, {CITY_CASE.format(col='city')} AS city_norm
        FROM read_csv_auto('{p("city_daily_context.csv")}', header=true)
    """)
    con.execute(f"CREATE OR REPLACE TABLE tickets AS SELECT * FROM read_csv_auto('{p('support_tickets.csv')}', header=true)")
    con.execute(f"CREATE OR REPLACE TABLE hourly AS SELECT * FROM read_csv_auto('{p('station_hourly_status.csv')}', header=true)")

    raw_count = con.execute(f"SELECT count(*) FROM {swap_reader(p('swap_events.csv'))}").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE raw_ev AS
        SELECT * FROM {swap_reader(p("swap_events.csv"))}
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE dedup_ev AS
        SELECT * EXCLUDE (prev_ts, prev_bin, prev_sync) FROM (
            SELECT *,
                   lag(event_ts) OVER w AS prev_ts,
                   lag(battery_in_id) OVER w AS prev_bin,
                   lag(sync_mode) OVER w AS prev_sync
            FROM raw_ev
            WINDOW w AS (PARTITION BY rider_id, station_id ORDER BY event_ts)
        )
        WHERE NOT (
            prev_ts IS NOT NULL
            AND date_diff('second', prev_ts, event_ts) BETWEEN 0 AND 180
            AND coalesce(battery_in_id,'NA') = coalesce(prev_bin,'NA')
            AND (sync_mode = 'offline_batch' OR prev_sync = 'offline_batch')
        )
    """)
    dedup_count = con.execute("SELECT count(*) FROM dedup_ev").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TABLE events AS
        SELECT e.*,
               e.event_ts + CASE WHEN e.station_firmware = 'v3.2.0'
                                  AND e.event_ts >= TIMESTAMP '2025-03-10'
                                  AND e.event_ts < TIMESTAMP '2025-04-15'
                                 THEN INTERVAL 330 MINUTE ELSE INTERVAL 0 MINUTE END AS ts_corrected,
               s.city_norm AS city, s.zone, s.location_type, s.host_type, s.expansion_wave,
               s.charger_generation, s.slots_2w, s.slots_3w, s.inventory_target_2w, s.inventory_target_3w,
               s.monthly_rent_inr, s.monthly_maintenance_inr, s.grid_tariff_inr_kwh,
               s.connectivity_tier, s.firmware_version, s.commissioned_date, s.is_test,
               (e.event_type = 'swap_completed') AS is_completed,
               CASE WHEN e.km_since_last_swap > 0 AND e.km_since_last_swap <= 300
                    THEN e.km_since_last_swap END AS km_valid
        FROM dedup_ev e
        LEFT JOIN stations s ON e.station_id = s.station_id
    """)
    con.execute("""
        ALTER TABLE events ADD COLUMN ts_bug BOOLEAN;
        UPDATE events SET ts_bug = (station_firmware = 'v3.2.0'
                                    AND event_ts >= TIMESTAMP '2025-03-10'
                                    AND event_ts < TIMESTAMP '2025-04-15');
    """)
    con.execute("""
        ALTER TABLE events ADD COLUMN date DATE;
        ALTER TABLE events ADD COLUMN hour INTEGER;
        ALTER TABLE events ADD COLUMN month VARCHAR;
        ALTER TABLE events ADD COLUMN week VARCHAR;
        ALTER TABLE events ADD COLUMN dow INTEGER;
        UPDATE events SET
            date = CAST(ts_corrected AS DATE),
            hour = EXTRACT(hour FROM ts_corrected),
            month = strftime(ts_corrected, '%Y-%m'),
            week = strftime(ts_corrected, '%Y-%W'),
            dow = EXTRACT(dow FROM ts_corrected);
    """)

    qc = {
        "swap_rows_raw": raw_count,
        "near_duplicates_removed": raw_count - dedup_count,
        "swap_rows_clean": dedup_count,
        "test_station_events": con.execute("SELECT count(*) FROM events WHERE is_test").fetchone()[0],
        "tz_bug_events_corrected": con.execute("SELECT count(*) FROM events WHERE ts_bug").fetchone()[0],
        "date_range_corrected": [str(x) for x in con.execute("SELECT min(ts_corrected), max(ts_corrected) FROM events").fetchone()],
        "km_invalid": con.execute("SELECT count(*) FROM events WHERE km_since_last_swap IS NOT NULL AND km_valid IS NULL").fetchone()[0],
        "soc_out_over_100": con.execute("SELECT count(*) FROM events WHERE soc_out_pct > 100").fetchone()[0],
        "event_type_counts": con.execute("SELECT event_type, count(*) FROM events GROUP BY 1 ORDER BY 2 DESC").fetchall(),
        "tariff_counts": con.execute("SELECT tariff_code, count(*) FROM events GROUP BY 1 ORDER BY 2 DESC").fetchall(),
        "home_city_variants": con.execute("SELECT count(DISTINCT lower(trim(home_city_raw))) AS raw_variants, count(DISTINCT home_city_norm) AS norm_cities FROM riders").fetchall(),
        "hourly_rows": con.execute("SELECT count(*) FROM hourly").fetchone()[0],
        "tickets": con.execute("SELECT count(*) FROM tickets").fetchone()[0],
    }
    con.execute("CHECKPOINT")
    con.close()
    with open(os.path.join(OUT, "qc.json"), "w") as f:
        json.dump(qc, f, indent=2, default=str)
    print(json.dumps(qc, indent=2, default=str))

if __name__ == "__main__":
    main()
