import pandas as pd, os

D = r"S:\project_Y\data\Hackathon Data Set _ Gradient"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

sw = pd.read_csv(os.path.join(D, "swap_events.csv"), nrows=500000,
                 parse_dates=["event_ts"])
print("swap dtypes:\n", sw.dtypes)
print(sw.head(3).to_string())
print("\nevent_type:\n", sw.event_type.value_counts(dropna=False))
print("\ntariff:\n", sw.tariff_code.value_counts(dropna=False))
print("\npayment_mode:\n", sw.payment_mode.value_counts(dropna=False))
print("\nsync_mode:\n", sw.sync_mode.value_counts(dropna=False))
print("\nfirmware:\n", sw.station_firmware.value_counts(dropna=False))
print("\nevent_ts range:", sw.event_ts.min(), sw.event_ts.max())
print("nulls:\n", sw.isna().sum().to_string())
print("queue stats:", sw.queue_wait_sec.describe().to_string())
print("km invalid <=0:", (sw.km_since_last_swap <= 0).sum(), " >300:", (sw.km_since_last_swap > 300).sum())
print("soc_in >100:", (sw.soc_in_pct > 100).sum(), "soh range:", sw.soh_in_pct.min(), sw.soh_in_pct.max())

st = pd.read_csv(os.path.join(D, "stations.csv"))
for c in ["city", "expansion_wave", "charger_generation", "location_type", "host_type", "connectivity_tier", "firmware_version"]:
    print("\nstations", c, ":\n", st[c].value_counts().to_string())
print("\ncommissioned range:", st.commissioned_date.min(), st.commissioned_date.max())
print("decommissioned:\n", st[st.decommissioned_date.notna()][["station_id","city","decommissioned_date","expansion_wave"]].to_string())
print("test stations:", st[st.station_id.str.startswith("STN-TST")][["station_id","city","expansion_wave"]].to_string())

bat = pd.read_csv(os.path.join(D, "batteries.csv"))
print("\nbattery supplier:\n", bat.supplier.value_counts().to_string())
print("\npack type:\n", bat.pack_type.value_counts().to_string())
print("\nretirement:\n", bat.retirement_reason.value_counts(dropna=False).to_string())
print("\ncurrent_soh:", bat.current_soh_pct.describe().to_string())
print(bat.groupby(["supplier","pack_type"]).current_soh_pct.agg(["count","mean","std"]).to_string())
print("\nlots:\n", bat.manufacturing_lot.value_counts().head(20).to_string())

rd = pd.read_csv(os.path.join(D, "riders.csv"))
print("\nhome_city raw:\n", rd.home_city.value_counts().to_string())
print("\nsignup channels:\n", rd.signup_channel.value_counts().to_string())
print("\nplan:\n", rd.plan_type.value_counts().to_string())

tk = pd.read_csv(os.path.join(D, "support_tickets.csv"))
print("\nticket category:\n", tk.category.value_counts().to_string())
print("\nchannel:\n", tk.channel.value_counts().to_string())
print("\nresolution:\n", tk.resolution_status.value_counts().to_string())

hs = pd.read_csv(os.path.join(D, "station_hourly_status.csv"), nrows=300000, parse_dates=["hour_start"])
print("\nhourly head:\n", hs.head(3).to_string())
print("\ntelemetry:\n", hs.telemetry_status.value_counts().to_string())
print("\nhourly nulls:\n", hs.isna().sum().to_string())
