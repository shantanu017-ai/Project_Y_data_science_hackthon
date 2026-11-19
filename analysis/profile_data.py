import pandas as pd, os, json

D = r"S:\project_Y\data\Hackathon Data Set _ Gradient"

def profile(name, nrows=None):
    path = os.path.join(D, name)
    df = pd.read_csv(path, nrows=nrows)
    print("=" * 100)
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df.dtypes.to_string())
    print("--- head")
    with pd.option_context("display.max_columns", 50, "display.width", 250):
        print(df.head(3).to_string())
    print("--- nulls")
    print(df.isna().sum().to_string())
    print()

for f in ["stations.csv", "fleet_partners.csv", "batteries.csv", "riders.csv",
          "support_tickets.csv", "city_daily_context.csv"]:
    profile(f)

with pd.option_context("display.max_columns", 50, "display.width", 250):
    profile("swap_events.csv", nrows=200000)
    profile("station_hourly_status.csv", nrows=200000)
