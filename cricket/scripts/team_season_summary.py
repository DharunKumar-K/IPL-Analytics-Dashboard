import os
import glob
import pandas as pd

# Folder containing player-level CSVs
data_folder = os.path.join(os.path.dirname(__file__), "data")

# Find CSVs
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"❌ No CSV files found in folder: {data_folder}")

# Use latest CSV
latest_csv = max(csv_files, key=os.path.getctime)
print(f"📁 Loading CSV file: {latest_csv}")

data = pd.read_csv(latest_csv)
print(f"✅ Data loaded: {data.shape}")

# Infer season from match_s_id if not present
if 'season' not in data.columns:
    if 'match_s_id' in data.columns:
        # Extract first 4 characters as season (adjust if format differs)
        data['season'] = data['match_s_id'].astype(str).str[:4]
        print(f"⚠️ 'season' column not found — inferred from 'match_s_id'")
    else:
        raise KeyError("❌ Cannot infer season: 'season' and 'match_s_id' not found!")

# Aggregate per team per season
if 'team' not in data.columns:
    raise KeyError("❌ 'team' column not found in dataset!")

team_summary = data.groupby(['season', 'team']).agg({
    'runs': 'sum',
    'wickets': 'sum',
    'match_s_id': 'nunique'  # number of matches
}).reset_index().rename(columns={'match_s_id': 'matches'})

# Save summary
output_file = os.path.join(data_folder, "team_season_summary.csv")
team_summary.to_csv(output_file, index=False)
print(f"🎯 Team season summary saved successfully at: {output_file}")
