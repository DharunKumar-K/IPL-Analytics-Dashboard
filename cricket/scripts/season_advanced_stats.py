import os
import pandas as pd

# Paths
data_folder = os.path.join(os.path.dirname(__file__), "../data")
latest_csv = os.path.join(data_folder, "season_summary.csv")

# Load CSV
if not os.path.exists(latest_csv):
    raise FileNotFoundError(f"❌ CSV not found: {latest_csv}")

data = pd.read_csv(latest_csv)
print(f"📁 Using latest CSV file: {latest_csv}")
print(f"✅ Data loaded: {data.shape}")

# Check if CSV has rows
if data.empty:
    raise ValueError(f"❌ CSV is empty: {latest_csv}. Cannot compute advanced stats.")

# Check required columns
required_cols = ['runs', 'wickets']
for col in required_cols:
    if col not in data.columns:
        raise KeyError(f"❌ Column '{col}' is required in your dataset!")

# Compute advanced stats
# Example: runs per match, wickets per match
data['runs_per_match'] = data['runs'] / data['match_s_id']
data['wickets_per_match'] = data['wickets'] / data['match_s_id']

# Save
output_folder = os.path.join(data_folder, "plots")
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "season_advanced_stats.csv")
data.to_csv(output_file, index=False)
print(f"🎯 Advanced season stats saved successfully at: {output_file}")
