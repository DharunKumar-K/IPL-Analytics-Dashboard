import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------
# 1️⃣ Load Data
# ------------------------
data_file = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/combined_player_match_s_format.csv"

try:
    df = pd.read_csv(data_file, parse_dates=['date'], low_memory=False)
    print(f"✅ Data loaded: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ CSV not found at: {data_file}")

# ------------------------
# 2️⃣ Check for Required Columns
# ------------------------
required_cols = ['venue', 'season', 'match_s_id', 'team', 'runs', 'wickets']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"❌ Missing required column: {col}")

# ------------------------
# 3️⃣ Standardize Team Names
# ------------------------
team_name_map = {
    'Delhi Capitals': 'Delhi Capitals',
    'Delhi Daredevils': 'Delhi Capitals',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Kings XI Punjab': 'Punjab Kings',
    'Punjab Kinks': 'Punjab Kings',
    'Punjab': 'Punjab Kings',
}
df['team'] = df['team'].replace(team_name_map)

# ------------------------
# 4️⃣ Setup Plot Directory
# ------------------------
plots_folder = "C:/Users/Dharun Kumar/PycharmProjects/cricket/plots/venues"
os.makedirs(plots_folder, exist_ok=True)
sns.set(style="whitegrid")

# ------------------------
# 5️⃣ Venue Summary Stats
# ------------------------
venue_stats = df.groupby('venue').agg(
    total_runs=('runs', 'sum'),
    total_wickets=('wickets', 'sum'),
    total_matches=('match_s_id', 'nunique')
).reset_index()

venue_stats['avg_runs_per_match'] = (venue_stats['total_runs'] / venue_stats['total_matches']).round(2)
venue_stats['avg_wickets_per_match'] = (venue_stats['total_wickets'] / venue_stats['total_matches']).round(2)

print("📊 Sample Venue Stats:")
print(venue_stats.head())

# ------------------------
# 6️⃣ Top 10 Venues by Average Runs
# ------------------------
top_venues = venue_stats.sort_values('avg_runs_per_match', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='avg_runs_per_match', y='venue', data=top_venues, palette="coolwarm")
plt.title("🏟️ Top 10 High-Scoring Venues")
plt.xlabel("Average Runs per Match")
plt.ylabel("Venue")
plt.tight_layout()
plt.savefig(f"{plots_folder}/top_venues_avg_runs.png")
plt.close()

# ------------------------
# 7️⃣ Top 10 Venues by Average Wickets
# ------------------------
top_wicket_venues = venue_stats.sort_values('avg_wickets_per_match', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='avg_wickets_per_match', y='venue', data=top_wicket_venues, palette="viridis")
plt.title("🎯 Top 10 Venues with Most Wickets per Match")
plt.xlabel("Average Wickets per Match")
plt.ylabel("Venue")
plt.tight_layout()
plt.savefig(f"{plots_folder}/top_venues_avg_wickets.png")
plt.close()

# ------------------------
# 8️⃣ Most Successful Teams per Venue
# ------------------------
team_venue_wins = df.groupby(['venue', 'team']).agg(
    total_runs=('runs', 'sum'),
    total_wickets=('wickets', 'sum')
).reset_index()

top_teams_per_venue = (
    team_venue_wins.sort_values(['venue', 'total_runs'], ascending=[True, False])
    .groupby('venue')
    .head(1)
    .reset_index(drop=True)
)

plt.figure(figsize=(12, 6))
sns.barplot(x='total_runs', y='venue', hue='team', data=top_teams_per_venue, dodge=False, palette="Set2")
plt.title("🏆 Top-Scoring Teams by Venue")
plt.xlabel("Total Runs Scored")
plt.ylabel("Venue")
plt.legend(title="Team", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{plots_folder}/top_team_per_venue.png")
plt.close()

print(f"✅ Venue stats and plots saved in: {plots_folder}/")
