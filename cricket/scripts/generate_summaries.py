import pandas as pd

# Load the combined player-match CSV
player_file = r"C:\Users\Dharun Kumar\PycharmProjects\cricket\data\combined_player_match_s_format.csv"
data = pd.read_csv(player_file)

# --------------------------
# 1️⃣ Team summary
# --------------------------
team_summary = data.groupby(['season', 'team']).agg(
    matches=('match_s_id', 'nunique'),
    runs_scored=('runs', 'sum'),
    wickets_taken=('wickets', 'sum'),
    balls_faced=('balls', 'sum'),
    balls_bowled=('balls_bowled', 'sum'),
    wins=('win_outcome', lambda x: (x=='win').sum())
).reset_index()

team_summary['avg_runs'] = team_summary['runs_scored'] / team_summary['matches']
team_summary['avg_rpo'] = (team_summary['runs_scored'] / team_summary['balls_faced']) * 6
team_summary['win_pct'] = (team_summary['wins'] / team_summary['matches']) * 100

team_summary.to_csv(r"C:\Users\Dharun Kumar\PycharmProjects\cricket\data\team_summary.csv", index=False)
print("✅ team_summary.csv created!")

# --------------------------
# 2️⃣ Venue summary
# --------------------------
venue_summary = data.groupby(['season', 'venue']).agg(
    matches=('match_s_id', 'nunique'),
    total_runs=('runs', 'sum'),
    total_wickets=('wickets', 'sum'),
    avg_runs=('runs', 'mean')
).reset_index()

venue_summary.to_csv(r"C:\Users\Dharun Kumar\PycharmProjects\cricket\data\venue_summary.csv", index=False)
print("✅ venue_summary.csv created!")

# --------------------------
# 3️⃣ Season summary
# --------------------------
season_summary = data.groupby(['season']).agg(
    total_matches=('match_s_id', 'nunique'),
    total_runs=('runs', 'sum'),
    total_wickets=('wickets', 'sum'),
    avg_runs=('runs', 'mean'),
    avg_wickets=('wickets', 'mean')
).reset_index()

season_summary.to_csv(r"C:\Users\Dharun Kumar\PycharmProjects\cricket\data\season_summary.csv", index=False)
print("✅ season_summary.csv created!")
