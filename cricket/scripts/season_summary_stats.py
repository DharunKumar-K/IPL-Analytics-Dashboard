# season_match_advanced.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------
# Config & paths
# ------------------------
DATA_CSV = r"C:/Users/Dharun Kumar/PycharmProjects/cricket/data/combined_player_match_s_format.csv"
PLOTS_DIR = r"C:/Users/Dharun Kumar/PycharmProjects/cricket/plots"
SUMMARY_CSV = r"C:/Users/Dharun Kumar/PycharmProjects/cricket/season_summary_stats.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set(style="whitegrid")

# ------------------------
# Load data (safe)
# ------------------------
try:
    df = pd.read_csv(DATA_CSV, parse_dates=['date'], low_memory=False)
    print("✅ Loaded:", df.shape)
except FileNotFoundError:
    raise FileNotFoundError(f"Input CSV not found: {DATA_CSV}")

# ------------------------
# Basic column normalization
# ------------------------
# detect team column (player-level 'team' is used to mean player's team)
possible_team_cols = ['team', 'bat_team', 'batting_team']
team_col = next((c for c in possible_team_cols if c in df.columns), None)
if team_col is None:
    raise KeyError(f"None of expected team columns found: {possible_team_cols}")

# Standardize team names (extend mapping as needed)
team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Delhi Capitals': 'Delhi Capitals',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Royal Challengers Banglore': 'Royal Challengers Bangalore',
    'RCB': 'Royal Challengers Bangalore',
    'Kings XI Punjab': 'Punjab Kings',
    'Punjab Kinks': 'Punjab Kings',
    'Punjab': 'Punjab Kings'
}
df[team_col] = df[team_col].replace(team_name_map)

# helper for safe filenames
def safe(s):
    return str(s).replace('/', '_').replace('\\', '_')

# ------------------------
# Derive / ensure helpful columns
# ------------------------
# Ensure match-level columns exist if present in combined CSV (they should per earlier conversation)
# We'll attempt to use these columns if available:
has_match_info = all(col in df.columns for col in ['match_s_id', 'match_id', 'season', 'team1', 'team2'])
if not has_match_info:
    print("⚠️ Warning: some match-level columns (match_s_id/match_id/season/team1/team2) are missing - some outputs may be incomplete.")

# Some columns used below: runs, wickets, balls, balls_bowled, runs_conceded, runs_team1, runs_team2, balls_team1, balls_team2, wickets_team1, wickets_team2, match_won_by, venue
# We will check presence before using.
available_cols = set(df.columns)

# ------------------------
# 1) Team Win % per Season
# uses match-level match_won_by and team1/team2 to compute matches played
# ------------------------
team_win_stats = None
if {'match_s_id', 'season', 'match_won_by', 'team1', 'team2'}.issubset(available_cols):
    matches = df[['match_s_id', 'season', 'match_id', 'team1', 'team2', 'match_won_by']].drop_duplicates(subset=['match_s_id'])
    # matches played per team per season
    m1 = matches[['season', 'team1', 'match_s_id']].rename(columns={'team1': 'team'})
    m2 = matches[['season', 'team2', 'match_s_id']].rename(columns={'team2': 'team'})
    matches_played = pd.concat([m1, m2], ignore_index=True).drop_duplicates().groupby(['season', 'team']).size().reset_index(name='matches_played')
    # wins per team
    wins = matches.groupby(['season', 'match_won_by']).size().reset_index(name='wins').rename(columns={'match_won_by': 'team'})
    team_win_stats = pd.merge(matches_played, wins, on=['season', 'team'], how='left').fillna({'wins': 0})
    team_win_stats['wins'] = team_win_stats['wins'].astype(int)
    team_win_stats['win_pct'] = round(team_win_stats['wins'] / team_win_stats['matches_played'] * 100, 2)
    # plot per season
    for season in sorted(team_win_stats['season'].unique()):
        sd = team_win_stats[team_win_stats['season'] == season].sort_values('win_pct', ascending=False)
        if sd.empty: continue
        plt.figure(figsize=(10,6))
        sns.barplot(x='win_pct', y='team', data=sd, palette='viridis', dodge=False, hue=None)
        plt.title(f"Team Win% - Season {season}")
        plt.xlabel("Win Percentage")
        plt.ylabel("Team")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"team_win_pct_{safe(season)}.png"))
        plt.close()

else:
    print("⚠️ Not enough match-level columns to compute win percentages (need match_won_by, team1, team2, match_s_id).")

# ------------------------
# 2) Team runs per over (avg) per season using runs_team1/runs_team2 and balls_team1/balls_team2
# ------------------------
if {'runs_team1','runs_team2','balls_team1','balls_team2','season','team1','team2'}.issubset(available_cols):
    match_level = df[['match_s_id','season','team1','team2','runs_team1','runs_team2','balls_team1','balls_team2']].drop_duplicates(subset=['match_s_id'])
    # create rows per team per match with runs and balls
    rows = []
    for _, r in match_level.iterrows():
        rows.append({'season': r['season'], 'team': r['team1'], 'runs': r['runs_team1'], 'balls': r['balls_team1'], 'match_s_id': r['match_s_id']})
        rows.append({'season': r['season'], 'team': r['team2'], 'runs': r['runs_team2'], 'balls': r['balls_team2'], 'match_s_id': r['match_s_id']})
    team_innings = pd.DataFrame(rows)
    # avoid division by zero
    team_innings['overs'] = team_innings['balls'].replace(0, np.nan) / 6.0
    team_innings['rpo'] = team_innings['runs'] / team_innings['overs']
    team_rpo = team_innings.groupby(['season','team']).agg(total_runs=('runs','sum'), total_overs=('overs','sum'), innings=('match_s_id','nunique')).reset_index()
    team_rpo['avg_rpo'] = round(team_rpo['total_runs'] / team_rpo['total_overs'], 2)
    # plot avg rpo per season
    for season in sorted(team_rpo['season'].unique()):
        sd = team_rpo[team_rpo['season'] == season].sort_values('avg_rpo', ascending=False)
        if sd.empty: continue
        plt.figure(figsize=(10,6))
        sns.barplot(x='avg_rpo', y='team', data=sd, palette='magma', dodge=False, hue=None)
        plt.title(f"Team Average Runs Per Over - Season {season}")
        plt.xlabel("Avg Runs Per Over")
        plt.ylabel("Team")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"team_avg_rpo_{safe(season)}.png"))
        plt.close()
else:
    print("⚠️ Missing team innings level columns (runs_team1/runs_team2/balls_team1/balls_team2) to compute runs-per-over.")

# ------------------------
# 3) Highest individual score per season (player per match)
# we use df which likely contains player-level rows per delivery; aggregate player x match
# ------------------------
if {'season','match_s_id','player','runs'}.issubset(available_cols):
    player_match = df.groupby(['season','match_s_id','player']).runs.sum().reset_index(name='player_match_runs')
    max_scores = player_match.loc[player_match.groupby('season')['player_match_runs'].idxmax()].reset_index(drop=True)
    # write top individual scores plot per season
    for _, row in max_scores.iterrows():
        s = row['season']; player = row['player']; runs = row['player_match_runs']
        # We'll create a small text file per season? Instead plot a bar with top scorer
        plt.figure(figsize=(6,3))
        sns.barplot(x=[runs], y=[player], palette='rocket', dodge=False, hue=None)
        plt.title(f"Highest Individual Score - {s}: {player} ({runs})")
        plt.xlabel("Runs")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"highest_individual_{safe(s)}.png"))
        plt.close()
else:
    print("⚠️ Not enough player-level data to calculate highest individual scores (need season, match_s_id, player, runs).")

# ------------------------
# 4) Best bowling figures (wickets & runs conceded) per season (player in a match)
# ------------------------
# we need wickets per player per match and runs conceded per player per match
if {'season','match_s_id','player','wickets','runs_conceded'}.issubset(available_cols):
    bowl_pm = df.groupby(['season','match_s_id','player']).agg(wickets=('wickets','sum'), runs_conceded=('runs_conceded','sum')).reset_index()
    # choose best by wickets descending, then runs_conceded ascending
    best_bowling = bowl_pm.sort_values(['season','wickets','runs_conceded'], ascending=[True, False, True]).groupby('season').first().reset_index()
    # plot best bowling bar per season
    for _, row in best_bowling.iterrows():
        s = row['season']; player = row['player']; w = row['wickets']; rc = row['runs_conceded']
        plt.figure(figsize=(6,3))
        sns.barplot(x=[w], y=[player], palette='mako', dodge=False, hue=None)
        plt.title(f"Best Bowling - {s}: {player} ({int(w)}/{int(rc)})")
        plt.xlabel("Wickets")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"best_bowling_{safe(s)}.png"))
        plt.close()
else:
    print("⚠️ Missing bowling per-match columns (wickets, runs_conceded) to compute best bowling figures.")

# ------------------------
# 5) Top strike rate and top economy (with minimum ball thresholds)
# ------------------------
# Build season-player aggregates
if {'season','player'}.issubset(available_cols) and ('runs' in available_cols):
    aggs = df.groupby(['season','player']).agg(
        total_runs=('runs','sum'),
        balls_faced=('balls','sum') if 'balls' in available_cols else ('runs','count'),
        balls_bowled=('balls_bowled','sum') if 'balls_bowled' in available_cols else 0,
        runs_conceded=('runs_conceded','sum') if 'runs_conceded' in available_cols else 0,
        wickets=('wickets','sum') if 'wickets' in available_cols else 0
    ).reset_index()
    # compute strike rate if balls_faced > 0
    aggs['strike_rate'] = np.where(aggs['balls_faced']>0, aggs['total_runs']/aggs['balls_faced']*100, np.nan)
    aggs['economy'] = np.where(aggs['balls_bowled']>0, aggs['runs_conceded']/(aggs['balls_bowled']/6), np.nan)
    # thresholds
    min_balls_bat = 100
    min_balls_bowl = 100
    for season in sorted(aggs['season'].unique()):
        sd = aggs[aggs['season']==season]
        # top strike rate (min balls)
        sr_candidates = sd[sd['balls_faced']>=min_balls_bat].sort_values('strike_rate', ascending=False).head(10)
        if not sr_candidates.empty:
            plt.figure(figsize=(10,6))
            sns.barplot(x='strike_rate', y='player', data=sr_candidates, palette='cool', dodge=False, hue=None)
            plt.title(f"Top Strike Rate (min {min_balls_bat} balls) - Season {season}")
            plt.xlabel("Strike Rate")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"top_strike_rate_{safe(season)}.png"))
            plt.close()
        # top economy (min balls bowled)
        ec_candidates = sd[sd['balls_bowled']>=min_balls_bowl].sort_values('economy', ascending=True).head(10)
        if not ec_candidates.empty:
            plt.figure(figsize=(10,6))
            sns.barplot(x='economy', y='player', data=ec_candidates, palette='cividis', dodge=False, hue=None)
            plt.title(f"Best Economy (min {min_balls_bowl} balls) - Season {season}")
            plt.xlabel("Economy")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"best_economy_{safe(season)}.png"))
            plt.close()
else:
    print("⚠️ Not enough columns for strike rate/economy (need balls, balls_bowled, runs_conceded).")

# ------------------------
# 6) Venue Insights: average first-innings score, avg wickets per innings, high scoring venues
# ------------------------
if {'venue','runs_team1','runs_team2','wickets_team1','wickets_team2','match_s_id'}.issubset(available_cols):
    match_level = df[['match_s_id','venue','runs_team1','runs_team2','wickets_team1','wickets_team2']].drop_duplicates(subset=['match_s_id'])
    # avg first-innings score by venue (runs_team1)
    venue_first = match_level.groupby('venue').agg(avg_first_innings=('runs_team1','mean'), matches=('match_s_id','nunique')).reset_index()
    venue_first = venue_first.sort_values('avg_first_innings', ascending=False).head(20)
    plt.figure(figsize=(12,8))
    sns.barplot(x='avg_first_innings', y='venue', data=venue_first, palette='viridis', dodge=False, hue=None)
    plt.title("Top venues by average first-innings score (top 20)")
    plt.xlabel("Avg First-Innings Runs")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "venue_avg_first_innings.png"))
    plt.close()

    # avg wickets per innings (combine both innings)
    match_level['avg_wickets_innings'] = (match_level['wickets_team1'] + match_level['wickets_team2']) / 2.0
    venue_wk = match_level.groupby('venue').agg(avg_wickets=('avg_wickets_innings','mean')).reset_index().sort_values('avg_wickets', ascending=False).head(20)
    plt.figure(figsize=(12,8))
    sns.barplot(x='avg_wickets', y='venue', data=venue_wk, palette='rocket', dodge=False, hue=None)
    plt.title("Venues by avg wickets per innings (top 20)")
    plt.xlabel("Avg Wickets per Innings")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "venue_avg_wickets.png"))
    plt.close()

    # most high-scoring venues by avg total match runs
    match_level['total_match_runs'] = match_level['runs_team1'] + match_level['runs_team2']
    venue_score = match_level.groupby('venue').agg(avg_match_runs=('total_match_runs','mean'), matches=('match_s_id','nunique')).reset_index().sort_values('avg_match_runs', ascending=False).head(20)
    plt.figure(figsize=(12,8))
    sns.barplot(x='avg_match_runs', y='venue', data=venue_score, palette='mako', dodge=False, hue=None)
    plt.title("Venues by average total match runs (top 20)")
    plt.xlabel("Avg Total Match Runs")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "venue_avg_total_runs.png"))
    plt.close()
else:
    print("⚠️ Venue-level match columns missing for venue insights (need venue, runs_team1/2, wickets_team1/2).")

# ------------------------
# 7) Match-level run rate comparison first vs second innings (if balls info present)
# ------------------------
if {'runs_team1','balls_team1','runs_team2','balls_team2','match_s_id'}.issubset(available_cols):
    ml = df[['match_s_id','runs_team1','balls_team1','runs_team2','balls_team2']].drop_duplicates(subset=['match_s_id'])
    ml['rpo_first'] = ml['runs_team1'] / (ml['balls_team1'].replace(0, np.nan) / 6.0)
    ml['rpo_second'] = ml['runs_team2'] / (ml['balls_team2'].replace(0, np.nan) / 6.0)
    ml = ml.dropna(subset=['rpo_first','rpo_second'])
    plt.figure(figsize=(8,8))
    plt.scatter(ml['rpo_first'], ml['rpo_second'], alpha=0.6)
    plt.plot([0, max(ml['rpo_first'].max(), ml['rpo_second'].max())],[0, max(ml['rpo_first'].max(), ml['rpo_second'].max())], color='red', linestyle='--')
    plt.xlabel("First Innings RPO")
    plt.ylabel("Second Innings RPO")
    plt.title("First vs Second Innings Runs Per Over (RPO)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "first_vs_second_rpo.png"))
    plt.close()
else:
    print("⚠️ Missing balls_team1/2 to compare first vs second innings run rates.")

# ------------------------
# 8) Season summary CSV (top-level)
# For each season: highest team score, lowest team score, top scorer and runs, top bowler and wickets, avg match runs
# ------------------------
summary_rows = []
seasons = sorted(df['season'].dropna().unique())
# For player-level aggregates we'll reuse player_match and bowl_pm if available
player_match_exists = {'season','match_s_id','player','runs'}.issubset(available_cols)
bowl_pm_exists = {'season','match_s_id','player','wickets','runs_conceded'}.issubset(available_cols)
match_level_cols_exist = {'match_s_id','runs_team1','runs_team2','season'}.issubset(available_cols)

for season in seasons:
    row = {'season': season}
    # highest & lowest team score in that season (from match-level fields)
    if match_level_cols_exist:
        ml_season = df[df['season']==season][['match_s_id','runs_team1','runs_team2']].drop_duplicates('match_s_id')
        # flatten scores
        scores = pd.concat([ml_season['runs_team1'], ml_season['runs_team2']], ignore_index=True)
        row['highest_team_score'] = int(scores.max()) if not scores.empty else None
        row['lowest_team_score'] = int(scores.min()) if not scores.empty else None
        row['avg_match_runs'] = round((ml_season['runs_team1'] + ml_season['runs_team2']).mean(),2) if not ml_season.empty else None
    else:
        row['highest_team_score'] = None
        row['lowest_team_score'] = None
        row['avg_match_runs'] = None

    # top scorer
    if player_match_exists:
        pm_s = player_match[player_match['season']==season] if 'player_match' in locals() else None
        if pm_s is not None and not pm_s.empty:
            top = pm_s.sort_values('player_match_runs', ascending=False).iloc[0]
            row['top_scorer'] = top['player']
            row['top_scorer_runs'] = int(top['player_match_runs'])
        else:
            row['top_scorer'] = None
            row['top_scorer_runs'] = None
    else:
        row['top_scorer'] = None
        row['top_scorer_runs'] = None

    # top bowler
    if bowl_pm_exists:
        bp = bowl_pm[bowl_pm['season']==season] if 'bowl_pm' in locals() else None
        if bp is not None and not bp.empty:
            topb = bp.sort_values(['wickets','runs_conceded'], ascending=[False, True]).iloc[0]
            row['top_bowler'] = topb['player']
            row['top_bowler_wickets'] = int(topb['wickets'])
        else:
            row['top_bowler'] = None
            row['top_bowler_wickets'] = None
    else:
        row['top_bowler'] = None
        row['top_bowler_wickets'] = None

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"💾 Season summary CSV saved: {SUMMARY_CSV}")

print("✅ Advanced stats & plots complete. Plots folder:", PLOTS_DIR)
