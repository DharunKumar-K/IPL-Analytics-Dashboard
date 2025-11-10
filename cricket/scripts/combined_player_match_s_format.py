import pandas as pd
import os

def create_combined_player_match_summary(cleaned_csv: str, match_csv: str, output_csv: str):
    print("📥 Loading cleaned data...")
    df = pd.read_csv(cleaned_csv, parse_dates=['date'], low_memory=False)
    match_info = pd.read_csv(match_csv)
    print(f"✅ Loaded cleaned matches: {df.shape}, match info: {match_info.shape}")

    # --- Create season-wise match number ---
    match_info['season_number'] = match_info['season'].rank(method='dense').astype(int)
    match_info['match_number_in_season'] = match_info.groupby('season')['match_id'].rank(method='first').astype(int)
    match_info['match_s_id'] = match_info.apply(
        lambda x: f"S{x['season_number']}_M{x['match_number_in_season']}", axis=1
    )

    # Merge S#_M# into main data
    df = pd.merge(df, match_info[['match_id','match_s_id']], on='match_id', how='left')
    df.drop(columns=['match_id'], inplace=True)  # remove original match_id

    # --- Batting summary per player per match ---
    df['balls_faced'] = 1
    batting_summary = df.groupby(['match_s_id','batter','batting_team']).agg(
        runs=('runs_batter','sum'),
        balls=('balls_faced','sum'),
        outs=('player_out', lambda x: x.notna().sum())
    ).reset_index()
    batting_summary.rename(columns={'batter':'player','batting_team':'team'}, inplace=True)

    # --- Bowling summary per player per match ---
    df['balls_bowled'] = 1
    wicket_df = df[df['player_out'].notna()]
    bowling_summary = df.groupby(['match_s_id','bowler','bowling_team']).agg(
        balls_bowled=('balls_bowled','sum'),
        runs_conceded=('runs_total','sum')
    ).reset_index()
    wickets_count = wicket_df.groupby(['match_s_id','bowler']).size().reset_index(name='wickets')
    bowling_summary = pd.merge(bowling_summary, wickets_count, on=['match_s_id','bowler'], how='left')
    bowling_summary['wickets'] = bowling_summary['wickets'].fillna(0).astype(int)
    bowling_summary.rename(columns={'bowler':'player','bowling_team':'team'}, inplace=True)

    # Merge batting & bowling stats
    player_summary = pd.merge(batting_summary, bowling_summary, on=['match_s_id','player','team'], how='outer')
    player_summary.fillna(0, inplace=True)

    # --- Derived metrics ---
    player_summary['strike_rate'] = player_summary.apply(
        lambda x: round((x['runs']/x['balls'])*100,2) if x['balls'] > 0 else 0, axis=1
    )
    player_summary['batting_average'] = player_summary.apply(
        lambda x: round((x['runs']/x['outs']),2) if x['outs'] > 0 else x['runs'], axis=1
    )
    player_summary['bowling_economy'] = player_summary.apply(
        lambda x: round((x['runs_conceded']/x['balls_bowled'])*6,2) if x['balls_bowled'] > 0 else 0, axis=1
    )
    player_summary['bowling_average'] = player_summary.apply(
        lambda x: round((x['runs_conceded']/x['wickets']),2) if x['wickets'] > 0 else x['runs_conceded'], axis=1
    )

    # --- Merge with match-level info ---
    combined_df = pd.merge(player_summary, match_info.drop(columns=['season_number','match_number_in_season','match_id']),
                           on='match_s_id', how='left')

    # Save final CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"💾 Combined player-match dataset saved at: {output_csv}")
    print("✅ Sample preview:")
    print(combined_df.head(15))


if __name__ == "__main__":
    cleaned_csv = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/cleaned_matches.csv"
    match_csv = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/match_summary.csv"
    output_csv = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/combined_player_match_s_format.csv"

    create_combined_player_match_summary(cleaned_csv, match_csv, output_csv)
