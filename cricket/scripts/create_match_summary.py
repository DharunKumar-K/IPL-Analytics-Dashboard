import pandas as pd
import os

def create_match_summary(cleaned_csv: str, output_csv: str):
    print("📥 Loading cleaned data...")
    df = pd.read_csv(cleaned_csv, parse_dates=['date'], low_memory=False)
    print("✅ Loaded:", df.shape, "rows x columns")

    # Clean and normalize season
    df['season'] = df['season'].astype(str).str.strip()

    def get_start_year(season):
        try:
            return int(season.split('/')[0])
        except:
            return int(season)

    df['season_start_year'] = df['season'].apply(get_start_year)

    match_summaries = []
    grouped = df.groupby('match_id')

    for match_id, group in grouped:
        row = group.iloc[0]
        date = row['date']
        season = row['season']
        venue = row['venue']
        city = row['city']
        player_of_match = row['player_of_match']
        match_won_by = row['match_won_by']
        win_outcome = row.get('win_outcome', 'Unknown')

        # Teams
        teams = group['batting_team'].unique()
        team1, team2 = teams[0], teams[1] if len(teams) > 1 else (teams[0], 'Unknown')

        # Aggregate stats by innings & team
        innings_summary = group.groupby(['innings','batting_team']).agg({
            'runs_total': 'sum',
            'runs_extras': 'sum',
            'player_out': lambda x: x.notna().sum()
        }).reset_index()

        # Count legal balls (exclude wides and no-balls)
        team1_legal_balls = group[(group['batting_team']==team1) &
                                  (~group['extra_type'].isin(['wides','noballs']))].shape[0]
        team2_legal_balls = group[(group['batting_team']==team2) &
                                  (~group['extra_type'].isin(['wides','noballs']))].shape[0]

        # Initialize stats
        runs_team1 = extras_team1 = wickets_team1 = 0
        runs_team2 = extras_team2 = wickets_team2 = 0

        for _, r in innings_summary.iterrows():
            if r['batting_team'] == team1:
                runs_team1 = r['runs_total']
                extras_team1 = r['runs_extras']
                wickets_team1 = r['player_out']
            else:
                runs_team2 = r['runs_total']
                extras_team2 = r['runs_extras']
                wickets_team2 = r['player_out']

        match_summaries.append({
            'date': date,
            'season': season,
            'venue': venue,
            'city': city,
            'team1': team1,
            'team2': team2,
            'runs_team1': runs_team1,
            'extras_team1': extras_team1,
            'wickets_team1': wickets_team1,
            'balls_team1': team1_legal_balls,
            'runs_team2': runs_team2,
            'extras_team2': extras_team2,
            'wickets_team2': wickets_team2,
            'balls_team2': team2_legal_balls,
            'player_of_match': player_of_match,
            'match_won_by': match_won_by,
            'win_outcome': win_outcome,
            'season_start_year': row['season_start_year']
        })

    summary_df = pd.DataFrame(match_summaries)
    summary_df = summary_df.sort_values(['season_start_year','date']).reset_index(drop=True)

    # Map seasons to numbers
    unique_seasons = sorted(summary_df['season_start_year'].unique())
    season_map = {year: idx+1 for idx, year in enumerate(unique_seasons)}
    summary_df['season_no'] = summary_df['season_start_year'].map(season_map)

    # Assign match number per season
    summary_df['season_match_no'] = summary_df.groupby('season_no').cumcount() + 1

    # Match code like S1_01, S2_01
    summary_df['match_code'] = summary_df.apply(
        lambda x: f"S{x['season_no']}_{x['season_match_no']:02d}", axis=1
    )

    # Remove match_id column
    if 'match_id' in summary_df.columns:
        summary_df = summary_df.drop(columns=['match_id'])

    # Reorder columns
    cols = ['match_code'] + [c for c in summary_df.columns if c != 'match_code']
    summary_df = summary_df[cols]

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"💾 Match summary saved at: {output_csv}")
    print("✅ Total matches summarized:", len(summary_df))

    # Sample preview
    print("\n--- SAMPLE ---")
    print(summary_df[['season','season_no','season_match_no','match_code',
                      'team1','runs_team1','extras_team1','balls_team1','wickets_team1',
                      'team2','runs_team2','extras_team2','balls_team2','wickets_team2',
                      'match_won_by']].head(15))


if __name__ == "__main__":
    cleaned_csv = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/cleaned_matches.csv"
    output_csv = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/match_summary_final.csv"
    create_match_summary(cleaned_csv, output_csv)
