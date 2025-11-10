import pandas as pd
import numpy as np
import os


def clean_and_save(input_path: str, output_path: str) -> None:
    """Clean cricket match dataset and export a simplified CSV."""

    print("📥 Loading data...")
    df = pd.read_csv(input_path, low_memory=False)
    print("✅ Loaded:", df.shape, "rows x columns")

    # 1️⃣ Drop unwanted columns
    drop_cols = [
        'Unnamed: 0', 'review_batter', 'team_reviewed', 'review_decision',
        'umpire', 'umpires_call', 'review_batter', 'method', 'superover_winner',
        'result_type', 'fielders', 'new_batter', 'next_batter'
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # 2️⃣ Normalize text placeholders
    df.replace(['NA', 'NaN', 'Unknown', 'none', 'None', '', ' '], np.nan, inplace=True)

    # 3️⃣ Convert date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 4️⃣ Clean strings (remove trailing/leading spaces)
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # 5️⃣ Ensure numeric types
    numeric_cols = [
        'over', 'ball', 'ball_no', 'runs_batter', 'balls_faced',
        'runs_extras', 'runs_total', 'runs_bowler', 'balls_per_over',
        'team_runs', 'team_balls', 'team_wicket'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # 6️⃣ Fill categorical NaNs meaningfully
    if 'wicket_kind' in df.columns:
        df['wicket_kind'] = df['wicket_kind'].fillna('No Wicket')
    if 'extra_type' in df.columns:
        df['extra_type'] = df['extra_type'].fillna('No Extra')
    if 'match_won_by' in df.columns:
        df['match_won_by'] = df['match_won_by'].replace(np.nan, 'Unknown')

    # 7️⃣ Optional: remove duplicate rows if any
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"🧹 Removed {before - len(df)} duplicate rows.")

    # 8️⃣ Reorder columns logically
    preferred_order = [
        'match_id', 'date', 'season', 'event_name', 'match_type', 'venue', 'city',
        'innings', 'batting_team', 'bowling_team',
        'over', 'ball', 'ball_no',
        'batter', 'non_striker', 'bowler',
        'runs_batter', 'runs_extras', 'runs_total',
        'wicket_kind', 'player_out',
        'extra_type', 'bat_pos', 'balls_faced',
        'team_runs', 'team_balls', 'team_wicket',
        'player_of_match', 'match_won_by', 'win_outcome',
        'toss_winner', 'toss_decision', 'gender', 'team_type'
    ]
    df = df[[c for c in preferred_order if c in df.columns]]

    # 9️⃣ Save cleaned CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned dataset saved at: {output_path}")
    print("✅ Final shape:", df.shape)

    # 10️⃣ Show a small preview
    print("\n--- SAMPLE ROWS ---")
    print(df.head(5))


if __name__ == "__main__":
    # 🔹 Absolute paths
    input_file = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/matches.csv"
    output_file = "C:/Users/Dharun Kumar/PycharmProjects/cricket/data/cleaned_matches.csv"

    clean_and_save(input_file, output_file)
