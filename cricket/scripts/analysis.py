# scripts/analysis.py
import pandas as pd
import numpy as np
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame with safe defaults.
    - low_memory=False reduces dtype guessing warnings.
    - keep default dtype inference; we will clean after loading.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        # Try alternative encoding if file isn't utf-8
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    print("✅ Loaded file:", path)
    print("Shape (rows, cols):", df.shape)
    return df


def inspect_df(df: pd.DataFrame, n_head: int = 5) -> None:
    """
    Print diagnostics to understand schema, missingness, and quick statistics.
    """
    print("\n--- COLUMNS ---")
    for i, c in enumerate(df.columns.tolist(), start=1):
        print(f"{i:02d}. {c}")

    print("\n--- INFO ---")
    print(df.info(memory_usage='deep'))

    print("\n--- FIRST", n_head, "ROWS ---")
    print(df.head(n_head))

    # Show sample last rows too (helpful to check footer rows or corrupt lines)
    print("\n--- LAST", min(n_head, len(df)), "ROWS ---")
    print(df.tail(n_head))

    # Missing values (counts + percentage)
    total = len(df)
    na_counts = df.isna().sum().sort_values(ascending=False)
    na_percent = (na_counts / total * 100).round(2)
    na_table = pd.concat([na_counts, na_percent], axis=1)
    na_table.columns = ['missing_count', 'missing_pct']
    print("\n--- MISSING VALUES (top 20) ---")
    print(na_table[na_table['missing_count'] > 0].head(20))

    # Basic unique counts for key columns
    for col in ['match_id', 'batting_team', 'bowling_team', 'venue', 'season']:
        if col in df.columns:
            print(f"\nUnique values in '{col}':", df[col].nunique())
            print("Sample unique (up to 10):", df[col].dropna().unique()[:10])

    # Distribution of rows per match (ball counts)
    if 'match_id' in df.columns:
        counts = df.groupby('match_id').size()
        print("\nRows per match (min, median, mean, max):",
              counts.min(), counts.median(), counts.mean(), counts.max())
        print("If max >> expected balls (e.g., > 300), there may be extras/metadata rows.")


def preview_value_counts(df: pd.DataFrame, col: str, top_n: int = 10) -> None:
    """Helper: show top value_counts for a column if it exists"""
    if col in df.columns:
        print(f"\nTop {top_n} value_counts for '{col}':")
        print(df[col].value_counts(dropna=False).head(top_n))
    else:
        print(f"\nColumn '{col}' not found in DataFrame.")


if __name__ == "__main__":
    # Quick local run when you run this module directly from PyCharm
    path = "../data/matches.csv"   # adjust if you run script from different cwd
    df = load_data(path)
    inspect_df(df)
    preview_value_counts(df, 'wicket_kind', top_n=15)
    preview_value_counts(df, 'extra_type', top_n=15)
    preview_value_counts(df, 'player_of_match', top_n=10)
