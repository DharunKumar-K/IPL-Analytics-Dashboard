# main.py
from scripts.analysis import load_data, inspect_df

if __name__ == "__main__":
    data_path = "data/matches.csv"   # path relative to project root
    df = load_data(data_path)
    inspect_df(df, n_head=5)
