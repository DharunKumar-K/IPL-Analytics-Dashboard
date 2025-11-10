"""
Streamlit Cricket Analytics Dashboard
Run: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide", page_title="Cricket Analytics Dashboard", initial_sidebar_state="expanded")

# ------------------------
# Helper Functions
# ------------------------
def infer_and_rename_cols(df):
    aliases = {
        "player": ["player", "batter", "batsman", "player_name"],
        "team": ["team", "team_name", "batting_team", "bat_team"],
        "match_s_id": ["match_s_id", "match_s", "s_match_id"],
        "match_id": ["match_id", "match"],
        "season": ["season", "Season", "year", "season_year"],
        "runs": ["runs", "runs_batter", "batter_runs", "player_runs"],
        "balls": ["balls", "balls_faced", "batter_balls"],
        "wickets": ["wickets", "bowler_wicket", "bowler_wickets"],
        "balls_bowled": ["balls_bowled", "balls_bowled_by", "bowler_balls"],
        "runs_conceded": ["runs_conceded", "runs_bowler", "conceded"],
        "match_won_by": ["match_won_by", "winner", "match_winner"],
        "venue": ["venue", "ground", "stadium"],
        "city": ["city"],
    }
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for target, possibles in aliases.items():
        for name in possibles:
            if name.lower() in cols_lower:
                rename_map[cols_lower[name.lower()]] = target
                break
    df = df.rename(columns=rename_map)
    return df

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, low_memory=False)

def load_data_prefer_path(default_path="data/combined_player_match_s_format.csv"):
    if os.path.exists(default_path):
        try:
            return load_csv(default_path)
        except Exception:
            return None
    return None

# ------------------------
# Title & Data Load
# ------------------------
st.title("🏏 IPL Advanced Analytics Dashboard")

df_raw = load_data_prefer_path()
if df_raw is None:
    st.error("❌ Data file missing. Place it in /data/ and name it correctly.")
    st.stop()

df = infer_and_rename_cols(df_raw.copy())

# Basic cleaning
for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype(str)
for c in ["runs", "balls", "balls_bowled", "wickets", "runs_conceded"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
if "season" not in df.columns and "match_s_id" in df.columns:
    df["season"] = df["match_s_id"].astype(str).str[:4]

# ------------------------
# Sidebar Filters
# ------------------------
seasons = sorted(df["season"].dropna().unique().astype(str))
selected_season = st.sidebar.selectbox("Select Season", seasons, index=len(seasons)-1)
min_balls = st.sidebar.number_input("Min balls faced / bowled", 0, 200, 100, 10)
min_innings = st.sidebar.number_input("Min innings", 1, 20, 7, 1)
top_n = st.sidebar.slider("Top N", 3, 30, 10)
show_raw = st.sidebar.checkbox("Show raw data")

season_df = df[df["season"].astype(str) == str(selected_season)].copy()
if season_df.empty:
    st.warning("No data for this season")
    st.stop()

# ------------------------
# Player Aggregations
# ------------------------
bats = season_df.groupby("player").agg(
    runs_total=("runs", "sum"),
    balls_total=("balls", "sum"),
    innings=("match_s_id", "nunique")
).reset_index()
bats["avg_score"] = (bats["runs_total"]/bats["innings"]).round(2)
bats["strike_rate"] = (bats["runs_total"]/bats["balls_total"]*100).round(2)
bats_eligible = bats[(bats["balls_total"] >= min_balls) & (bats["innings"] >= min_innings)]

bowl = season_df.groupby("player").agg(
    wickets_total=("wickets", "sum"),
    balls_bowled_total=("balls_bowled", "sum"),
    runs_conceded_total=("runs_conceded", "sum"),
    innings=("match_s_id","nunique")
).reset_index()
bowl["economy"] = (bowl["runs_conceded_total"]/(bowl["balls_bowled_total"]/6)).round(2)
bowl_eligible = bowl[(bowl["balls_bowled_total"] >= min_balls) & (bowl["innings"] >= min_innings)]

top_bats = bats_eligible.sort_values("runs_total", ascending=False).head(top_n)
top_bowl = bowl_eligible.sort_values("wickets_total", ascending=False).head(top_n)

# ------------------------
# Team Stats
# ------------------------
team = season_df.groupby("team").agg(
    runs_total=("runs", "sum"),
    wickets_total=("wickets", "sum"),
    matches=("match_id", "nunique")
).reset_index()
team["avg_runs"] = (team["runs_total"]/team["matches"]).round(1)
team["avg_wickets"] = (team["wickets_total"]/team["matches"]).round(1)
if "match_won_by" in season_df.columns:
    wins = season_df[["match_id", "team", "match_won_by"]].drop_duplicates()
    win_count = wins.groupby("team").apply(lambda g: (g["match_won_by"] == g["team"]).sum()).reset_index(name="wins")
    team = team.merge(win_count, on="team", how="left")
    team["win_pct"] = (team["wins"]/team["matches"]*100).round(1)
else:
    team["win_pct"] = np.nan

# ------------------------
# Venue Stats
# ------------------------
venue = season_df.groupby("venue").agg(
    total_runs=("runs", "sum"),
    total_wickets=("wickets", "sum"),
    matches=("match_id","nunique")
).reset_index()
venue["avg_score"] = (venue["total_runs"]/venue["matches"]/2).round(1)
venue["avg_wickets"] = (venue["total_wickets"]/venue["matches"]/2).round(1)

# ------------------------
# Charts Section
# ------------------------
st.markdown(f"### 📅 Season {selected_season} Overview")

# Player Stats
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 🏆 Top Run Scorers (Horizontal Bar)")
    fig = px.bar(top_bats, y="player", x="runs_total", orientation="h",
                 text="runs_total", color="runs_total", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("#### 🎯 Top Wicket Takers (Funnel Chart)")
fig = px.funnel(
    top_bowl,
    y="player",
    x="wickets_total",
    color="player",  # Use player names as categorical colors
    title="Top Wicket Takers"
)
st.plotly_chart(fig, use_container_width=True)

# Strike Rate & Economy (Independent)
st.markdown("#### ⚡ Strike Rate & Economy (Min 100 balls, Min 7 innings)")
sr_top = bats_eligible.sort_values("strike_rate", ascending=False).head(top_n)
econ_top = bowl_eligible.sort_values("economy").head(top_n)

col_sr, col_econ = st.columns(2)
with col_sr:
    fig = px.scatter(sr_top, x="innings", y="strike_rate", size="runs_total", color="runs_total",
                     hover_name="player", title="Best Strike Rates")
    st.plotly_chart(fig, use_container_width=True)
with col_econ:
    fig = px.scatter(econ_top, x="innings", y="economy", size="wickets_total", color="wickets_total",
                     hover_name="player", title="Best Economies")
    st.plotly_chart(fig, use_container_width=True)

# Team Performance
st.markdown("#### 🏟️ Team Performance Overview")
team_fig = go.Figure()
team_fig.add_trace(go.Bar(x=team["team"], y=team["avg_runs"], name="Avg Runs"))
team_fig.add_trace(go.Scatter(x=team["team"], y=team["avg_wickets"], mode="markers+lines", name="Avg Wickets", marker=dict(color="red", size=12)))
st.plotly_chart(team_fig, use_container_width=True)

# Team Win %
st.markdown("#### 🏆 Team Win % (Pie Chart)")
win_fig = px.pie(team, names="team", values="win_pct", title="Team Win % Share")
st.plotly_chart(win_fig, use_container_width=True)

# Venue Performance
st.markdown("#### 🏠 Venue Performance")
venue_fig = px.bar(venue.sort_values("avg_score", ascending=False), x="venue", y="avg_score",
                   text="avg_score", title="Avg Score by Venue")
st.plotly_chart(venue_fig, use_container_width=True)

venue_fig2 = px.scatter(venue.sort_values("avg_wickets", ascending=False), x="venue", y="avg_wickets",
                        size="avg_wickets", color="avg_wickets", text="avg_wickets", title="Avg Wickets by Venue")
st.plotly_chart(venue_fig2, use_container_width=True)

# Player Trends (Match-wise)
st.markdown("#### 📈 Player Trends (Match-wise)")
trend1, trend2 = st.columns(2)
with trend1:
    top5_bats = top_bats["player"].head(5).tolist()
    trend_df = season_df[season_df["player"].isin(top5_bats)].groupby(["match_s_id","player"]).runs.sum().reset_index()
    fig = px.line(trend_df, x="match_s_id", y="runs", color="player", markers=True, title="Top 5 Batsmen Trends")
    st.plotly_chart(fig, use_container_width=True)

with trend2:
    top5_bowls = top_bowl["player"].head(5).tolist()
    trend_df_b = season_df[season_df["player"].isin(top5_bowls)].groupby(["match_s_id","player"]).wickets.sum().reset_index()
    fig = px.line(trend_df_b, x="match_s_id", y="wickets", color="player", markers=True, title="Top 5 Bowlers Trends")
    st.plotly_chart(fig, use_container_width=True)

# Optional Raw Data
if show_raw:
    st.subheader("Raw Data (Filtered)")
    st.dataframe(season_df)
