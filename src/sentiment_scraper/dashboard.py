from datetime import datetime
import json
import pathlib
import pandas as pd
import plotly.express as px
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Global AI Sentiment & Trend Tracker",
    page_icon="🌐",
    layout="wide",
)

# Title & Description
st.title("🌐 Global AI Sentiment & Trend Tracker")
st.markdown(
    "Monitoring public sentiment, breaking news, and developer discussions across global AI ecosystems."
)


# 1. Load, Transform, and Deduplicate Data
@st.cache_data
def load_sentiment_data() -> pd.DataFrame:
  file_path = pathlib.Path("sentiment_history.json")
  if not file_path.exists():
    return pd.DataFrame()

  try:
    with open(file_path, "r", encoding="utf-8") as f:
      runs = json.load(f)
  except json.JSONDecodeError:
    return pd.DataFrame()

  flat_records = []
  if isinstance(runs, dict):
    runs = [runs]

  for run in runs:
    records = run.get("records", [])
    for record in records:
      flat_records.append(record)

  if not flat_records:
    return pd.DataFrame()

  df = pd.DataFrame(flat_records)
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df["date"] = df["timestamp"].dt.date

  # Deduplicate based on unique article URL (keeping the latest capture)
  df = df.sort_values("timestamp", ascending=False).drop_duplicates(subset=["url"], keep="first")
  return df


df = load_sentiment_data()

if df.empty:
  st.warning(
      "No sentiment history found. Run your scraper first using `uv run src/sentiment_scraper/scraper.py`!"
  )
else:
  # 2. Sidebar Filters
  st.sidebar.header("🔍 Dashboard Filters")

  min_date = df["date"].min()
  max_date = df["date"].max()
  date_range = st.sidebar.date_input(
      "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
  )

  all_entities = sorted(df["entity"].unique().tolist())
  selected_entities = st.sidebar.multiselect(
      "Select Entities / Ecosystems", options=all_entities, default=all_entities
  )

  all_sources = sorted(df["source"].unique().tolist())
  selected_sources = st.sidebar.multiselect(
      "Select News Sources", options=all_sources, default=all_sources
  )

  # Apply Filters Safely
  if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
        & (df["entity"].isin(selected_entities))
        & (df["source"].isin(selected_sources))
    ]
  else:
    filtered_df = df[
        df["entity"].isin(selected_entities) & df["source"].isin(selected_sources)
    ]

  # 3. Top-Level Metric Summary Cards (Hero Metrics)
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Total Unique Articles", len(filtered_df))
  with col2:
    avg_score = (
        filtered_df["score"].mean() if not filtered_df.empty else 0.0
    )
    st.metric("Overall Avg Sentiment Score", f"{avg_score:+.4f}")
  with col3:
    unique_entities_count = filtered_df["entity"].nunique()
    st.metric("Active Entities Tracked", unique_entities_count)
  with col4:
    unique_sources_count = filtered_df["source"].nunique()
    st.metric("Active Sources", unique_sources_count)

  st.markdown("---")

  # 4. Main Visualizations
  if filtered_df.empty:
    st.info(
        "No records match your selected filter criteria. Try expanding your date"
        " range or selections."
    )
  else:
    col_left, col_right = st.columns([2, 1])

    with col_left:
      st.subheader("📈 Sentiment Trends Over Time")
      daily_agg = (
          filtered_df.groupby(["date", "entity"])["score"].mean().reset_index()
      )

      fig_trend = px.line(
          daily_agg,
          x="date",
          y="score",
          color="entity",
          markers=True,
          labels={"score": "Avg Compound Sentiment", "date": "Date"},
          title="Daily Average Sentiment by Ecosystem",
      )
      fig_trend.update_layout(hovermode="x unified", legend_title="Entity")
      st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
      st.subheader("📊 Mention Volume by Entity")
      entity_counts = filtered_df["entity"].value_counts().reset_index()
      entity_counts.columns = ["entity", "count"]

      fig_bar = px.bar(
          entity_counts,
          x="count",
          y="entity",
          orientation="h",
          text="count",
          labels={"count": "Mention Count", "entity": "Entity"},
          title="Total Mentions Share",
      )
      fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
      st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # 5. Interactive Evidence Ledger (Article Table with Clickable Links)
    st.subheader("📑 Article Evidence Ledger & Direct Links")
    st.markdown(
        "Explore individual unique articles captured by the pipeline, check their"
        " sentiment score, and click the link to read the original source."
    )

    display_df = filtered_df[
        ["timestamp", "entity", "source", "sentiment", "score", "title", "url"]
    ].copy()
    display_df = display_df.sort_values(by="timestamp", ascending=False)

    st.dataframe(
        display_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(
                "Captured At", format="YYYY-MM-DD HH:mm"
            ),
            "score": st.column_config.NumberColumn(
                "Score", format="%.4f"
            ),
            "url": st.column_config.LinkColumn(
                "Source Link", display_text="🔗 Read Article"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )