import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import Client, create_client

# Page Config
st.set_page_config(
    page_title="Global AI Sentiment Tracker",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling (Dark-mode optimized)
st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1rem; color: #94A3B8; margin-bottom: 1.5rem; }
        .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def load_sentiment_data() -> pd.DataFrame:
  """Fetch sentiment log records directly from Supabase via the API client."""
  url = st.secrets.get("SUPABASE_URL")
  key = st.secrets.get("SUPABASE_KEY")

  if not url or not key:
    st.error(
        "Missing SUPABASE_URL or SUPABASE_KEY in .streamlit/secrets.toml"
    )
    return pd.DataFrame()

  try:
    supabase: Client = create_client(url, key)
    # Fetch the most recent 100 records to keep performance snappy and UI clean
    response = (
        supabase.table("sentiment_logs")
        .select("*")
        .order("timestamp", desc=True)
        .limit(100)
        .execute()
    )

    if not response.data:
      return pd.DataFrame()

    df = pd.DataFrame(response.data)

    if "timestamp" in df.columns:
      df["timestamp"] = pd.to_datetime(df["timestamp"])
      df["date"] = df["timestamp"].dt.date
      df = df.sort_values(by="timestamp", ascending=False)

    return df
  except Exception as e:
    st.error(f"Error querying Supabase: {e}")
    return pd.DataFrame()


def main():
  st.markdown(
      '<div class="main-header">🌐 Global AI Sentiment & Trend Tracker</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Monitoring public sentiment, breaking news, and'
      " developer discussions across global AI ecosystems.</div>",
      unsafe_allow_html=True,
  )

  df = load_sentiment_data()

  if df.empty:
    st.warning(
        "⚠️ No sentiment history found in Supabase. Check your connection"
        " secrets in `.streamlit/secrets.toml` or run `scraper.py` first!"
    )
    return

  # Sidebar Filters
  st.sidebar.header("🔍 Filters")

  all_entities = sorted(df["entity"].unique().tolist())
  selected_entities = st.sidebar.multiselect(
      "Select Entities / Ecosystems",
      options=all_entities,
      default=all_entities,
  )

  min_date = df["date"].min()
  max_date = df["date"].max()

  if min_date == max_date:
    date_range = (min_date, max_date)
  else:
    date_range = st.sidebar.date_input(
        "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

  # Apply Filters
  if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["entity"].isin(selected_entities))
        & (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ]
  else:
    filtered_df = df[df["entity"].isin(selected_entities)]

  if filtered_df.empty:
    st.info("No matching articles found for the selected filters.")
    return

  # 1. Metric Cards
  total_articles = len(filtered_df)
  avg_compound = filtered_df["score"].mean()
  positive_pct = (filtered_df["sentiment"] == "Positive").mean() * 100
  negative_pct = (filtered_df["sentiment"] == "Negative").mean() * 100

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Analyzed Mentions", f"{total_articles}")
  m2.metric("Avg Sentiment Score", f"{avg_compound:+.3f}")
  m3.metric("Positive Mentions", f"{positive_pct:.1f}%")
  m4.metric("Negative Mentions", f"{negative_pct:.1f}%")

  st.markdown("---")

  # 2. Charts Layout
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("📊 Average Sentiment by Entity")
    entity_agg = (
        filtered_df.groupby("entity")["score"]
        .mean()
        .reset_index()
        .sort_values(by="score", ascending=False)
    )
    fig_bar = px.bar(
        entity_agg,
        x="score",
        y="entity",
        orientation="h",
        color="score",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        labels={"score": "Avg Score", "entity": "Ecosystem"},
    )
    fig_bar.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

  with col2:
    st.subheader("📈 Sentiment Share Breakdown")
    sentiment_counts = (
        filtered_df["sentiment"].value_counts().reset_index()
    )
    sentiment_counts.columns = ["sentiment", "count"]
    fig_pie = px.pie(
        sentiment_counts,
        values="count",
        names="sentiment",
        color="sentiment",
        color_discrete_map={
            "Positive": "#2E7D32",
            "Neutral": "#757575",
            "Negative": "#C62828",
        },
        hole=0.4,
    )
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

  # 3. Article Log Table
  st.subheader("📰 Recent Articles & Sentiment Logs")

  display_columns = [
      "timestamp",
      "entity",
      "source",
      "sentiment",
      "score",
      "title",
      "url",
  ]
  existing_cols = [c for c in display_columns if c in filtered_df.columns]

  st.dataframe(
      filtered_df[existing_cols],
      column_config={
          "url": st.column_config.LinkColumn("Article Link"),
          "score": st.column_config.NumberColumn(
              "Compound Score", format="%+.4f"
          ),
          "timestamp": st.column_config.DatetimeColumn(
              "Timestamp", format="YYYY-MM-DD HH:mm UTC"
          ),
      },
      use_container_width=True,
      hide_index=True,
  )

  st.markdown("---")

  # 4. Data Sources & Community Feedback Section
  with st.expander("📡 Tracked Feeds, Sources & Methodology"):
    st.markdown("""
        This dashboard aggregates public news, research updates, and developer discussions in real-time from the following curated sources:
        """)

    active_sources = (
        sorted(filtered_df["source"].unique().tolist())
        if not filtered_df.empty
        else []
    )

    if active_sources:
      cols = st.columns(3)
      for i, source in enumerate(active_sources):
        with cols[i % 3]:
          st.markdown(f"- ✅ `{source}`")
    else:
      st.info("No sources currently active in the database view.")

    st.markdown("""
        ---
        **Have a suggestion for a new source?**  
        We are always looking to expand our coverage of global AI ecosystems, research labs, and developer communities. If there is a specific RSS feed, blog, or publication you think should be added to this tracker, please feel free to reach out or suggest it!
        """)


if __name__ == "__main__":
  main()