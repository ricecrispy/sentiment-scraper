from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import pathlib

from collections import defaultdict
import feedparser
from supabase import Client, create_client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment credentials for Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 1. Multi-Tier Feed Portfolio (Mainstream + Global Open Source / Developer Research)
FEEDS: dict[str, str] = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Hacker News": "https://news.ycombinator.com/rss",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    "r/LocalLLaMA": "https://www.reddit.com/r/LocalLLaMA/.rss",
    "r/MachineLearning": "https://www.reddit.com/r/MachineLearning/.rss",
    "r/ArtificialIntelligence": "https://www.reddit.com/r/ArtificialIntelligence/.rss",
}

META_BLOCKLIST: tuple[str, ...] = (
    "megathread",
    "project showcase",
    "weekly",
    "bi-weekly",
    "ama",
    "poll",
)

# 2. Resilient Brand-Root & Prefix Mapping (Future-proof against new model versions)
ENTITY_MAPPING: dict[str, tuple[str, ...]] = {
    "OpenAI": ("openai", "chatgpt", "gpt-"),
    "Google / DeepMind": ("google ai", "gemini", "deepmind"),
    "Anthropic": ("anthropic", "claude"),
    "Meta / Llama Ecosystem": ("meta ai", "llama", "huggingface"),
    "xAI / Grok": ("xai", "grok"),
    "DeepSeek": ("deepseek",),
    "Alibaba / Qwen": ("alibaba", "qwen"),
    "Zhipu / GLM": ("zhipu", "glm-"),
    "Moonshot / Kimi": ("moonshot", "kimi"),
}

# 3. Automatically derive TARGET_KEYWORDS from the mapping (Zero duplication)
TARGET_KEYWORDS: tuple[str, ...] = tuple(
    kw for keywords in ENTITY_MAPPING.values() for kw in keywords
)


@dataclass(frozen=True)
class SentimentRecord:
  timestamp: str
  entity: str
  source: str
  title: str
  url: str
  score: float
  sentiment: str


def identify_entity(text: str) -> str:
  """Map text to a specific primary tech giant/ecosystem using brand roots."""
  text_lower = text.lower()
  for entity, keywords in ENTITY_MAPPING.items():
    if any(kw in text_lower for kw in keywords):
      return entity
  return "General AI"


def fetch_single_feed(source_name: str, url: str) -> tuple[str, list]:
  """Helper to fetch and parse a single feed concurrently in a background thread."""
  try:
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", 0) == 1 and not feed.entries:
      print(
          f"[Warning] Failed to parse feed cleanly for {source_name}:"
          f" {feed.get('bozo_exception', 'Unknown error')}"
      )
      return source_name, []
    return source_name, feed.entries
  except Exception as e:
    print(f"[Error] Network or parsing exception for {source_name}: {e}")
    return source_name, []


def fetch_and_analyze() -> list[SentimentRecord]:
  analyzer = SentimentIntensityAnalyzer()
  collected_data: list[SentimentRecord] = []

  # Generate a unified UTC timestamp for this run batch
  run_timestamp = datetime.now(timezone.utc).isoformat()

  print(
      "Fetching and analyzing global multi-source AI sentiment"
      " concurrently...\n"
  )

  with ThreadPoolExecutor(max_workers=len(FEEDS)) as executor:
    future_to_source = {
        executor.submit(fetch_single_feed, name, url): name
        for name, url in FEEDS.items()
    }

    for future in as_completed(future_to_source):
      source_name, entries = future.result()

      for entry in entries[:20]:
        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))
        link = entry.get("link", "")
        combined_text = f"{title} {summary}"
        text_lower = combined_text.lower()

        # Heuristic Filter A: Skip very short entries
        if len(combined_text.strip()) < 40:
          continue

        # Heuristic Filter B: Exclude forum meta-noise
        if any(block in text_lower for block in META_BLOCKLIST):
          continue

        # Strict Keyword Filter (derived from ENTITY_MAPPING)
        if any(kw in text_lower for kw in TARGET_KEYWORDS):
          vs = analyzer.polarity_scores(combined_text)
          compound_score = vs["compound"]

          if compound_score >= 0.05:
            sentiment = "Positive"
          elif compound_score <= -0.05:
            sentiment = "Negative"
          else:
            sentiment = "Neutral"

          entity = identify_entity(combined_text)

          collected_data.append(
              SentimentRecord(
                  timestamp=run_timestamp,
                  entity=entity,
                  source=source_name,
                  title=title,
                  url=link,
                  score=compound_score,
                  sentiment=sentiment,
              )
          )

  return collected_data


def export_to_supabase(records: list[dict]):
  """Upserts records into Supabase PostgreSQL, relying on the 'url' UNIQUE constraint."""
  if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Supabase credentials missing from environment variables.")
    return

  supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

  try:
    response = supabase.table("sentiment_logs").upsert(
        records, on_conflict="url", ignore_duplicates=True
    ).execute()
    print("✅ Successfully synced records to Supabase.")
  except Exception as e:
    print(f"❌ Error syncing to Supabase: {e}")


def generate_summary_report(results: list[SentimentRecord]) -> None:
  if not results:
    print("No matching articles found to summarize.")
    return

  print("=" * 65)
  print(" GLOBAL PUBLIC SENTIMENT TRACKER: AGGREGATE SUMMARY REPORT")
  print("=" * 65)

  entity_scores = defaultdict(list)
  for item in results:
    entity_scores[item.entity].append(item.score)

  print(f"\nTotal Analyzed Mentions: {len(results)}\n")
  print("-" * 65)
  print(
      f"{'Entity / Ecosystem':<30} | {'Mentions':<10} | {'Avg Compound'}"
      " Score"
  )
  print("-" * 65)

  for entity, scores in entity_scores.items():
    avg_score = sum(scores) / len(scores)
    print(f"{entity:<30} | {len(scores):<10} | {avg_score:+.4f}")

  print("-" * 65)
  print("\nDetailed Article Breakdown:")
  print("-" * 65)
  for item in results:
    print(
        f"[{item.sentiment} ({item.score:+.4f})] ({item.entity})"
        f" {item.source}"
    )
    print(f"-> {item.title}")
    print(f"   Link: {item.url}\n")


if __name__ == "__main__":
  records = fetch_and_analyze()
  generate_summary_report(records)

  # Sync to Supabase Cloud Database
  supabase_payload = [asdict(r) for r in records]
  export_to_supabase(supabase_payload)