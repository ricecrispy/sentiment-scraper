from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Multi-Tier Feed Portfolio (Mainstream + Developer Communities + Research)
FEEDS: dict[str, str] = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Hacker News": "https://news.ycombinator.com/rss",
    "r/LocalLLaMA": "https://www.reddit.com/r/LocalLLaMA/.rss",
    "r/MachineLearning": "https://www.reddit.com/r/MachineLearning/.rss",
    "r/ArtificialIntelligence": "https://www.reddit.com/r/ArtificialIntelligence/.rss",
}

META_BLOCKLIST: tuple[str, ...] = (
    "megathread", "project showcase", "weekly", "bi-weekly", "ama", "poll"
)

# 2. Expanded Single Source of Truth for Global AI Ecosystems
ENTITY_MAPPING: dict[str, tuple[str, ...]] = {
    "OpenAI": ("openai", "chatgpt", "gpt-4", "gpt-5"),
    "Google / DeepMind": ("google ai", "gemini", "deepmind"),
    "Anthropic": ("anthropic", "claude"),
    "Meta / Llama Ecosystem": ("llama 3", "llama 2", "huggingface"),
    "xAI / Grok": ("xai", "grok"),
    "DeepSeek": ("deepseek",),
    "Alibaba / Qwen": ("alibaba", "qwen"),
    "Zhipu / GLM": ("zhipu", "glm-4", "glm-5"),
    "Moonshot / Kimi": ("moonshot", "kimi ai"),
}

# 3. Automatically derive TARGET_KEYWORDS from the mapping (Zero duplication)
TARGET_KEYWORDS: tuple[str, ...] = tuple(
    kw for keywords in ENTITY_MAPPING.values() for kw in keywords
)

@dataclass(frozen=True)
class SentimentRecord:
    source: str
    entity: str
    title: str
    score: float
    sentiment: str

def identify_entity(text: str) -> str:
    """Map text to a specific primary tech giant/ecosystem using the centralized mapping."""
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
            print(f"[Warning] Failed to parse feed cleanly for {source_name}: {feed.get('bozo_exception', 'Unknown error')}")
            return source_name, []
        return source_name, feed.entries
    except Exception as e:
        print(f"[Error] Network or parsing exception for {source_name}: {e}")
        return source_name, []

def fetch_and_analyze() -> list[SentimentRecord]:
    analyzer = SentimentIntensityAnalyzer()
    collected_data: list[SentimentRecord] = []

    print("Fetching and analyzing global multi-source AI sentiment concurrently...\n")

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
                combined_text = f"{title} {summary}"
                text_lower = combined_text.lower()

                if len(combined_text.strip()) < 40:
                    continue

                if any(block in text_lower for block in META_BLOCKLIST):
                    continue

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
                            source=source_name,
                            entity=entity,
                            title=title,
                            score=compound_score,
                            sentiment=sentiment
                        )
                    )

    return collected_data

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
    print(f"{'Entity / Ecosystem':<30} | {'Mentions':<10} | {'Avg Compound Score':<15}")
    print("-" * 65)

    for entity, scores in entity_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"{entity:<30} | {len(scores):<10} | {avg_score:+.4f}")
    
    print("-" * 65)
    print("\nDetailed Article Breakdown:")
    print("-" * 65)
    for item in results:
        print(f"[{item.sentiment} ({item.score:+.4f})] ({item.entity}) {item.source}")
        print(f"-> {item.title}\n")

if __name__ == "__main__":
    records = fetch_and_analyze()
    generate_summary_report(records)