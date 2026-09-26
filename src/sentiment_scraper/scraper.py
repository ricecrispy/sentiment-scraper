import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict

# 1. Multi-Tier Portfolio (Mainstream + Developer Communities + Research)
FEEDS = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Hacker News": "https://news.ycombinator.com/rss",
    "r/LocalLLaMA": "https://www.reddit.com/r/LocalLLaMA/.rss",
    "r/MachineLearning": "https://www.reddit.com/r/MachineLearning/.rss",
    "r/ArtificialIntelligence": "https://www.reddit.com/r/ArtificialIntelligence/.rss",
}

def identify_entity(text):
    """Map text to a specific primary tech giant/ecosystem for aggregation."""
    text = text.lower()
    if any(k in text for k in ["openai", "chatgpt", "gpt-4", "gpt-5"]):
        return "OpenAI"
    elif any(k in text for k in ["google ai", "gemini", "deepmind"]):
        return "Google / DeepMind"
    elif any(k in text for k in ["anthropic", "claude"]):
        return "Anthropic"
    elif any(k in text for k in ["localllama", "llama 3", "llama 2", "huggingface"]):
        return "Open Source / LLaMA Ecosystem"
    return "General AI"

def fetch_and_analyze():
    analyzer = SentimentIntensityAnalyzer()
    collected_data = []

    print("Fetching and analyzing scaled multi-source AI sentiment...\n")

    for source_name, url in FEEDS.items():
        feed = feedparser.parse(url)
        
        # Increased fetch depth to 20 entries per feed for a richer dataset
        for entry in feed.entries[:20]:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            combined_text = f"{title} {summary}"
            text_lower = combined_text.lower()

            # Heuristic Filter A: Skip very short entries (noise/stubs)
            if len(combined_text.strip()) < 40:
                continue

            # Heuristic Filter B: Exclude forum meta-noise
            meta_blocklist = ["megathread", "project showcase", "weekly", "bi-weekly", "ama", "poll"]
            if any(block in text_lower for block in meta_blocklist):
                continue

            # Strict Keyword Filter
            target_keywords = [
                "openai", "chatgpt", "gpt-4", "gpt-5", 
                "google ai", "gemini", "deepmind", 
                "anthropic", "claude", 
                "localllama", "llama 3", "llama 2", "huggingface"
            ]
            
            if any(kw in text_lower for kw in target_keywords):
                vs = analyzer.polarity_scores(combined_text)
                compound_score = vs["compound"]

                if compound_score >= 0.05:
                    sentiment = "Positive"
                elif compound_score <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                entity = identify_entity(combined_text)

                collected_data.append({
                    "source": source_name,
                    "entity": entity,
                    "title": title,
                    "score": compound_score,
                    "sentiment": sentiment
                })

    return collected_data

def generate_summary_report(results):
    if not results:
        print("No matching articles found to summarize.")
        return

    print("=" * 60)
    print(" PUBLIC SENTIMENT TRACKER: AGGREGATE SUMMARY REPORT")
    print("=" * 60)

    # Group scores by entity
    entity_scores = defaultdict(list)
    for item in results:
        entity_scores[item["entity"]].append(item["score"])

    print(f"\nTotal Analyzed Mentions: {len(results)}\n")
    print("-" * 60)
    print(f"{'Entity / Ecosystem':<30} | {'Mentions':<10} | {'Avg Compound Score':<15}")
    print("-" * 60)

    for entity, scores in entity_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"{entity:<30} | {len(scores):<10} | {avg_score:+.4f}")
    
    print("-" * 60)
    print("\nDetailed Article Breakdown:")
    print("-" * 60)
    for item in results:
        print(f"[{item['sentiment']} ({item['score']:+.4f})] ({item['entity']}) {item['source']}")
        print(f"-> {item['title']}\n")

if __name__ == "__main__":
    results = fetch_and_analyze()
    generate_summary_report(results)