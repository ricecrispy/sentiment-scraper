import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Define a multi-source configuration (Balanced portfolio of feeds)
FEEDS = {
    "Tech News (General)": "https://techcrunch.com/feed/",
    "The Verge (AI/Tech)": "https://www.theverge.com/rss/index.xml",
}

def fetch_articles():
    analyzer = SentimentIntensityAnalyzer()
    collected_data = []

    print("Fetching and analyzing recent tech discussions...\n")

    for source_name, url in FEEDS.items():
        feed = feedparser.parse(url)
        # Grab the latest 5 entries from each feed to keep it lightweight
        for entry in feed.entries[:5]:
            title = entry.title
            summary = entry.get(
                "summary", entry.get("description", "")
            )
            combined_text = f"{title} {summary}"

            # Simple keyword filter to focus specifically on AI tech giants
            keywords = ["openai", "google", "gemini", "chatgpt", "anthropic", "claude"]
            if any(kw in combined_text.lower() for kw in keywords):
                
                # 2. Run sentiment analysis immediately
                vs = analyzer.polarity_scores(combined_text)
                compound_score = vs["compound"] # Ranges from -1.0 (extreme negative) to +1.0 (extreme positive)

                # Classify sentiment bucket
                if compound_score >= 0.05:
                    sentiment = "Positive"
                elif compound_score <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                collected_data.append({
                    "source": source_name,
                    "title": title,
                    "score": compound_score,
                    "sentiment": sentiment
                })

    return collected_data

if __name__ == "__main__":
    results = fetch_articles()
    
    # 3. Display raw results
    for item in results:
        print(f"[{item['sentiment']} ({item['score']})] {item['source']}")
        print(f"-> {item['title']}\n")