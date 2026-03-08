import requests
import json
import time

OUTPUT_FILE = "crawled_docs.json"
CRAWL_DELAY = 0.5  # seconds between requests

TOPICS = [
    # Core programmatic
    "Online advertising",
    "Targeted advertising",
    "Digital marketing",
    "Programmatic advertising",
    "Demand-side platform",
    "Supply-side platform",
    "Real-time bidding",
    "Ad exchange",
    "Ad serving",
    "Header bidding",
    # Data & audiences
    "Data management platform",
    "Customer data platform",
    "Lookalike audience",
    "Retargeting",
    "Behavioral targeting",
    "Contextual advertising",
    # Metrics & measurement
    "Cost per mille",
    "Click-through rate",
    "Conversion tracking",
    "Attribution (marketing)",
    "Viewability",
    # Deals & inventory
    "Private marketplace",
    "Programmatic direct",
    # Brand safety & fraud
    "Brand safety",
    "Ad fraud",
    "Ad blocking",
    # Privacy & regulation
    "General Data Protection Regulation",
    "California Consumer Privacy Act",
    "Children's Online Privacy Protection Act",
    "Internet privacy",
    "HTTP cookie",
    "Third-party cookie",
    "Consent management platform",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"


def fetch_article(title, session):
    """Fetch plain text of a Wikipedia article via the API."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
    }
    try:
        response = session.get(WIKI_API, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [error] {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            print(f"  — not found on Wikipedia")
            return None
        text = page.get("extract", "")
        if len(text.split()) < 100:
            print(f"  — too short, skipping")
            return None
        slug = page["title"].replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{slug}"
        return {
            "title": page["title"],
            "url": url,
            "text": " ".join(text.split())
        }
    return None


def crawl():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "AdTechDocsAssistant/1.0 (portfolio project; contact via rwhyte.com)"
    })

    print("=== AdTech Knowledge Crawler (Wikipedia) ===\n")
    print(f"Topics to fetch: {len(TOPICS)}\n")

    results = []
    skipped = 0

    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i}/{len(TOPICS)}] {topic}")
        time.sleep(CRAWL_DELAY)

        article = fetch_article(topic, session)
        if article:
            results.append(article)
            print(f"  ✓ {article['title']} ({len(article['text'].split())} words)")
        else:
            skipped += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Done ===")
    print(f"Articles fetched: {len(results)}")
    print(f"Skipped: {skipped}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    crawl()
    