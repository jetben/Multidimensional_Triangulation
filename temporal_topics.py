# ci_cd_pipeline.py
# Full pipeline: fetch Stack Overflow posts (OR across tags) + GitHub repos + run BERTopic topic modeling

import argparse
import os
import time
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# -------------------------------
# Helpers
# -------------------------------
def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(" ", strip=True)


def _so_epoch(date_str):
    if not date_str:
        return None
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def _fetch_so_single_tag(tag, pages=3, pagesize=50, site="stackoverflow",
                         fromdate=None, todate=None):
    """Fetch SO questions for a single tag. Returns a DataFrame."""
    url = "https://api.stackexchange.com/2.3/questions"
    all_posts = []
    for page in range(1, pages + 1):
        print(f"[SO] Tag '{tag}' Page {page}/{pages}")
        params = {
            "page": page,
            "pagesize": pagesize,
            "order": "desc",
            "sort": "creation",
            "tagged": tag,            # single tag only
            "site": site,
            "filter": "withbody"
        }
        if fromdate:
            params["fromdate"] = fromdate
        if todate:
            params["todate"] = todate

        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"❌ SO error {resp.status_code}: {resp.text[:300]}")
            break

        data = resp.json()
        if "backoff" in data:
            # Respect backoff (seconds)
            bo = int(data["backoff"])
            print(f"⏳ SO backoff requested: sleeping {bo}s…")
            time.sleep(bo)

        for item in data.get("items", []):
            all_posts.append({
                "id": item["question_id"],
                "title": item.get("title", ""),
                "body": clean_html(item.get("body", "")),
                "created_at": item["creation_date"]
            })

        if not data.get("has_more", False):
            break
        time.sleep(0.5)

    df = pd.DataFrame(all_posts)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s").dt.strftime("%Y-%m-%d")
    return df


def fetch_so_posts_or(tags_csv, pages=3, pagesize=50, site="stackoverflow",
                      fromdate_str=None, todate_str=None):
    """
    Fetch SO posts for ANY of the given tags (OR logic).
    tags_csv: "github-actions,jenkins,gitlab-ci"
    """
    tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
    if not tags:
        return pd.DataFrame()

    fromdate = _so_epoch(fromdate_str) if fromdate_str else None
    todate = _so_epoch(todate_str) if todate_str else None

    frames = []
    for tag in tags:
        df = _fetch_so_single_tag(tag, pages=pages, pagesize=pagesize, site=site,
                                  fromdate=fromdate, todate=todate)
        if not df.empty:
            df["tag"] = tag
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # de-duplicate by question id
    out = out.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return out


def fetch_github_repos(query, pages=2, per_page=30, token=None):
    url = "https://api.github.com/search/repositories"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        print("ℹ️ No GitHub token provided. You are limited to ~60 requests/hour.")

    all_repos = []
    for page in range(1, pages + 1):
        print(f"[GitHub] Page {page}/{pages} for query: {query}")
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page, "page": page}
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"❌ GitHub error {resp.status_code}: {resp.text[:300]}")
            break
        data = resp.json()
        for item in data.get("items", []):
            all_repos.append({
                "id": item["id"],
                "name": item["full_name"],
                "description": item.get("description", ""),
                "stars": item["stargazers_count"],
                "language": item.get("language", ""),
                "created_at": item["created_at"][:10],
                "updated_at": item["updated_at"][:10],
                "url": item["html_url"]
            })
        time.sleep(1)
    return pd.DataFrame(all_repos)


# -------------------------------
# BERTopic pipeline
# -------------------------------
def run_topic_modeling(df, outdir="outputs", emb_model="all-MiniLM-L6-v2", nr_bins=20):
    os.makedirs(outdir, exist_ok=True)

    # Prepare docs
    df["text_clean"] = (df["title"].astype(str) + ". " + df["body"].astype(str)).str.replace(r"\s+", " ", regex=True)
    docs = df["text_clean"].tolist()
    timestamps = pd.to_datetime(df["created_at"], errors="coerce").tolist()

    # Encode
    print("🔤 Encoding documents with SentenceTransformers...")
    encoder = SentenceTransformer(emb_model)
    embeddings = encoder.encode(docs, show_progress_bar=True, normalize_embeddings=True)

    # Model
    print("🧠 Running BERTopic...")
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=5)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
    topics, _ = topic_model.fit_transform(docs, embeddings)

    # Save outputs
    info = topic_model.get_topic_info()
    info.to_csv(os.path.join(outdir, "topics.csv"), index=False)

    df_out = df.copy()
    df_out["topic"] = topics
    df_out.to_csv(os.path.join(outdir, "docs_with_topics.csv"), index=False)

    print("📈 Computing temporal topic trends...")
    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=nr_bins)
    topics_over_time.to_csv(os.path.join(outdir, "topics_over_time.csv"), index=False)

    # Plot topic sizes
    sizes = info[info["Topic"] >= 0][["Topic", "Count"]].sort_values("Count", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sizes)), sizes["Count"])
    plt.xticks(range(len(sizes)), sizes["Topic"].astype(str), rotation=90)
    plt.title("Topic Sizes (Number of Documents)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_topic_sizes.png"), dpi=150)
    plt.close()

    # Save top terms per topic
    top_terms = []
    for t in sizes["Topic"].tolist():
        words = [w for w, _ in topic_model.get_topic(t)[:10]]
        top_terms.append({"topic": t, "top_terms": ", ".join(words)})
    pd.DataFrame(top_terms).to_csv(os.path.join(outdir, "topic_top_terms.csv"), index=False)

    print(f"✅ Topic modeling finished. Results in {outdir}")


# -------------------------------
# Main
# -------------------------------
def main(args):
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # Fetch Stack Overflow with OR across tags
    so_df = fetch_so_posts_or(
        args.so_tags, pages=args.so_pages, pagesize=args.so_pagesize,
        fromdate_str=args.so_fromdate, todate_str=args.so_todate
    )
    if not so_df.empty:
        so_path = os.path.join("data", "so_posts.csv")
        so_df.to_csv(so_path, index=False)
        print(f"✅ Saved {len(so_df)} Stack Overflow posts to {so_path}")
    else:
        print("⚠️ No SO posts fetched.")
        return

    # Fetch GitHub (uses env var if CLI arg not provided)
    token = args.gh_token or os.getenv("GITHUB_TOKEN")
    gh_df = fetch_github_repos(args.gh_query, pages=args.gh_pages, per_page=args.gh_pagesize, token=token)
    if not gh_df.empty:
        gh_path = os.path.join("data", "github_repos.csv")
        gh_df.to_csv(gh_path, index=False)
        print(f"✅ Saved {len(gh_df)} GitHub repos to {gh_path}")
    else:
        print("⚠️ No GitHub repos fetched.")

    # Run topic modeling on SO posts
    run_topic_modeling(so_df, outdir=args.outdir, emb_model=args.emb_model, nr_bins=args.nr_bins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Stack Overflow
    parser.add_argument("--so_tags", type=str,
                        default="github-actions,continuous-integration,travis-ci,jenkins,gitlab-ci",
                        help="Comma-separated tags; fetched as OR across tags")
    parser.add_argument("--so_pages", type=int, default=5)
    parser.add_argument("--so_pagesize", type=int, default=50)
    parser.add_argument("--so_fromdate", type=str, default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--so_todate", type=str, default=None, help="YYYY-MM-DD (optional)")
    # GitHub
    parser.add_argument("--gh_query", type=str, default="github-actions language:yaml")
    parser.add_argument("--gh_pages", type=int, default=3)
    parser.add_argument("--gh_pagesize", type=int, default=30)
    parser.add_argument("--gh_token", type=str, default=None, help="GitHub personal access token (optional; env GITHUB_TOKEN used if unset)")
    # Outputs
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--emb_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--nr_bins", type=int, default=20)
    args = parser.parse_args()

    main(args)
