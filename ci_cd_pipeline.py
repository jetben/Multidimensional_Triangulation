# fetch_so_rich.py (Python 3.9 compatible)
# Usage:
#   python fetch_so_rich.py --tags "github-actions,continuous-integration,jenkins,gitlab-ci,travis-ci" \
#       --pages 5 --pagesize 50 --fromdate 2017-01-01 --todate 2025-12-31 --out data/so_posts.csv

import argparse
import os
import time
import requests
import pandas as pd
from typing import Optional
from datetime import datetime
from bs4 import BeautifulSoup


def clean_html(s: str) -> str:
    if not s:
        return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)


def _to_epoch(dstr: Optional[str]):
    """YYYY-MM-DD -> epoch seconds (int) for Stack Exchange API."""
    if not dstr:
        return None
    return int(datetime.strptime(dstr, "%Y-%m-%d").timestamp())


def fetch_so_posts_or_rich(tags_csv: str,
                           pages: int = 3,
                           pagesize: int = 50,
                           fromdate: Optional[int] = None,
                           todate: Optional[int] = None,
                           site: str = "stackoverflow") -> pd.DataFrame:
    """
    Fetch Stack Overflow questions using OR over tags and return a DataFrame with
    all fields needed for survival/resolution analysis.
    Includes: id, title, body, created_at, last_activity_date, is_answered,
              answer_count, accepted_answer_id, score, view_count, tags, source_tag.
    """
    tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
    all_rows = []
    url = "https://api.stackexchange.com/2.3/questions"

    for tag in tags:
        for page in range(1, pages + 1):
            params = {
                "page": page,
                "pagesize": pagesize,
                "order": "desc",
                "sort": "creation",
                "tagged": tag,
                "site": site,
                # 'withbody' includes body + common metadata (is_answered, last_activity_date, etc.)
                "filter": "withbody",
            }
            if fromdate is not None:
                params["fromdate"] = fromdate
            if todate is not None:
                params["todate"] = todate

            print(f"[SO] Tag '{tag}' Page {page}/{pages}")
            r = requests.get(url, params=params)
            if r.status_code != 200:
                print(f"❌ SO {r.status_code}: {r.text[:200]}")
                break
            data = r.json()

            if "backoff" in data:
                bo = int(data["backoff"])
                print(f"⏳ Backoff requested: sleeping {bo}s")
                time.sleep(bo)

            for it in data.get("items", []):
                created_ts = it.get("creation_date")
                last_act_ts = it.get("last_activity_date", created_ts)
                row = {
                    "id": it.get("question_id"),
                    "title": it.get("title", ""),
                    "body": clean_html(it.get("body", "")),
                    # keep raw epochs + ISO dates for convenience
                    "created_at_epoch": created_ts,
                    "last_activity_epoch": last_act_ts,
                    "created_at": (
                        datetime.utcfromtimestamp(created_ts).strftime("%Y-%m-%d")
                        if created_ts else None
                    ),
                    "last_activity_date": (
                        datetime.utcfromtimestamp(last_act_ts).strftime("%Y-%m-%d")
                        if last_act_ts else None
                    ),
                    "is_answered": it.get("is_answered", False),
                    "answer_count": it.get("answer_count", 0),
                    "accepted_answer_id": it.get("accepted_answer_id"),
                    "score": it.get("score", 0),
                    "view_count": it.get("view_count", 0),
                    "tags": ",".join(it.get("tags", [])),
                    "source_tag": tag,
                }
                all_rows.append(row)

            if not data.get("has_more", False):
                break
            time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        # ensure numeric id for merging with docs_with_topics.csv
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        # normalize ISO date strings
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["last_activity_date"] = pd.to_datetime(df["last_activity_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", type=str, required=True,
                    help="Comma-separated SO tags; OR logic is used across them")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--pagesize", type=int, default=50)
    ap.add_argument("--fromdate", type=str, default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--todate", type=str, default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--out", type=str, default="data/so_posts.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = fetch_so_posts_or_rich(
        tags_csv=args.tags,
        pages=args.pages,
        pagesize=args.pagesize,
        fromdate=_to_epoch(args.fromdate),
        todate=_to_epoch(args.todate)
    )
    if df.empty:
        print("⚠️ No posts fetched.")
        return
    df.to_csv(args.out, index=False)
    print(f"✅ Saved {len(df)} posts to {args.out}")


if __name__ == "__main__":
    main()
