#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub Actions Health & Failures (with verbose logging)

Usage examples:
  python github_logs_analysis.py --repo pytorch/pytorch --outdir outputs --pages 3 --verbose 1
  python github_logs_analysis.py --repo owner/repo --from 2024-01-01 --to 2025-12-31 --verbose 1
  python github_logs_analysis.py --repo owner/repo --link_so 1

Notes:
  - Reads token from --token or env GITHUB_TOKEN
  - Writes CSVs and PNGs into --outdir (default: outputs)
  - Metrics: failure rate, success rate, retry rate, MTTR (hours), flakiness ratio
  - Verbose mode prints URLs, status codes, and items/page

Dependencies:
  pip install pandas numpy matplotlib requests python-dateutil
"""

import os
import re
import time
import json
import math
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import isoparse

API_BASE = "https://api.github.com"
DEFAULT_PER_PAGE = 100

# ---------------------------
# Logging helpers
# ---------------------------
def log(verbose: int, *args):
    if verbose:
        print(*args)

def gh_headers(token):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def handle_rate_limit(resp, verbose=0):
    # If we hit rate limit, sleep until reset if possible
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset:
            wait = max(0, int(reset) - int(time.time())) + 2
            log(verbose, f"⏳ Rate limit hit. Sleeping {wait}s until reset …")
            time.sleep(wait)
            return True
    return False

def fetch_paginated(url, headers, params=None, pages=10, verbose=0):
    """Generic GitHub pagination helper with verbose logging and rate-limit handling."""
    params = dict(params or {})
    all_items = []
    for page in range(1, pages + 1):
        p = params.copy()
        p["per_page"] = p.get("per_page", DEFAULT_PER_PAGE)
        p["page"] = page

        log(verbose, f"GET {url} page={page} per_page={p['per_page']} …")
        resp = requests.get(url, headers=headers, params=p)
        if handle_rate_limit(resp, verbose):
            # retry once after sleeping
            resp = requests.get(url, headers=headers, params=p)

        log(verbose, f"  -> HTTP {resp.status_code}")
        if resp.status_code != 200:
            text = resp.text[:400].replace("\n", " ")
            print(f"❌ HTTP {resp.status_code}: {text}")
            break

        data = resp.json()
        # Runs endpoint returns {"total_count":..., "workflow_runs":[...]}
        # Jobs endpoint returns {"total_count":..., "jobs":[...]}
        items = data.get("workflow_runs") or data.get("jobs") or data
        if isinstance(items, dict):  # last fallback
            items = items.get("items", [])
        if not isinstance(items, list):
            items = []

        log(verbose, f"  -> items this page: {len(items)}")
        all_items.extend(items)

        if len(items) < p["per_page"]:
            log(verbose, "  -> last page (less than per_page).")
            break

        time.sleep(0.35)  # be nice to the API
    log(verbose, f"TOTAL items fetched: {len(all_items)}")
    return all_items

# ---------------------------
# Fetchers
# ---------------------------
def fetch_runs(repo, token=None, created_from=None, created_to=None, pages=10, verbose=0):
    url = f"{API_BASE}/repos/{repo}/actions/runs"
    headers = gh_headers(token)
    runs = fetch_paginated(url, headers, params={}, pages=pages, verbose=verbose)
    df = pd.DataFrame(runs)
    if df.empty:
        return df

    # Keep relevant columns even if missing
    keep = [
        "id","name","event","status","conclusion","run_attempt","run_number",
        "head_sha","head_branch","created_at","updated_at","run_started_at","html_url"
    ]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan

    # Parse datetimes
    for col in ["created_at","updated_at","run_started_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration (sec): updated - run_started (or created)
    base = df["run_started_at"].fillna(df["created_at"])
    df["duration_sec"] = (df["updated_at"] - base).dt.total_seconds()

    # Filter by date (client-side)
    if created_from:
        df = df[df["created_at"] >= pd.to_datetime(created_from)]
    if created_to:
        df = df[df["created_at"] <= pd.to_datetime(created_to)]

    df = df.reset_index(drop=True)
    log(verbose, f"Runs after date filter: {len(df)}")
    return df

def fetch_jobs_for_runs(repo, run_ids, token=None, pages=5, verbose=0):
    headers = gh_headers(token)
    rows = []
    for idx, rid in enumerate(run_ids, 1):
        url = f"{API_BASE}/repos/{repo}/actions/runs/{rid}/jobs"
        log(verbose, f"[{idx}/{len(run_ids)}] Fetching jobs for run {rid} …")
        jobs = fetch_paginated(url, headers, params={}, pages=pages, verbose=verbose)
        for j in jobs:
            row = {
                "run_id": rid,
                "job_id": j.get("id"),
                "name": j.get("name"),
                "status": j.get("status"),
                "conclusion": j.get("conclusion"),
                "started_at": j.get("started_at"),
                "completed_at": j.get("completed_at"),
                "steps": json.dumps([
                    {"name": s.get("name"), "conclusion": s.get("conclusion")}
                    for s in (j.get("steps") or [])
                ])
            }
            rows.append(row)
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["started_at","completed_at"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["duration_sec"] = (df["completed_at"] - df["started_at"]).dt.total_seconds()
    return df

# ---------------------------
# Metrics
# ---------------------------
def compute_repo_metrics(runs_df):
    if runs_df.empty:
        return pd.DataFrame([{
            "runs": 0, "success_rate": np.nan, "failure_rate": np.nan,
            "avg_duration_sec": np.nan, "retry_rate": np.nan,
            "flaky_ratio": np.nan, "mttr_hours": np.nan
        }])

    df = runs_df.copy()
    df["month"] = df["created_at"].dt.to_period("M").astype(str)

    total = len(df)
    success = (df["conclusion"] == "success").sum()
    failures = (df["conclusion"] == "failure").sum()
    success_rate = success / total if total else np.nan
    failure_rate = failures / total if total else np.nan

    # Retry proxy: run_attempt > 1
    retry_rate = (df["run_attempt"].fillna(1) > 1).mean()

    # Flaky: same head_sha fails then succeeds within 48h
    flaky = 0
    g = df.sort_values("created_at").groupby("head_sha", dropna=True)
    for sha, block in g:
        conc = block["conclusion"].tolist()
        times = block["created_at"].tolist()
        for i, c in enumerate(conc):
            if c == "failure":
                # look ahead for first success
                for j in range(i+1, len(conc)):
                    if conc[j] == "success" and (times[j] - times[i]).total_seconds() <= 48*3600:
                        flaky += 1
                        break
    flaky_ratio = flaky / max(1, failures)

    # MTTR: failure to next success (hours)
    mttr_list = []
    sorted_df = df.sort_values("created_at")
    for i, row in sorted_df.iterrows():
        if row["conclusion"] == "failure":
            next_success = sorted_df[(sorted_df["created_at"] > row["created_at"]) &
                                     (sorted_df["conclusion"] == "success")].head(1)
            if not next_success.empty:
                delta = (next_success["created_at"].iloc[0] - row["created_at"]).total_seconds()
                mttr_list.append(delta/3600.0)
    mttr_hours = float(np.mean(mttr_list)) if mttr_list else np.nan

    avg_dur = float(np.nanmean(df["duration_sec"]))

    return pd.DataFrame([{
        "runs": total,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "retry_rate": float(retry_rate),
        "flaky_ratio": float(flaky_ratio),
        "mttr_hours": mttr_hours,
        "avg_duration_sec": avg_dur
    }])

def compute_monthly_metrics(runs_df):
    if runs_df.empty:
        return pd.DataFrame()
    df = runs_df.copy()
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    out = df.groupby("month").apply(lambda g: pd.Series({
        "runs": len(g),
        "success_rate": (g["conclusion"] == "success").mean(),
        "failure_rate": (g["conclusion"] == "failure").mean(),
        "retry_rate": (g["run_attempt"].fillna(1) > 1).mean(),
        "avg_duration_sec": g["duration_sec"].mean()
    })).reset_index()
    return out.sort_values("month")

# ---------------------------
# Failure taxonomy (heuristic)
# ---------------------------
TAXON_PATTERNS = {
    "dependency": r"(module not found|no such file|could not resolve|pip install failed|npm err|gradle .* failed)",
    "infra": r"(timeout|network|connection reset|no space left|runner (unavailable|offline))",
    "tests_flaky": r"(flake|flaky|intermittent|random fail|rerun passed)",
    "secrets": r"(invalid credentials|permission denied|auth failed|unauthorized|secret|token|github_token)",
    "config_yaml": r"(yaml|workflow file|invalid syntax|unexpected key|matrix)",
    "cache_artifact": r"(cache|artifact|restore|save cache failed)"
}

def classify_failure_from_jobs(jobs_df):
    if jobs_df.empty:
        return pd.DataFrame(columns=["category","count"])
    df = jobs_df.copy()
    df["text"] = df["name"].fillna("") + " " + df["steps"].fillna("")
    cats = []
    for _, row in df.iterrows():
        text = str(row["text"]).lower()
        hit = None
        for cat, pat in TAXON_PATTERNS.items():
            if re.search(pat, text):
                hit = cat
                break
        if hit:
            cats.append(hit)
    ser = pd.Series(cats).value_counts()
    return ser.reset_index().rename(columns={"index":"category","value":"count"})

# ---------------------------
# Optional: link with SO monthly topic shares
# ---------------------------
def link_with_so_monthly(outdir, monthly_metrics_csv="metrics_monthly.csv",
                         so_monthly_csv="topics_monthly_share.csv"):
    mm_path = os.path.join(outdir, monthly_metrics_csv)
    so_path = os.path.join(outdir, so_monthly_csv)
    if not (os.path.exists(mm_path) and os.path.exists(so_path)):
        print("ℹ️ Skipping SO linkage (monthly files not found).")
        return None
    mm = pd.read_csv(mm_path)
    so = pd.read_csv(so_path)  # month, topic, count, total, share

    # If topic_top_terms.csv exists, flag "flaky-like" topics by terms
    tt_path = os.path.join(outdir, "topic_top_terms.csv")
    if os.path.exists(tt_path):
        tt = pd.read_csv(tt_path)
        tt["flag_flaky"] = tt["top_terms"].str.contains(r"(test|flake|retry|fail|ci)", case=False, regex=True)
        flaky_topics = set(tt[tt["flag_flaky"]]["topic"].astype(int).tolist())
        flaky_share = so[so["topic"].isin(flaky_topics)].groupby("month")["share"].mean().rename("flaky_topic_share")
    else:
        flaky_share = so.groupby("month")["share"].mean().rename("flaky_topic_share")

    merged = mm.merge(flaky_share.reset_index(), on="month", how="left")
    merged.to_csv(os.path.join(outdir, "link_so_github_monthly.csv"), index=False)

    corr = merged[["failure_rate","retry_rate","flaky_topic_share"]].corr()
    corr.to_csv(os.path.join(outdir, "link_so_github_corr.csv"), index=False)
    print("🔗 Saved linkage: link_so_github_monthly.csv, link_so_github_corr.csv")
    return merged

# ---------------------------
# Plots
# ---------------------------
def plot_monthly(mm, outdir):
    if mm is None or mm.empty:
        return
    plt.figure(figsize=(11,6))
    x = mm["month"]
    plt.plot(x, mm["failure_rate"], marker="o", label="Failure rate")
    plt.plot(x, mm["retry_rate"], marker="o", label="Retry rate")
    plt.plot(x, mm["success_rate"], marker="o", label="Success rate")
    plt.xticks(rotation=45, ha="right"); plt.legend()
    plt.title("Workflow Health (Monthly)")
    plt.ylabel("Rate")
    plt.tight_layout()
    p = os.path.join(outdir, "viz_github_monthly_rates.png")
    plt.savefig(p, dpi=150); plt.close()
    print("📈 saved:", p)

def plot_taxonomy(tax_df, outdir):
    if tax_df is None or tax_df.empty:
        return
    plt.figure(figsize=(8,5))
    d = tax_df.sort_values("count", ascending=False)
    plt.bar(d["category"], d["count"])
    plt.title("Failure Taxonomy (Heuristic)")
    plt.tight_layout()
    p = os.path.join(outdir, "viz_failure_taxonomy.png")
    plt.savefig(p, dpi=150); plt.close()
    print("📊 saved:", p)

def plot_linkage(link_df, outdir):
    if link_df is None or link_df.empty:
        return
    plt.figure(figsize=(11,6))
    x = link_df["month"]
    plt.plot(x, link_df["failure_rate"], marker="o", label="Failure rate (GitHub)")
    plt.plot(x, link_df["flaky_topic_share"], marker="o", label="Flaky-topic share (SO)")
    plt.xticks(rotation=45, ha="right"); plt.legend()
    plt.title("SO Discourse vs GitHub Failures (Monthly)")
    plt.tight_layout()
    p = os.path.join(outdir, "viz_link_so_github.png")
    plt.savefig(p, dpi=150); plt.close()
    print("🔗 saved:", p)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--token", default=None, help="GitHub token (optional; env GITHUB_TOKEN used if not provided)")
    ap.add_argument("--pages", type=int, default=10, help="API pages (x100 items per page)")
    ap.add_argument("--from", dest="date_from", default=None, help="Created from (YYYY-MM-DD)")
    ap.add_argument("--to", dest="date_to", default=None, help="Created to (YYYY-MM-DD)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--link_so", type=int, default=0, help="1 = link with SO monthly shares if available")
    ap.add_argument("--verbose", type=int, default=0, help="1 = print verbose HTTP logs")
    args = ap.parse_args()

    token = args.token or os.getenv("GITHUB_TOKEN")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Repo: {args.repo} | Outdir: {args.outdir} | Pages: {args.pages} | Verbose: {args.verbose}")

    # Fetch runs
    print("⬇️ Fetching workflow runs …")
    runs_df = fetch_runs(args.repo, token=token, created_from=args.date_from,
                         created_to=args.date_to, pages=args.pages, verbose=args.verbose)
    runs_path = os.path.join(args.outdir, "github_actions_runs.csv")
    runs_df.to_csv(runs_path, index=False)
    print(f"✅ saved: {runs_path} ({len(runs_df)} rows)")

    # If no runs, stop early with a clear note
    if runs_df.empty:
        print("ℹ️ No workflow runs returned by the API. Check repo name, token perms, or increase --pages.")
        return

    # Fetch jobs/steps
    print("⬇️ Fetching jobs/steps …")
    jobs_df = fetch_jobs_for_runs(args.repo, runs_df["id"].tolist(), token=token,
                                  pages=5, verbose=args.verbose)
    jobs_path = os.path.join(args.outdir, "github_actions_jobs.csv")
    jobs_df.to_csv(jobs_path, index=False)
    print(f"✅ saved: {jobs_path} ({len(jobs_df)} rows)")

    # Metrics
    repo_metrics = compute_repo_metrics(runs_df)
    repo_metrics.to_csv(os.path.join(args.outdir, "metrics_repo.csv"), index=False)
    monthly = compute_monthly_metrics(runs_df)
    monthly.to_csv(os.path.join(args.outdir, "metrics_monthly.csv"), index=False)
    print("✅ saved: metrics_repo.csv, metrics_monthly.csv")

    # Taxonomy
    tax_df = classify_failure_from_jobs(jobs_df)
    tax_df.to_csv(os.path.join(args.outdir, "failure_taxonomy.csv"), index=False)
    print("✅ saved: failure_taxonomy.csv")

    # Plots
    plot_monthly(monthly, args.outdir)
    plot_taxonomy(tax_df, args.outdir)

    # Optional linkage with SO monthly topic shares
    link_df = None
    if args.link_so:
        link_df = link_with_so_monthly(args.outdir)
        plot_linkage(link_df, args.outdir)

    print("\n📂 Outputs in:", os.path.abspath(args.outdir))
    print(" - github_actions_runs.csv")
    print(" - github_actions_jobs.csv")
    print(" - metrics_repo.csv")
    print(" - metrics_monthly.csv")
    print(" - failure_taxonomy.csv")
    print(" - viz_github_monthly_rates.png")
    print(" - viz_failure_taxonomy.png")
    if args.link_so:
        print(" - link_so_github_monthly.csv")
        print(" - link_so_github_corr.csv")
        print(" - viz_link_so_github.png")

if __name__ == "__main__":
    main()


