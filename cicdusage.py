#!/usr/bin/env python3
"""
CI/CD Detector & Feature Extractor (GitHub, 2020–2025)

What it does
------------
- Queries GitHub Search API for repositories created in each year (2020–2025).
- For each repo, checks for presence of common CI/CD config files:
  GitHub Actions, GitLab CI, Jenkins, Azure Pipelines, CircleCI, Travis CI.
- Parses configs to extract pipeline-level features:
  jobs/steps, matrix usage, cache, runner type, triggers, container/services, etc.
- Saves a CSV with one row per (repo, tool) detection.

Notes
-----
- This script samples repositories per year (configurable) to stay within rate limits.
- You can re-run with higher limits or narrower language filters if desired.
- For very large studies, shard by language or use GraphQL/bulk jobs.

Setup
-----
pip install requests pyyaml tqdm pandas tenacity python-dateutil
export GITHUB_TOKEN=ghp_xxx   # or set in code below

Output
------
ci_detection_results.csv  (append-safe; will overwrite by default)

Author: you ✨
"""

import os
import time
import csv
import re
import json
from datetime import datetime, timedelta, timezone
from dateutil.parser import isoparse
from typing import Dict, Any, List, Optional, Tuple, Set

import requests
import yaml
from tqdm import tqdm
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# -------------------- Configuration --------------------

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()  # put your token here if not using env var
if not GITHUB_TOKEN:
    # fallback: paste token string if you prefer (less secure)
    GITHUB_TOKEN = "REPLACE_WITH_YOUR_TOKEN"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

YEARS = list(range(2020, 2026))  # inclusive: 2020..2025
# To limit data: we’ll sample up to this many repos per year.
MAX_REPOS_PER_YEAR = 300  # increase if you have higher rate limits
# Search query base; you can add language filters, e.g., "language:Python"
BASE_QUERY = ""  # leave blank for all languages
# Sort by stars (stable-ish sampling) or "updated" if you prefer
SEARCH_SORT = "stars"  # or "updated"
SEARCH_ORDER = "desc"

OUTPUT_CSV = "ci_detection_results.csv"

# Toggle verbose logging of parsing errors
VERBOSE_PARSE_ERRORS = False

# -------------------- Helper: rate-limit aware requests --------------------

def handle_rate_limit(resp: requests.Response):
    """Sleep until reset if secondary or primary rate-limited."""
    if resp.status_code == 403:
        # Try to detect primary rate limit
        reset = resp.headers.get("X-RateLimit-Reset")
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining == "0" and reset:
            reset_ts = int(reset)
            now = int(time.time())
            sleep_s = max(0, reset_ts - now) + 5
            print(f"[RATE LIMIT] Sleeping {sleep_s}s until primary rate limit resets...")
            time.sleep(sleep_s)
            return True
        # Secondary rate limit (abuse detection)
        # Back off a bit
        print("[RATE LIMIT/SECONDARY] 403 received; sleeping 30s...")
        time.sleep(30)
        return True
    return False

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def gh_get(url: str, params: Dict[str, Any] = None) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
    if resp.status_code in (403, 502, 503, 504):
        if handle_rate_limit(resp):
            raise RuntimeError("Retry due to rate limit")
        else:
            resp.raise_for_status()
    elif resp.status_code >= 400:
        resp.raise_for_status()
    return resp

# -------------------- GitHub search & repo helpers --------------------

def search_repos_created_in_year(year: int, max_repos: int) -> List[Dict[str, Any]]:
    """Search repositories created in a given year; return a sampled list up to max_repos."""
    q = f"created:{year}-01-01..{year}-12-31 {BASE_QUERY}".strip()
    per_page = 100
    repos: List[Dict[str, Any]] = []
    page = 1
    while len(repos) < max_repos and page <= 10:  # 10*100 = 1000 Search API cap
        url = "https://api.github.com/search/repositories"
        params = {"q": q, "sort": SEARCH_SORT, "order": SEARCH_ORDER,
                  "per_page": per_page, "page": page}
        r = gh_get(url, params)
        data = r.json()
        items = data.get("items", [])
        if not items:
            break
        repos.extend(items)
        if len(items) < per_page:
            break
        page += 1
    # Sample to cap
    return repos[:max_repos]

def list_dir_contents(owner: str, repo: str, path: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = gh_get(url)
    if r.status_code == 200:
        try:
            data = r.json()
            if isinstance(data, list):
                return data
        except Exception:
            return None
    return None

def get_file_text(owner: str, repo: str, path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = gh_get(url)
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, dict) and data.get("encoding") == "base64":
            import base64
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except Exception:
                return None
        # Raw content redirect (if any)
    return None

# -------------------- Parsers for CI tools --------------------

def safe_yaml_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        return yaml.safe_load(text)
    except Exception as e:
        if VERBOSE_PARSE_ERRORS:
            print("YAML parse error:", e)
        return None

def parse_github_actions(yml_text: str) -> Dict[str, Any]:
    data = safe_yaml_load(yml_text) or {}
    out = {
        "jobs_count": 0, "steps_count": 0, "matrix_used": False,
        "runner_self_hosted": False, "cache_used": False,
        "container_used": False, "services_used": False,
        "trig_push": False, "trig_pr": False, "trig_schedule": False, "trig_tags": False,
        "dag_depth": None
    }
    on = data.get("on") or {}
    if isinstance(on, list):
        out["trig_push"] = "push" in on
        out["trig_pr"] = "pull_request" in on
        out["trig_schedule"] = "schedule" in on
        out["trig_tags"] = any(k in ("create", "push") for k in on)  # heuristic
    elif isinstance(on, dict):
        out["trig_push"] = "push" in on
        out["trig_pr"] = "pull_request" in on
        out["trig_schedule"] = "schedule" in on
        # tag heuristic:
        push = on.get("push", {})
        if isinstance(push, dict):
            tags = push.get("tags") or []
            out["trig_tags"] = bool(tags)

    jobs = data.get("jobs") or {}
    out["jobs_count"] = len(jobs)
    max_depth = 0
    for _, job in jobs.items():
        # runs-on
        runs_on = job.get("runs-on")
        if isinstance(runs_on, list):
            if any(isinstance(x, str) and "self-hosted" in x for x in runs_on):
                out["runner_self_hosted"] = True
        elif isinstance(runs_on, str) and "self-hosted" in runs_on:
            out["runner_self_hosted"] = True

        # container / services
        if "container" in job:
            out["container_used"] = True
        if "services" in job and job["services"]:
            out["services_used"] = True

        # strategy.matrix
        strat = job.get("strategy") or {}
        if "matrix" in strat:
            out["matrix_used"] = True

        steps = job.get("steps") or []
        out["steps_count"] += len(steps)
        # cache
        for s in steps:
            if isinstance(s, dict):
                uses = s.get("uses", "")
                if isinstance(uses, str) and "actions/cache" in uses:
                    out["cache_used"] = True
        # DAG depth (heuristic: count needs chains)
        needs = job.get("needs")
        if needs is None:
            depth = 1
        elif isinstance(needs, list):
            depth = 1 + len(needs)
        else:
            depth = 2
        if depth > max_depth:
            max_depth = depth
    out["dag_depth"] = max_depth if out["jobs_count"] else None
    return out

def parse_gitlab_ci(yml_text: str) -> Dict[str, Any]:
    data = safe_yaml_load(yml_text) or {}
    out = {
        "jobs_count": 0, "steps_count": None, "matrix_used": False,
        "runner_self_hosted": None, "cache_used": False,
        "container_used": False, "services_used": False,
        "trig_push": None, "trig_pr": None, "trig_schedule": None, "trig_tags": None,
        "dag_depth": None
    }
    # GitLab jobs: top-level keys excluding reserved keywords
    reserved = {"stages", "image", "services", "variables", "cache", "before_script", "after_script",
                "default", "include", "workflow"}
    jobs = [k for k, v in (data.items() if isinstance(data, dict) else [])
            if isinstance(v, dict) and ("script" in v or "stage" in v) and k not in reserved]
    out["jobs_count"] = len(jobs)
    # image/services/cache
    if "image" in data: out["container_used"] = True
    if "services" in data and data["services"]: out["services_used"] = True
    if "cache" in data: out["cache_used"] = True
    # matrix-ish: GitLab uses parallel:matrix or rules: matrix
    for k in jobs:
        job = data.get(k, {})
        if isinstance(job, dict):
            if "parallel" in job and isinstance(job["parallel"], dict) and "matrix" in job["parallel"]:
                out["matrix_used"] = True
    return out

def parse_azure_pipelines(yml_text: str) -> Dict[str, Any]:
    data = safe_yaml_load(yml_text) or {}
    out = {
        "jobs_count": 0, "steps_count": 0, "matrix_used": False,
        "runner_self_hosted": None, "cache_used": False,
        "container_used": False, "services_used": None,
        "trig_push": False, "trig_pr": False, "trig_schedule": False, "trig_tags": None,
        "dag_depth": None
    }
    # jobs may be at root or under stages
    def count_steps(obj) -> int:
        if isinstance(obj, list):
            return sum(count_steps(x) for x in obj)
        if isinstance(obj, dict):
            s = 0
            for k, v in obj.items():
                if k == "steps" and isinstance(v, list):
                    s += len(v)
                else:
                    s += count_steps(v)
            return s
        return 0

    jobs = []
    if isinstance(data, dict):
        if "jobs" in data and isinstance(data["jobs"], list):
            jobs = data["jobs"]
        elif "stages" in data and isinstance(data["stages"], list):
            # gather all jobs in stages
            for st in data["stages"]:
                if isinstance(st, dict):
                    if "jobs" in st and isinstance(st["jobs"], list):
                        jobs.extend(st["jobs"])
    out["jobs_count"] = len(jobs)
    out["steps_count"] = count_steps(data)
    # matrix:
    def find_matrix(d):
        if isinstance(d, dict):
            if "strategy" in d and isinstance(d["strategy"], dict) and "matrix" in d["strategy"]:
                return True
            return any(find_matrix(v) for v in d.values())
        if isinstance(d, list):
            return any(find_matrix(x) for x in d)
        return False
    out["matrix_used"] = find_matrix(data)
    # container: 'container:' key or container image in job
    def find_container(d):
        if isinstance(d, dict):
            if "container" in d: return True
            return any(find_container(v) for v in d.values())
        if isinstance(d, list):
            return any(find_container(x) for x in d)
        return False
    out["container_used"] = find_container(data)
    # triggers
    if "trigger" in data and data["trigger"]: out["trig_push"] = True
    if "pr" in data and data["pr"]: out["trig_pr"] = True
    if "schedules" in data and data["schedules"]: out["trig_schedule"] = True
    # cache (Azure has Cache@2 task)
    text = yml_text.lower()
    out["cache_used"] = "cache@2" in text or "cache:" in text
    return out

def parse_circleci(yml_text: str) -> Dict[str, Any]:
    data = safe_yaml_load(yml_text) or {}
    out = {
        "jobs_count": 0, "steps_count": 0, "matrix_used": False,
        "runner_self_hosted": None, "cache_used": False,
        "container_used": False, "services_used": None,
        "trig_push": None, "trig_pr": None, "trig_schedule": None, "trig_tags": None,
        "dag_depth": None
    }
    jobs = (data.get("jobs") or {})
    if isinstance(jobs, dict):
        out["jobs_count"] = len(jobs)
        for jname, jdef in jobs.items():
            if isinstance(jdef, dict):
                steps = jdef.get("steps") or []
                out["steps_count"] += len(steps)
                # executors -> docker section implies container usage
                exec_name = jdef.get("executor")
                if exec_name and "executors" in data and exec_name in data["executors"]:
                    ex = data["executors"][exec_name]
                    if isinstance(ex, dict) and "docker" in ex:
                        out["container_used"] = True
                if "docker" in jdef:
                    out["container_used"] = True
    text = yml_text.lower()
    out["cache_used"] = ("save_cache" in text) or ("restore_cache" in text)
    # matrix: reusable config with parameters/repeat? heuristic:
    out["matrix_used"] = "matrix" in text
    return out

def parse_travis(yml_text: str) -> Dict[str, Any]:
    data = safe_yaml_load(yml_text) or {}
    out = {
        "jobs_count": None, "steps_count": None, "matrix_used": False,
        "runner_self_hosted": None, "cache_used": False,
        "container_used": None, "services_used": None,
        "trig_push": None, "trig_pr": None, "trig_schedule": None, "trig_tags": None,
        "dag_depth": None
    }
    text = yml_text.lower()
    out["cache_used"] = "cache:" in text or "ccache" in text
    out["matrix_used"] = "matrix:" in text or "jobs:" in text
    return out

def parse_jenkinsfile(groovy_text: str) -> Dict[str, Any]:
    text = groovy_text
    # Heuristics only; Jenkinsfiles vary widely
    stages = len(re.findall(r"\bstage\s*\(", text))
    parallel = len(re.findall(r"\bparallel\s*\{", text))
    docker = len(re.findall(r"\bdocker\s*\{|\bdocker\.image", text))
    agent_label = "self-hosted" if re.search(r"agent\s*\{\s*label\s+'self", text) else None
    matrix = bool(re.search(r"matrix\s*\{", text))
    return {
        "jobs_count": stages if stages else None,
        "steps_count": None,
        "matrix_used": matrix,
        "runner_self_hosted": agent_label is not None,
        "cache_used": None,
        "container_used": docker > 0,
        "services_used": None,
        "trig_push": None,
        "trig_pr": None,
        "trig_schedule": None,
        "trig_tags": None,
        "dag_depth": None
    }

# -------------------- Detection logic --------------------

CI_TARGETS = [
    ("github_actions_dir", ".github/workflows/"),
    ("gitlab_ci", ".gitlab-ci.yml"),
    ("jenkinsfile", "Jenkinsfile"),
    ("azure_pipelines", "azure-pipelines.yml"),
    ("circleci", ".circleci/config.yml"),
    ("travis", ".travis.yml"),
]

def detect_and_extract(owner: str, repo: str) -> List[Dict[str, Any]]:
    """Return a list of {tool, feature_dict} for this repo."""
    detections: List[Dict[str, Any]] = []

    # GitHub Actions (directory)
    gha_dir = list_dir_contents(owner, repo, ".github/workflows")
    if gha_dir:
        for item in gha_dir:
            if item.get("type") == "file" and item["name"].lower().endswith((".yml", ".yaml")):
                text = get_file_text(owner, repo, f".github/workflows/{item['name']}")
                if not text:
                    continue
                feats = parse_github_actions(text)
                detections.append({"tool": "github_actions", "file": item["name"], **feats})

    # Single-file tools
    single_files = [
        ("gitlab_ci", ".gitlab-ci.yml", parse_gitlab_ci),
        ("jenkins", "Jenkinsfile", parse_jenkinsfile),
        ("azure_pipelines", "azure-pipelines.yml", parse_azure_pipelines),
        ("circleci", ".circleci/config.yml", parse_circleci),
        ("travis", ".travis.yml", parse_travis),
    ]
    for tool, path, parser in single_files:
        text = get_file_text(owner, repo, path)
        if text:
            feats = parser(text)
            detections.append({"tool": tool, "file": path.split("/")[-1], **feats})

    return detections

# -------------------- Main pipeline --------------------

def main():
    if not GITHUB_TOKEN or GITHUB_TOKEN.startswith("REPLACE_"):
        raise SystemExit("Please set GITHUB_TOKEN environment variable or paste your token into the script.")

    rows = []
    for year in YEARS:
        print(f"\n=== Year {year} ===")
        repos = search_repos_created_in_year(year, MAX_REPOS_PER_YEAR)
        print(f"Sampled {len(repos)} repos for {year} (sorted by {SEARCH_SORT}).")

        for item in tqdm(repos, desc=f"Scanning repos {year}"):
            full = item["full_name"]  # owner/repo
            owner, repo = full.split("/", 1)
            created_at = item.get("created_at")
            language = item.get("language")
            stars = item.get("stargazers_count", 0)
            forks = item.get("forks_count", 0)
            is_org = 1 if item.get("owner", {}).get("type") == "Organization" else 0

            try:
                detections = detect_and_extract(owner, repo)
            except Exception as e:
                if VERBOSE_PARSE_ERRORS:
                    print(f"Parse error in {full}: {e}")
                detections = []

            if not detections:
                # record a negative if desired; here we skip non-CI repos to keep file small
                continue

            for det in detections:
                rows.append({
                    "year": year,
                    "repo_full_name": full,
                    "created_at": created_at,
                    "primary_language": language,
                    "stars": stars,
                    "forks": forks,
                    "is_org": is_org,
                    "tool": det.get("tool"),
                    "file": det.get("file"),
                    "jobs_count": det.get("jobs_count"),
                    "steps_count": det.get("steps_count"),
                    "matrix_used": det.get("matrix_used"),
                    "runner_self_hosted": det.get("runner_self_hosted"),
                    "cache_used": det.get("cache_used"),
                    "container_used": det.get("container_used"),
                    "services_used": det.get("services_used"),
                    "trig_push": det.get("trig_push"),
                    "trig_pr": det.get("trig_pr"),
                    "trig_schedule": det.get("trig_schedule"),
                    "trig_tags": det.get("trig_tags"),
                    "dag_depth": det.get("dag_depth"),
                })

        # Write out incrementally each year
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"Saved {len(rows)} detections so far to {OUTPUT_CSV}")

    print("\nDone. You can now analyze `ci_detection_results.csv` in pandas/Excel.")


if __name__ == "__main__":
    main()
