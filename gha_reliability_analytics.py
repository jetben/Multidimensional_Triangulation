import os
import json
import argparse
from datetime import datetime
from dateutil import parser as dtparser
import pandas as pd
import matplotlib.pyplot as plt

# Optional but recommended for huge JSON
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False


# -----------------------------
# Robust field extraction helpers
# -----------------------------
def get_nested(d, path, default=None):
    """Safely get nested dict value using a list path."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def first_present(record, candidates):
    """
    candidates: list of paths, where each path is a list of keys.
    returns first non-empty value found.
    """
    for path in candidates:
        v = get_nested(record, path, None)
        if v is not None and v != "":
            return v
    return None

def parse_dt(s):
    if not s:
        return None
    try:
        return dtparser.parse(s)
    except Exception:
        return None

def ensure_outdir(outdir):
    outdir_abs = os.path.abspath(outdir)
    os.makedirs(outdir_abs, exist_ok=True)
    return outdir_abs


# -----------------------------
# Streaming reader
# -----------------------------
def stream_records(runs_path):
    """
    Supports:
    1) JSON array: [ {...}, {...} ]
    2) NDJSON: one JSON object per line
    """
    with open(runs_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # Case A: JSON array
        if first_char == "[":
            if HAS_IJSON:
                for obj in ijson.items(f, "item"):
                    yield obj
            else:
                # fallback (not recommended for huge files)
                data = json.load(f)
                for obj in data:
                    yield obj

        # Case B: NDJSON
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


# -----------------------------
# Main analytics
# -----------------------------
def compute_metrics(runs_path, outdir, max_rows=None):
    outdir = ensure_outdir(outdir)
    print("📂 Output folder:", outdir)

    processed = 0
    kept = 0
    skipped_no_year = 0

    # Aggregators
    year_total = {}
    year_failed = {}
    year_retry_gt1 = {}
    year_attempt_sum = {}
    year_duration_sum_min = {}
    year_duration_count = {}
    retry_hist = {}  # attempt -> count
    year_logsize_sum = {}
    year_logsize_count = {}

    # Field candidates (based on your screenshot)
    created_candidates = [
        ["metadata", "created_at"],
        ["created_at"],
        ["run_started_at"],
        ["metadata", "run_started_at"]
    ]
    completed_candidates = [
        ["metadata", "completed_at"],
        ["completed_at"],
        ["updated_at"],
        ["metadata", "updated_at"]
    ]
    conclusion_candidates = [
        ["metadata", "conclusion"],
        ["conclusion"],
        ["metadata", "status"],
        ["status"]
    ]
    attempt_candidates = [
        ["run_attempt"],
        ["metadata", "run_attempt"],
        ["attempt"],
        ["metadata", "attempt"]
    ]
    logsize_candidates = [
        ["total_logs_size"],
        ["metadata", "total_logs_size"],
        ["log_insights", "total_logs_size"],
        ["metadata", "log_insights", "total_logs_size"]
    ]

    for rec in stream_records(runs_path):
        processed += 1
        if max_rows and processed > max_rows:
            break

        created_s = first_present(rec, created_candidates)
        created_dt = parse_dt(created_s)
        if not created_dt:
            skipped_no_year += 1
            continue

        year = created_dt.year

        conclusion = first_present(rec, conclusion_candidates)
        conclusion = (conclusion or "").lower().strip()

        attempt = first_present(rec, attempt_candidates)
        try:
            attempt = int(attempt) if attempt is not None else 1
        except Exception:
            attempt = 1

        completed_s = first_present(rec, completed_candidates)
        completed_dt = parse_dt(completed_s)

        # Duration (MTTR-style proxy) if timestamps exist
        duration_min = None
        if created_dt and completed_dt:
            delta = completed_dt - created_dt
            # Ignore negative or absurd values
            if delta.total_seconds() >= 0 and delta.total_seconds() <= 60 * 60 * 24 * 7:
                duration_min = delta.total_seconds() / 60.0

        logsize = first_present(rec, logsize_candidates)
        try:
            logsize = float(logsize) if logsize is not None else None
        except Exception:
            logsize = None

        # Update year totals
        year_total[year] = year_total.get(year, 0) + 1
        year_attempt_sum[year] = year_attempt_sum.get(year, 0) + attempt

        # Failure (treat "failure" or "failed" as failed)
        is_failed = conclusion in ("failure", "failed")
        if is_failed:
            year_failed[year] = year_failed.get(year, 0) + 1

        # Flakiness proxy: attempt > 1
        if attempt > 1:
            year_retry_gt1[year] = year_retry_gt1.get(year, 0) + 1

        # Retry histogram
        retry_hist[attempt] = retry_hist.get(attempt, 0) + 1

        # Duration aggregation
        if duration_min is not None:
            year_duration_sum_min[year] = year_duration_sum_min.get(year, 0.0) + duration_min
            year_duration_count[year] = year_duration_count.get(year, 0) + 1

        # Log size aggregation
        if logsize is not None:
            year_logsize_sum[year] = year_logsize_sum.get(year, 0.0) + logsize
            year_logsize_count[year] = year_logsize_count.get(year, 0) + 1

        kept += 1

    print(f"✅ Processed: {processed:,}")
    print(f"✅ Kept (has created_at/year): {kept:,}")
    print(f"⚠️ Skipped (missing timestamps/year): {skipped_no_year:,}")

    if kept == 0:
        print("\n❌ No usable records found. Your timestamp fields may use different names.")
        print("   Try printing one record and share the keys.")
        return

    # Build yearly dataframe
    years = sorted(year_total.keys())
    rows = []
    for y in years:
        total = year_total.get(y, 0)
        failed = year_failed.get(y, 0)
        flaky = year_retry_gt1.get(y, 0)

        failure_rate = (failed / total) if total else 0.0
        flakiness_rate = (flaky / total) if total else 0.0

        # MTTR proxy:
        # - if duration exists: mean duration minutes
        # - else: mean attempts as MTTR proxy (and mean log size)
        mean_duration = None
        if year_duration_count.get(y, 0) > 0:
            mean_duration = year_duration_sum_min[y] / year_duration_count[y]

        mean_attempts = year_attempt_sum.get(y, 0) / total if total else None

        mean_logsize = None
        if year_logsize_count.get(y, 0) > 0:
            mean_logsize = year_logsize_sum[y] / year_logsize_count[y]

        rows.append({
            "year": y,
            "total_runs": total,
            "failed_runs": failed,
            "failure_rate": round(failure_rate, 6),
            "flaky_runs_attempt_gt1": flaky,
            "flakiness_proxy": round(flakiness_rate, 6),
            "mean_attempts": round(mean_attempts, 4) if mean_attempts is not None else None,
            "mean_duration_min": round(mean_duration, 4) if mean_duration is not None else None,
            "mean_log_size": round(mean_logsize, 4) if mean_logsize is not None else None
        })

    df_year = pd.DataFrame(rows)

    # Retry distribution DF
    df_retry = pd.DataFrame(
        [{"run_attempt": k, "count": v} for k, v in sorted(retry_hist.items())]
    )

    # Save CSVs
    df_year.to_csv(os.path.join(outdir, "reliability_by_year.csv"), index=False)
    df_retry.to_csv(os.path.join(outdir, "retry_distribution.csv"), index=False)

    print("\n🧾 Saved CSVs:")
    print(" -", os.path.join(outdir, "reliability_by_year.csv"))
    print(" -", os.path.join(outdir, "retry_distribution.csv"))

    # -----------------------------
    # Figures (300 DPI)
    # -----------------------------
    # Failure rate by year
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_year["year"], df_year["failure_rate"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Failure rate (failed / total)")
    plt.title("Failure Rate by Year (GitHub Actions Runs)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "fig_failure_rate_by_year.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Flakiness proxy by year
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_year["year"], df_year["flakiness_proxy"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Flakiness proxy (% runs with attempt > 1)")
    plt.title("Flakiness Proxy by Year")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "fig_flakiness_proxy_by_year.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Retry distribution
    plt.figure(figsize=(8, 4.5))
    plt.bar(df_retry["run_attempt"].astype(str), df_retry["count"])
    plt.xlabel("run_attempt")
    plt.ylabel("Count")
    plt.title("Retry Distribution (run_attempt histogram)")
    plt.savefig(os.path.join(outdir, "fig_retry_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # MTTR proxy figure: duration if available else attempts/log size
    plt.figure(figsize=(8, 4.5))
    if df_year["mean_duration_min"].notna().any():
        plt.plot(df_year["year"], df_year["mean_duration_min"], marker="o")
        plt.ylabel("Mean duration (minutes)")
        plt.title("MTTR Proxy by Year (Mean Run Duration)")
    else:
        plt.plot(df_year["year"], df_year["mean_attempts"], marker="o", label="Mean attempts")
        if df_year["mean_log_size"].notna().any():
            plt.plot(df_year["year"], df_year["mean_log_size"], marker="o", label="Mean log size")
        plt.legend()
        plt.ylabel("Proxy value")
        plt.title("MTTR Proxies by Year (Attempts / Log Size)")

    plt.xlabel("Year")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "fig_mttr_proxy_by_year.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n🖼️ Saved Figures (300 DPI):")
    print(" - fig_failure_rate_by_year.png")
    print(" - fig_flakiness_proxy_by_year.png")
    print(" - fig_retry_distribution.png")
    print(" - fig_mttr_proxy_by_year.png")

    print("\n✅ Done. Open the folder:", outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_path", type=str, default="data/runs.json",
                        help="Path to runs.json (default: data/runs.json)")
    parser.add_argument("--outdir", type=str, default="outputs_tri",
                        help="Output folder (default: outputs_tri)")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Process only first N rows (debug)")
    args = parser.parse_args()

    compute_metrics(args.runs_path, args.outdir, args.max_rows)


if __name__ == "__main__":
    main()
