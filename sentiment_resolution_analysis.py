import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def load_so_data(so_csv, outdir):
    # prefer explicit path, else try common locations
    candidates = [so_csv, os.path.join(outdir, "so_posts.csv"), os.path.join("data", "so_posts.csv")]
    for c in candidates:
        if c and os.path.exists(c):
            print(f"📥 Using Stack Overflow CSV: {c}")
            return pd.read_csv(c)
    raise FileNotFoundError(f"Could not find so_posts.csv. Tried: {candidates}")

def load_topics(docs_with_topics_csv, outdir):
    candidates = [docs_with_topics_csv, os.path.join(outdir, "docs_with_topics.csv")]
    for c in candidates:
        if c and os.path.exists(c):
            print(f"📥 Using docs_with_topics CSV: {c}")
            return pd.read_csv(c)
    raise FileNotFoundError(f"Could not find docs_with_topics.csv. Tried: {candidates}")

def compute_sentiment(df, text_field="body"):
    sid = SentimentIntensityAnalyzer()
    return df[text_field].astype(str).apply(lambda x: sid.polarity_scores(x)["compound"])

def build_survival_data(df):
    # best-effort survival: use created_at; if last_activity_date missing, use created_at (0 days)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if "last_activity_date" in df.columns:
        df["last_activity_date"] = pd.to_datetime(df["last_activity_date"], errors="coerce")
    else:
        df["last_activity_date"] = df["created_at"]
    df["duration"] = (df["last_activity_date"] - df["created_at"]).dt.days.fillna(0)
    df["event"] = df["is_answered"].astype(int) if "is_answered" in df.columns else 0
    return df

def survival_plot(df, out_path):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10,6))
    any_plotted = False
    for topic in sorted(df["topic"].dropna().unique()):
        sub = df[df["topic"] == topic]
        if len(sub) < 15:  # skip tiny topics
            continue
        kmf.fit(durations=sub["duration"], event_observed=sub["event"], label=f"T{int(topic)}")
        kmf.plot_survival_function(ci_show=False)
        any_plotted = True
    if not any_plotted:
        print("ℹ️ Not enough per-topic data to draw survival curves (all tiny or censored). "
              "Saving an overall curve instead.")
        kmf.fit(durations=df["duration"], event_observed=df["event"], label="All topics")
        kmf.plot_survival_function(ci_show=False)
    plt.title("Survival Analysis of Time-to-Resolution")
    plt.xlabel("Days"); plt.ylabel("P(Question still unresolved)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"✅ saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--so_csv", type=str, default=None, help="Path to so_posts.csv (optional)")
    ap.add_argument("--docs_with_topics_csv", type=str, default=None, help="Path to docs_with_topics.csv (optional)")
    ap.add_argument("--text_field", type=str, default="body")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    so_df = load_so_data(args.so_csv, args.outdir)
    dwt_df = load_topics(args.docs_with_topics_csv, args.outdir)

    # merge on id, keep topic
    df = pd.merge(so_df, dwt_df[["id","topic"]], on="id", how="left")

    # sentiment
    try:
        from nltk import download as nltk_download
        nltk_download("vader_lexicon", quiet=True)
    except Exception:
        pass
    df["sentiment"] = compute_sentiment(df, text_field=args.text_field)
    so_sent_path = os.path.join(args.outdir, "so_with_sentiment.csv")
    df.to_csv(so_sent_path, index=False)
    print(f"✅ saved: {so_sent_path}")

    # survival
    df = build_survival_data(df)
    surv_img = os.path.join(args.outdir, "viz_resolution_survival.png")
    survival_plot(df, surv_img)

    # summary
    summary = df.groupby("topic", dropna=False).agg(
        mean_sentiment=("sentiment","mean"),
        mean_duration_days=("duration","mean"),
        resolution_rate=("event","mean")
    ).reset_index()
    summary_path = os.path.join(args.outdir, "resolution_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ saved: {summary_path}")

    print("\n📂 Outputs created:")
    print("  -", so_sent_path)
    print("  -", surv_img)
    print("  -", summary_path)
    print(f"\n🔎 Check the folder: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()

