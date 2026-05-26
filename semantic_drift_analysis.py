# semantic_drift_analysis.py
# Usage:
#   python semantic_drift_analysis.py --outdir outputs \
#       --terms workflow_dispatch,matrix,secrets,cache,artifact,runner,pipeline \
#       --eras "2017-01-01:2020-03-31,2020-04-01:2022-06-30,2022-07-01:2025-12-31"

import argparse, os, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cosine, jensenshannon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Helpers
# ----------------------------
def parse_eras(eras_str):
    """
    "2017-01-01:2020-03-31,2020-04-01:2022-06-30,2022-07-01:2025-12-31"
    -> [("2017-01-01","2020-03-31"), ("2020-04-01","2022-06-30"), ("2022-07-01","2025-12-31")]
    """
    out = []
    for part in eras_str.split(","):
        a, b = part.split(":")
        out.append((a.strip(), b.strip()))
    return out

def sentence_split(text):
    # light-weight sentence split
    return re.split(r'(?<=[.!?])\s+', text)

def extract_term_contexts(df, term, era_mask, max_per_doc=3):
    """
    For each doc in era_mask, extract up to N sentences that contain term (case-insensitive).
    Returns list of strings (contexts).
    """
    term_re = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    contexts = []
    for _, row in df.loc[era_mask, ["text_clean"]].iterrows():
        sents = sentence_split(str(row["text_clean"]))
        hits = [s for s in sents if term_re.search(s)]
        if hits:
            contexts.extend(hits[:max_per_doc])
    return contexts

def embed_centroid(texts, encoder, batch_size=64):
    if not texts:
        return None
    vecs = encoder.encode(texts, show_progress_bar=False, batch_size=batch_size, normalize_embeddings=True)
    return np.mean(vecs, axis=0)

def safe_cosine(u, v):
    if u is None or v is None:
        return np.nan
    # cosine distance -> convert to similarity or keep as distance
    return cosine(u, v)  # 0 means same, 1 means orthogonal

def js_divergence(p, q):
    # p, q are arrays that sum to 1; return Jensen-Shannon divergence
    return jensenshannon(p, q, base=2.0)**2  # square of distance is divergence

def normalize_counts(counts):
    s = counts.sum()
    return counts / s if s > 0 else counts

# ----------------------------
# Main analysis
# ----------------------------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model outputs
    docs = pd.read_csv(outdir / "docs_with_topics.csv")  # expects: title, body, text_clean, created_at, topic
    top_terms = pd.read_csv(outdir / "topic_top_terms.csv")  # (optional for reporting)
    # Ensure date
    docs["created_at"] = pd.to_datetime(docs["created_at"], errors="coerce")
    docs = docs.dropna(subset=["created_at"])
    docs["year"] = docs["created_at"].dt.year

    # Prepare terms to track
    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    eras = parse_eras(args.eras)  # list of (start, end)
    era_labels = [f"{a} to {b}" for a,b in eras]

    # Encoder (same as modeling, aligned space)
    encoder = SentenceTransformer(args.emb_model)

    # Build era masks
    era_masks = []
    for a, b in eras:
        mask = (docs["created_at"] >= pd.to_datetime(a)) & (docs["created_at"] <= pd.to_datetime(b))
        era_masks.append(mask)

    # 1) Term-level semantic drift: centroid per era, cosine distance between eras
    rows = []
    for term in terms:
        era_centroids = []
        for mask in era_masks:
            contexts = extract_term_contexts(docs, term, mask, max_per_doc=args.max_per_doc)
            c = embed_centroid(contexts, encoder)
            era_centroids.append(c)
        # pairwise distances between eras
        for (i, c1), (j, c2) in combinations(list(enumerate(era_centroids)), 2):
            dist = safe_cosine(c1, c2)  # cosine distance (0: same, 1: orthogonal)
            rows.append({
                "term": term,
                "era_i": era_labels[i],
                "era_j": era_labels[j],
                "cosine_distance": dist,
                "contexts_i": len(extract_term_contexts(docs, term, era_masks[i], max_per_doc=args.max_per_doc)),
                "contexts_j": len(extract_term_contexts(docs, term, era_masks[j], max_per_doc=args.max_per_doc)),
            })
    drift_df = pd.DataFrame(rows)
    drift_df.to_csv(outdir / "semantic_drift_terms.csv", index=False)
    print("saved:", outdir / "semantic_drift_terms.csv")

    # Plot drift (average across pairs) per term
    mean_drift = drift_df.groupby("term")["cosine_distance"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    plt.bar(mean_drift.index, mean_drift.values)
    plt.ylabel("Mean cosine distance across eras (higher = more drift)")
    plt.title("Semantic Drift of CI/CD Terms Across Eras")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "viz_semantic_drift_terms.png", dpi=150)
    plt.close()
    print("saved:", outdir / "viz_semantic_drift_terms.png")

    # 2) Contrastive era topic prevalence: JS divergence between topic distributions per era
    # Build topic distributions per era
    topic_ids = docs["topic"].dropna().astype(int)
    K = topic_ids[topic_ids >= 0].nunique()  # exclude -1 (outliers)
    all_topics = sorted(list(set(topic_ids[topic_ids >= 0].unique())))

    era_topic_vecs = []
    for mask in era_masks:
        subset = docs[mask & (docs["topic"] >= 0)]
        counts = subset["topic"].value_counts().reindex(all_topics, fill_value=0).astype(float).values
        era_topic_vecs.append(normalize_counts(counts))

    # Pairwise JS divergence between eras
    js_rows = []
    for (i, p), (j, q) in combinations(list(enumerate(era_topic_vecs)), 2):
        div = js_divergence(p, q)
        js_rows.append({"era_i": era_labels[i], "era_j": era_labels[j], "js_divergence": float(div)})
    js_df = pd.DataFrame(js_rows)
    js_df.to_csv(outdir / "contrastive_era_topic_js.csv", index=False)
    print("saved:", outdir / "contrastive_era_topic_js.csv")

    # Bar chart JS divergence
    if not js_df.empty:
        plt.figure(figsize=(8,5))
        labels = [f"{r['era_i']} vs\n{r['era_j']}" for _, r in js_df.iterrows()]
        plt.bar(range(len(js_df)), js_df["js_divergence"].values)
        plt.xticks(range(len(js_df)), labels, rotation=0)
        plt.ylabel("Jensen-Shannon divergence")
        plt.title("Contrastive Era Topic Distributions")
        plt.tight_layout()
        plt.savefig(outdir / "viz_contrastive_era_js.png", dpi=150)
        plt.close()
        print("saved:", outdir / "viz_contrastive_era_js.png")

    # 3) (Optional) Term prevalence per era (sanity check)
    prev_rows = []
    for term in terms:
        for idx, mask in enumerate(era_masks):
            # count docs mentioning term
            term_re = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            count = docs.loc[mask, "text_clean"].apply(lambda s: bool(term_re.search(str(s)))).sum()
            total = mask.sum()
            prev_rows.append({
                "term": term,
                "era": era_labels[idx],
                "docs_with_term": int(count),
                "docs_total": int(total),
                "share": (count/total) if total > 0 else 0.0
            })
    prev_df = pd.DataFrame(prev_rows)
    prev_df.to_csv(outdir / "term_prevalence_by_era.csv", index=False)
    print("saved:", outdir / "term_prevalence_by_era.csv")

    # Plot term share by era
    if not prev_df.empty:
        pivot = prev_df.pivot(index="era", columns="term", values="share").fillna(0.0)
        plt.figure(figsize=(12,6))
        for term in pivot.columns:
            plt.plot(pivot.index, pivot[term].values, marker="o", label=term)
        plt.ylabel("Share of docs mentioning term")
        plt.title("Term Prevalence by Era")
        plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / "viz_term_prevalence_by_era.png", dpi=150)
        plt.close()
        print("saved:", outdir / "viz_term_prevalence_by_era.png")

    print("✅ Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--emb_model", type=str, default="all-MiniLM-L6-v2",
                    help="Use the same encoder as your topic model for aligned spaces")
    ap.add_argument("--terms", type=str,
                    default="workflow_dispatch,matrix,secrets,cache,artifact,runner,pipeline")
    ap.add_argument("--eras", type=str,
                    default="2017-01-01:2020-03-31,2020-04-01:2022-06-30,2022-07-01:2025-12-31")
    ap.add_argument("--max_per_doc", type=int, default=3, help="Max context sentences per doc per term")
    args = ap.parse_args()
    main(args)
