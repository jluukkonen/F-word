import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import os

# Paths
HUMAN_FILE = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/to_label.xlsx"
CHATGPT_FILE = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/chatgpt_labels.tsv"
OUTPUT_DIR = "/Volumes/United/Work/F-word/swearing-nlp/results"

def main():
    # Load data
    human = pd.read_excel(HUMAN_FILE)
    chatgpt = pd.read_csv(CHATGPT_FILE, sep="\t")
    
    # Rename columns for clarity
    human = human.rename(columns={"label": "human_label"})
    chatgpt = chatgpt.rename(columns={"label": "chatgpt_label"})
    
    # Merge on 'id' — only IDs that exist in both
    merged = pd.merge(human, chatgpt, on="id", how="inner")
    
    # Clean: drop rows where either label is missing
    merged = merged.dropna(subset=["human_label", "chatgpt_label"])
    merged = merged[merged["human_label"] != ""]
    merged = merged[merged["chatgpt_label"] != ""]
    
    # Normalize labels to lowercase
    merged["human_label"] = merged["human_label"].str.strip().str.lower()
    merged["chatgpt_label"] = merged["chatgpt_label"].str.strip().str.lower()
    
    # Note: ChatGPT used "neutral" once — map it to "ambiguous" for consistency
    merged["chatgpt_label"] = merged["chatgpt_label"].replace("neutral", "ambiguous")
    
    n = len(merged)
    print(f"=" * 60)
    print(f"INTER-RATER RELIABILITY: Human vs. ChatGPT")
    print(f"=" * 60)
    print(f"Overlapping labeled samples: {n}")
    print()
    
    # ── Overall Agreement ──
    agree = (merged["human_label"] == merged["chatgpt_label"]).sum()
    pct = agree / n * 100
    print(f"Overall Agreement: {agree}/{n} ({pct:.1f}%)")
    print()
    
    # ── Cohen's Kappa ──
    labels = sorted(set(merged["human_label"].unique()) | set(merged["chatgpt_label"].unique()))
    kappa = cohen_kappa_score(merged["human_label"], merged["chatgpt_label"], labels=labels)
    print(f"Cohen's Kappa (κ): {kappa:.3f}")
    
    # Interpret kappa
    if kappa < 0:
        interp = "Less than chance agreement"
    elif kappa < 0.21:
        interp = "Slight agreement"
    elif kappa < 0.41:
        interp = "Fair agreement"
    elif kappa < 0.61:
        interp = "Moderate agreement"
    elif kappa < 0.81:
        interp = "Substantial agreement"
    else:
        interp = "Almost perfect agreement"
    print(f"Interpretation: {interp}")
    print()
    
    # ── Label Distribution Comparison ──
    print(f"{'Label':<15} {'Human':>8} {'ChatGPT':>10}")
    print("-" * 35)
    for label in labels:
        h_count = (merged["human_label"] == label).sum()
        c_count = (merged["chatgpt_label"] == label).sum()
        print(f"{label:<15} {h_count:>8} {c_count:>10}")
    print()
    
    # ── Confusion Matrix ──
    cm = confusion_matrix(merged["human_label"], merged["chatgpt_label"], labels=labels)
    print("Confusion Matrix (rows=Human, cols=ChatGPT):")
    header = f"{'':>15}" + "".join(f"{l[:8]:>10}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>15}" + "".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        print(row)
    print()
    
    # ── Per-Label Agreement (treating human as "reference") ──
    print("Per-Label Classification Report (Human = Reference):")
    print(classification_report(merged["human_label"], merged["chatgpt_label"], 
                                labels=labels, target_names=labels, zero_division=0))
    
    # ── Disagreement Analysis by Tie Strength ──
    merged["agree"] = merged["human_label"] == merged["chatgpt_label"]
    print("Agreement by Network Tie Strength:")
    tie_agreement = merged.groupby("network_tie_strength")["agree"].agg(["sum", "count"])
    tie_agreement["pct"] = (tie_agreement["sum"] / tie_agreement["count"] * 100).round(1)
    for tie, row in tie_agreement.iterrows():
        print(f"  {tie}: {int(row['sum'])}/{int(row['count'])} ({row['pct']}%)")
    print()
    
    # ── Most Interesting Disagreements ──
    disagree = merged[~merged["agree"]].copy()
    print(f"Total disagreements: {len(disagree)}")
    print()
    print("Sample Disagreements (first 10):")
    print("-" * 80)
    for _, row in disagree.head(10).iterrows():
        text_preview = str(row["text"])[:80] + "..." if len(str(row["text"])) > 80 else str(row["text"])
        print(f"  ID: {row['id']} | Sub: r/{row['subreddit']} ({row['network_tie_strength']})")
        print(f"  Text: {text_preview}")
        print(f"  Human: {row['human_label']}  |  ChatGPT: {row['chatgpt_label']}")
        print()
    
    # ── Save full results ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save merged comparison
    merged.to_csv(os.path.join(OUTPUT_DIR, "interrater_comparison.tsv"), sep="\t", index=False)
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, "interrater_summary.txt"), "w") as f:
        f.write(f"INTER-RATER RELIABILITY: Human vs. ChatGPT\n")
        f.write(f"Overlapping samples: {n}\n")
        f.write(f"Overall Agreement: {agree}/{n} ({pct:.1f}%)\n")
        f.write(f"Cohen's Kappa: {kappa:.3f} ({interp})\n")
    
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
