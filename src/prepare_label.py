import pandas as pd
import os

INPUT_FILE = "data/raw/reddit_fuck.tsv"
OUTPUT_FILE = "data/labeled/to_label.tsv"
SAMPLE_SIZE = 300

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run collect.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE, sep="\t")
    
    # We want a balance of strong-tie and weak-tie
    strong_tie = df[df["network_tie_strength"] == "strong-tie"]
    weak_tie = df[df["network_tie_strength"] == "weak-tie"]
    
    print(f"Total strong-tie available: {len(strong_tie)}")
    print(f"Total weak-tie available: {len(weak_tie)}")
    
    # Sample 150 from each if possible
    n_per_group = SAMPLE_SIZE // 2
    
    strong_sample = strong_tie.sample(n=min(n_per_group, len(strong_tie)), random_state=42)
    weak_sample = weak_tie.sample(n=min(n_per_group, len(weak_tie)), random_state=42)
    
    labeled_df = pd.concat([strong_sample, weak_sample])
    
    # Shuffle the final dataset
    labeled_df = labeled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add empty label column
    labeled_df["label"] = ""
    
    # Reorder columns slightly for better labeling experience
    cols = ["id", "subreddit", "network_tie_strength", "parent_text", "text", "label"]
    labeled_df = labeled_df[cols]
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    labeled_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    
    print(f"\nPrepared {len(labeled_df)} samples for labeling in {OUTPUT_FILE}")
    print("Labels to use: aggression, bonding, emphasis, frustration, or ambiguous.")

if __name__ == "__main__":
    main()
