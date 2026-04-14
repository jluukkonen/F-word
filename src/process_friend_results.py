import pandas as pd
import re
import os
import sys

# Paths
HUMAN_50_SOURCE = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/human_rater_50.tsv"

def normalize(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def process_csv(csv_path):
    source_df = pd.read_csv(HUMAN_50_SOURCE, sep="\t")
    friend_df = pd.read_csv(csv_path)
    
    if len(friend_df) == 0:
        print(f"Error: CSV {csv_path} is empty.")
        return
    
    # Mapping based on alphanumeric sequences of the raw text and subreddit
    id_map = {}
    for _, row in source_df.iterrows():
        key = normalize(row['text'])[:50]
        id_map[key] = row['id']
        
    print(f"Prepared mapping for {len(id_map)} items using {csv_path}")
    
    for row_idx in range(len(friend_df)):
        results = []
        found_count = 0
        timestamp = friend_df.iloc[row_idx]['Timestamp']
        # Identify respondent by row index (0=Friend 1, 1=Friend 2/Professor, etc.)
        respondent_id = row_idx + 1
        output_file = f"/Volumes/United/Work/F-word/swearing-nlp/data/labeled/friend{respondent_id}_labels.tsv"
        
        for col in friend_df.columns:
            if col.lower() == "timestamp":
                continue
                
            # Clean the column header
            if "Comment:" in col:
                comment_part = col.split("Comment:")[1].strip()
            else:
                comment_part = col
                
            col_key = normalize(comment_part)[:50]
            
            # Match ID
            match_id = None
            if col_key in id_map:
                match_id = id_map[col_key]
            else:
                for k, id_val in id_map.items():
                    if col_key[:30] in k or k[:30] in col_key:
                        match_id = id_val
                        break
            
            if match_id:
                label = friend_df.iloc[row_idx][col]
                if pd.notna(label):
                    clean_label = str(label).split("(")[0].strip().lower()
                    results.append({"id": match_id, "label": clean_label})
                    found_count += 1
        
        # Save results for this respondent
        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        out_df.to_csv(output_file, sep="\t", index=False)
        print(f"Respondent {respondent_id} ({timestamp}): Mapped {found_count} items. Saved to {output_file}")

if __name__ == "__main__":
    combined_csv = "/Volumes/United/Work/F-word/results/The Many Shapes of the F-Word 3.csv"
    process_csv(combined_csv)
