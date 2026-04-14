import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def preprocess():
    # File paths
    input_file = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/to_label.xlsx"
    output_dir = "/Volumes/United/Work/F-word/swearing-nlp/data/processed"
    label_map_path = os.path.join(output_dir, "label_map.json")
    
    # Define valid categories
    valid_categories = ["aggression", "bonding", "emphasis", "frustration", "ambiguous"]
    label_map = {label: i for i, label in enumerate(sorted(valid_categories))}
    
    print(f"Loading data from {input_file}...")
    df = pd.read_excel(input_file)
    
    # 1. Drop missing text or label
    initial_len = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"Dropped {initial_len - len(df)} rows with missing text or label.")
    
    # 2. Clean labels
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    
    # Filter only valid categories
    df = df[df['label'].isin(valid_categories)]
    print(f"Remaining rows after filtering valid categories: {len(df)}")
    
    # 3. Format input text
    def format_text(row):
        parent = str(row['parent_text']).strip()
        comment = str(row['text']).strip()
        
        if not parent or parent.lower() in ['nan', '[context not found]', 'none']:
            return f"[COMMENT] {comment}"
        else:
            return f"[CONTEXT] {parent} [COMMENT] {comment}"
            
    df['formatted_text'] = df.apply(format_text, axis=1)
    
    # 4. Map labels to integers
    df['label_id'] = df['label'].map(label_map)
    
    # Save label map
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Saved label map to {label_map_path}")
    
    # 5. Stratified train/test split
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['label_id'], 
        random_state=42
    )
    
    # 6. Print class distributions
    print("\nClass Distribution - Train:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nClass Distribution - Test:")
    print(test_df['label'].value_counts(normalize=True))
    
    # 7. Save as CSVs
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df[['id', 'formatted_text', 'label', 'label_id']].to_csv(train_path, index=False)
    test_df[['id', 'formatted_text', 'label', 'label_id']].to_csv(test_path, index=False)
    
    print(f"\nSaved {len(train_df)} train samples to {train_path}")
    print(f"Saved {len(test_df)} test samples to {test_path}")

if __name__ == "__main__":
    preprocess()
