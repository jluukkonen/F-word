import pandas as pd
import numpy as np
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

def evaluate():
    # Paths
    test_path = "/Volumes/United/Work/F-word/swearing-nlp/data/processed/test.csv"
    model_dir = "/Volumes/United/Work/F-word/swearing-nlp/models/pilot_v1"
    label_map_path = "/Volumes/United/Work/F-word/swearing-nlp/data/processed/label_map.json"
    results_path = "/Volumes/United/Work/F-word/swearing-nlp/results/pilot_evaluation.xlsx"
    
    gemini_path = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/gemini_labels.tsv"
    chatgpt_path = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/chatgpt_labels.tsv"
    
    # Load test data and label map
    test_df = pd.read_csv(test_path)
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    rev_label_map = {v: k for k, v in label_map.items()}
    labels_list = [rev_label_map[i] for i in range(len(label_map))]
    
    # Device handling
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # 1. Predict on test set
    print(f"Predicting on {len(test_df)} test samples...")
    all_preds = []
    
    batch_size = 8
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df['formatted_text'].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
    
    test_df['pred_id'] = all_preds
    test_df['pred_label'] = test_df['pred_id'].map(rev_label_map)
    
    print("\nSample Predictions vs True Labels:")
    print(test_df[['id', 'label', 'pred_label', 'label_id', 'pred_id']].head(10))
    
    # 2. Compute metrics for our model
    ft_acc = accuracy_score(test_df['label_id'], test_df['pred_id'])
    ft_kappa = cohen_kappa_score(test_df['label_id'], test_df['pred_id'])
    ft_report = classification_report(test_df['label_id'], test_df['pred_id'], target_names=labels_list, output_dict=True)
    ft_cm = confusion_matrix(test_df['label_id'], test_df['pred_id'])
    
    # 3. LLM Baselines comparison
    print("Computing LLM baselines on the SAME test set...")
    
    def get_llm_metrics(path, name):
        llm_df = pd.read_csv(path, sep='\t')
        # Clean labels to match our categories
        llm_df['label'] = llm_df['label'].astype(str).str.lower().str.strip()
        
        # Merge with test set to get same items
        merged = test_df[['id', 'label', 'label_id']].merge(llm_df, on='id', suffixes=('_expert', f'_{name}'))
        
        # Some items might be missing if LLM failed or ID mismatch
        if len(merged) < len(test_df):
            print(f"Warning: Only {len(merged)} matches found for {name} (expected {len(test_df)})")
            
        # Map LLM labels to IDs
        merged['label_id_llm'] = merged[f'label_{name}'].map(label_map)
        
        # Drop rows where LLM label is unknown (not in our 5 categories)
        merged = merged.dropna(subset=['label_id_llm'])
        
        acc = accuracy_score(merged['label_id'], merged['label_id_llm'])
        kappa = cohen_kappa_score(merged['label_id'], merged['label_id_llm'])
        cm = confusion_matrix(merged['label_id'], merged['label_id_llm'], labels=range(len(labels_list)))
        
        return acc, kappa, cm, len(merged)

    gemini_acc, gemini_kappa, gemini_cm, gemini_count = get_llm_metrics(gemini_path, "gemini")
    chatgpt_acc, chatgpt_kappa, chatgpt_cm, chatgpt_count = get_llm_metrics(chatgpt_path, "chatgpt")
    
    # 4. Summary Table
    summary_data = [
        {"Model": "Fine-tuned twitter-roberta", "Kappa": ft_kappa, "Accuracy": ft_acc, "N": len(test_df)},
        {"Model": "Gemini 3.1 Pro (zero-shot)", "Kappa": gemini_kappa, "Accuracy": gemini_acc, "N": gemini_count},
        {"Model": "ChatGPT 5.4 (zero-shot)", "Kappa": chatgpt_kappa, "Accuracy": chatgpt_acc, "N": chatgpt_count},
        {"Model": "Random baseline", "Kappa": 0.0, "Accuracy": 1/len(labels_list), "N": len(test_df)}
    ]
    summary_df = pd.DataFrame(summary_data)
    
    print("\n=== EVALUATION SUMMARY ===")
    print(summary_df)
    
    # 5. Save to Excel
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Per-label F1 scores
        f1_data = {label: ft_report[label]['f1-score'] for label in labels_list}
        pd.DataFrame(list(f1_data.items()), columns=['Label', 'F1-Score']).to_excel(writer, sheet_name='Per-label F1', index=False)
        
        # Confusion Matrices
        def save_cm(cm, name):
            df_cm = pd.DataFrame(cm, index=labels_list, columns=labels_list)
            df_cm.to_excel(writer, sheet_name=f'CM {name}')
            
        save_cm(ft_cm, "Fine-tuned")
        save_cm(gemini_cm, "Gemini")
        save_cm(chatgpt_cm, "ChatGPT")
        
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    evaluate()
