import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score, accuracy_score
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.grid'] = False
sns.set_context("talk", font_scale=0.9)

COLORS = {'aggression': '#E63946', 'bonding': '#457B9D', 'emphasis': '#F4A261', 'frustration': '#6A4C93', 'ambiguous': '#94A3B8'}
MODEL_COLORS = ['#34D399', '#22D3EE', '#8B5CF6']
ORDER = ['aggression', 'bonding', 'emphasis', 'frustration', 'ambiguous']
RATERS = ['expert', 'friend1', 'friend2', 'friend3', 'gemini', 'chatgpt']
RATER_MAP = {
    'expert': 'Principal Annotator',
    'friend1': 'Peer A (Random)',
    'friend2': 'Peer B (Lecturer)',
    'friend3': 'Peer C (English Major)',
    'gemini': 'Gemini 3.1 Pro',
    'chatgpt': 'ChatGPT 5.4'
}

FIGURES_DIR = 'results/figures/'
EXCEL_OUT = 'results/analysis_summary.xlsx'

def setup_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def load_all_data():
    print("Loading data...")
    expert_df = pd.read_excel('data/labeled/to_label.xlsx').dropna(subset=['label'])
    expert_df['id'] = expert_df['id'].astype(str).str.strip().str.lower()
    
    gemini_df = pd.read_csv('data/labeled/gemini_labels.tsv', sep='\t')
    gemini_df['id'] = gemini_df['id'].astype(str).str.strip().str.lower()
    
    chatgpt_df = pd.read_csv('data/labeled/chatgpt_labels.tsv', sep='\t')
    chatgpt_df['id'] = chatgpt_df['id'].astype(str).str.strip().str.lower()
    
    peers = {}
    for i in range(1, 4):
        fname = f'data/labeled/friend{i}_labels.tsv'
        if os.path.exists(fname):
            pf = pd.read_csv(fname, sep='\t')
            pf['id'] = pf['id'].astype(str).str.strip().str.lower()
            peers[f'friend{i}'] = pf
            
    merged = expert_df[['id', 'subreddit', 'network_tie_strength', 'text', 'label']].rename(columns={'label': 'expert'})
    merged = merged.merge(gemini_df.rename(columns={'label': 'gemini'}), on='id', how='left')
    merged = merged.merge(chatgpt_df.rename(columns={'label': 'chatgpt'}), on='id', how='left')
    for key, pf in peers.items():
        merged = merged.merge(pf.rename(columns={'label': key}), on='id', how='left')
    return merged

def section_a_distributions(df):
    results = {}
    print("Running Section A: Distributions...")
    # A1. Overall
    plt.figure(figsize=(10, 6))
    counts = df['expert'].value_counts().reindex(ORDER)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=[COLORS[l] for l in ORDER], legend=False)
    plt.title('Overall Distribution (Principal Annotator Labels)', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xlabel('Pragmatic Function')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'a1_label_distribution.png'), dpi=300); plt.close()
    
    # A2. Subreddit Distribution (Stacked)
    sub_df = pd.crosstab(df['subreddit'], df['expert'], normalize='index') * 100
    sub_df = sub_df.reindex(columns=ORDER)
    sub_total = df['subreddit'].value_counts()
    sub_df = sub_df.loc[sub_total.index]
    plt.figure(figsize=(12, 7))
    sub_df.plot(kind='barh', stacked=True, color=[COLORS[l] for l in ORDER], ax=plt.gca())
    plt.title('Pragmatic Function Distribution by Subreddit', fontsize=14, fontweight='bold')
    plt.xlabel('Proportion (%)')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'a2_distribution_by_subreddit.png'), dpi=300); plt.close()
    results['subreddit_dist'] = sub_df

    # A3. Tie Strength
    tie_df = pd.crosstab(df['network_tie_strength'], df['expert'], normalize='index') * 100
    tie_df = tie_df.reindex(columns=ORDER)
    tie_df.plot(kind='bar', color=[COLORS[l] for l in ORDER], figsize=(10, 6))
    plt.title('Distribution by Tie Strength', fontsize=14, fontweight='bold')
    plt.ylabel('Proportion (%)'); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'a3_distribution_by_tie_strength.png'), dpi=300); plt.close()
    
    contingency = pd.crosstab(df['network_tie_strength'], df['expert'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    v = np.sqrt(chi2 / (contingency.sum().sum() * (min(contingency.shape) - 1)))
    results['chi_square'] = pd.DataFrame({'Metric': ['Chi-Square', 'p-value', 'Cramer\'s V'], 'Value': [chi2, p, v]})
    results['overall_counts'] = counts.to_frame()
    return results

def section_b_disagreements(df):
    results = {}
    print("Running Section B: Disagreements...")
    sub_raters = ['friend1', 'friend2', 'friend3', 'gemini', 'chatgpt']
    agreement_data = []
    # B1 Fix: NaN-aware labeling
    for label in ORDER:
        label_row = []
        for rater in sub_raters:
            subset = df[df['expert'] == label]
            valid_rater = subset[subset[rater].notna()]
            if not valid_rater.empty:
                acc = (valid_rater[rater] == label).mean() * 100
            else:
                acc = np.nan
            label_row.append(acc)
        agreement_data.append(label_row)
    agreement_df = pd.DataFrame(agreement_data, index=ORDER, columns=[RATER_MAP[r] for r in sub_raters])
    plt.figure(figsize=(10, 6))
    sns.heatmap(agreement_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False)
    plt.title('Label-Level Agreement with P.A. (%)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'b1_label_level_agreement.png'), dpi=300); plt.close()
    results['label_agreement'] = agreement_df

    # B2/B3 Confusion Matrices
    for rater in ['gemini', 'chatgpt']:
        valid = df[df[rater].notna()]
        cm = confusion_matrix(valid['expert'], valid[rater], labels=ORDER, normalize='true')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(cm, index=ORDER, columns=ORDER), annot=True, fmt=".2f", cmap="Purples")
        plt.title(f'Confusion Matrix: P.A. vs. {RATER_MAP[rater]} (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'b{"2" if rater=="gemini" else "3"}_expert_vs_{rater}_cm.png'), dpi=300)
        plt.close()

    # B4 Confused Pairs
    valid_gemini = df[df['gemini'].notna()]
    cp_df = pd.DataFrame([{'P.A.': r['expert'], 'Gemini': r['gemini'], 'Text': r['text']} for _, r in valid_gemini.iterrows() if r['expert'] != r['gemini']])
    pair_counts = cp_df.groupby(['P.A.', 'Gemini']).size().sort_values(ascending=False).head(5)
    results['confused_pairs'] = pair_counts.reset_index(name='Count')
    examples = []
    for (exp, gem), _ in pair_counts.items():
        for _, row in cp_df[(cp_df['P.A.'] == exp) & (cp_df['Gemini'] == gem)].head(2).iterrows():
            examples.append({'P.A.': exp, 'Gemini': gem, 'Comment': row['Text']})
    results['confused_examples'] = pd.DataFrame(examples)

    # B5 Hard Items Table
    all_rater_cols = ['expert', 'friend1', 'friend2', 'friend3', 'gemini', 'chatgpt']
    df['unique_labels'] = df[all_rater_cols].apply(lambda x: x.dropna().nunique(), axis=1)
    hard_items = df.sort_values('unique_labels', ascending=False).head(10)
    plt.figure(figsize=(14, 8)); plt.axis('off')
    tbl = plt.table(cellText=hard_items[['id', 'expert', 'friend1', 'friend2', 'friend3', 'gemini']].values,
                  colLabels=['ID', 'P.A.', 'Peer A', 'Peer B', 'Peer C', 'Gemini'],
                  loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    plt.title('Top 10 "Hard Items" (Maximum Disagreement)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'b5_hard_items_table.png'), dpi=300); plt.close()
    results['hard_items'] = hard_items[['id', 'text'] + all_rater_cols]
    
    return results

def section_c_transformer(df):
    results = {}
    print("Running Section C: Transformer Analysis...")
    if os.path.exists('results/training_log.json'):
        logs = json.load(open('results/training_log.json')); eval_logs = [l for l in logs if 'eval_kappa' in l]
        if eval_logs:
            epochs, kappas, losses = [l['epoch'] for l in eval_logs], [l['eval_kappa'] for l in eval_logs], [l['eval_loss'] for l in eval_logs]
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(epochs, losses, color='tab:red', marker='o', label='Eval Loss'); ax1.set_ylabel('Loss', color='tab:red')
            ax2 = ax1.twinx(); ax2.plot(epochs, kappas, color='tab:blue', marker='s', label='Eval Kappa'); ax2.set_ylabel('Kappa (κ)', color='tab:blue')
            ax2.axhline(y=0.415, color='gray', linestyle='--', alpha=0.7); ax2.text(epochs[-1] + 0.1, 0.415, 'Gemini κ=0.415', va='center', color='gray', fontsize=10)
            ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
            plt.title('Training Dynamics', fontsize=14, fontweight='bold'); plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'c1_training_curve.png'), dpi=300); plt.close()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = 'models/pilot_v1/'
    tokenizer = AutoTokenizer.from_pretrained(model_dir); model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device); model.eval()
    test_df = pd.read_csv('data/processed/test.csv'); test_ids = set(test_df['id'].str.lower())
    all_preds, all_conf = [], []
    for i in range(0, len(test_df), 8):
        batch = test_df['formatted_text'].iloc[i:i+8].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1); conf, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy()); all_conf.extend(conf.cpu().numpy())
    rev_label_map = {v: k for k, v in json.load(open('data/processed/label_map.json')).items()}
    test_df['pred_label'] = [rev_label_map[p] for p in all_preds]; test_df['confidence'] = all_conf
    
    ai_test = df[df['id'].isin(test_ids)].copy()
    models = ['Fine-tuned', 'Gemini 3.1 Pro', 'ChatGPT 5.4']
    ft_kappa, ft_acc, ft_f1 = cohen_kappa_score(test_df['label'], test_df['pred_label'], labels=ORDER), accuracy_score(test_df['label'], test_df['pred_label']), f1_score(test_df['label'], test_df['pred_label'], labels=ORDER, average='weighted')
    gem_test = ai_test[ai_test['gemini'].notna()]
    gem_kappa, gem_acc, gem_f1 = (cohen_kappa_score(gem_test['expert'], gem_test['gemini'], labels=ORDER), accuracy_score(gem_test['expert'], gem_test['gemini']), f1_score(gem_test['expert'], gem_test['gemini'], labels=ORDER, average='weighted')) if not gem_test.empty else (0,0,0)
    gpt_test = ai_test[ai_test['chatgpt'].notna()]
    gpt_kappa, gpt_acc, gpt_f1 = (cohen_kappa_score(gpt_test['expert'], gpt_test['chatgpt'], labels=ORDER), accuracy_score(gpt_test['expert'], gpt_test['chatgpt']), f1_score(gpt_test['expert'], gpt_test['chatgpt'], labels=ORDER, average='weighted')) if not gpt_test.empty else (0,0,0)
    
    metrics_df = pd.DataFrame({
        'Model': models * 3, 'Metric': ['Kappa']*3 + ['Accuracy']*3 + ['Weighted F1']*3,
        'Value': [ft_kappa, gem_kappa, gpt_kappa, ft_acc, gem_acc, gpt_acc, ft_f1, gem_f1, gpt_f1]
    })
    plt.figure(figsize=(10, 6)); sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model', palette=MODEL_COLORS)
    plt.axhline(y=0.2, color='gray', linestyle=':', label='Random baseline'); plt.title('Benchmarking (n=60)', fontsize=14, fontweight='bold'); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'c3_model_comparison.png'), dpi=300); plt.close()
    
    f1_data = []
    reports = {'Fine-tuned': classification_report(test_df['label'], test_df['pred_label'], labels=ORDER, output_dict=True), 'Gemini 3.1 Pro': classification_report(gem_test['expert'], gem_test['gemini'], labels=ORDER, output_dict=True, zero_division=0) if not gem_test.empty else None, 'ChatGPT 5.4': classification_report(gpt_test['expert'], gpt_test['chatgpt'], labels=ORDER, output_dict=True, zero_division=0) if not gpt_test.empty else None}
    for m, r in reports.items():
        if r:
            for l in ORDER: f1_data.append({'Model': m, 'Label': l, 'F1': r[l]['f1-score']})
    plt.figure(figsize=(12, 6)); sns.barplot(data=pd.DataFrame(f1_data), x='Label', y='F1', hue='Model', palette=MODEL_COLORS)
    plt.title('Per-Label F1 (Shared Test Set)', fontsize=14, fontweight='bold'); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'c4_per_label_f1.png'), dpi=300); plt.close()
    results['test_predictions'] = test_df
    return results

def section_d_rater_comparative(df):
    results = {}
    print("Running Section D: Rater Comparative...")
    kappa_matrix = pd.DataFrame(index=RATERS, columns=RATERS, dtype=float)
    for r1 in RATERS:
        for r2 in RATERS:
            overlap = df[df[r1].notna() & df[r2].notna()]
            kappa_matrix.loc[r1, r2] = cohen_kappa_score(overlap[r1], overlap[r2], labels=ORDER) if len(overlap) >= 10 else 0
    dist_matrix = 1 - kappa_matrix.fillna(0).values; np.fill_diagonal(dist_matrix, 0); dist_matrix = (dist_matrix + dist_matrix.T) / 2
    linked = linkage(squareform(dist_matrix), 'ward')
    plt.figure(figsize=(10, 6)); dendrogram(linked, labels=[RATER_MAP[r] for r in RATERS], leaf_font_size=12)
    plt.title('Rater Clustering (Dendrogram)', fontsize=14, fontweight='bold'); plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'd2_rater_dendrogram.png'), dpi=300); plt.close()
    plt.figure(figsize=(10, 8)); sns.heatmap(kappa_matrix.rename(index=RATER_MAP, columns=RATER_MAP), annot=True, fmt=".2f", cmap="RdYlGn", cbar=False)
    plt.title('Pairwise Inter-Rater Reliability (Kappa)', fontsize=14, fontweight='bold'); plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, 'd3_pairwise_kappa_heatmap.png'), dpi=300); plt.close()
    results['kappa_matrix'] = kappa_matrix.rename(index=RATER_MAP, columns=RATER_MAP)
    return results

def section_e_confidence(test_df):
    print("Running Section E: Confidence Analysis...")
    plt.figure(figsize=(10, 6)); sns.histplot(test_df['confidence'], bins=20, kde=True, color='#34D399')
    plt.title('Model Confidence Distribution', fontsize=14, fontweight='bold'); plt.xlabel('Confidence Score'); plt.ylabel('Count'); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'e1_model_confidence.png'), dpi=300); plt.close()

def section_f_qualitative(df, test_df):
    results = {}
    print("Running Section F: Qualitative Samples...")
    results['bonding_samples'] = df[(df['expert'] == 'bonding') & (df['friend3'] == 'bonding') & (df['gemini'] != 'bonding')][['id', 'text', 'expert', 'friend3', 'gemini']].head(5)
    results['confidence_errors'] = test_df[(test_df['confidence'] > 0.9) & (test_df['label'] != test_df['pred_label'])][['id', 'formatted_text', 'label', 'pred_label', 'confidence']].head(5)
    return results

if __name__ == "__main__":
    setup_dirs(); df = load_all_data()
    a_res, b_res, c_res = section_a_distributions(df), section_b_disagreements(df), section_c_transformer(df)
    d_res = section_d_rater_comparative(df)
    section_e_confidence(c_res['test_predictions'])
    f_res = section_f_qualitative(df, c_res['test_predictions'])
    with pd.ExcelWriter(EXCEL_OUT) as writer:
        a_res['subreddit_dist'].to_excel(writer, sheet_name='Dist by Subreddit')
        a_res['chi_square'].to_excel(writer, sheet_name='Chi-Square', index=False)
        b_res['label_agreement'].to_excel(writer, sheet_name='Label Agreement')
        b_res['confused_pairs'].to_excel(writer, sheet_name='Confused Pairs', index=False)
        b_res['confused_examples'].to_excel(writer, sheet_name='Confused Examples', index=False)
        b_res['hard_items'].to_excel(writer, sheet_name='Hard Items', index=False)
        d_res['kappa_matrix'].to_excel(writer, sheet_name='Kappa Matrix')
        f_res['bonding_samples'].to_excel(writer, sheet_name='Bonding Errors', index=False)
        f_res['confidence_errors'].to_excel(writer, sheet_name='Confidence Errors', index=False)
    print(f"✓ All saved to {FIGURES_DIR} and {EXCEL_OUT}")
