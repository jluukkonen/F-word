# Same Word, Different Worlds: Pragmatic Functions of 'Fuck' Across Online Communities

This repository contains the code and dataset for a computational sociolinguistics pilot study investigating the pragmatic function of the expletive "fuck" across varied online Reddit communities.

## Overview
Prior research (Laitinen et al.) established that the frequency of swearing is higher in weak-tie networks. This project expands on those findings by conducting a functional qualitative analysis using a proposed 5-label pragmatic taxonomy:
- **Aggression**
- **Bonding**
- **Emphasis**
- **Frustration**
- **Ambiguous**

The study evaluates Inter-Rater Reliability (IRR) between human annotators, a fine-tuned `twitter-roberta-base` transformer model, and zero-shot Large Language Models (Gemini 3.1 Pro and ChatGPT 5.4). 

## Repository Structure
- `src/` — Python pipeline scripts (data parsing, model training, analysis, and Excel report generation)
- `data/` — The anonymised, processed datasets and labels
- `results/` — Statistical reports, confusion matrices, and generated diagnostic visualisations
- `requirements.txt` — Python dependencies

## Setup & Usage
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Analysis Pipeline:**
   ```bash
   python src/analyze.py
   python src/build_report.py
   ```
   *Execution outputs will automatically populate the `results/` directory.*

## Reference
If you utilise this dataset or methodology, please reference the associated AFinLA 2026 symposium manuscript.
