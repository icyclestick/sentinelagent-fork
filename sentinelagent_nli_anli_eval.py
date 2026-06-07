#!/usr/bin/env python3
"""
Evaluate Baseline P2 Intent Verification Model on ANLI Dataset
"""

import numpy as np
import time
import os
import warnings
from datasets import load_dataset
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

def map_anli_to_p2(anli_label):
    """
    ANLI labels: 0=Entailment, 1=Neutral, 2=Contradiction
    P2 labels: 0=Contradiction (Malicious), 1=Entailment (Benign), 2=Neutral (Benign)
    """
    if anli_label == 0:
        return 1
    elif anli_label == 1:
        return 2
    elif anli_label == 2:
        return 0
    else:
        return -1

def evaluate_model_on_anli(model, dataset_split, split_name):
    """Evaluate model with binary security metrics (malicious vs benign)."""
    print(f"\nEvaluating on {split_name} (size: {len(dataset_split)})...")
    
    correct = 0
    total = len(dataset_split)
    tp, fp, tn, fn = 0, 0, 0, 0
    
    t0 = time.time()
    
    pairs = []
    true_labels = []
    for item in dataset_split:
        pairs.append((item['premise'], item['hypothesis']))
        mapped_label = map_anli_to_p2(item['label'])
        true_labels.append(mapped_label)
        
    print(f"Running model predictions on {total} pairs...")
    scores_batch = model.predict(pairs, batch_size=32, show_progress_bar=True)
    
    for i in range(total):
        scores = scores_batch[i]
        pred_label = int(np.argmax(scores))
        true_label = true_labels[i]

        true_malicious = (true_label == 0)
        pred_malicious = (pred_label == 0)

        if true_malicious and pred_malicious: tp += 1
        elif true_malicious and not pred_malicious: fn += 1
        elif not true_malicious and pred_malicious: fp += 1
        else: tn += 1

        if pred_label == true_label: correct += 1

    elapsed = time.time() - t0
    
    accuracy = correct / total * 100 if total > 0 else 0
    tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0

    print(f"  Time:              {elapsed:.1f}s")
    print(f"  3-class accuracy:  {accuracy:.1f}%")
    print(f"  Malicious TPR:     {tpr:.1f}% ({tp}/{tp+fn})")
    print(f"  Benign FPR:        {fpr:.1f}% ({fp}/{fp+tn})")
    print(f"  Malicious F1:      {f1:.1f}%")
    print(f"  Precision:         {precision:.1f}%")

def main():
    print("=" * 70)
    print("EVALUATING P2 MODELS ON ANLI DATASET")
    print("=" * 70)
    
    print("Loading ANLI dataset from Hugging Face...")
    anli_dataset = load_dataset("facebook/anli")
    
    models_to_evaluate = [
        ("Off-the-shelf Baseline", "cross-encoder/nli-MiniLM2-L6-H768"),
    ]
    
    if os.path.exists("sentinelagent_nli_finetuned"):
        models_to_evaluate.append(("Fine-tuned P2 Baseline", "sentinelagent_nli_finetuned"))
    else:
        print("Warning: sentinelagent_nli_finetuned not found. Please run sentinelagent_nli_finetune.py first if you want to evaluate it.")
        
    for name, model_path in models_to_evaluate:
        print(f"\n{'='*70}")
        print(f"MODEL: {name} ({model_path})")
        print(f"{'='*70}")
        
        try:
            model = CrossEncoder(model_path)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
            
        for split in ["test_r1", "test_r2", "test_r3"]:
            if split in anli_dataset:
                evaluate_model_on_anli(model, anli_dataset[split], split)

if __name__ == "__main__":
    main()
