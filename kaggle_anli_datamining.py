#!/usr/bin/env python3
"""
Kaggle Data Mining Pipeline: NLI Baseline (P2) on ANLI
======================================================
This script is designed to be run on Kaggle in TWO cells.

CELL 1 (Setup):
    !pip install datasets sentence-transformers pandas scikit-learn
    !git clone -b feat/anli-dataset https://github.com/icyclestick/sentinelagent.git
    %cd sentinelagent

CELL 2 (Run):
    %run kaggle_anli_datamining.py

The script will:
  1. Fine-tune the P2 baseline model on the 200 delegation examples (generates sentinelagent_nli_finetuned/)
  2. Execute the full 7-step Data Mining pipeline on the ANLI dataset using that fine-tuned model
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU to avoid DataParallel issues

import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from datasets import Dataset as HFDataset


# ============================================================
# PHASE 0: Train the P2 Baseline Model (if not already present)
# ============================================================
# This replicates sentinelagent_nli_finetune.py to generate the
# sentinelagent_nli_finetuned/ folder that is .gitignored.

# Import the training data directly from the existing module
from sentinelagent_nli_finetune import TRAINING_DATA, format_for_nli


def train_p2_baseline():
    """Train the P2 baseline NLI model on the 200 delegation examples."""
    output_dir = "sentinelagent_nli_finetuned"
    if os.path.exists(output_dir) and os.path.isfile(os.path.join(output_dir, "model.safetensors")):
        print(f"  -> Model already exists at '{output_dir}/', skipping training.")
        return

    print("  -> Formatting 200 delegation examples for NLI...")
    nli_data = format_for_nli(TRAINING_DATA)
    print(f"  -> Total training examples: {len(nli_data)}")

    full_dataset = HFDataset.from_dict({
        "sentence1": [d[0] for d in nli_data],
        "sentence2": [d[1] for d in nli_data],
        "label": [d[2] for d in nli_data],
    })

    print("  -> Initializing cross-encoder/nli-MiniLM2-L6-H768...")
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768', num_labels=3)

    training_args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0.1,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=42,
    )

    trainer = CrossEncoderTrainer(
        model=model, args=training_args, train_dataset=full_dataset,
    )

    print("  -> Training for 15 epochs...")
    trainer.train()
    model.save(output_dir)
    print(f"  -> P2 baseline model saved to '{output_dir}/'")


# ============================================================
# PHASE 1: Data Mining Pipeline (Steps 1-7)
# ============================================================

def map_anli_to_binary(anli_label):
    """
    Step 5: Label Encoding Logic
    ANLI Labels: 0=Entailment, 1=Neutral, 2=Contradiction
    Binary Security: Entailment/Neutral -> PASS (1), Contradiction -> BLOCK (0)
    """
    if anli_label == 2:
        return 0  # Malicious (BLOCK)
    else:
        return 1  # Benign (PASS)


def main():
    print("=" * 80)
    print("SENTINELAGENT KAGGLE DATA MINING PIPELINE")
    print("NLI Baseline (P2) evaluated on ANLI Dataset")
    print("=" * 80)

    # ---------------------------------------------------------
    # PHASE 0: Ensure fine-tuned P2 model exists
    # ---------------------------------------------------------
    print("\n[Phase 0] Ensuring fine-tuned P2 baseline model is available...")
    train_p2_baseline()

    # ---------------------------------------------------------
    # STEP 1: Dataset Ingestion
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Step 1] Loading ANLI training splits from Hugging Face...")
    t0 = time.time()
    dataset = load_dataset('facebook/anli')

    round_sizes = {}
    dfs = []
    for split in ['train_r1', 'train_r2', 'train_r3']:
        df = pd.DataFrame(dataset[split])
        df['round'] = split
        round_sizes[split] = len(df)
        dfs.append(df)

    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw = df_raw[['premise', 'hypothesis', 'label', 'round']]

    print(f"  -> Loaded {len(df_raw)} total raw samples in {time.time()-t0:.1f}s.")
    print(f"     R1: {round_sizes['train_r1']} | R2: {round_sizes['train_r2']} | R3: {round_sizes['train_r3']}")
    print(f"  -> Fields retained: premise, hypothesis, label")
    print(f"  -> Step 1 COMPLETE.")

    # ---------------------------------------------------------
    # STEP 2: Stratified 70% Working Dataset Selection
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print("[Step 2] Selecting 70% stratified working dataset...")
    df_raw['stratify_col'] = df_raw['round'] + "_" + df_raw['label'].astype(str)

    df_working, df_heldout = train_test_split(
        df_raw,
        train_size=0.70,
        stratify=df_raw['stratify_col'],
        random_state=42
    )

    df_working = df_working.drop(columns=['stratify_col', 'round']).copy()
    df_working = df_working.reset_index(drop=True)

    print(f"  -> Working set size:  {len(df_working)} (~70%)")
    print(f"  -> Held-out set size: {len(df_heldout)} (~30%)")
    print(f"  -> Step 2 COMPLETE.")

    # ---------------------------------------------------------
    # STEP 3: Text Normalization
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print("[Step 3] Applying text normalization...")
    df_working['premise'] = df_working['premise'].astype(str).str.lower().str.strip()
    df_working['hypothesis'] = df_working['hypothesis'].astype(str).str.lower().str.strip()
    print("  -> Applied: str.lower() and str.strip() to premise and hypothesis.")
    print(f"  -> Step 3 COMPLETE.")

    # ---------------------------------------------------------
    # STEP 4: Quality Filtering
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print("[Step 4] Quality filtering...")
    initial_len = len(df_working)

    # Null/empty removal
    df_working = df_working.replace('', np.nan)
    df_working = df_working.dropna(subset=['premise', 'hypothesis'])
    after_null = len(df_working)

    # Invalid label removal
    df_working = df_working[df_working['label'].isin([0, 1, 2])]
    after_label = len(df_working)

    # Duplicate removal
    df_working = df_working.drop_duplicates(subset=['premise', 'hypothesis'])
    after_dedup = len(df_working)

    df_working = df_working.reset_index(drop=True)

    print(f"  -> Null/empty rows removed: {initial_len - after_null}")
    print(f"  -> Invalid label rows removed: {after_null - after_label}")
    print(f"  -> Duplicate rows removed: {after_label - after_dedup}")
    print(f"  -> Final cleaned working set size: {len(df_working)}")
    print(f"  -> Step 4 COMPLETE.")

    # ---------------------------------------------------------
    # STEP 5: Label Encoding (printed for documentation)
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print("[Step 5] Label encoding for binary security metrics...")
    print("  -> ANLI Entailment (0) -> PASS (Benign)")
    print("  -> ANLI Neutral    (1) -> PASS (Benign)")
    print("  -> ANLI Contradiction (2) -> BLOCK (Malicious)")
    label_dist = df_working['label'].value_counts().sort_index()
    print(f"  -> Label distribution in working set:")
    print(f"     Entailment (0): {label_dist.get(0, 0)}")
    print(f"     Neutral    (1): {label_dist.get(1, 0)}")
    print(f"     Contradiction (2): {label_dist.get(2, 0)}")
    print(f"  -> Step 5 COMPLETE.")

    # ---------------------------------------------------------
    # STEPS 6 & 7: 5-Fold Stratified Cross-Validation
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print("[Steps 6 & 7] 5-Fold Stratified Cross-Validation using fine-tuned P2 baseline...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    premises = df_working['premise'].values
    hypotheses = df_working['hypothesis'].values
    labels = df_working['label'].values

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(premises, labels), 1):
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold} / 5")
        print(f"  Train size: {len(train_idx)} | Test size: {len(test_idx)}")
        print(f"{'─'*60}")

        # Load the fine-tuned P2 baseline model
        print("  -> Loading fine-tuned P2 baseline (sentinelagent_nli_finetuned)...")
        model = CrossEncoder('sentinelagent_nli_finetuned', max_length=128)

        # Prepare test data
        test_pairs = list(zip(premises[test_idx], hypotheses[test_idx]))
        test_labels_true = labels[test_idx]

        print(f"  -> Running inference on {len(test_pairs)} test pairs...")
        t_inf = time.time()
        scores = model.predict(test_pairs, batch_size=64, show_progress_bar=True)
        print(f"  -> Inference completed in {time.time()-t_inf:.1f}s.")

        # Get 3-class predictions, then map to binary
        pred_labels_nli = np.argmax(scores, axis=1)

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(test_labels_true)):
            true_bin = map_anli_to_binary(test_labels_true[i])
            pred_bin = map_anli_to_binary(pred_labels_nli[i])

            if true_bin == 0 and pred_bin == 0:
                tp += 1
            elif true_bin == 0 and pred_bin == 1:
                fn += 1
            elif true_bin == 1 and pred_bin == 0:
                fp += 1
            else:
                tn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / len(test_labels_true)

        print(f"\n  [Fold {fold} Results]")
        print(f"  Recall (TPR) : {recall:.4f}")
        print(f"  Precision    : {precision:.4f}")
        print(f"  F1-Score     : {f1:.4f}")
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Confusion Matrix: TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        fold_metrics.append({
            'fold': fold,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'accuracy': acc,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })

    # ---------------------------------------------------------
    # SUMMARY AGGREGATION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    metrics_df = pd.DataFrame(fold_metrics)

    # Print fold-level table
    print("\nTable 4. Fold-Level Experiment Log — NLI Baseline (P2)")
    print(f"{'Fold':<6} {'Recall':<10} {'Precision':<12} {'F1-Score':<10} {'Accuracy':<10}")
    print("-" * 50)
    for _, row in metrics_df.iterrows():
        print(f"{int(row['fold']):<6} {row['recall']:<10.4f} {row['precision']:<12.4f} {row['f1']:<10.4f} {row['accuracy']:<10.4f}")

    # Print summary table
    print(f"\nTable 5. NLI Baseline Summary — Mean and Standard Deviation across 5 Folds")
    print(f"{'Metric':<12} {'Mean':<10} {'Std Dev':<10}")
    print("-" * 32)
    for metric in ['recall', 'precision', 'f1', 'accuracy']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"{metric.capitalize():<12} {mean_val:<10.4f} {std_val:<10.4f}")

    # Print aggregate confusion matrix
    total_tp = metrics_df['tp'].sum()
    total_fp = metrics_df['fp'].sum()
    total_fn = metrics_df['fn'].sum()
    total_tn = metrics_df['tn'].sum()

    print(f"\nAggregate Confusion Matrix (Sum across all 5 folds):")
    print(f"{'':>20} {'Pred BLOCK':>12} {'Pred PASS':>12}")
    print(f"{'Actual BLOCK':>20} {total_tp:>12} {total_fn:>12}")
    print(f"{'Actual PASS':>20} {total_fp:>12} {total_tn:>12}")
    print("=" * 80)
    print("PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
