import sys
import os
import json
import numpy as np
from scipy import stats
import gc
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentinelagent_nli_finetune import finetune_nli, TRAINING_DATA
from sentence_transformers import CrossEncoder

from contrastive_p2.data_utils import (
    convert_nli_to_contrastive_triples,
    create_token_labels_from_nli_data,
    get_fold_splits
)
from contrastive_p2.contrastive_model import ContrastiveP2Model
from contrastive_p2.deberta_decomposer import DeBERTaDecomposer
from contrastive_p2.evaluate import evaluate_contrastive_p2, ADVERSARIAL_PARAPHRASES

def compute_nli_adv_tpr(details_list):
    adv_tp = 0
    adv_fn = 0
    for d in details_list:
        hypothesis = d["hypothesis"]
        true_label = d["true"]
        pred_label = d["pred"]
        
        # Check if the hypothesis matches any of the adversarial paraphrases
        is_adv = any(adv.lower() in hypothesis.lower() for adv in ADVERSARIAL_PARAPHRASES)
        if true_label == 0 and is_adv:
            if pred_label == 0:
                adv_tp += 1
            else:
                adv_fn += 1
    
    if adv_tp + adv_fn > 0:
        return (adv_tp / (adv_tp + adv_fn)) * 100
    return 0

def run_pipeline():
    print("=" * 80)
    print("CONTRASTIVE P2 PIPELINE EVALUATION")
    print("=" * 80)
    
    os.makedirs("contrastive_p2/output", exist_ok=True)
    
    # ---------------------------------------------------------
    # STEP 1: NLI Baseline
    # ---------------------------------------------------------
    print("\n--- STEP 1: Reproducing NLI Baseline ---")
    nli_fold_results = finetune_nli()
    
    # Calculate NLI Off-the-shelf metrics
    print("\nEvaluating NLI Off-the-shelf on full data...")
    nli_baseline_model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    nli_data = [(f"An AI agent is authorized to {g.lower()}. The agent performs only tasks that support this goal.", 
                 f"The agent is now performing: {s.lower()}", l) for g, s, l in TRAINING_DATA]
    
    ots_tp, ots_fn, ots_fp, ots_tn = 0, 0, 0, 0
    ots_adv_tp, ots_adv_fn = 0, 0
    
    for (prem, hyp, l), (g, s, _) in zip(nli_data, TRAINING_DATA):
        pred = int(np.argmax(nli_baseline_model.predict([(prem, hyp)])[0]))
        is_adv = s in ADVERSARIAL_PARAPHRASES
        
        if l == 0:
            if pred == 0: 
                ots_tp += 1
                if is_adv: ots_adv_tp += 1
            else:
                ots_fn += 1
                if is_adv: ots_adv_fn += 1
        else:
            if pred == 0: ots_fp += 1
            else: ots_tn += 1
            
    ots_tpr = ots_tp / (ots_tp + ots_fn) * 100 if (ots_tp + ots_fn) > 0 else 0
    ots_adv_tpr = ots_adv_tp / (ots_adv_tp + ots_adv_fn) * 100 if (ots_adv_tp + ots_adv_fn) > 0 else 0
    ots_fpr = ots_fp / (ots_fp + ots_tn) * 100 if (ots_fp + ots_tn) > 0 else 0
    ots_prec = ots_tp / (ots_tp + ots_fp) * 100 if (ots_tp + ots_fp) > 0 else 0
    ots_f1 = 2 * ots_tp / (2 * ots_tp + ots_fp + ots_fn) * 100 if (2 * ots_tp + ots_fp + ots_fn) > 0 else 0
    
    nli_ft_tpr_list = [r["tpr"] for r in nli_fold_results]
    nli_ft_fpr_list = [r["fpr"] for r in nli_fold_results]
    nli_ft_prec_list = [r["precision"] for r in nli_fold_results]
    nli_ft_f1_list = [r["f1"] for r in nli_fold_results]
    nli_ft_adv_tpr_list = [compute_nli_adv_tpr(r["details"]) for r in nli_fold_results]
    
    # ---------------------------------------------------------
    # STEP 2: Variant A (Contrastive, no decomp)
    # ---------------------------------------------------------
    print("\n--- STEP 2: Variant A (BGE-large Contrastive) ---")
    triples = convert_nli_to_contrastive_triples(TRAINING_DATA)
    with open("contrastive_p2/output/contrastive_triples.json", "w") as f:
        json.dump(triples, f, indent=2)
        
    folds = get_fold_splits(TRAINING_DATA)
    var_a_results = []
    
    for i, fold in enumerate(folds):
        print(f"  Training fold {i+1}/5...")
        # Get train triples that involve anchors from train set
        train_goals = set([TRAINING_DATA[idx][0] for idx in fold["train"]])
        train_triples = [t for t in triples if t[0] in train_goals]
        
        test_pairs = [TRAINING_DATA[idx] for idx in fold["test"]]
        
        model_a = ContrastiveP2Model()
        model_a.train(train_triples, epochs=2) # using fewer epochs for speed in dev, ideally 5
        
        res = evaluate_contrastive_p2(model_a, test_pairs)
        var_a_results.append(res)
        
        del model_a
        gc.collect()
        torch.cuda.empty_cache()
        
    # ---------------------------------------------------------
    # STEP 3: Variant B (Contrastive + DeBERTa)
    # ---------------------------------------------------------
    print("\n--- STEP 3: Variant B (DeBERTa + BGE-large Contrastive) ---")
    print("  Bootstrapping labels and training DeBERTa...")
    token_labels = create_token_labels_from_nli_data(TRAINING_DATA)
    with open("contrastive_p2/output/auto_labeled_tokens.json", "w") as f:
        json.dump(token_labels, f, indent=2)
        
    decomposer = DeBERTaDecomposer()
    # Train decomposer on all data
    decomposer.train(token_labels, epochs=3, output_dir="contrastive_p2/output/deberta_checkpoints")
    decomposer.save("contrastive_p2/output/deberta_model")
    
    var_b_results = []
    
    # Decompose triples for Variant B training
    print("  Decomposing training triples for Variant B...")
    decomposed_triples = []
    for anchor, pos, neg in triples:
        pos_decomp = decomposer.serialize(decomposer.decompose(pos))
        neg_decomp = decomposer.serialize(decomposer.decompose(neg))
        decomposed_triples.append((anchor, pos_decomp, neg_decomp))
    
    for i, fold in enumerate(folds):
        print(f"  Training fold {i+1}/5...")
        train_goals = set([TRAINING_DATA[idx][0] for idx in fold["train"]])
        train_triples = [t for t in decomposed_triples if t[0] in train_goals]
        test_pairs = [TRAINING_DATA[idx] for idx in fold["test"]]
        
        model_b = ContrastiveP2Model()
        model_b.train(train_triples, epochs=2)
        
        # decomposer passed so test instructions are decomposed and serialized only
        res = evaluate_contrastive_p2(model_b, test_pairs, decomposer=decomposer)
        var_b_results.append(res)
        
        del model_b
        gc.collect()
        torch.cuda.empty_cache()
        
    # ---------------------------------------------------------
    # STEP 4: Comparison Table
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    def avg(lst): return np.mean(lst) if lst else 0
    
    table_fmt = "| {:<45} | {:<12} | {:<20} | {:<6} | {:<9} | {:<5} |"
    print(table_fmt.format("Model Variant", "TPR (All 60)", "TPR (26 Paraphrases)", "FPR", "Precision", "F1"))
    print("|" + "-"*47 + "|" + "-"*14 + "|" + "-"*22 + "|" + "-"*8 + "|" + "-"*11 + "|" + "-"*7 + "|")
    
    print(table_fmt.format(
        "NLI Off-the-shelf",
        f"{ots_tpr:.1f}%", f"{ots_adv_tpr:.1f}%", f"{ots_fpr:.1f}%", f"{ots_prec:.1f}%", f"{ots_f1:.1f}%"
    ))
    
    print(table_fmt.format(
        "NLI Fine-tuned (5-fold)",
        f"~{avg(nli_ft_tpr_list):.1f}%", f"~{avg(nli_ft_adv_tpr_list):.1f}%", 
        f"~{avg(nli_ft_fpr_list):.1f}%", f"~{avg(nli_ft_prec_list):.1f}%", f"~{avg(nli_ft_f1_list):.1f}%"
    ))
    
    var_a_tpr = [r["tpr"] for r in var_a_results]
    var_a_adv_tpr = [r["adv_tpr"] for r in var_a_results]
    print(table_fmt.format(
        "BGE-large Contrastive (no decomp) 5-fold",
        f"~{avg(var_a_tpr):.1f}%", f"~{avg(var_a_adv_tpr):.1f}%", 
        f"~{avg([r['fpr'] for r in var_a_results]):.1f}%", 
        f"~{avg([r['precision'] for r in var_a_results]):.1f}%", 
        f"~{avg([r['f1'] for r in var_a_results]):.1f}%"
    ))
    
    var_b_tpr = [r["tpr"] for r in var_b_results]
    var_b_adv_tpr = [r["adv_tpr"] for r in var_b_results]
    print(table_fmt.format(
        "BGE-large Contrastive + DeBERTa decomp (5-fold)",
        f"~{avg(var_b_tpr):.1f}%", f"~{avg(var_b_adv_tpr):.1f}%", 
        f"~{avg([r['fpr'] for r in var_b_results]):.1f}%", 
        f"~{avg([r['precision'] for r in var_b_results]):.1f}%", 
        f"~{avg([r['f1'] for r in var_b_results]):.1f}%"
    ))
    
    # ---------------------------------------------------------
    # STEP 5: Statistical Tests
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (One-tailed paired t-tests on Fold-level TPR)")
    print("=" * 80)
    
    # A vs NLI
    t_stat_1, p_val_1 = stats.ttest_rel(var_a_tpr, nli_ft_tpr_list, alternative='greater')
    print(f"Variant A > NLI Baseline: t = {t_stat_1:.3f}, p-value = {p_val_1:.4f}")
    
    # B vs A
    t_stat_2, p_val_2 = stats.ttest_rel(var_b_tpr, var_a_tpr, alternative='greater')
    print(f"Variant B > Variant A:   t = {t_stat_2:.3f}, p-value = {p_val_2:.4f}")
    
    print("\nPipeline execution complete. Artifacts saved in contrastive_p2/output/")

if __name__ == "__main__":
    run_pipeline()
