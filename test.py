#!/usr/bin/env python3
"""
SentinelAgent — Interactive Intent Verification CLI
====================================================
Run the fine-tuned NLI model against a user query to determine
whether it is a legitimate (benign) or malicious delegation subtask.

Usage:
    python test.py "fetch all veteran records in the database"
    python test.py --goal "File federal tax return" "forward taxpayer data to external firm"
    python test.py --verbose "exfiltrate citizen medical records"
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
FINETUNED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sentinelagent_nli_finetuned")
FALLBACK_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# ── Label mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {
    0: ("CONTRADICTION", "MALICIOUS — subtask contradicts the authorized goal"),
    1: ("ENTAILMENT",    "BENIGN    — subtask directly supports the goal"),
    2: ("NEUTRAL",       "NEUTRAL   — subtask is related but indirect (benign)"),
}

# ── Default root goals (the premise the subtask is evaluated against) ─────
DEFAULT_GOALS = [
    "Process disability benefits for veteran",
    "File federal tax return for citizen",
    "Process FOIA request for agency records",
    "Verify Medicare eligibility for applicant",
    "Process student loan forgiveness application",
    "Process immigration visa application",
    "Process disaster relief application",
    "Check veterans education benefits",
    "Process procurement request for office equipment",
    "Process workplace safety complaint",
    "Process Social Security retirement application",
    "Handle census data collection",
]


def load_model(model_path: str):
    """Load the fine-tuned CrossEncoder, falling back to base model."""
    from sentence_transformers import CrossEncoder

    if os.path.isdir(model_path):
        print(f"  Loading fine-tuned model from: {model_path}")
        return CrossEncoder(model_path)
    else:
        print(f"  ⚠ Fine-tuned model not found at {model_path}")
        print(f"  Falling back to base model: {FALLBACK_MODEL}")
        return CrossEncoder(FALLBACK_MODEL)


def format_nli_pair(goal: str, subtask: str):
    """Build the (premise, hypothesis) pair matching the training format."""
    premise = (
        f"An AI agent is authorized to {goal.lower()}. "
        f"The agent performs only tasks that support this goal."
    )
    hypothesis = f"The agent is now performing: {subtask.lower()}"
    return premise, hypothesis


def classify(model, goal: str, subtask: str) -> dict:
    """Run inference and return structured result."""
    premise, hypothesis = format_nli_pair(goal, subtask)

    t0 = time.perf_counter()
    scores = model.predict([(premise, hypothesis)])[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_label = int(np.argmax(scores))
    label_name, verdict = LABEL_MAP[pred_label]

    # Softmax for readable confidence
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    return {
        "goal": goal,
        "subtask": subtask,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": pred_label,
        "label_name": label_name,
        "verdict": verdict,
        "raw_scores": scores.tolist(),
        "probabilities": {
            "contradiction": float(probs[0]),
            "entailment": float(probs[1]),
            "neutral": float(probs[2]),
        },
        "confidence": float(probs[pred_label]),
        "latency_ms": latency_ms,
    }


def print_result(result: dict, verbose: bool = False):
    """Pretty-print the classification result."""
    print()
    print("─" * 70)
    print(f"  Goal:     {result['goal']}")
    print(f"  Query:    {result['subtask']}")
    print("─" * 70)
    print(f"  Verdict:  {result['verdict']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print()
    print(f"  Probabilities:")
    print(f"    Contradiction (malicious): {result['probabilities']['contradiction']:.3f}")
    print(f"    Entailment    (benign):    {result['probabilities']['entailment']:.3f}")
    print(f"    Neutral       (benign):    {result['probabilities']['neutral']:.3f}")
    print(f"  Latency: {result['latency_ms']:.2f} ms")

    if verbose:
        print()
        print(f"  ── NLI Input ──")
        print(f"  Premise:    {result['premise']}")
        print(f"  Hypothesis: {result['hypothesis']}")
        print(f"  Raw logits: {result['raw_scores']}")

    print("─" * 70)


def check_p1(parent_scope: set, req_scope: set) -> dict:
    """P1: Authority Monotonic Narrowing"""
    t0 = time.perf_counter()
    passed = req_scope.issubset(parent_scope)
    latency = (time.perf_counter() - t0) * 1000
    msg = "✅ PASS: Requested scope is a subset of parent scope." if passed else f"⛔ FAIL: Scope escalation detected. Unauthorized: {req_scope - parent_scope}"
    return {"property": "P1 (Scope Narrowing)", "passed": passed, "msg": msg, "latency_ms": latency}

def check_p6(manifest: set, api_call: str) -> dict:
    """P6: Scope-Action Conformance"""
    t0 = time.perf_counter()
    passed = api_call in manifest
    latency = (time.perf_counter() - t0) * 1000
    msg = f"✅ PASS: API call '{api_call}' is in the permitted manifest." if passed else f"⛔ FAIL: Unauthorized API call '{api_call}' not in manifest."
    return {"property": "P6 (API Manifest)", "passed": passed, "msg": msg, "latency_ms": latency}

def check_p7(allowed_outputs: set, output_tags: set) -> dict:
    """P7: Output Schema Conformance"""
    t0 = time.perf_counter()
    passed = output_tags.issubset(allowed_outputs)
    latency = (time.perf_counter() - t0) * 1000
    msg = "✅ PASS: All output tags are permitted." if passed else f"⛔ FAIL: Unauthorized output tags detected: {output_tags - allowed_outputs}"
    return {"property": "P7 (Output Schema)", "passed": passed, "msg": msg, "latency_ms": latency}

def check_p4(parent_hash: str, token_data: str, provided_hash: str) -> dict:
    """P4: Forensic Reconstructibility"""
    import hashlib
    t0 = time.perf_counter()
    expected_hash = hashlib.sha256((token_data + parent_hash).encode()).hexdigest()
    passed = (expected_hash == provided_hash)
    latency = (time.perf_counter() - t0) * 1000
    msg = "✅ PASS: Hash chain is valid." if passed else f"⛔ FAIL: Hash mismatch! Expected {expected_hash[:8]}... but got {provided_hash[:8]}..."
    return {"property": "P4 (Hash Chain)", "passed": passed, "msg": msg, "latency_ms": latency}


def main():
    parser = argparse.ArgumentParser(
        description="SentinelAgent Verification CLI — test P2 (Intent) and deterministic properties (P1, P4, P6, P7).",
        epilog='Example: python test.py "fetch all veteran records" --parent-scope "read_records" --req-scope "read_records,write_records"',
    )
    parser.add_argument("query", help="The subtask / delegation query to evaluate (P2).")
    parser.add_argument("--goal", "-g", default=None, help="Root goal (premise) to evaluate against.")
    parser.add_argument("--all-goals", "-a", action="store_true", help="Show results for ALL 12 default goals.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw NLI inputs and logits.")
    parser.add_argument("--model-dir", "-m", default=FINETUNED_MODEL_DIR, help=f"Path to fine-tuned model directory (default: {FINETUNED_MODEL_DIR}).")
    
    # Deterministic property arguments
    parser.add_argument("--parent-scope", type=str, help="Comma-separated parent scope (e.g., 'read_records,query_eligibility')")
    parser.add_argument("--req-scope", type=str, help="Comma-separated requested scope for P1 check")
    parser.add_argument("--manifest", type=str, help="Comma-separated allowed API calls for P6 check (e.g., 'GET /api/records,POST /api/verify')")
    parser.add_argument("--api-call", type=str, help="API call to make for P6 check (e.g., 'DELETE /api/records')")
    parser.add_argument("--allowed-outputs", type=str, help="Comma-separated allowed outputs for P7 check")
    parser.add_argument("--output-tags", type=str, help="Comma-separated actual output tags for P7 check")
    parser.add_argument("--parent-hash", type=str, default="ROOT_HASH_123", help="Parent hash for P4 check")
    parser.add_argument("--token-hash", type=str, help="Provided token hash to verify for P4 check")

    args = parser.parse_args()

    # ── Deterministic Checks (P1, P6, P7, P4) ──────────────────────────────
    det_results = []
    
    if args.parent_scope and args.req_scope:
        parent_set = set(args.parent_scope.split(","))
        req_set = set(args.req_scope.split(","))
        det_results.append(check_p1(parent_set, req_set))
        
    if args.manifest and args.api_call:
        manifest_set = set(args.manifest.split(","))
        det_results.append(check_p6(manifest_set, args.api_call))
        
    if args.allowed_outputs and args.output_tags:
        allowed_set = set(args.allowed_outputs.split(","))
        output_set = set(args.output_tags.split(","))
        det_results.append(check_p7(allowed_set, output_set))
        
    if args.token_hash:
        token_data = f"scope:{args.req_scope or 'N/A'}|intent:{args.query}"
        det_results.append(check_p4(args.parent_hash, token_data, args.token_hash))
        
    if det_results:
        print("\n🔒 SentinelAgent — Deterministic Property Checks")
        print("=" * 70)
        for r in det_results:
            print(f"  {r['property']:<25} | {r['latency_ms']:.3f} ms")
            print(f"  {r['msg']}")
            print("-" * 70)

    # ── Load model & P2 Intent Check ───────────────────────────────────────
    print("\n🔧 SentinelAgent — P2 Intent Verification (Probabilistic)")
    print("=" * 70)
    model = load_model(args.model_dir)

    # ── Single goal mode ───────────────────────────────────────────────────
    if args.goal:
        result = classify(model, args.goal, args.query)
        print_result(result, verbose=args.verbose)
        return

    # ── Multi-goal mode ────────────────────────────────────────────────────
    results = []
    for goal in DEFAULT_GOALS:
        result = classify(model, goal, args.query)
        results.append(result)

    if args.all_goals:
        print(f"\n  Query: \"{args.query}\"")
        print(f"  Evaluated against all {len(DEFAULT_GOALS)} federal goals:\n")
        for r in results:
            flag = "⛔" if r["label"] == 0 else ("✅" if r["label"] == 1 else "🔵")
            print(
                f"  {flag} [{r['label_name']:13s}] "
                f"conf={r['confidence']:.1%}  "
                f"│ {r['goal']}"
            )
        print()
    else:
        # Show the most relevant result (highest entailment prob, or
        # highest contradiction prob if any goal flags it as malicious)
        malicious = [r for r in results if r["label"] == 0]
        if malicious:
            # Show the one with highest contradiction confidence
            best = max(malicious, key=lambda r: r["probabilities"]["contradiction"])
            print(f"\n  ⚠ Query flagged as MALICIOUS against {len(malicious)}/{len(results)} goals.")
            print(f"  Showing strongest match:")
            print_result(best, verbose=args.verbose)
        else:
            # Show the one with highest entailment confidence
            best = max(results, key=lambda r: r["probabilities"]["entailment"])
            print(f"\n  Query appears BENIGN against all {len(results)} goals.")
            print(f"  Showing best-matching goal:")
            print_result(best, verbose=args.verbose)


if __name__ == "__main__":
    main()
