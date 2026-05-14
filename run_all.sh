#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================================="
echo "    SentinelAgent Full System Verification Suite"
echo "    Architecture Flow: P2 -> P1 -> P6 -> P7 -> P4"
echo "=========================================================================="
echo ""

# Activate the virtual environment
source venv/bin/activate

# --------------------------------------------------------------------------
# 1. P2: Intent Entailment Preservation (Probabilistic)
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "▶ MODULE 1: P2 (Intent Preservation) — NLI Semantic Verification"
echo "   Script: sentinelagent_nli_finetune.py"
echo "=========================================================================="
python sentinelagent_nli_finetune.py
echo ""

# --------------------------------------------------------------------------
# 2. P1: Authority Monotonic Narrowing (Scope enforcement)
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "▶ MODULE 2: P1 (Scope Narrowing) & System Simulation"
echo "   Script: sentinelagent_simulation.py"
echo "=========================================================================="
python sentinelagent_simulation.py
echo ""

# --------------------------------------------------------------------------
# 3. P6 & P7: Scope-Action Conformance & Output Schema Validation
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "▶ MODULE 3: P6 (API Manifest) & P7 (Output Schema) — Real HTTP DAS"
echo "   Scripts: sentinelagent_das_prototype.py, sentinelagent_redteam.py, "
echo "            sentinelagent_redteam_independent.py"
echo "=========================================================================="
echo ">>> Running DelegationBench v4 (516 Scenarios) ..."
python sentinelagent_das_prototype.py
echo ""
echo ">>> Running Black-box Red Team (30 Attacks) ..."
python sentinelagent_redteam.py
echo ""
echo ">>> Running Independent Red Team (45 Attacks) ..."
python sentinelagent_redteam_independent.py
echo ""

# --------------------------------------------------------------------------
# 4. P4: Forensic Reconstructibility & P5 Cascade Containment
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "▶ MODULE 4: P4 (Forensic Logging) & P5 (Cascade Containment) — Formal Proofs"
echo "   Scripts: sentinelagent_theorems.py, sentinelagent_robustness.py"
echo "=========================================================================="
echo ">>> Running 5 Meta-Theorems Executable Proofs ..."
python sentinelagent_theorems.py
echo ""
echo ">>> Running DAS Robustness Tests (29 Edge Cases) ..."
python sentinelagent_robustness.py
echo ""

# --------------------------------------------------------------------------
# 5. Additional System Fault Tolerance Checks
# --------------------------------------------------------------------------
echo "=========================================================================="
echo "▶ MODULE 5: DAS Infrastructure Fault Tolerance (2-of-3 Threshold Signing)"
echo "   Script: sentinelagent_fault_tolerance.py"
echo "=========================================================================="
python sentinelagent_fault_tolerance.py
echo ""

echo "=========================================================================="
echo "✅ ALL MODULES EXECUTED SUCCESSFULLY!"
echo "   System Flow: P2 (Intent) -> P1 (Scope) -> P6/P7 (API/Outputs) -> P4/P5 (Logging/Revocation)"
echo "=========================================================================="
