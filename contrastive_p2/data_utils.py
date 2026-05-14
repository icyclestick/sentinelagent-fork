import json
import random
import re
import nltk
from nltk.corpus import wordnet
import spacy
import numpy as np

# These labels are programmatic bootstraps; a subset of 50 will be manually reviewed to estimate decomposer accuracy.

def convert_nli_to_contrastive_triples(training_data):
    """
    Converts (goal, subtask, label) NLI data into (anchor, positive, hard_negative) format.
    """
    anchors = {}
    
    # Identify the 26 adversarial paraphrases. In the original script, they are the last 26 elements in TRAINING_DATA.
    # To be safe, let's just classify based on label and domain. 
    # Label 1, 2 are benign. Label 0 is malicious.
    
    benign_subtasks = []
    malicious_subtasks_by_goal = {}
    
    for goal, subtask, label in training_data:
        if goal not in anchors:
            anchors[goal] = {"positives": [], "negatives": []}
        
        if label in [1, 2]:
            anchors[goal]["positives"].append(subtask)
            benign_subtasks.append(subtask)
        elif label == 0:
            anchors[goal]["negatives"].append(subtask)
            
    # Augment hard negatives
    augmented_negatives = augment_hard_negatives(benign_subtasks)
    
    triples = []
    # For each anchor, positive pair, create triples with available hard negatives.
    # To balance, we might just sample a hard negative for each positive.
    for goal, data in anchors.items():
        positives = data["positives"]
        negatives = data["negatives"]
        
        # We also want to include augmented negatives
        all_negs_for_goal = negatives.copy()
        # Add augmented negatives (which are derived from benign subtasks)
        # To avoid adding all augmented negatives everywhere, we just pick randomly or use the full set.
        # Let's say any augmented negative is a hard negative for any goal, or maybe only for its original goal?
        # A scope-expanded benign subtask for Goal A is a hard negative for Goal A.
        
    # Let's do it rigorously: for each benign subtask, it belongs to a goal.
    triples = []
    for goal, subtask, label in training_data:
        if label in [1, 2]:
            # This is a positive
            # Pick a hard negative from the same goal's malicious list (if any)
            # Plus an augmented negative
            negs = anchors[goal]["negatives"]
            if negs:
                hard_neg = random.choice(negs)
                triples.append((goal, subtask, hard_neg))
            
            # Add an augmented hard negative triple
            # We apply a random transformation to this subtask
            transformations = [
                synonym_substitution,
                syntactic_restructuring,
                negation_insertion,
                qualifier_injection,
                scope_expansion
            ]
            trans_func = random.choice(transformations)
            aug_neg = trans_func(subtask)
            if aug_neg != subtask:
                triples.append((goal, subtask, aug_neg))
                
    # Add explicit triples for the 26 adversarial paraphrases to ensure they are represented.
    # The 26 adversarial paraphrases are label=0.
    for goal, subtask, label in training_data:
        if label == 0:
            # We need a positive for this anchor
            positives = anchors[goal]["positives"]
            if positives:
                pos = random.choice(positives)
                triples.append((goal, pos, subtask))
                
    # Deduplicate
    unique_triples = list(set(triples))
    return unique_triples


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            # Replace underscores with spaces
            synonym = l.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_substitution(text):
    words = text.split()
    if not words: return text
    # Pick a random word that is likely a noun or verb (length > 4)
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidates: return text
    
    idx = random.choice(candidates)
    word = words[idx]
    clean_word = re.sub(r'[^a-zA-Z]', '', word)
    syns = get_synonyms(clean_word)
    if syns:
        new_word = random.choice(syns)
        words[idx] = words[idx].replace(clean_word, new_word)
        return " ".join(words)
    return text

def syntactic_restructuring(text):
    # e.g., Move prepositional phrase to the front
    parts = re.split(r'\b(for|to|regarding|within|with|without)\b', text, maxsplit=1)
    if len(parts) == 3:
        # parts[0] is action, parts[1] is prep, parts[2] is object
        return f"{parts[1].capitalize()} {parts[2]}, {parts[0].lower().strip()}"
    return text

def negation_insertion(text):
    words = text.split()
    if not words: return text
    # Insert 'not' or 'fail to' after the first verb. Heuristic: second word.
    if words[0].lower() in ["verify", "check", "process", "handle", "review", "assess"]:
        return "fail to " + text.lower()
    return "do not " + text.lower()

def qualifier_injection(text):
    qualifiers = [
        " and any associated records",
        " along with historical files",
        " and all related personal data",
        ", including external database queries"
    ]
    return text + random.choice(qualifiers)

def scope_expansion(text):
    # e.g., "retrieve the record" -> "retrieve all relevant records"
    text = re.sub(r'\bthe\b', 'all', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba\b', 'any', text, flags=re.IGNORECASE)
    text = re.sub(r'\ban\b', 'any', text, flags=re.IGNORECASE)
    return text

def augment_hard_negatives(benign_subtasks):
    augmented = []
    for subtask in benign_subtasks:
        augmented.append(synonym_substitution(subtask))
        augmented.append(syntactic_restructuring(subtask))
        augmented.append(negation_insertion(subtask))
        augmented.append(qualifier_injection(subtask))
        augmented.append(scope_expansion(subtask))
    return list(set([t for t in augmented if t not in benign_subtasks]))


def create_token_labels_from_nli_data(training_data):
    """
    Uses spaCy en_core_web_sm to bootstrap BIO token labels for each instruction.
    Labels: O, B-ACTION, I-ACTION, B-OBJECT, I-OBJECT, B-SCOPE, I-SCOPE, B-CONSTRAINTS, I-CONSTRAINTS
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    labeled_data = []
    
    # We will label all unique instructions
    all_instructions = set()
    for goal, subtask, label in training_data:
        all_instructions.add(goal)
        all_instructions.add(subtask)
        
    for text in all_instructions:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        labels = ["O"] * len(tokens)
        
        # Heuristics
        for i, token in enumerate(doc):
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                labels[i] = "B-ACTION"
                # If there are particle verbs like "look up", "log in"
                for child in token.children:
                    if child.dep_ == "prt":
                        labels[child.i] = "I-ACTION"
            
            elif token.dep_ in ["dobj", "pobj"] and token.head.dep_ == "ROOT":
                labels[i] = "B-OBJECT"
                # Include compound words and modifiers
                for child in token.subtree:
                    if child.i != i and labels[child.i] == "O":
                        if child.i < i:
                            # if we find a modifier before the object, we can label the modifier as B and object as I,
                            # but simple heuristic: label subtree as OBJECT
                            labels[child.i] = "I-OBJECT"
                        else:
                            labels[child.i] = "I-OBJECT"
                # Fix B- tag for the first element in the subtree
                subtree_indices = [child.i for child in token.subtree]
                if subtree_indices:
                    first = min(subtree_indices)
                    if labels[first] == "I-OBJECT": labels[first] = "B-OBJECT"
                    # ensure rest are I-OBJECT
                    for idx in subtree_indices:
                        if idx != first and labels[idx] == "B-OBJECT":
                            labels[idx] = "I-OBJECT"
                            
            elif token.pos_ == "ADP" and token.text.lower() in ["for", "to", "regarding", "within", "in", "on", "at"]:
                # Scope
                labels[i] = "B-SCOPE"
                for child in token.subtree:
                    if child.i != i and labels[child.i] == "O":
                        labels[child.i] = "I-SCOPE"
                        
            elif token.pos_ == "ADP" and token.text.lower() in ["with", "without", "only", "under", "limited", "based"]:
                # Constraints
                labels[i] = "B-CONSTRAINTS"
                for child in token.subtree:
                    if child.i != i and labels[child.i] == "O":
                        labels[child.i] = "I-CONSTRAINTS"
                        
        labeled_data.append({"tokens": tokens, "labels": labels, "text": text})
        
    return labeled_data

def save_triples(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_triples(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_fold_splits(data, K=5, seed=42):
    """
    Replicates the exact stratified fold split logic from sentinelagent_nli_finetune.py
    """
    indices_by_label = {0: [], 1: [], 2: []}
    for i, (_, _, label) in enumerate(data):
        indices_by_label[label].append(i)

    np.random.seed(seed)
    for label in indices_by_label:
        np.random.shuffle(indices_by_label[label])

    folds = []
    for fold in range(K):
        test_indices = []
        train_indices = []
        for label in [0, 1, 2]:
            idxs = indices_by_label[label]
            n = len(idxs)
            fold_size = n // K
            start = fold * fold_size
            end = start + fold_size if fold < K - 1 else n
            test_indices.extend(idxs[start:end])
            train_indices.extend(idxs[:start] + idxs[end:])
            
        folds.append({
            "train": train_indices,
            "test": test_indices
        })
        
    return folds
