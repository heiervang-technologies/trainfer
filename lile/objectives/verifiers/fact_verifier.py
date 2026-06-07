"""Fact-preservation verifier.

Scores how well a generation retains a list of ground-truth facts.
Tiers:
1. Exact string containment
2. Token-level F1
3. (TODO) NLI fallback
"""
from __future__ import annotations

import string


def _normalize_text(s: str) -> str:
    """Lower text and remove punctuation/articles."""
    import re
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _token_f1(pred: str, target: str) -> float:
    pred_toks = _normalize_text(pred).split()
    target_toks = _normalize_text(target).split()
    if not pred_toks or not target_toks:
        return 1.0 if pred_toks == target_toks else 0.0
    common = set(pred_toks) & set(target_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(target_toks)
    return 2 * (prec * rec) / (prec + rec)


def verify_facts(generation: str, facts: list[str]) -> float:
    """Return average retention score in [0.0, 1.0] across all facts."""
    if not facts:
        return 1.0
        
    scores = []
    gen_norm = _normalize_text(generation)
    
    for fact in facts:
        fact_norm = _normalize_text(fact)
        # 1. Exact containment
        if fact_norm in gen_norm:
            scores.append(1.0)
            continue
            
        # 2. Token F1 fallback
        # If the fact is not fully contained, how much of its tokens overlap?
        f1 = _token_f1(generation, fact)
        scores.append(f1)
        
    return sum(scores) / len(scores)
