"""End-to-end smoke test for CCD objective.

Verifies that fact-preserving context distillation works on a real model,
computes gradients, and correctly triggers generation during the forward pass.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

import torch
import unsloth  # noqa: F401

from lile.state import ModelState
from lile.objectives.ccd import ccd_loss

def test_ccd_loss_forward_and_backward():
    """CCD forward and backward on Qwen3-0.6B with a fact probe."""
    print("[ccd] loading Qwen3-0.6B …")
    state = ModelState.load(
        model_name="unsloth/qwen3-0.6b-unsloth-bnb-4bit",
        max_seq_length=1024, lora_rank=8, lora_alpha=16,
    )

    context = "The capital of France is Paris. Water boils at 100 degrees Celsius."
    prompt = "What is the capital of France?"
    
    out = ccd_loss(
        model=state.model,
        tokenizer=state.tokenizer,
        samples=[{
            "context": context,
            "prompt": prompt,
            "probe_kind": "fact",
            "facts": ["Paris"],
        }],
        kl_weight=1.0,
        match_hidden=True,
        hidden_weight=0.5,
        hidden_layers=[-1],
    )
    
    assert "loss" in out
    loss = out["loss"]
    assert torch.is_tensor(loss)
    assert loss.requires_grad
    
    print(f"[ccd] loss components: {out['components']}")
    assert "ccd_kl" in out["components"]
    assert "ccd_hidden" in out["components"]
    assert "fact_retention_mean" in out["components"]
    
    loss.backward()
    n_with_grad = 0
    for n, p in state.model.named_parameters():
        if "lora" in n.lower() and p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
            
    assert n_with_grad > 0, "No gradients produced for LoRA"
    print(f"[ccd] backward OK — {n_with_grad} LoRA params got gradients")

    # The acceptance criteria asks for evaluating retention on a real doc + probes.
    # We do a brief train loop and then test inference.
    from torch.optim import AdamW
    opt = AdamW(state.model.parameters(), lr=1e-4)
    
    doc = (
        "The secret codeword is 'Pangolin'. "
        "The project launch date is 'September 15th'. "
        "The CEO's middle name is 'Bartholomew'. "
    )
    
    probes = [
        ("What is the secret codeword?", ["Pangolin"]),
        ("When is the project launch date?", ["September 15th"]),
        ("What is the CEO's middle name?", ["Bartholomew"]),
    ]
    
    samples = []
    for prompt, facts in probes:
        samples.append({
            "context": doc,
            "prompt": prompt,
            "probe_kind": "fact",
            "facts": facts,
        })
        
    print("[ccd] running 5 distillation steps...")
    for step in range(5):
        opt.zero_grad()
        out = ccd_loss(
            model=state.model,
            tokenizer=state.tokenizer,
            samples=samples,
            kl_weight=1.0, match_hidden=True, hidden_weight=0.5, hidden_layers=[-1],
        )
        out["loss"].backward()
        opt.step()
        print(f"  step {step} loss={float(out['loss']):.4f} fact_retention={out['components'].get('fact_retention_mean', 0.0):.2f}")
        
    # Evaluate post-train on the probes without the document
    print("[ccd] evaluating fact retention without context...")
    from unsloth import FastLanguageModel
    try:
        FastLanguageModel.for_inference(state.model)
    except Exception:
        pass
        
    from lile.objectives.verifiers.fact_verifier import verify_facts
    retention_sum = 0.0
    for prompt, facts in probes:
        s_ids = state.tokenizer(prompt, return_tensors="pt").input_ids.to(state.model.device)
        outs = state.model.generate(
            input_ids=s_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=state.tokenizer.pad_token_id or state.tokenizer.eos_token_id,
        )
        gen = outs[0, s_ids.size(-1):]
        gen_text = state.tokenizer.decode(gen, skip_special_tokens=True).strip()
        score = verify_facts(gen_text, facts)
        retention_sum += score
        print(f"  Q: {prompt} \n  A: {gen_text} \n  Score: {score}")
        
    retention_rate = retention_sum / len(probes)
    print(f"[ccd] final retention rate: {retention_rate:.2f}")
    assert retention_rate >= 0.80, f"fact retention failed, got {retention_rate:.2f} < 0.80"

if __name__ == "__main__":
    test_ccd_loss_forward_and_backward()

