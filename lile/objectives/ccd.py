"""Context Distillation (CCD) objective — fact-preserving.

Internalizes a long context block (system prompt, doc) into the live LoRA residual
so future inferences can drop the context.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ._utils import _to_int_list, pad_and_stack


def ccd_loss(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    kl_weight: float = 1.0,
    match_hidden: bool = True,
    hidden_weight: float = 0.5,
    hidden_layers: list[int] | None = None,
    sft_weight: float = 1.0,
    **_: Any,
) -> dict[str, Any]:
    """Fact-preserving Context Distillation.

    Samples:
    {
      "context": str,
      "prompt": str,
      "response": str | None,
      ...
    }
    """
    if not samples:
        raise ValueError("ccd_loss requires at least one sample")

    if not hasattr(model, "disable_adapter"):
        raise ValueError("ccd_loss requires a model with disable_adapter() (e.g. PEFT)")

    if hidden_layers is None:
        hidden_layers = [-1, -2, -3, -4]

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    # 1. Tokenize Teacher (context + prompt) and Student (prompt)
    teacher_tokenized = []
    student_tokenized = []
    has_response = False

    for s in samples:
        context = s.get("context", "")
        prompt = s.get("prompt", "")
        response = s.get("response")

        # Teacher: context + prompt
        # We need to compute logits over the prompt positions.
        # But wait, Askell baseline: KL(π_teacher || π_student) over the probe span.
        # So teacher needs to output logits for the prompt tokens, given context.
        # Let's tokenize context and prompt separately to find lengths.
        # Since we use build_chat_inputs, we can just concat them.

        s_ids = _to_int_list(tokenizer(text=prompt, add_special_tokens=False).input_ids)
        if getattr(tokenizer, "chat_template", None):
            is_processor = hasattr(tokenizer, "image_processor") or hasattr(
                tokenizer, "video_processor"
            )
            c_p = context + "\n\n" + prompt
            user_content_t = [{"type": "text", "text": c_p}] if is_processor else c_p
            messages_t = [{"role": "user", "content": user_content_t}]
            t_ids = _to_int_list(
                tokenizer.apply_chat_template(
                    messages_t,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )
            )

            user_content_s = (
                [{"type": "text", "text": prompt}] if is_processor else prompt
            )
            messages_s = [{"role": "user", "content": user_content_s}]
            s_ids = _to_int_list(
                tokenizer.apply_chat_template(
                    messages_s,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )
            )
        else:
            t_ids = _to_int_list(
                tokenizer(
                    text=context + "\n\n" + prompt, add_special_tokens=False
                ).input_ids
            )
            s_ids = _to_int_list(
                tokenizer(text=prompt, add_special_tokens=False).input_ids
            )

        teacher_tokenized.append({"input_ids": torch.tensor(t_ids, dtype=torch.long)})
        student_tokenized.append({"input_ids": torch.tensor(s_ids, dtype=torch.long)})
        if response:
            has_response = True

    t_batch = pad_and_stack(teacher_tokenized, pad_id=pad_id)
    s_batch = pad_and_stack(student_tokenized, pad_id=pad_id)
    t_ids = t_batch["input_ids"].to(device)
    t_mask = t_batch["attention_mask"].to(device)
    s_ids = s_batch["input_ids"].to(device)
    s_mask = s_batch["attention_mask"].to(device)

    # Calculate span of the prompt for student and teacher.
    # We want to match from the end, backward by len(s_ids) tokens, because student is just prompt.
    # But wait, left-padding vs right-padding. pad_and_stack does right padding.
    # So valid tokens are at the beginning.
    # The teacher length is `L_t`, student length is `L_s`. The prompt is at the end of the teacher sequence before padding!
    # Wait, t_ids is right padded. The valid sequence length for sample i is sum(t_mask[i]).
    # For student, it is sum(s_mask[i]).
    # We want to match the last sum(s_mask[i]) tokens of teacher with student.

    # 2. Forward passes
    # Teacher forward (no grad)
    with torch.no_grad():
        with model.disable_adapter():
            t_out = model(
                input_ids=t_ids,
                attention_mask=t_mask,
                use_cache=False,
                output_hidden_states=match_hidden,
            )
            t_logits = t_out.logits
            t_hiddens = t_out.hidden_states if match_hidden else None

    # Student forward (with grad)
    s_out = model(
        input_ids=s_ids,
        attention_mask=s_mask,
        use_cache=False,
        output_hidden_states=match_hidden,
    )
    s_logits = s_out.logits
    s_hiddens = s_out.hidden_states if match_hidden else None

    # 3. Compute losses over the probe span (prompt)
    B = len(samples)
    kl_mean_sum = 0.0
    mse_mean_sum = 0.0

    fact_retention_sum = 0.0
    valid_facts = 0
    sample_kl_weights = [1.0] * B

    if any(s.get("probe_kind") == "fact" and s.get("facts") for s in samples):
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)
        except Exception:
            pass

        for i, s in enumerate(samples):
            if s.get("probe_kind") == "fact" and s.get("facts"):
                facts = s["facts"]
                s_id_tensor = student_tokenized[i]["input_ids"].unsqueeze(0).to(device)
                attn_tensor = torch.ones_like(s_id_tensor)

                outs = model.generate(
                    input_ids=s_id_tensor,
                    attention_mask=attn_tensor,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
                prompt_len = s_id_tensor.size(-1)
                gen = outs[0, prompt_len:]
                gen_text = tokenizer.decode(gen, skip_special_tokens=True).strip()

                from .verifiers.fact_verifier import verify_facts

                score = verify_facts(gen_text, facts)
                fact_retention_sum += score
                valid_facts += 1

                # Upweight lost facts, downweight retained ones
                sample_kl_weights[i] = 1.5 - score

        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_training(model)
        except Exception:
            pass

    for i in range(B):
        L_t = int(t_mask[i].sum().item())
        L_s = int(s_mask[i].sum().item())

        # probe span is the last L_s tokens of the teacher (before padding)
        # But wait, apply_chat_template might add prefix/suffix differently?
        # Assuming the prompt part is exactly L_s tokens at the end.
        start_t = L_t - L_s
        if start_t < 0:
            start_t = 0

        from .kl import chunked_kl_div
        t_chunk = t_logits[i, start_t : L_t - 1]
        s_chunk = s_logits[i, 0 : L_s - 1]
        kl_per_token = chunked_kl_div(input_logits=s_chunk, target_logits=t_chunk)
        kl = kl_per_token.mean()

        kl_mean_sum += kl * sample_kl_weights[i]

        if match_hidden:
            sample_mse = 0.0
            for layer_idx in hidden_layers:
                # hiddens are tuple of length num_layers + 1
                t_h = t_hiddens[layer_idx][i, start_t:L_t]  # type: ignore
                s_h = s_hiddens[layer_idx][i, 0:L_s]  # type: ignore
                sample_mse += F.mse_loss(s_h, t_h)
            mse_mean_sum += sample_mse / len(hidden_layers)

    kl_loss = kl_weight * (kl_mean_sum / B)
    loss = kl_loss
    components = {"ccd_kl": float((kl_mean_sum / B).detach().cpu())}  # type: ignore

    if match_hidden:
        hidden_loss = hidden_weight * (mse_mean_sum / B)
        loss += hidden_loss
        components["ccd_hidden"] = float((mse_mean_sum / B).detach().cpu())  # type: ignore

    if has_response:
        # 4. Optional Response SFT term
        sft_samples = [s for s in samples if s.get("response")]
        if sft_samples:
            from .sft import sft_loss

            sft_res = sft_loss(model, tokenizer, sft_samples, span_prefix=None)
            sft_l = sft_res["loss"]
            loss += sft_weight * sft_l
            components["ccd_sft"] = float(sft_l.detach().cpu())

    if valid_facts > 0:
        components["fact_retention_mean"] = float(fact_retention_sum / valid_facts)

    return {
        "loss": loss,
        "components": components,
    }
