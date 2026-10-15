"""CPU serialization of projection weights without a second GPU base.

Bitsandbytes Params4bit metadata travels with packed weights. Dense projections
in mixed-precision quantized models retain their existing floating-point dtype.
"""

import torch


def quantized_layers(state):
    return {name: base for name, base in projection_layers(state).items()
            if getattr(base.weight, "quant_state", None) is not None}


def projection_layers(state):
    result = {}
    for name, module in state.model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            base = getattr(module, "base_layer", None)
            weight = getattr(base, "weight", None)
            if weight is not None:
                result[name] = base
    return result


def pack(base):
    weight = base.weight
    if getattr(weight, "quant_state", None) is None:
        if not weight.is_floating_point():
            raise ValueError("unsupported projection weight representation")
        return {"weight": weight.data.detach().cpu().contiguous().clone()}
    if not hasattr(weight.quant_state, "as_dict"):
        raise ValueError("quantized checkpoint requires bitsandbytes QuantState")
    result = {"weight": weight.data.detach().cpu().contiguous().clone()}
    result.update({k: v.detach().cpu().contiguous().clone()
                   for k, v in weight.quant_state.as_dict(packed=True).items()})
    return result


def validate(base, tensors):
    if tensors["weight"].shape != base.weight.shape or tensors["weight"].dtype != base.weight.dtype:
        raise ValueError("quantized weight storage shape/dtype mismatch")
    if getattr(base.weight, "quant_state", None) is None:
        if set(tensors) != {"weight"} or not tensors["weight"].is_floating_point():
            raise ValueError("dense projection cannot receive quantized weights")
        return None
    from bitsandbytes.functional import QuantState

    quant = QuantState.from_dict({k: v for k, v in tensors.items() if k != "weight"}, device="cpu")
    old = base.weight.quant_state
    for field in ("shape", "dtype", "blocksize", "quant_type", "nested"):
        if getattr(quant, field) != getattr(old, field):
            raise ValueError(f"quantization {field} differs from live projection")
    return quant


def apply(base, tensors):
    quant = validate(base, tensors)
    # Small scale/codebook allocation precedes the in-place packed-weight copy.
    # Retain Parameter identity for Unsloth's bound forward references.
    if quant is not None:
        quant.to(base.weight.device)
    with torch.no_grad():
        base.weight.data.copy_(tensors["weight"])
    if quant is not None:
        base.weight.quant_state = quant
        base.quant_state = quant
