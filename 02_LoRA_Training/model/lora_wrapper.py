"""LoRA (Low-Rank Adaptation) wrapper for model fine-tuning.

Uses PEFT library to apply LoRA adapters to target modules.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Optional


def apply_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
) -> nn.Module:
    """Apply LoRA to a model.

    Args:
        model: PyTorch model to apply LoRA to
        r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling parameter (default: 16)
        target_modules: List of module names to apply LoRA to
            For AASIST/ConformerTCM: ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "LL"]
        lora_dropout: Dropout probability for LoRA layers (default: 0.0)
        bias: Bias type ("none", "all", "lora_only")

    Returns:
        Model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "LL"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def apply_multi_lora(
    model: nn.Module,
    num_domains: int = 8,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
) -> nn.Module:
    """Apply multiple LoRA adapters for domain-specific training.

    Creates num_domains named LoRA adapters (domain_0 .. domain_{n-1}).
    The first adapter (domain_0) is set as active by default.

    Args:
        model: PyTorch model to apply LoRA to
        num_domains: Number of domain-specific adapters (default: 8)
        r: LoRA rank
        lora_alpha: LoRA alpha scaling parameter
        target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias type

    Returns:
        PeftModel with multiple named adapters
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "LL"]

    # Create the first adapter
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    model = get_peft_model(model, lora_config, adapter_name="domain_0")

    # Add remaining adapters
    for i in range(1, num_domains):
        adapter_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        model.add_adapter(f"domain_{i}", adapter_config)

    # Set first adapter as active
    model.set_adapter("domain_0")
    model.print_trainable_parameters()

    print(f"Multi-LoRA: {num_domains} adapters created (domain_0 .. domain_{num_domains-1})")
    return model


def switch_lora_adapter(model: nn.Module, domain_id: int) -> None:
    """Switch active LoRA adapter by domain ID.

    Args:
        model: PeftModel with multiple adapters
        domain_id: Domain index to activate
    """
    adapter_name = f"domain_{domain_id}"
    if isinstance(model, PeftModel):
        model.set_adapter(adapter_name)
    else:
        raise ValueError("Model is not a PeftModel. Cannot switch adapters.")


def load_lora_weights(model: nn.Module, adapter_path: str) -> nn.Module:
    """Load LoRA adapter weights from checkpoint.

    Args:
        model: Base model
        adapter_path: Path to LoRA adapter checkpoint

    Returns:
        Model with loaded LoRA weights
    """
    model = PeftModel.from_pretrained(model, adapter_path)
    return model


def merge_and_unload_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base model and unload adapter.

    Args:
        model: Model with LoRA applied

    Returns:
        Model with merged weights (no adapter)
    """
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters.

    Args:
        model: Model with LoRA applied
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters.

    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module) -> int:
    """Get number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module) -> None:
    """Print model parameter information.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = get_trainable_params(model)
    frozen_params = total_params - trainable_params

    print(f"=" * 50)
    print(f"Model Parameter Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    print(f"=" * 50)
