"""
ADD Model Wrapper with domain-specific LoRA switching.
- Pretrained ADD 모델(AASIST/ConformerTCM) 로드
- 도메인별 LoRA 체크포인트를 미리 캐싱
- domain_id에 따라 LoRA 스위칭 후 추론
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import yaml
import importlib

LORA_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../02_LoRA_Training"))


class ADDModelWrapper:
    def __init__(self, model_module, pretrained_path, config_path, lora_config, device="cuda"):
        """
        Args:
            model_module: "aasist" or "conformertcm"
            pretrained_path: path to pretrained checkpoint
            config_path: path to model config yaml
            lora_config: dict with r, alpha, target_modules, checkpoints (D1~D7)
            device: cuda device
        """
        self.device = device
        self.model_module_name = model_module
        self.lora_r = lora_config["r"]
        self.lora_alpha = lora_config["alpha"]
        self.target_modules = lora_config.get("target_modules",
            ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "LL"])

        # Add LoRA training path for model imports
        if LORA_BASE not in sys.path:
            sys.path.insert(0, LORA_BASE)

        # Load config
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_cfg = config["model"]

        # Create args namespace for model constructor
        class Args:
            pass
        args = Args()
        for key, value in model_cfg.items():
            if key != "name":
                setattr(args, key, value)

        # Build base model (load from 02_LoRA_Training/model/)
        import types
        lora_model_dir = os.path.join(LORA_BASE, "model")

        # Swap model package to point to 02_LoRA_Training/model/
        saved = {}
        for k in list(sys.modules.keys()):
            if k == "model" or k.startswith("model."):
                saved[k] = sys.modules.pop(k)

        fake_pkg = types.ModuleType("model")
        fake_pkg.__path__ = [lora_model_dir]
        fake_pkg.__package__ = "model"
        sys.modules["model"] = fake_pkg

        sys.path.insert(0, LORA_BASE)
        model_mod = importlib.import_module(f"model.{model_module}")
        importlib.import_module("model.lora_wrapper")
        sys.path.remove(LORA_BASE)

        # Also keep lora_wrapper accessible
        lora_mods = {}
        for k in list(sys.modules.keys()):
            if k == "model" or k.startswith("model."):
                lora_mods[k] = sys.modules.pop(k)
        self._lora_mods = lora_mods

        # Restore local model modules
        sys.modules.update(saved)

        # Save for rebuilding
        self._model_mod = model_mod
        self._args = args
        self._device = device

        self.base_model = model_mod.Model(args, device).to(device)

        # Load pretrained weights
        ckpt = torch.load(pretrained_path, map_location="cpu")
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        self.base_model.load_state_dict(ckpt)
        print(f"[ADD] Loaded {model_module} from {pretrained_path}")

        # Cache base model state for resetting
        self.base_state = copy.deepcopy(self.base_model.state_dict())

        # Pre-load all LoRA checkpoints into memory
        self.lora_states = {}
        for domain_key, ckpt_path in lora_config["checkpoints"].items():
            abs_path = os.path.join(os.path.dirname(__file__), "..", ckpt_path) \
                if not os.path.isabs(ckpt_path) else ckpt_path
            abs_path = os.path.abspath(abs_path)
            if os.path.exists(abs_path):
                self.lora_states[domain_key] = torch.load(abs_path, map_location="cpu")
                print(f"[ADD] Cached LoRA {domain_key}: {abs_path}")
            else:
                print(f"[WARN] LoRA {domain_key} not found: {abs_path}")

        # Current active domain
        self.current_domain = None
        self.model = None  # will be set by apply_domain()

        # Apply base (no LoRA) initially
        self._apply_base()

    def _rebuild_base(self):
        """Rebuild a fresh base model from scratch."""
        # Free previous model memory
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

        model_mod = self._model_mod
        model = model_mod.Model(self._args, self._device).to(self._device)
        model.load_state_dict(self.base_state)
        return model

    def _apply_base(self):
        """Reset to base model (no LoRA) for D0/clean."""
        self.model = self._rebuild_base()
        self.model.eval()
        self.current_domain = 0

    def _apply_lora(self, domain_id):
        """Apply LoRA for given domain. Rebuilds model from scratch each time."""
        apply_lora = self._lora_mods["model.lora_wrapper"].apply_lora

        domain_key = f"D{domain_id}"
        if domain_key not in self.lora_states:
            print(f"[WARN] No LoRA for {domain_key}, using base model")
            self._apply_base()
            return

        # Build fresh base model
        base = self._rebuild_base()

        # Apply LoRA structure
        model = apply_lora(
            base,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=0.0,
        )

        # Load LoRA weights
        model.load_state_dict(self.lora_states[domain_key])
        model.eval()
        self.model = model
        self.current_domain = domain_id

    def apply_domain(self, domain_id):
        """Switch to the given domain. D0 = base, D1~D7 = LoRA."""
        if domain_id == self.current_domain:
            return  # already applied

        if domain_id == 0:
            self._apply_base()
        else:
            self._apply_lora(domain_id)

    @torch.no_grad()
    def predict(self, waveforms):
        """
        Args:
            waveforms: [batch, length] tensor
        Returns:
            scores: [batch, 2] tensor (spoof, bonafide logits)
        """
        waveforms = waveforms.to(self.device)
        out = self.model(waveforms)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.cpu()
