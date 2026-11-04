"""
Model Configuration

Resolves the model weights/cache directory and points every model library at it.
Models are loaded directly from HuggingFace and auto-downloaded on first use (no
custom downloader layer).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _default_cache_dir() -> Path:
    """Default model weights/cache directory.

    Always the project's ``models_weights/`` folder unless ``MODEL_WEIGHTS_DIR``
    is set.  Keeping weights inside the project tree makes builds, containers,
    and runpod volumes predictable and self-contained.
    """
    env_dir = os.environ.get("MODEL_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir)
    # langswap/model_config.py -> project root is two parents up.
    return Path(__file__).resolve().parent.parent / "models_weights"


# Resolve the weights/cache directory and point every model library at it so
# HuggingFace / torch.hub downloads all land in the project's models_weights/.
MODEL_WEIGHTS_DIR = str(_default_cache_dir())
Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
for _var in (
    "MODEL_WEIGHTS_DIR",
    "TRANSFORMERS_CACHE",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TORCH_HOME",
):
    os.environ[_var] = MODEL_WEIGHTS_DIR

# vLLM and torch.compile/Inductor write large JIT/compile artifacts (the
# ~58 s "torch.compile took ..." step that both the Qwen3-ASR and OmniVoice
# engines pay).  By default these land in ~/.cache, which is ephemeral on
# serverless containers, so the compile is re-paid on every cold start.
# Redirect them under the (persisted) weights dir so a warmed compile cache
# survives container restarts.  setdefault keeps any explicit user override.
_vllm_cache = os.path.join(MODEL_WEIGHTS_DIR, "vllm_cache")
_inductor_cache = os.path.join(MODEL_WEIGHTS_DIR, "torchinductor_cache")
for _d in (_vllm_cache, _inductor_cache):
    Path(_d).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("VLLM_CACHE_ROOT", _vllm_cache)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _inductor_cache)


__all__ = ["MODEL_WEIGHTS_DIR"]
