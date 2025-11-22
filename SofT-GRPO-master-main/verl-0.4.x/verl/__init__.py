# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import os
import sys
import types
from functools import partial
from pathlib import Path

import pkg_resources
import torch.nn as nn
from packaging.version import parse as parse_version
from pkg_resources import DistributionNotFound

from .protocol import DataProto
from .utils.device import is_npu_available
from .utils.logging_utils import set_basic_config

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]

if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    import importlib

    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()

if is_npu_available:
    package_name = 'transformers'
    required_version_spec = '4.51.0'
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        installed = parse_version(installed_version)
        required = parse_version(required_version_spec)

        if not installed >= required:
            raise ValueError(f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is {installed}.")
    except DistributionNotFound:
        raise ImportError(
            f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}")


def _ensure_transformers_modeling_layers():
    """Backfill transformers.modeling_layers for HF builds that lack it."""
    try:
        import importlib
        module_name = "transformers.modeling_layers"
        if importlib.util.find_spec(module_name):
            return
        import transformers  # noqa: F401  # ensure base package exists
    except ImportError:
        return

    class GradientCheckpointingLayer(nn.Module):
        """Minimal implementation that peft expects."""

        gradient_checkpointing = False

        def __call__(self, *args, **kwargs):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
            return super().__call__(*args, **kwargs)

    module = types.ModuleType(module_name)
    module.GradientCheckpointingLayer = GradientCheckpointingLayer
    module.nn = nn
    module.partial = partial
    sys.modules[module_name] = module


_ensure_transformers_modeling_layers()


def _install_flash_attn_stub(error):
    """Register lightweight flash_attn python stubs so imports succeed."""
    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "flash_attn backend is unavailable or failed to load "
            f"(original error: {error}). Install a compatible flash-attn wheel "
            "or keep attn_implementation='eager'."
        )

    import importlib.machinery as _machinery

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__spec__ = _machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn_module.__version__ = "unavailable"
    flash_attn_module.flash_attn_with_kvcache = _unavailable
    flash_attn_module.flash_attn_varlen_func = _unavailable
    flash_attn_module.flash_attn_func = _unavailable
    flash_attn_module.__dict__["flash_attn_unpadded_func"] = _unavailable

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = _machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    for name in ("index_first_axis", "pad_input", "unpad_input", "rearrange"):
        setattr(bert_padding, name, _unavailable)

    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.__spec__ = _machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    for name in ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"):
        setattr(interface, name, _unavailable)

    sys.modules["flash_attn"] = flash_attn_module
    sys.modules["flash_attn.bert_padding"] = bert_padding
    sys.modules["flash_attn.flash_attn_interface"] = interface


def _ensure_flash_attn_modules():
    """Ensure importing flash_attn never crashes even if the wheel is missing/broken."""
    try:
        import importlib
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError("flash_attn spec not found")
        import flash_attn  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment specific
        logging.getLogger(__name__).warning(
            "flash_attn backend unavailable; falling back to python stubs. "
            "Error: %s",
            exc,
        )
        _install_flash_attn_stub(exc)


_ensure_flash_attn_modules()


def _ensure_triton_cache_module():
    """Provide missing Triton cache helpers expected by sglang."""
    try:
        import importlib
        if importlib.util.find_spec("triton.runtime.cache") is None:
            return
        from pathlib import Path
        import os
        import triton.runtime.cache as cache_mod  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env packages
        logging.getLogger(__name__).warning("Unable to prepare Triton cache helpers: %s", exc)
        return

    def _resolve_dir(env_var: str, fallback: str) -> str:
        path = os.getenv(env_var, "").strip()
        if not path:
            path = str(Path.home() / ".triton" / fallback)
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return str(path_obj)

    if not hasattr(cache_mod, "default_cache_dir"):
        cache_mod.default_cache_dir = lambda: _resolve_dir("TRITON_CACHE_DIR", "cache")
    if not hasattr(cache_mod, "default_dump_dir"):
        cache_mod.default_dump_dir = lambda: _resolve_dir("TRITON_DUMP_DIR", "dump")
    if not hasattr(cache_mod, "default_override_dir"):
        cache_mod.default_override_dir = lambda: _resolve_dir("TRITON_OVERRIDE_DIR", "override")


_ensure_triton_cache_module()
