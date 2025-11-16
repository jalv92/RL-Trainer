"""
Helper utilities for retrieving action masks from wrapped environments.

Gymnasium warns against accessing wrapper attributes directly (e.g.,
``env.action_masks``). These helpers centralize the logic for safely reaching
the underlying environment implementation without triggering deprecation
warnings.
"""

from __future__ import annotations

from typing import Any, Iterable
import numpy as np


def get_action_masks(env: Any):
    """
    Return the current action masks for an environment, unwrapping wrappers
    safely when necessary.
    """
    # VecEnv support: env_method returns a list (one entry per sub-env)
    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        masks = env_method("action_masks")
        if isinstance(masks, (list, tuple)):
            return masks[0]
        return masks

    # Prefer Gymnasium's helper if available to avoid deprecated attribute access
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        attr = get_wrapper_attr("action_masks")
        if callable(attr):
            return attr()
        if attr is not None:
            return attr

    # Fall back to the base environment
    base_env = getattr(env, "unwrapped", env)
    action_masks = getattr(base_env, "action_masks", None)
    if callable(action_masks):
        return action_masks()
    if action_masks is not None:
        return action_masks

    raise AttributeError("Environment does not expose action_masks()")


def ensure_vecenv_action_masks(vec_env: Any) -> Any:
    """
    Attach an action_masks() method to VecEnv wrappers (VecNormalize, etc.)
    so upstream code doesn't trigger Gym's deprecated attribute traversal.
    """
    if hasattr(vec_env, "action_masks"):
        return vec_env

    def _vec_action_masks():
        masks = vec_env.env_method("action_masks")
        if isinstance(masks, (list, tuple)):
            # VecEnv env_method returns list per sub-environment
            try:
                return np.stack(masks)
            except Exception:
                return masks
        return masks

    setattr(vec_env, "action_masks", _vec_action_masks)
    return vec_env
