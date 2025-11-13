import os
import sys
import torch
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hybrid_policy_with_adapter import HybridAgentPolicyWithAdapter  # noqa: E402


def _fake_lr_schedule(_progress: float) -> float:
    return 3e-4


def _make_policy():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(261,), dtype=np.float32)
    action_space = spaces.Discrete(6)
    return HybridAgentPolicyWithAdapter(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=_fake_lr_schedule,
        hybrid_agent=None,
        base_obs_dim=228,
    )


def test_adapter_enforces_feature_shape():
    policy = _make_policy()
    obs = torch.randn(3, 261)

    features = policy.extract_features(obs)
    assert features.shape == (3, 228)

    vf_features = policy.extract_features(obs, policy.vf_features_extractor)
    assert vf_features.shape == (3, 228)


def test_predict_values_handles_full_observation_dim():
    policy = _make_policy()
    obs = torch.randn(5, 261)
    values = policy.predict_values(obs)
    assert values.shape == (5, 1)
