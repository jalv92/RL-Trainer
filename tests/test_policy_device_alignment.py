import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import sys

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_phase3_llm import setup_hybrid_model, PHASE3_CONFIG  # noqa: E402
from hybrid_agent import HybridTradingAgent  # noqa: E402


class MinimalPhase3Env(gym.Env):
    """Simple deterministic env that mimics Phase 3 observation/action spaces."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(261,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(261, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(261, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return np.ones(self.action_space.n, dtype=bool)


def _make_env():
    return ActionMasker(MinimalPhase3Env(), lambda env: env.action_masks())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this device test")
def test_transfer_wrapped_policy_matches_model_device():
    """Ensure setup_hybrid_model keeps the wrapped policy on the RL model's device."""
    env = DummyVecEnv([_make_env])

    base_model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])),
    )

    agent_config = {
        "fusion": {"use_neural_fusion": False, "always_on_thinking": False},
        "risk": {},
        "llm_workers": 1,
    }
    hybrid_agent = HybridTradingAgent(rl_model=None, llm_model=None, config=agent_config)

    config = PHASE3_CONFIG.copy()
    config["policy_kwargs"] = config["policy_kwargs"].copy()
    config["device"] = "cuda"

    model = setup_hybrid_model(env, hybrid_agent, config=config, base_model=base_model)
    hybrid_agent.set_rl_model(model)

    policy_devices = {param.device.type for param in model.policy.parameters()}
    assert policy_devices == {model.device.type}
    assert model.policy._get_policy_device().type == model.device.type

    env.close()
