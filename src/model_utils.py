"""
Model Management Utilities for RL Training System

This module provides utilities for detecting, loading, and managing
trained models in the models directory.
"""

import os
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize


def detect_models_in_folder(model_dir: str = 'models', phase: Optional[str] = None) -> List[Dict]:
    """
    Detect all trained models in the models directory.

    Args:
        model_dir: Path to models directory (default: 'models')
        phase: Filter by phase ('phase1' or 'phase2'). If None, return all models.

    Returns:
        List of dicts with model information:
        [
            {
                'path': 'models/phase1_foundational_final.zip',
                'name': 'phase1_foundational_final',
                'type': 'phase1',
                'size_mb': 6.97,
                'modified': datetime object,
                'modified_str': '2025-10-28 14:30:15',
                'vecnorm_path': 'models/phase1_vecnorm.pkl' or None
            },
            ...
        ]
    """
    models = []

    # Search for all .zip files in models directory
    search_pattern = os.path.join(model_dir, '**', '*.zip')
    model_files = glob.glob(search_pattern, recursive=True)

    for model_path in model_files:
        # Get basic file info
        path_obj = Path(model_path)
        name = path_obj.stem  # filename without extension
        size_bytes = path_obj.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        modified_timestamp = path_obj.stat().st_mtime
        modified_dt = datetime.fromtimestamp(modified_timestamp)

        # Determine model type from path or name
        model_type = None
        if 'phase1' in model_path.lower():
            model_type = 'phase1'
        elif 'phase2' in model_path.lower():
            model_type = 'phase2'
        else:
            # Try to infer from directory structure
            if 'phase1' in str(path_obj.parent):
                model_type = 'phase1'
            elif 'phase2' in str(path_obj.parent):
                model_type = 'phase2'

        # Skip if we're filtering by phase and this doesn't match
        if phase and model_type != phase:
            continue

        # Look for corresponding VecNormalize file
        vecnorm_path = None

        # Check for vecnorm file with same base name
        vecnorm_candidate1 = str(path_obj.parent / f"{name}_vecnorm.pkl")
        vecnorm_candidate2 = str(path_obj.parent / f"{model_type}_vecnorm.pkl")
        vecnorm_candidate3 = os.path.join(model_dir, f"{model_type}_vecnorm.pkl")

        for candidate in [vecnorm_candidate1, vecnorm_candidate2, vecnorm_candidate3]:
            if os.path.exists(candidate):
                vecnorm_path = candidate
                break

        models.append({
            'path': model_path,
            'name': name,
            'type': model_type or 'unknown',
            'size_mb': size_mb,
            'modified': modified_dt,
            'modified_str': modified_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'vecnorm_path': vecnorm_path
        })

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)

    return models


def load_model_auto(model_path: str, device: str = 'auto') -> Tuple[object, str]:
    """
    Load a model automatically detecting its type (Phase 1 PPO or Phase 2 MaskablePPO).

    Args:
        model_path: Path to .zip model file
        device: Device to load model on ('cuda', 'cpu', or 'auto')

    Returns:
        Tuple of (model, model_type) where model_type is 'phase1' or 'phase2'

    Raises:
        ValueError: If model cannot be loaded
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect device if needed
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Try loading as Phase 2 first (MaskablePPO)
    try:
        print(f"Attempting to load as Phase 2 (MaskablePPO) model...")
        model = MaskablePPO.load(model_path, device=device)
        print(f"✓ Successfully loaded as Phase 2 model")
        return model, 'phase2'
    except Exception as e:
        print(f"  Not a Phase 2 model: {str(e)[:50]}...")

    # Try loading as Phase 1 (PPO)
    try:
        print(f"Attempting to load as Phase 1 (PPO) model...")
        model = PPO.load(model_path, device=device)
        print(f"✓ Successfully loaded as Phase 1 model")
        return model, 'phase1'
    except Exception as e:
        print(f"  Not a Phase 1 model: {str(e)[:50]}...")

    raise ValueError(f"Could not load model from {model_path}. Model may be corrupted or incompatible.")


def load_vecnormalize(vecnorm_path: str, env) -> VecNormalize:
    """
    Load VecNormalize statistics from a .pkl file.

    Args:
        vecnorm_path: Path to VecNormalize .pkl file
        env: Environment to wrap with VecNormalize

    Returns:
        VecNormalize wrapped environment

    Raises:
        FileNotFoundError: If vecnorm file doesn't exist
    """
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    print(f"Loading VecNormalize stats from {vecnorm_path}...")
    env = VecNormalize.load(vecnorm_path, env)
    print("✓ VecNormalize stats loaded successfully")

    return env


def display_model_selection(models: List[Dict], phase_filter: Optional[str] = None) -> int:
    """
    Display a list of models and get user selection.

    Args:
        models: List of model info dicts from detect_models_in_folder()
        phase_filter: Optional phase filter to show in title

    Returns:
        Index of selected model in the models list

    Raises:
        ValueError: If user input is invalid
    """
    if not models:
        print("\n⚠ No models found in the models directory.")
        if phase_filter:
            print(f"   (Filtered for: {phase_filter})")
        return -1

    # Display header
    print("\n" + "="*70)
    if phase_filter:
        print(f"Available {phase_filter.upper()} Models:")
    else:
        print("Available Models:")
    print("="*70)

    # Display each model
    for i, model_info in enumerate(models, 1):
        type_label = model_info['type'].upper() if model_info['type'] != 'unknown' else 'UNKNOWN'
        vecnorm_status = "✓" if model_info['vecnorm_path'] else "✗"

        print(f"\n{i}. {model_info['name']}")
        print(f"   Type: {type_label} | Size: {model_info['size_mb']:.2f} MB | VecNorm: {vecnorm_status}")
        print(f"   Modified: {model_info['modified_str']}")
        print(f"   Path: {model_info['path']}")

    print("\n" + "="*70)
    print(f"0. Cancel")
    print("="*70)

    # Get user input
    while True:
        try:
            choice = input(f"\nSelect model to continue training (0-{len(models)}): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                print("Operation cancelled.")
                return -1
            elif 1 <= choice_num <= len(models):
                return choice_num - 1  # Convert to 0-indexed
            else:
                print(f"⚠ Please enter a number between 0 and {len(models)}")
        except ValueError:
            print("⚠ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return -1


def get_model_save_name(default_name: str) -> str:
    """
    Prompt user for a model save name.

    Args:
        default_name: Suggested default name

    Returns:
        User-chosen model name (without .zip extension)
    """
    print("\n" + "="*70)
    print("Model Training Complete!")
    print("="*70)
    print(f"\nDefault save name: {default_name}")
    print("Enter a custom name, or press Enter to use the default.")
    print("Note: Do not include .zip extension (it will be added automatically)")
    print("="*70)

    while True:
        try:
            user_input = input("\nModel save name: ").strip()

            # Use default if empty
            if not user_input:
                return default_name

            # Remove .zip extension if user included it
            if user_input.endswith('.zip'):
                user_input = user_input[:-4]

            # Validate name (no special characters that would break filesystem)
            if any(char in user_input for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
                print("⚠ Model name contains invalid characters. Please use only letters, numbers, hyphens, and underscores.")
                continue

            return user_input

        except KeyboardInterrupt:
            print(f"\n\nUsing default name: {default_name}")
            return default_name


def validate_model_environment_compatibility(model_type: str, environment_type: str) -> bool:
    """
    Validate that a model type is compatible with an environment type.

    Args:
        model_type: 'phase1' or 'phase2'
        environment_type: 'phase1' or 'phase2'

    Returns:
        True if compatible, False otherwise
    """
    if model_type == environment_type:
        return True

    print(f"\n⚠ ERROR: Model type ({model_type}) does not match environment type ({environment_type})")
    print(f"   Phase 1 models can only continue training in Phase 1 environments.")
    print(f"   Phase 2 models can only continue training in Phase 2 environments.")
    print(f"   For transfer learning from Phase 1 to Phase 2, use 'Training Pod' instead.")

    return False


if __name__ == "__main__":
    # Test the detection function
    print("Testing model detection...")
    models = detect_models_in_folder()

    if models:
        print(f"\nFound {len(models)} model(s):")
        for model in models:
            print(f"  - {model['name']} ({model['type']}) - {model['modified_str']}")
    else:
        print("No models found.")

    # Test display
    if models:
        display_model_selection(models)
