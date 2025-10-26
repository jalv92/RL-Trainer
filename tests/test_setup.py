#!/usr/bin/env python3
"""
AI Trainer Setup Verification Script

This script verifies that all dependencies are installed correctly
and that the project structure is properly configured.

Usage:
    python test_setup.py
"""

import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def check_python_version():
    """Check Python version"""
    print("\n[1/6] Checking Python version...")
    required_version = (3, 11, 9)
    current_version = sys.version_info

    version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
    required_str = f"{required_version[0]}.{required_version[1]}.{required_version[2]}"

    if current_version >= required_version:
        print(f"  ✓ Python {version_str} (>= {required_str} required)")
        return True
    else:
        print(f"  ✗ Python {version_str} (>= {required_str} required)")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n[2/6] Checking dependencies...")

    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'gymnasium': 'Gymnasium',
        'matplotlib': 'Matplotlib',
        'stable_baselines3': 'Stable Baselines3',
        'sb3_contrib': 'SB3 Contrib (MaskablePPO)',
        'tensorflow': 'TensorFlow',
        'colorama': 'Colorama',
        'tqdm': 'tqdm',
    }

    all_installed = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_installed = False

    return all_installed


def check_project_structure():
    """Check if project structure is correct"""
    print("\n[3/6] Checking project structure...")

    project_root = Path(__file__).parent
    required_dirs = [
        ('src', True),
        ('data', True),
        ('models', True),
        ('logs', True),
        ('results', True),
        ('tensorboard_logs', True),
        ('tests', True),
        ('docs', True),
    ]

    required_files = [
        ('main.py', True),
        ('requirements.txt', True),
        ('src/environment_phase1.py', True),
        ('src/environment_phase2.py', True),
        ('src/train_phase1.py', True),
        ('src/train_phase2.py', True),
        ('src/evaluate_phase2.py', True),
        ('src/feature_engineering.py', True),
        ('src/apex_compliance_checker.py', True),
    ]

    all_exist = True

    for dir_name, required in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        elif required:
            print(f"  ✗ {dir_name}/ - MISSING")
            all_exist = False
        else:
            print(f"  - {dir_name}/ - Optional (not found)")

    for file_name, required in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        elif required:
            print(f"  ✗ {file_name} - MISSING")
            all_exist = False
        else:
            print(f"  - {file_name} - Optional (not found)")

    return all_exist


def check_imports():
    """Check if src modules can be imported"""
    print("\n[4/6] Checking module imports...")

    # Add src to path
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'
    sys.path.insert(0, str(src_dir))

    modules = [
        'environment_phase1',
        'environment_phase2',
        'feature_engineering',
        'technical_indicators',
        'kl_callback',
        'apex_compliance_checker',
    ]

    all_imported = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - IMPORT ERROR: {e}")
            all_imported = False
        except Exception as e:
            print(f"  ⚠ {module} - WARNING: {e}")

    return all_imported


def check_gpu_availability():
    """Check if GPU is available"""
    print("\n[5/6] Checking GPU availability...")

    # Check TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ TensorFlow GPU available ({len(gpus)} GPU(s))")
            for i, gpu in enumerate(gpus):
                print(f"    - GPU {i}: {gpu.name}")
        else:
            print(f"  - TensorFlow GPU not available (CPU mode)")
    except Exception as e:
        print(f"  ⚠ TensorFlow GPU check failed: {e}")

    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ PyTorch CUDA available")
            print(f"    - Device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  - PyTorch CUDA not available (CPU mode)")
    except ImportError:
        print(f"  - PyTorch not installed (optional)")
    except Exception as e:
        print(f"  ⚠ PyTorch CUDA check failed: {e}")

    return True  # GPU is optional


def check_paths():
    """Check if path configurations are correct"""
    print("\n[6/6] Checking path configurations...")

    project_root = Path(__file__).parent

    # Check data paths
    data_dir = project_root / 'data'
    if data_dir.exists():
        print(f"  ✓ Data directory: {data_dir}")
    else:
        print(f"  ✗ Data directory: {data_dir} - MISSING")
        return False

    # Check models paths
    models_dir = project_root / 'models'
    if models_dir.exists():
        print(f"  ✓ Models directory: {models_dir}")
    else:
        print(f"  ✗ Models directory: {models_dir} - MISSING")
        return False

    return True


def main():
    """Main verification function"""
    print_header("AI TRAINER SETUP VERIFICATION")

    print("\nThis script will verify your AI Trainer installation.")

    results = []

    # Run all checks
    results.append(("Python version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Project structure", check_project_structure()))
    results.append(("Module imports", check_imports()))
    results.append(("GPU availability", check_gpu_availability()))
    results.append(("Path configurations", check_paths()))

    # Print summary
    print_header("VERIFICATION SUMMARY")

    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ All checks passed! Your AI Trainer is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python main.py' to start the interactive menu")
        print("  2. Or process data: 'python src/update_training_data.py --market ES'")
        print("  3. Then train: 'python src/train_phase1.py'")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check project structure matches README.md")
        print("  - Ensure all files were copied correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
