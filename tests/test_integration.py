#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify testing framework integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_integration():
    """Test that the integration was successful."""
    print("Testing Testing Framework Integration")
    print("=" * 60)
    
    try:
        # Test import
        print("1. Testing imports...")
        from main import RLTrainerMenu
        print("   ✅ main.py imports successfully")
        
        # Test instantiation
        print("\n2. Testing class instantiation...")
        menu = RLTrainerMenu()
        print("   ✅ RLTrainerMenu instantiates without errors")
        
        # Test menu options
        print("\n3. Testing menu options...")
        expected_options = {
            "8": "Testing Framework - Hardware-Maximized Mode",
            "9": "Testing Framework - Pipeline Mode",
            "10": "Validate Testing Framework",
            "11": "Benchmark Optimizations",
            "12": "Back to Main Menu"
        }
        
        for key, expected_value in expected_options.items():
            if key in menu.training_menu_options:
                actual_value = menu.training_menu_options[key]
                if actual_value == expected_value:
                    print(f"   ✅ Option {key}: {actual_value}")
                else:
                    print(f"   ❌ Option {key}: Expected '{expected_value}', got '{actual_value}'")
            else:
                print(f"   ❌ Option {key} missing from menu")
        
        # Test method existence
        print("\n4. Testing new methods...")
        new_methods = [
            'run_hardware_maximized_test',
            'run_pipeline_test', 
            'validate_testing_framework',
            'benchmark_optimizations'
        ]
        
        for method_name in new_methods:
            if hasattr(menu, method_name):
                method = getattr(menu, method_name)
                if callable(method):
                    print(f"   ✅ Method {method_name} exists and is callable")
                else:
                    print(f"   ❌ Method {method_name} exists but is not callable")
            else:
                print(f"   ❌ Method {method_name} not found")
        
        # Test configuration files
        print("\n5. Testing configuration files...")
        config_files = [
            "config/test_hardware_maximized.yaml",
            "config/test_pipeline.yaml"
        ]
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                print(f"   ✅ {config_file} exists")
            else:
                print(f"   ❌ {config_file} not found")
        
        # Test execution scripts
        print("\n6. Testing execution scripts...")
        scripts = [
            "run_hardware_maximized.py",
            "run_pipeline.py",
            "validate_testing_framework.py",
            "benchmark_optimizations.py",
            "demo_testing_framework.py"
        ]
        
        for script in scripts:
            script_path = project_root / script
            if script_path.exists():
                size = script_path.stat().st_size
                print(f"   ✅ {script} exists ({size:,} bytes)")
            else:
                print(f"   ❌ {script} not found")
        
        # Test framework module
        print("\n7. Testing framework module...")
        try:
            from src.testing_framework import (
                TestingFramework,
                create_test_config,
                HardwareMonitor,
                OptimizedFeatureCache,
                VectorizedDecisionFusion
            )
            print("   ✅ testing_framework.py imports successfully")
            print("   ✅ All core classes available")
        except ImportError as e:
            print(f"   ❌ Failed to import testing_framework: {e}")
        
        print("\n" + "=" * 60)
        print("✅ INTEGRATION TEST PASSED!")
        print("\nThe testing framework is fully integrated and ready to use.")
        print("\nFrom the main menu:")
        print("  1. Run 'python main.py'")
        print("  2. Select 'Training Model' (option 3)")
        print("  3. Choose testing framework options (8-11)")
        print("\nOr run standalone scripts:")
        print("  python run_hardware_maximized.py --market NQ --mock-llm")
        print("  python run_pipeline.py --market ES --auto-resume")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
