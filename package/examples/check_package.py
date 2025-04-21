#!/usr/bin/env python3
"""
Simple script to check if the G-DIG package is installed correctly.
"""
import sys
import importlib

def check_import(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    print("Checking G-DIG package installation...\n")
    
    # List of modules to check
    modules = [
        "gdig",
        "gdig.gdig",
        "gdig.cli",
        "gdig.nngeometry",
        "gdig.nngeometry.object",
        "gdig.nngeometry.generator",
        "gdig.dataset",
    ]
    
    # Check each module
    all_success = True
    for module in modules:
        success = check_import(module)
        all_success = all_success and success
    
    print("\nSummary:")
    if all_success:
        print("✅ G-DIG package is correctly installed and ready to use!")
    else:
        print("❌ There were some issues with importing G-DIG modules.")
        print("   Please check the output above and fix any missing dependencies.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 