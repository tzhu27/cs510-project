#!/usr/bin/env python3
"""
Setup script to copy necessary files from the original G-DIG project to the new package structure.
"""
import os
import shutil
import sys
import glob

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def copy_directory(src, dst):
    """Copy directory tree."""
    if os.path.exists(dst):
        print(f"Directory exists, removing: {dst}")
        shutil.rmtree(dst)
    
    shutil.copytree(src, dst)
    print(f"Copied directory: {src} -> {dst}")

def copy_file(src, dst):
    """Copy a single file."""
    shutil.copy2(src, dst)
    print(f"Copied file: {src} -> {dst}")

def main():
    """Main setup function."""
    # Create basic package structure
    create_directory("gdig")
    create_directory("gdig/nngeometry")
    create_directory("gdig/dataset")
    create_directory("examples")
    
    # Copy nngeometry directory
    if os.path.exists("nngeometry"):
        # Copy all subdirectories
        for item in os.listdir("nngeometry"):
            if os.path.isdir(os.path.join("nngeometry", item)):
                copy_directory(os.path.join("nngeometry", item), os.path.join("gdig/nngeometry", item))
            else:
                # Copy Python files
                if item.endswith(".py"):
                    copy_file(os.path.join("nngeometry", item), os.path.join("gdig/nngeometry", item))
    else:
        print("Warning: nngeometry directory not found. Please copy it manually.")
    
    # Copy dataset directory
    if os.path.exists("dataset"):
        # Copy all subdirectories
        for item in os.listdir("dataset"):
            if os.path.isdir(os.path.join("dataset", item)):
                copy_directory(os.path.join("dataset", item), os.path.join("gdig/dataset", item))
            else:
                # Copy Python files
                if item.endswith(".py"):
                    copy_file(os.path.join("dataset", item), os.path.join("gdig/dataset", item))
    else:
        print("Warning: dataset directory not found. Please copy it manually.")
    
    # Create empty __init__.py files if they don't exist
    for path in [
        "gdig/nngeometry/__init__.py",
        "gdig/dataset/__init__.py",
        "gdig/dataset/data/__init__.py",
        "gdig/dataset/prompt_maker/__init__.py",
        "gdig/dataset/utils/__init__.py",
        "gdig/nngeometry/object/__init__.py",
        "gdig/nngeometry/generator/__init__.py",
    ]:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            create_directory(directory)
        
        if not os.path.exists(path):
            with open(path, "w") as f:
                print(f"Created empty __init__.py: {path}")

    print("\nSetup complete!")
    print("You may now run: pip install -e .")
    print("Check README.md for usage instructions.")

if __name__ == "__main__":
    main() 