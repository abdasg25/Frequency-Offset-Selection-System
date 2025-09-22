#!/usr/bin/env python3
"""
Repository cleanup verification script.
Verifies that all rule-based code has been completely removed.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_for_rule_based_artifacts():
    """Check for any remaining rule-based code or artifacts."""
    
    print("🧹 ML-Based Frequency Selection Repository Cleanup Verification")
    print("=" * 60)
    
    # Define rule-based keywords to search for
    rule_based_keywords = [
        "center_biased",
        "CenterBias",
        "rule_based",
        "rule-based",
        "center_bias",
        "final_frequency_selection",
        "run_final_frequency"
    ]
    
    # Files that should not exist
    forbidden_files = [
        "processing/center_biased_selector.py",
        "processing/frequency_analysis.py",
        "scripts/run_final_frequency_selection.py",
        "scripts/test_center_biased_selector.py",
        "scripts/verify_frequency_mapping.py",
        "outputs/final_frequency_selection_results/",
        "outputs/center_biased_selector_results.json"
    ]
    
    # Check for forbidden files
    print("\n📁 Checking for deleted rule-based files...")
    deleted_files = []
    remaining_files = []
    
    for file_path in forbidden_files:
        full_path = Path(file_path)
        if full_path.exists():
            remaining_files.append(file_path)
            print(f"   ❌ FOUND: {file_path}")
        else:
            deleted_files.append(file_path)
            print(f"   ✅ DELETED: {file_path}")
    
    # Check for rule-based code in remaining files
    print(f"\n🔍 Searching for rule-based keywords in Python files...")
    
    project_files = []
    for py_file in Path(".").rglob("*.py"):
        # Skip virtual environment files
        if "evn/" not in str(py_file) and "__pycache__" not in str(py_file):
            project_files.append(py_file)
    
    files_with_keywords = []
    
    for py_file in project_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            found_keywords = []
            for keyword in rule_based_keywords:
                if keyword in content:
                    found_keywords.append(keyword)
            
            if found_keywords:
                files_with_keywords.append((py_file, found_keywords))
                print(f"   ❌ {py_file}: {', '.join(found_keywords)}")
        
        except (UnicodeDecodeError, PermissionError):
            # Skip binary files or files we can't read
            continue
    
    if not files_with_keywords:
        print("   ✅ No rule-based keywords found in Python files!")
    
    # Check current file structure
    print(f"\n📂 Current clean ML-focused file structure:")
    
    important_dirs = [
        "models/",
        "data/", 
        "scripts/",
        "processing/",
        "configs/"
    ]
    
    for dir_name in important_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"\n   📁 {dir_name}")
            for file_path in sorted(dir_path.glob("*.py")):
                print(f"      📄 {file_path.name}")
    
    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"   ✅ Deleted files: {len(deleted_files)}")
    print(f"   ❌ Remaining forbidden files: {len(remaining_files)}")
    print(f"   ❌ Files with rule-based keywords: {len(files_with_keywords)}")
    
    if remaining_files or files_with_keywords:
        print(f"\n⚠️  CLEANUP INCOMPLETE!")
        print(f"   Please address the remaining items above.")
        return False
    else:
        print(f"\n🎉 CLEANUP COMPLETE!")
        print(f"   Repository is now 100% ML-focused.")
        print(f"\n🚀 Ready for ML training:")
        print(f"   1. Train: python scripts/train_ml_frequency_selector.py")
        print(f"   2. Inference: python scripts/run_ml_frequency_selection.py")
        print(f"   3. Evaluate: python scripts/evaluate_ml_performance.py")
        return True

if __name__ == "__main__":
    success = check_for_rule_based_artifacts()
    sys.exit(0 if success else 1)