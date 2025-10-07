"""
Validate Databricks Bundle Configuration
Run this to check if your bundle is properly configured
"""

import os
import sys
from pathlib import Path

def check_file(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"[OK] {description}: {file_path} ({size} bytes)")
        return True
    else:
        print(f"[ERROR] {description}: {file_path} NOT FOUND")
        return False

def check_yaml_syntax(file_path):
    """Check YAML syntax"""
    try:
        import yaml
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        print(f"[OK] YAML syntax valid: {file_path}")
        return True
    except Exception as e:
        print(f"[ERROR] YAML syntax error in {file_path}: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("  Databricks Bundle Validation")
    print("=" * 70)
    print()
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"Project Root: {project_root}")
    print()
    
    all_checks = []
    
    # Check main bundle file
    print("[*] Checking Bundle Configuration...")
    all_checks.append(check_file(
        project_root / "databricks.yml",
        "Main bundle config"
    ))
    
    # Check resources
    print("\n[*] Checking Resources...")
    all_checks.append(check_file(
        project_root / "resources" / "jobs.yml",
        "Jobs configuration"
    ))
    
    # Check notebooks
    print("\n[*] Checking Notebooks...")
    notebooks = [
        "01_data_preparation.py",
        "02_train_rl_model.py",
        "03_clinical_dashboard.py",
        "04_continuous_learning.py",
        "05_api_service.py"
    ]
    
    for notebook in notebooks:
        all_checks.append(check_file(
            project_root / "notebooks" / notebook,
            f"Notebook {notebook}"
        ))
    
    # Check data
    print("\n[*] Checking Data...")
    all_checks.append(check_file(
        project_root / "Clinical Main Data for Databricks.csv",
        "Training data"
    ))
    
    # Check YAML syntax (if PyYAML available)
    print("\n[*] Checking YAML Syntax...")
    try:
        import yaml
        check_yaml_syntax(project_root / "databricks.yml")
        check_yaml_syntax(project_root / "resources" / "jobs.yml")
    except ImportError:
        print("[!] PyYAML not installed. Run: pip install pyyaml")
    
    # Summary
    print("\n" + "=" * 70)
    if all(all_checks):
        print("[SUCCESS] ALL CHECKS PASSED!")
        print("\nYour bundle is properly configured.")
        print("\nNext steps:")
        print("1. Reload VS Code window (Ctrl+Shift+P -> 'Reload Window')")
        print("2. Check Bundle Resource Explorer in VS Code sidebar")
        print("3. Or deploy manually via Command Palette")
    else:
        print("[FAILED] SOME CHECKS FAILED!")
        print("\nPlease fix the issues above and try again.")
    print("=" * 70)
    
    return 0 if all(all_checks) else 1

if __name__ == "__main__":
    sys.exit(main())
