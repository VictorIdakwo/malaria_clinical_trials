"""
Setup script for Malaria RL Clinical Trial System
This script initializes the Databricks environment and prepares the system for deployment.
"""

import subprocess
import sys
import json
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 70)
    print("  🦟 Malaria RL Clinical Trial System - Setup")
    print("=" * 70)
    print()

def check_databricks_cli():
    """Check if Databricks CLI is installed and configured"""
    print("📋 Checking Databricks CLI...")
    try:
        result = subprocess.run(
            ["databricks", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Databricks CLI installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Databricks CLI not found or not configured")
        print("   Install: pip install databricks-cli")
        print("   Configure: databricks configure")
        return False

def check_python_version():
    """Check Python version"""
    print("\n📋 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} detected")
        print("   Required: Python 3.9 or higher")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_config_file():
    """Create configuration file"""
    print("\n⚙️  Creating configuration file...")
    
    config = {
        "catalog": "eha",
        "schema": "malaria_catalog",
        "volume": "clinical_trial",
        "data_path": "/Volumes/eha/malaria_catalog/clinical_trial/data/Clinical Main Data for Databricks.csv",
        "experiment_path": "/Shared/malaria_rl_experiment"
    }
    
    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(config, indent=2, fp=f)
    
    print(f"✅ Configuration saved to {config_path}")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 70)
    print("  ✅ Setup Complete!")
    print("=" * 70)
    print("\n📝 Next Steps:\n")
    print("1. Upload your data to Databricks:")
    print("   databricks fs cp 'Clinical Main Data for Databricks.csv' \\")
    print("     dbfs:/Volumes/eha/malaria_catalog/clinical_trial/data/")
    print()
    print("2. Deploy the DAB bundle:")
    print("   databricks bundle deploy --target dev")
    print()
    print("3. Run the data preparation notebook:")
    print("   Open notebooks/01_data_preparation.py in Databricks")
    print()
    print("4. Train the initial model:")
    print("   Open notebooks/02_train_rl_model.py in Databricks")
    print()
    print("5. Launch the dashboard:")
    print("   Open notebooks/03_clinical_dashboard.py in Databricks")
    print("   Run with Streamlit")
    print()
    print("📚 For more information, see README.md")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    checks = [
        ("Python Version", check_python_version()),
        ("Databricks CLI", check_databricks_cli()),
        ("Dependencies", install_dependencies()),
        ("Configuration", create_config_file())
    ]
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
