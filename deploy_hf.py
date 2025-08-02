#!/usr/bin/env python3
"""
Hugging Face Spaces Deployment Script
Automates the deployment process to Hugging Face Spaces
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_git_status():
    """Check if git repository is clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print("⚠️  Warning: You have uncommitted changes")
            print("   Consider committing them before deployment")
            return False
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: Not a git repository")
        return False


def check_required_files():
    """Check if all required files exist."""
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "src/app.py",
        "src/model.py",
        "src/data_generator.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ All required files present")
    return True


def get_hf_username():
    """Get Hugging Face username from user."""
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return None
    return username


def setup_hf_remote(username):
    """Set up Hugging Face remote repository."""
    space_name = "electricity-consumption-predictor"
    remote_url = f"https://huggingface.co/spaces/{username}/{space_name}"

    try:
        # Check if remote already exists
        result = subprocess.run(
            ["git", "remote", "get-url", "hf"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ HF remote already exists: {result.stdout.strip()}")
            return True

        # Add new remote
        subprocess.run(["git", "remote", "add", "hf", remote_url], check=True)
        print(f"✅ Added HF remote: {remote_url}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up HF remote: {e}")
        return False


def deploy_to_hf():
    """Deploy the application to Hugging Face Spaces."""
    print("🚀 Starting Hugging Face Spaces deployment...")

    # Pre-deployment checks
    if not check_git_status():
        response = input("Continue anyway? (y/N): ").lower()
        if response != "y":
            return False

    if not check_required_files():
        return False

    # Get username and setup remote
    username = get_hf_username()
    if not username:
        return False

    if not setup_hf_remote(username):
        return False

    # Commit changes if needed
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Deploy to Hugging Face Spaces"], check=True
        )
        print("✅ Committed changes")
    except subprocess.CalledProcessError:
        print("⚠️  No changes to commit")

    # Push to Hugging Face
    try:
        print("📤 Pushing to Hugging Face Spaces...")
        subprocess.run(["git", "push", "hf", "main"], check=True)
        print("✅ Successfully pushed to Hugging Face Spaces!")

        space_url = f"https://huggingface.co/spaces/{username}/electricity-consumption-predictor"
        print(f"\n🎉 Your app is now live at: {space_url}")
        print("\n📋 Next steps:")
        print("1. Visit the URL above to verify deployment")
        print("2. Check the 'Logs' tab for any issues")
        print("3. Test all features of your app")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error pushing to HF: {e}")
        return False


def main():
    """Main deployment function."""
    print("⚡ Electricity Consumption Predictor - HF Deployment")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("src/app.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        return 1

    # Run deployment
    if deploy_to_hf():
        print("\n🎊 Deployment completed successfully!")
        return 0
    else:
        print("\n💥 Deployment failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
