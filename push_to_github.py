#!/usr/bin/env python3
"""
Quick script to push the optimized AI Interview System to GitHub
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        return False

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed or not in PATH")
        print("   Please install Git from https://git-scm.com/")
        return False

def setup_git_config():
    """Setup git configuration if not already set"""
    try:
        # Check if user.name is set
        result = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True)
        if not result.stdout.strip():
            name = input("Enter your name for Git commits: ")
            run_command(f'git config --global user.name "{name}"', "Setting Git username")
        
        # Check if user.email is set
        result = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True)
        if not result.stdout.strip():
            email = input("Enter your email for Git commits: ")
            run_command(f'git config --global user.email "{email}"', "Setting Git email")
        
        return True
    except Exception as e:
        print(f"❌ Failed to setup Git config: {e}")
        return False

def main():
    """Main function to push code to GitHub"""
    print("🚀 AI Interview System - GitHub Deployment")
    print("=" * 50)
    
    # Check if git is installed
    if not check_git_installed():
        return False
    
    # Setup git config
    if not setup_git_config():
        return False
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project directory.")
        return False
    
    print("\n📁 Files to be uploaded:")
    files_to_upload = [
        "app.py",
        "config.py", 
        "utils.py",
        "health_check.py",
        "deploy.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        ".streamlit/config.toml",
        ".streamlit/secrets.toml.template",
        "setup_github.md"
    ]
    
    for file in files_to_upload:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️  {file} (missing)")
    
    # Confirm upload
    confirm = input("\n🤔 Do you want to proceed with uploading to GitHub? (y/N): ")
    if confirm.lower() not in ['y', 'yes']:
        print("❌ Upload cancelled")
        return False
    
    # Initialize git repository
    if not Path(".git").exists():
        if not run_command("git init", "Initializing Git repository"):
            return False
    
    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return False
    
    # Create commit
    commit_message = "feat: Add optimized AI Interview System with enhanced performance and deployment features"
    if not run_command(f'git commit -m "{commit_message}"', "Creating commit"):
        # Check if there are no changes to commit
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("ℹ️  No changes to commit")
        else:
            return False
    
    # Add remote origin
    repo_url = "https://github.com/nams2002/Interview-Bot.git"
    
    # Check if remote already exists
    result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    if result.returncode != 0:
        if not run_command(f"git remote add origin {repo_url}", "Adding GitHub remote"):
            return False
    else:
        print("ℹ️  Remote origin already exists")
    
    # Set main branch
    if not run_command("git branch -M main", "Setting main branch"):
        return False
    
    # Push to GitHub
    print("\n🚀 Pushing to GitHub...")
    print("   Note: You may be prompted for GitHub credentials")
    
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        print("\n💡 If authentication failed, try:")
        print("   1. Use GitHub CLI: gh auth login")
        print("   2. Use personal access token instead of password")
        print("   3. Configure SSH keys for GitHub")
        return False
    
    print("\n🎉 Successfully uploaded to GitHub!")
    print(f"📍 Repository URL: {repo_url}")
    
    # Deployment instructions
    print("\n📋 Next Steps for Streamlit Cloud Deployment:")
    print("1. Go to https://share.streamlit.io")
    print("2. Sign in with GitHub")
    print("3. Click 'New app'")
    print("4. Select repository: nams2002/Interview-Bot")
    print("5. Set main file: app.py")
    print("6. Click 'Deploy!'")
    print("7. Add your OpenAI API key in app secrets")
    
    print("\n🔐 Don't forget to configure your API keys in Streamlit Cloud secrets!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ All done! Your AI Interview System is ready for deployment!")
    else:
        print("\n❌ Deployment preparation failed. Please check the errors above.")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
