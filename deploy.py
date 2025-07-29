#!/usr/bin/env python3
"""
Deployment script for AI Interview System
Helps with setup and configuration for different deployment environments
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = [
        "interview_reports",
        "session_data",
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_secrets():
    """Setup Streamlit secrets file"""
    secrets_path = Path(".streamlit/secrets.toml")
    template_path = Path(".streamlit/secrets.toml.template")
    
    if not secrets_path.exists() and template_path.exists():
        print("🔐 Setting up secrets file...")
        
        # Copy template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get API keys from user
        openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        eden_key = input("Enter your Eden AI API key (optional, press Enter to skip): ").strip()
        
        # Replace placeholders
        content = template_content
        if openai_key:
            content = content.replace("your-openai-api-key-here", openai_key)
        if eden_key:
            content = content.replace("your-eden-ai-api-key-here", eden_key)
        
        with open(secrets_path, 'w') as f:
            f.write(content)
        
        print("✅ Secrets file created")
    else:
        print("ℹ️  Secrets file already exists or template not found")

def download_models():
    """Download required models"""
    print("🤖 Downloading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✅ YOLO model ready")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    required_packages = [
        'streamlit',
        'cv2',
        'numpy',
        'pandas',
        'mediapipe',
        'openai',
        'ultralytics',
        'gtts',
        'pygame',
        'sklearn',
        'requests',
        'nltk'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def setup_nltk():
    """Download required NLTK data"""
    print("📚 Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to setup NLTK: {e}")
        return False

def create_run_script():
    """Create a run script for easy startup"""
    script_content = """#!/bin/bash
# Run script for AI Interview System

echo "🚀 Starting AI Interview System..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "👋 Application stopped"
"""
    
    with open("run.sh", "w") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("run.sh", 0o755)
    
    print("✅ Run script created (run.sh)")

def create_docker_files():
    """Create Docker configuration files"""
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libgthread-2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p interview_reports session_data

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    docker_compose_content = """version: '3.8'

services:
  interview-system:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ENABLE_CORS=false
    volumes:
      - ./interview_reports:/app/interview_reports
      - ./session_data:/app/session_data
    restart: unless-stopped
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("✅ Docker files created")

def main():
    """Main deployment function"""
    print("🎤 AI Interview System - Deployment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Setup NLTK
    setup_nltk()
    
    # Download models
    download_models()
    
    # Setup secrets
    setup_secrets()
    
    # Create run script
    create_run_script()
    
    # Create Docker files
    create_docker_files()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your API keys in .streamlit/secrets.toml")
    print("2. Run the application with: streamlit run app.py")
    print("3. Or use the run script: ./run.sh")
    print("4. For Docker deployment: docker-compose up")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
