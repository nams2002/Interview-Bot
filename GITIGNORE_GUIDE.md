# 📁 .gitignore File Guide for AI Interview System

## 🛡️ What's Protected

Your `.gitignore` file is comprehensive and protects sensitive data while keeping your repository clean. Here's what it excludes:

### 🔐 **Security & Secrets**
- `.streamlit/secrets.toml` - Your API keys and secrets
- `.env*` files - Environment variables
- `api_keys.txt`, `openai_key.txt` - Any API key files
- `credentials/`, `.secrets/` - Credential directories
- `private_keys/` - Private key storage

### 🤖 **AI Interview System Specific**
- `interview_reports/` - Generated interview reports
- `session_data/` - User session data
- `yolov8n.pt` - YOLO model file (large)
- `speech_*.mp3` - Generated speech files
- `temp_audio_*` - Temporary audio files
- `interview_system.log` - Application logs

### 🧠 **AI Models & Data**
- `*.pt`, `*.pth` - PyTorch model files
- `*.h5` - Keras/TensorFlow models
- `*.onnx` - ONNX model files
- `*.pkl`, `*.pickle` - Serialized Python objects
- `models/`, `checkpoints/` - Model directories

### 📊 **Generated Content**
- `*.mp3`, `*.wav` - Audio files
- `*.mp4`, `*.avi` - Video files
- `*.csv`, `*.xlsx` - Data exports (except requirements)
- `reports/`, `exports/` - Generated reports
- `backups/` - Backup files

### 🐍 **Python Standard**
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python files
- `venv/`, `env/` - Virtual environments
- `.pytest_cache/` - Test cache
- `*.egg-info/` - Package info

### 💻 **Development Tools**
- `.vscode/`, `.idea/` - IDE settings
- `*.log` - Log files
- `.cache/` - Cache directories
- `node_modules/` - Node.js dependencies
- `.DS_Store` - macOS system files

### 🔧 **Build & Distribution**
- `build/`, `dist/` - Build artifacts
- `*.egg`, `*.whl` - Python packages
- `target/` - Build targets
- `*.zip`, `*.tar.gz` - Archives

## ✅ **What's Included**

The following important files ARE tracked:

### 📋 **Configuration Files**
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration
- `.streamlit/secrets.toml.template` - Secrets template
- `config.py` - Application configuration

### 📚 **Source Code**
- `app.py` - Main application
- `utils.py` - Utility functions
- `health_check.py` - Health monitoring
- `deploy.py` - Deployment script

### 📖 **Documentation**
- `README.md` - Project documentation
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `*.md` files - All markdown documentation

### 🐳 **Deployment Files**
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose setup
- `push_to_github.py` - GitHub deployment script

## 🚨 **Important Notes**

### ⚠️ **Never Commit These**
- Real API keys or secrets
- User personal data
- Large model files (>100MB)
- Generated reports with sensitive info
- Audio/video recordings

### ✅ **Safe to Commit**
- Template files (`.template` extension)
- Configuration without secrets
- Documentation and guides
- Source code and scripts
- Requirements and dependencies

## 🔧 **Customization**

If you need to track specific files that are currently ignored, you can:

### Add Exceptions
```gitignore
# Ignore all CSV files except specific ones
*.csv
!important_config.csv
!sample_data.csv
```

### Temporary Override
```bash
# Force add a normally ignored file
git add -f filename.csv
```

## 📝 **Best Practices**

1. **Review before committing**: Always check what you're committing
   ```bash
   git status
   git diff --cached
   ```

2. **Use git check-ignore**: Check if a file is ignored
   ```bash
   git check-ignore filename.txt
   ```

3. **Clean up regularly**: Remove old ignored files
   ```bash
   git clean -fdx  # Be careful with this command!
   ```

4. **Update as needed**: Add new patterns when you add new file types

## 🛠️ **Troubleshooting**

### File Still Being Tracked?
If a file is already tracked by git, adding it to `.gitignore` won't stop tracking it:

```bash
# Stop tracking but keep the file
git rm --cached filename.txt

# Stop tracking a directory
git rm -r --cached directory/
```

### Check What's Ignored
```bash
# See all ignored files
git status --ignored

# Check if specific file is ignored
git check-ignore -v filename.txt
```

## 🎯 **Summary**

Your `.gitignore` file is production-ready and includes:
- ✅ **307 lines** of comprehensive exclusions
- ✅ **Security-focused** - protects API keys and secrets
- ✅ **AI/ML optimized** - handles model files and data
- ✅ **Streamlit specific** - covers Streamlit deployment needs
- ✅ **Cross-platform** - works on Windows, macOS, Linux

This ensures your repository stays clean, secure, and deployment-ready! 🚀
