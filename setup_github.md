# 🚀 GitHub Repository Setup Guide

## Step 1: Initialize Git Repository

Open your terminal/command prompt in the project directory and run:

```bash
# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Optimized AI Interview System"
```

## Step 2: Connect to GitHub Repository

```bash
# Add your GitHub repository as remote origin
git remote add origin https://github.com/nams2002/Interview-Bot.git

# Verify the remote was added correctly
git remote -v
```

## Step 3: Push Code to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Alternative: If Repository Already Has Content

If your repository already has some files, you might need to pull first:

```bash
# Pull existing content (if any)
git pull origin main --allow-unrelated-histories

# Then push your changes
git push origin main
```

## Step 4: Set Up Streamlit Cloud Deployment

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Select your repository**: `nams2002/Interview-Bot`
5. **Set main file path**: `app.py`
6. **Click "Deploy!"**

## Step 5: Configure Secrets in Streamlit Cloud

After deployment, add your API keys:

1. **Go to your app settings** in Streamlit Cloud
2. **Click on "Secrets"**
3. **Add the following**:

```toml
[openai]
api_key = "your-openai-api-key-here"

[eden_ai]
api_key = "your-eden-ai-api-key-here"
```

## Step 6: Test Your Deployment

Your app will be available at: `https://interview-bot-[random-string].streamlit.app`

## Troubleshooting

### If you get authentication errors:
```bash
# Configure git with your GitHub credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Use GitHub CLI for easier authentication
gh auth login
```

### If you get merge conflicts:
```bash
# Force push (only if you're sure about overwriting)
git push origin main --force
```

### If deployment fails:
1. Check the logs in Streamlit Cloud
2. Ensure all dependencies are in requirements.txt
3. Verify the main file path is correct

## File Structure in Repository

```
Interview-Bot/
├── app.py                    # Main application
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── health_check.py           # System monitoring
├── deploy.py                 # Deployment script
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── .gitignore               # Git ignore rules
├── .streamlit/
│   ├── config.toml          # Streamlit config
│   └── secrets.toml.template # Secrets template
└── setup_github.md          # This guide
```

## Next Steps After Deployment

1. **Test the application** with your API keys
2. **Share the URL** with users
3. **Monitor performance** using the health check dashboard
4. **Update documentation** as needed
5. **Set up monitoring** for production use

## Commands Summary

```bash
# Quick setup commands
git init
git add .
git commit -m "Initial commit: Optimized AI Interview System"
git remote add origin https://github.com/nams2002/Interview-Bot.git
git branch -M main
git push -u origin main
```

## Support

If you encounter any issues:
1. Check the GitHub repository issues
2. Review Streamlit Cloud logs
3. Verify API key configuration
4. Test locally first with `streamlit run app.py`

---

**Your optimized AI Interview System is now ready for deployment! 🎉**
