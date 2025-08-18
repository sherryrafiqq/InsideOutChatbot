# PythonAnywhere Free Tier Setup Guide

This guide provides detailed steps for deploying your InsideOut Chatbot API on PythonAnywhere's **free tier** account.

## ⚠️ Free Tier Limitations

**Important limitations to be aware of:**
- **512 MB RAM** limit
- **1 CPU** core
- **1 GB disk space**
- **1 web app** allowed
- **No custom domains** (only `yourusername.pythonanywhere.com`)
- **No HTTPS** (only HTTP)
- **Limited external network access** (whitelist required)
- **No background tasks**
- **No cron jobs**

## 📋 Prerequisites

1. **PythonAnywhere free account** (sign up at pythonanywhere.com)
2. **Supabase project** with database tables set up
3. **Cohere API key** for AI features
4. **All project files** ready for upload

## 🚀 Step-by-Step Setup

### Step 1: Access Your PythonAnywhere Dashboard

1. **Log in** to your PythonAnywhere account
2. **Go to the Dashboard** (main page after login)
3. **Note your username** - you'll need this for URLs

### Step 2: Upload Project Files

1. **Click on "Files"** in the top navigation
2. **Create a new directory**:
   - Click **"New directory"**
   - Name it: `IoT-Sentiment-Analysis`
   - Click **"Create"**
3. **Navigate into the directory** by clicking on it
4. **Upload all project files**:
   - Click **"Upload a file"**
   - Upload these files one by one:
     - `main.py`
     - `wsgi.py`
     - `requirements.txt`
     - `pythonanywhere_config.py`
     - `test_deployment.py`
     - Any other project files

### Step 3: Set Up Virtual Environment

1. **Go to "Consoles"** in the top navigation
2. **Click "New console"** → **"Bash"**
3. **Wait for the console to load** (may take a few seconds)
4. **Navigate to your project directory**:
   ```bash
   cd IoT-Sentiment-Analysis
   ```
5. **Create a virtual environment**:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.9 insideout_env
   ```
   *(This may take 1-2 minutes)*
6. **Activate the virtual environment**:
   ```bash
   workon insideout_env
   ```
7. **Verify activation** - you should see `(insideout_env)` at the start of your prompt
8. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
   *(This may take 5-10 minutes on free tier)*

### Step 4: Configure Environment Variables

1. **Go to "Web"** in the top navigation
2. **Click "Add a new web app"**
3. **Choose your domain**: `yourusername.pythonanywhere.com`
4. **Select "Manual configuration"** (not Django/Flask)
5. **Choose Python version**: `3.9`
6. **Click "Next"**
7. **In the web app configuration**:
   - **Source code**: `/home/yourusername/IoT-Sentiment-Analysis`
   - **Working directory**: `/home/yourusername/IoT-Sentiment-Analysis`
   - **WSGI configuration file**: Click the link to edit it
8. **Replace the WSGI file content** with:
   ```python
   import sys
   path = '/home/yourusername/IoT-Sentiment-Analysis'
   if path not in sys.path:
       sys.path.append(path)
   
   from wsgi import application
   ```
   *(Replace `yourusername` with your actual username)*
9. **Save the WSGI file**
10. **Go to "Environment variables"** section
11. **Add these environment variables**:
    ```
    COHERE_API_KEY=your-actual-cohere-api-key
    SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co/
    SUPABASE_KEY=your-actual-supabase-key
    JWT_SECRET=your-super-secret-jwt-key-change-me-in-production
    DEBUG=False
    PYTHONANYWHERE_SITE=yourusername.pythonanywhere.com
    ```
    *(Replace `yourusername` with your actual username)*

### Step 5: Configure Virtual Environment for Web App

1. **In the web app configuration**, find **"Virtual environment"**
2. **Enter the path**: `/home/yourusername/.virtualenvs/insideout_env`
   *(Replace `yourusername` with your actual username)*
3. **Click "Save"**

### Step 6: Reload the Web App

1. **Click the green "Reload" button** at the top of the web app page
2. **Wait for the reload to complete** (may take 30-60 seconds)
3. **Check the error logs** if the reload fails:
   - Click **"Error log"** to see any issues
   - Common issues and solutions are listed below

### Step 7: Test Your Deployment

1. **Open a new browser tab**
2. **Go to your API URL**: `http://yourusername.pythonanywhere.com`
3. **You should see a JSON response** like:
   ```json
   {
     "success": true,
     "message": "InsideOut Chatbot API is running!",
     "data": {
       "version": "1.0.0",
       "status": "healthy",
       "ai_available": true,
       "database_connected": true,
       "environment": "production"
     }
   }
   ```

### Step 8: Run the Test Script

1. **Go back to "Consoles"**
2. **Open your Bash console** (or start a new one)
3. **Navigate to your project**:
   ```bash
   cd IoT-Sentiment-Analysis
   ```
4. **Activate your virtual environment**:
   ```bash
   workon insideout_env
   ```
5. **Edit the test script** to use your URL:
   ```bash
   nano test_deployment.py
   ```
6. **Change the BASE_URL** line to:
   ```python
   BASE_URL = "http://yourusername.pythonanywhere.com"
   ```
   *(Replace `yourusername` with your actual username)*
7. **Save and exit** (Ctrl+X, then Y, then Enter)
8. **Run the test script**:
   ```bash
   python test_deployment.py
   ```

## 🔧 Troubleshooting Common Issues

### Issue 1: Import Errors
**Symptoms**: Error logs show import failures
**Solution**:
1. **Check virtual environment** is activated in web app
2. **Verify all packages installed**:
   ```bash
   workon insideout_env
   pip list
   ```
3. **Reinstall requirements** if needed:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

### Issue 2: Memory Errors
**Symptoms**: "Out of memory" or "Killed" errors
**Solution**:
1. **Reduce worker count** in main.py (already set to 1)
2. **Optimize imports** (already done)
3. **Consider upgrading** to paid tier if persistent

### Issue 3: Database Connection Issues
**Symptoms**: "Database connection not available"
**Solution**:
1. **Check Supabase credentials** in environment variables
2. **Verify network access** (free tier has limitations)
3. **Test connection** in console:
   ```bash
   workon insideout_env
   python -c "from main import supabase; print('Connected' if supabase else 'Failed')"
   ```

### Issue 4: AI Model Loading Issues
**Symptoms**: "AI service is not available"
**Solution**:
1. **Check Cohere API key** in environment variables
2. **Verify API key is valid** and has credits
3. **Test API key** in console:
   ```bash
   workon insideout_env
   python -c "import os; print('Key set' if os.getenv('COHERE_API_KEY') else 'No key')"
   ```

### Issue 5: Permission Errors
**Symptoms**: "Permission denied" errors
**Solution**:
1. **Check file permissions**:
   ```bash
   ls -la /home/yourusername/IoT-Sentiment-Analysis/
   ```
2. **Fix permissions** if needed:
   ```bash
   chmod 644 *.py
   chmod 644 requirements.txt
   ```

## 📊 Monitoring Your App

### Check Error Logs
1. **Go to "Web"** → your web app
2. **Click "Error log"** to see recent errors
3. **Click "Server log"** to see general server activity

### Check Resource Usage
1. **Go to "Account"** in top navigation
2. **View "CPU seconds used"** and **"Disk space used"**
3. **Monitor daily usage** to stay within free tier limits

### Test API Endpoints
Use these URLs to test your API:
- **Health check**: `http://yourusername.pythonanywhere.com/health`
- **Root endpoint**: `http://yourusername.pythonanywhere.com/`
- **API docs** (if DEBUG=True): `http://yourusername.pythonanywhere.com/docs`

## 🔒 Security Considerations for Free Tier

1. **Use strong JWT secrets** (change the default)
2. **Validate all inputs** (already implemented)
3. **Monitor for abuse** (free tier has rate limits)
4. **Keep dependencies updated** regularly
5. **Use environment variables** for all secrets

## 📈 Performance Optimization for Free Tier

1. **Minimize memory usage**:
   - AI models load on-demand
   - Database connections are pooled
   - Static files are served efficiently

2. **Optimize for single CPU**:
   - Single worker configuration
   - Async operations where possible
   - Efficient database queries

3. **Stay within limits**:
   - Monitor CPU usage
   - Keep disk usage under 1GB
   - Avoid long-running operations

## 🆘 Getting Help

If you encounter issues:

1. **Check PythonAnywhere documentation**: https://help.pythonanywhere.com/
2. **Review error logs** in the Web tab
3. **Test in console** first before web app
4. **Use the test script** to verify functionality
5. **Check this guide** for common solutions

## ✅ Success Checklist

- [ ] Files uploaded to PythonAnywhere
- [ ] Virtual environment created and activated
- [ ] Requirements installed successfully
- [ ] Web app configured with correct paths
- [ ] Environment variables set
- [ ] WSGI file updated
- [ ] Web app reloaded successfully
- [ ] Root endpoint returns JSON response
- [ ] Health check endpoint works
- [ ] Test script passes all tests
- [ ] Database connection established
- [ ] AI models loading correctly

## 🎯 Next Steps After Deployment

1. **Update your Flutter app** to use the new API URL
2. **Test all features** thoroughly
3. **Monitor performance** and usage
4. **Set up logging** for debugging
5. **Consider upgrading** to paid tier for production use

Your API should now be accessible at: `http://yourusername.pythonanywhere.com` 