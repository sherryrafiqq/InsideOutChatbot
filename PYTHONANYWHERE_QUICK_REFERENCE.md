# PythonAnywhere Free Tier Quick Reference

## 🚀 Essential Commands

### Console Commands
```bash
# Navigate to project
cd IoT-Sentiment-Analysis

# Create virtual environment
mkvirtualenv --python=/usr/bin/python3.9 insideout_env

# Activate virtual environment
workon insideout_env

# Install requirements
pip install -r requirements.txt

# Check installed packages
pip list

# Test Python imports
python -c "from main import app; print('App loaded successfully')"

# Run test script
python test_deployment.py
```

### File Paths (Replace `yourusername` with your actual username)
```
Project Directory: /home/yourusername/IoT-Sentiment-Analysis
Virtual Environment: /home/yourusername/.virtualenvs/insideout_env
WSGI File: /var/www/yourusername_pythonanywhere_com_wsgi.py
```

## ⚙️ Environment Variables

Add these in Web App → Environment variables:
```
COHERE_API_KEY=your-actual-cohere-api-key
SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co/
SUPABASE_KEY=your-actual-supabase-key
JWT_SECRET=your-super-secret-jwt-key-change-me-in-production
DEBUG=False
PYTHONANYWHERE_SITE=yourusername.pythonanywhere.com
```

## 🔧 WSGI Configuration

Replace the WSGI file content with:
```python
import sys
path = '/home/yourusername/IoT-Sentiment-Analysis'
if path not in sys.path:
    sys.path.append(path)

from wsgi import application
```

## 📊 Web App Settings

- **Source code**: `/home/yourusername/IoT-Sentiment-Analysis`
- **Working directory**: `/home/yourusername/IoT-Sentiment-Analysis`
- **Virtual environment**: `/home/yourusername/.virtualenvs/insideout_env`
- **Python version**: 3.9

## 🧪 Testing URLs

- **API Base**: `http://yourusername.pythonanywhere.com`
- **Health Check**: `http://yourusername.pythonanywhere.com/health`
- **Root Endpoint**: `http://yourusername.pythonanywhere.com/`

## 🔍 Troubleshooting Commands

### Check Virtual Environment
```bash
workon insideout_env
which python
pip list | grep fastapi
```

### Test Database Connection
```bash
workon insideout_env
python -c "from main import supabase; print('Connected' if supabase else 'Failed')"
```

### Test AI Models
```bash
workon insideout_env
python -c "from main import AI_AVAILABLE; print('AI Available:', AI_AVAILABLE)"
```

### Check Environment Variables
```bash
workon insideout_env
python -c "import os; print('COHERE_KEY:', 'Set' if os.getenv('COHERE_API_KEY') else 'Not set')"
```

### Check File Permissions
```bash
ls -la /home/yourusername/IoT-Sentiment-Analysis/
chmod 644 *.py requirements.txt
```

## 📝 Common Error Solutions

### Import Error
```bash
workon insideout_env
pip install -r requirements.txt --force-reinstall
```

### Memory Error
- Check if using single worker (already configured)
- Monitor memory usage in Account tab
- Consider upgrading to paid tier

### Permission Error
```bash
chmod 644 /home/yourusername/IoT-Sentiment-Analysis/*.py
chmod 644 /home/yourusername/IoT-Sentiment-Analysis/requirements.txt
```

### Database Connection Error
- Verify Supabase credentials in environment variables
- Check network access (free tier limitations)
- Test connection in console

## 🔄 Reload Process

1. **Save all changes** in web app configuration
2. **Click green "Reload" button**
3. **Wait 30-60 seconds** for reload to complete
4. **Check error logs** if reload fails
5. **Test endpoints** in browser

## 📈 Resource Monitoring

- **CPU Usage**: Account → CPU seconds used
- **Disk Usage**: Account → Disk space used
- **Memory**: Monitor in error logs for "Killed" messages
- **Network**: Check for external access limitations

## 🆘 Emergency Commands

### Reset Virtual Environment
```bash
deactivate
rmvirtualenv insideout_env
mkvirtualenv --python=/usr/bin/python3.9 insideout_env
workon insideout_env
pip install -r requirements.txt
```

### Check All Services
```bash
workon insideout_env
python -c "
from main import app, supabase, AI_AVAILABLE
print('App:', 'OK' if app else 'FAIL')
print('Database:', 'OK' if supabase else 'FAIL')
print('AI:', 'OK' if AI_AVAILABLE else 'FAIL')
"
```

### Quick Health Check
```bash
curl http://yourusername.pythonanywhere.com/health
```

## 🎯 Success Indicators

✅ **Virtual environment** shows `(insideout_env)` in prompt  
✅ **Requirements install** without errors  
✅ **Web app reloads** successfully  
✅ **Root endpoint** returns JSON response  
✅ **Health check** shows all services connected  
✅ **Test script** passes all tests  

## 📞 Support Resources

- **PythonAnywhere Help**: https://help.pythonanywhere.com/
- **Error Logs**: Web → your web app → Error log
- **Server Logs**: Web → your web app → Server log
- **Account Usage**: Account tab for resource monitoring 