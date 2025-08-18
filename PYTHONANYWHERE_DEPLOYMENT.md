# PythonAnywhere Deployment Guide

This guide will help you deploy your InsideOut Chatbot API to PythonAnywhere.

## Prerequisites

- PythonAnywhere account (free or paid)
- Supabase project with database tables set up
- Cohere API key for AI features

## Step 1: Upload Files to PythonAnywhere

1. **Access your PythonAnywhere dashboard**
2. **Go to the Files tab**
3. **Create a new directory** (e.g., `IoT-Sentiment-Analysis`)
4. **Upload all project files** to this directory:
   - `main.py`
   - `wsgi.py`
   - `requirements.txt`
   - `pythonanywhere_config.py`
   - Any other project files

## Step 2: Set Up Virtual Environment

1. **Go to the Consoles tab**
2. **Start a new Bash console**
3. **Navigate to your project directory**:
   ```bash
   cd IoT-Sentiment-Analysis
   ```
4. **Create a virtual environment**:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.9 insideout_env
   ```
5. **Activate the virtual environment**:
   ```bash
   workon insideout_env
   ```
6. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## Step 3: Configure Environment Variables

1. **Go to the Web tab**
2. **Click on your web app** (or create a new one)
3. **Go to the Environment variables section**
4. **Add the following environment variables**:

```
COHERE_API_KEY=your-actual-cohere-api-key
SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co/
SUPABASE_KEY=your-actual-supabase-key
JWT_SECRET=your-super-secret-jwt-key-change-me-in-production
DEBUG=False
PYTHONANYWHERE_SITE=yourusername.pythonanywhere.com
```

## Step 4: Configure Web App

1. **In the Web tab, configure your web app**:
   - **Source code**: `/home/yourusername/IoT-Sentiment-Analysis`
   - **Working directory**: `/home/yourusername/IoT-Sentiment-Analysis`
   - **WSGI configuration file**: Click on the WSGI configuration file link
   - **Replace the content** with the path to your `wsgi.py` file

2. **Update the WSGI configuration** to point to your application:
   ```python
   import sys
   path = '/home/yourusername/IoT-Sentiment-Analysis'
   if path not in sys.path:
       sys.path.append(path)
   
   from wsgi import application
   ```

## Step 5: Set Up Database Tables

Make sure your Supabase project has the following tables:

### Users Table
```sql
CREATE TABLE users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Emotion Logs Table
```sql
CREATE TABLE emotion_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    emotion VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Character Emotions Table
```sql
CREATE TABLE character_emotions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    character_name VARCHAR(100) NOT NULL,
    emotion VARCHAR(20) NOT NULL,
    led_color VARCHAR(7) NOT NULL,
    sound_file VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Step 6: Test Your Application

1. **Reload your web app** in the Web tab
2. **Check the error logs** if there are any issues
3. **Test the health endpoint**: `https://yourusername.pythonanywhere.com/health`
4. **Test the root endpoint**: `https://yourusername.pythonanywhere.com/`

## Step 7: Configure CORS for Your Flutter App

Update the CORS configuration in `main.py` to include your Flutter app's domain:

```python
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://your-flutter-app-domain.com",  # Add your actual domain
    "https://yourusername.pythonanywhere.com"  # Your PythonAnywhere domain
]
```

## Step 8: SSL Certificate (Optional)

If you have a paid PythonAnywhere account, you can set up SSL:
1. **Go to the Web tab**
2. **Click on your domain**
3. **Enable HTTPS**

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed in your virtual environment
2. **Database Connection Issues**: Verify your Supabase credentials and network connectivity
3. **Permission Errors**: Check file permissions in PythonAnywhere
4. **Memory Issues**: Consider upgrading to a paid plan if you hit memory limits

### Debugging

1. **Check error logs** in the Web tab
2. **Use the console** to test imports and connections
3. **Test locally** first to ensure everything works
4. **Check environment variables** are set correctly

### Performance Optimization

1. **Use connection pooling** for database connections
2. **Implement caching** for frequently accessed data
3. **Optimize AI model usage** to reduce API calls
4. **Monitor memory usage** and optimize accordingly

## API Endpoints

Your API will be available at:
- **Base URL**: `https://yourusername.pythonanywhere.com`
- **Health Check**: `GET /health`
- **Documentation**: `GET /docs` (if DEBUG=True)

## Security Considerations

1. **Change default JWT secret** in production
2. **Use HTTPS** for all communications
3. **Implement rate limiting** for API endpoints
4. **Validate all inputs** thoroughly
5. **Use secure environment variables** for sensitive data

## Monitoring

1. **Set up logging** to monitor application health
2. **Monitor API usage** and performance
3. **Set up alerts** for critical errors
4. **Regular backups** of your database

## Support

If you encounter issues:
1. **Check PythonAnywhere documentation**
2. **Review error logs** in the Web tab
3. **Test endpoints** individually
4. **Verify all configurations** are correct

## Next Steps

After successful deployment:
1. **Update your Flutter app** to use the new API URL
2. **Test all features** thoroughly
3. **Monitor performance** and usage
4. **Set up monitoring** and alerts
5. **Plan for scaling** if needed 