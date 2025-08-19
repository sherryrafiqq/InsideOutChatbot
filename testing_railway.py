import requests

url = "https://web-production-29fb.up.railway.app/chat"
headers = {
    "Authorization": "Bearer YOUR_TOKEN_HERE",
    "Content-Type": "application/json"
}
data = { "message": "Hello from Python!" }

response = requests.post(url, headers=headers, json=data)
print(response.json())
