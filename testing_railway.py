import requests

url = "https://web-production-29fb.up.railway.app/chat"
headers = {
    "Content-Type": "application/json"
}
data = { "message": "Hello from Python!" }

response = requests.post(url, headers=headers, json=data)
print(response.json())
