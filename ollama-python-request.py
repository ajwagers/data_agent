import requests
import json

def generate_ollama_response(prompt, model="tinyllama",stream=True):
    url = "http://192.168.0.202:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": stream
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers,stream=stream)
    
    if response.status_code == 200:
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        full_response += content
                        print(content, end='', flush=True)
                    if data.get('done', False):
                        print("\n--- Response complete ---")
                        break
            return full_response
        else:
            data = response.json()
            if 'message' in data and 'content' in data['message']:
                return data['message']['content']
            else:
                return "Error: Unepected Response Format"
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
prompt = "Why is the sky blue?"
model = "tinyllama"
stream = True

result = generate_ollama_response(prompt,model,stream)
if not stream:
    print(result)
