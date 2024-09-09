import requests
import json

def generate_ollama_response(prompt, model="tinyllama"):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        # Parse the response content
        response_lines = response.text.strip().split('\n')
        full_response = ""
        for line in response_lines:
            try:
                data = json.loads(line)
                if 'response' in data:
                    full_response += data['response']
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
        return full_response
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
prompt = "Why is the sky blue?"
result = generate_ollama_response(prompt)
print(result)
