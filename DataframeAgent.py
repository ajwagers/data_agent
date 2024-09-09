import os
import pandas as pd
import requests
import json

from langchain.llms.base import LLM
#from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Any, List, Mapping, Optional

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Create a custom LLM class that uses the Ollama API
class OllamaLLM(LLM):
    model_name: str = "llama3"
    base_url: str = 'http://localhost:11434/api/chat'

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(self.base_url, headers=headers, json=payload)  
        
        if response.status_code == 200:
            data = response.json()
            if 'message' in data and 'content' in data['message']:
                return data['message']['content']
            else:
                return "Error: Unexpected Response Format"
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    @property
    def _llm_type(self) -> str:
        return "custom_ollama"

ollama_llm = OllamaLLM(model="llama3")

agent = create_pandas_dataframe_agent(ollama_llm, iris, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

print("Welcome to the Interactive Dataset Query System!")
print("You can ask questions about the iris dataset, and the AI will answer them.")
print("Type 'exit' to quit the program.")

while True:
    user_input = input("\nEnter your question about the iris dataset: ")
    
    if user_input.lower() == 'exit':
        print("Thank you for using the Interactive Dataset Query System. Goodbye!")
        break
    
    try:
        response = agent.invoke(user_input)
        print("\nAnswer:", response['output'])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()