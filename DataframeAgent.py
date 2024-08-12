import os
import pandas as pd
import requests
import json

from openai import OpenAI
from langchain.llms.base import LLM
#from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from typing import Any, List, Mapping, Optional

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Set up the OpenAI client for Ollama
client = OpenAI(
    base_url='http://localhost:8080/v1/chat/completions',
    api_key='ollama',  # required, but unused
)

# Create a custom LLM class that uses the Ollama API
class OllamaLLM(LLM):
    model_name: str = "tinyllama"
    base_url: str = 'localhost:8080/v1/chat/completions'

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)  
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    #def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
    #    response = requests.post(
    #        f"{self.base_url}/chat",
    #        json={
    #            "model": self.model_name,
    #            "messages": [
    #                {"role": "user", "content": prompt}
    #            ]
    #        }
    #    )
    #    response.raise_for_status()
    #    return response.json()['message']['content']
    
    @property
    def _llm_type(self) -> str:
        return "custom_ollama"

llm = OllamaLLM()
agent = create_pandas_dataframe_agent(
    llm, 
    iris, 
    verbose=True,
    agent_type="openai-tools",
    allow_dangerous_code=True,
    handle_parsing_errors=True
)

print("Welcome to the Interactive Iris Dataset Query System!")
print("You can ask questions about the iris dataset, and the AI will answer them.")
print("Type 'exit' to quit the program.")

while True:
    user_input = input("\nEnter your question about the iris dataset: ")
    
    if user_input.lower() == 'exit':
        print("Thank you for using the Interactive Iris Dataset Query System. Goodbye!")
        break
    
    try:
        response = agent.invoke(user_input)
        print("\nAnswer:", response['output'])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Full error details:")
        import traceback
        traceback.print_exc()