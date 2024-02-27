import os
import time

from .BaseLLM import BaseLLM
from openai import OpenAI

class MoonshotAPI(BaseLLM):
    def __init__(self, model="moonshot-v1-8k", base_url="https://api.moonshot.cn/v1", verbose=False):
        super(MoonshotAPI, self).__init__()
        
        self.api_key = os.environ['moonshot_key']
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        self.verbose = verbose
        self.model_name = model
        self.messages = []
        
        if self.verbose:
            print('Model name:', self.model_name)
            if len(self.api_key) > 8:
                print('Found API key', self.api_key[:4], '****', self.api_key[-4:])
            else:
                print('Found API key but too short.')
    
    def initialize_message(self):
        self.messages = []
    
    def ai_message(self, payload):
        if len(self.messages) == 0:
            self.system_message("请根据我的要求进行角色扮演:")
        elif len(self.messages) % 2 == 1:
            self.messages.append({"role":"system","content":payload})
        elif len(self.messages)% 2 == 0:
            self.messages[-1]["content"] += "\n"+ payload
    
    def system_message(self, payload):
        self.messages.append({"role": "system", "content": payload})
    
    def user_message(self, payload):
        self.messages.append({"role": "user", "content": payload})
    
    def get_response(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            ai_response = response.choices[0].message.content
            if self.verbose:
                print(f'Response: {ai_response}')
            return ai_response
        except Exception as e:
            if self.verbose:
                print(f'Error getting response from Moonshot: {e}')
            return ''

    def print_prompt(self):
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
