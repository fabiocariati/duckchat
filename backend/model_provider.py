import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import ollama


load_dotenv()


class ModelProviderController:
    PROVIDERS = {
        "ollama": {
            "models": [m.model for m in ollama.list().models],
            "chat_class": ChatOllama,
            "params": {"base_url": os.getenv("OLLAMA_HOST")},
        },
        "openai": {
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo" ],
            "chat_class": ChatOpenAI,
            "params": {},
        },
    }

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(self):
        self.provider = "ollama"

    def add_parameter(self, key, value):
        self.PROVIDERS[self.provider]["params"][key] = value

    def get_providers(self):
        return list(self.PROVIDERS.keys())

    def get_chat(self):
        provider = self.PROVIDERS[self.provider]
        if provider == "openai" and not provider["params"].get("api_key"):
            raise ValueError("API key is required for OpenAI provider. Please set it.")
        return provider["chat_class"](**provider["params"], temperature=0.1)
    
    def get_models(self):
        return self.PROVIDERS[self.provider]["models"]
