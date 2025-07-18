import os
from dotenv import find_dotenv, load_dotenv
from langchain_ollama import ChatOllama

from vellum.utils import config


load_dotenv(find_dotenv())
OLLAMA_URL = os.environ['OLLAMA_URL']
OLLAMA_API_KEY = os.environ['OLLAMA_API_KEY']

chat_model = ChatOllama(
    base_url=OLLAMA_URL,
    client_kwargs={
        'headers': {
            'Authorization': f'Bearer {OLLAMA_API_KEY}',
            'Content-Type': 'application/json',
        },
    },
    model=config.chat_model,
)
