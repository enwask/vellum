import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI

from vellum.utils import config

load_dotenv(find_dotenv())
OPENAI_ENDPOINT = os.environ['OPENAI_ENDPOINT']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']

chat_model = AzureChatOpenAI(
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
    temperature=0,
    model=config.chat_model,
)
