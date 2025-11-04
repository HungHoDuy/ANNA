import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
class LLM_initializer():
    def __init__(self):
        load_dotenv()
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=100,
            max_retries=2,
        )



