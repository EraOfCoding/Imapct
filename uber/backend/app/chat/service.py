from pydantic import BaseSettings

from app.config import database

from .adapters.llm_service import LLMService


class Config(BaseSettings):
    HERE_API_KEY: str


class Service:
    def __init__(self):
        self.llm_service = LLMService()


def get_service():
    svc = Service()
    return svc