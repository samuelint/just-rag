from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai_llm = ChatOpenAI(model="gpt-3.5-turbo")
