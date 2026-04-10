from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from importlib.metadata import version


load_dotenv()

# langchain_version = version("dotenv")
# print(f"Dotenv version: {langchain_version}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("Hello from langchain!")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    response = llm.invoke("What is the capital of India?")
    print(response.content)
    total_tokens = response.response_metadata["token_usage"]["total_tokens"]
    print(f"Total tokens: {total_tokens}")
    


if __name__ == "__main__":
    main()
