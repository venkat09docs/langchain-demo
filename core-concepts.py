"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model


load_dotenv()

def demo_basic_components():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Combine components into a chain
    chain = prompt | model | parser

    # Component 3: Run the chain
    response = chain.invoke({"country": "France"})
    print(response)

def demo_basic_batch():
    """Demonstrates a basic batch using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Combine components into a batch
    chain = prompt | model | parser

    # Component 3: Run the batch
    # Batch - run with multiple inputs
    inputs = [
        {"country": "India"},
        {"country": "France"},
        {"country": "United States"},
        {"country": "Germany"},
        {"country": "Japan"},        
    ]

    responses = chain.batch(inputs)

    for response in responses:
        print(response)


def demo_streaming():
    """Demonstrates streaming using LCEL and Runnables."""

     # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("Explain the {concept} in 500 words")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Component 2: Combine components into a chain
    chain = prompt | model | parser

    # Component 3: Run the chain with streaming
    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"concept": "Langchain"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming

def demo_init_chat():
    # the univeral way to initialize a mode
    prompt = ChatPromptTemplate.from_template("Explain the {concept} in 500 words")
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)

    parser = StrOutputParser()

    # Component 2: Combine components into a chain
    chain = prompt | model | parser

    # Component 3: Run the chain with streaming
    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"concept": "Langchain"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming



if __name__ == "__main__":
    #demo_basic_components()
    #demo_basic_batch()
    #demo_streaming()
    demo_init_chat()
