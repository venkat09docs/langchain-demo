"""
Working with LLMs in LangChain V.1
Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

def demo_multi_models():
    prompt = "Explain recursion in one sentence."

    models = {
            "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
            ),
            "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        )
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")

def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content="You are a helpful receptionist. Always answer like a receptionist."),
        HumanMessage(content="What's the weather like today?")
    ]

    print("Using message objects:")
    print(f"Messages: {messages[0]} | {messages[1]}")

    response = model.invoke(messages)

    print(f"Response from the Pirate: {response.content}")


def demo_dynamic_messages():
    prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

    messages = prompt.format_messages(adjective="funny", topic="chickens")

    # print(messages)

    # multi-message templates
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "Translate the following text: {text}"),
        ]
    )

    messages = prompt.format_messages(
        input_language="English", output_language="Telugu", text="I love programming."
    )

    print(messages)

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(messages)
    print(response.content)


def demo_fewshot_prompt_template():
    """ Fewshot Prompt """

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    fewshot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(final_prompt.format_messages(input="tall"))

    print(response.content)

# Reusable components
# system_prompt = ChatPromptTemplate.from_messages([("system", "You are a {role}.")])

# user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

# Combine
# full_prompt = system_prompt + user_prompt


# fin = full_prompt.format_messages(role="helpful assistant", question="What is AI?")

# print(fin)

def demo_prompt_composition():
    """Compose prompts from reusable parts."""

    # Reusable system prompt
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )

    # Reusable task prompt
    task = ChatPromptTemplate.from_messages([("human", "{task}")])

    # Combine
    full_prompt = persona + task

    # Test different combinations
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    chain = full_prompt | model 

    # As a pirate
    response = chain.invoke(
        {
            "role": "pirate captain",
            "tone": "adventurous",
            "task": "Tell me about your ship",
        }
    )

    print(f"Pirate: {response.content[:100]}...")

    # As a scientist
    response = chain.invoke(
        {
            "role": "scientist",
            "tone": "precise and academic",
            "task": "Explain photosynthesis",
        }
    )

    print(f"\nScientist: {response.content[:100]}...")

def demo_messages_placeholder():
    """Use MessagesPlaceholder for dynamic conversation history."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Simulate conversation history
    history = [
        HumanMessage(content="My name is Venkat"),
        AIMessage(content="Nice to meet you, Venkat!"),
    ]

    messages = prompt.format_messages(history=history, question="What's my name?")

    print("With history placeholder:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content[:50]}...")

    # Execute
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | model
    response = chain.invoke({"history": history, "question": "What's my name?"})
    print(f"\nResponse: {response.content}")




if __name__ == "__main__":
    # demo_multi_models()
    # demo_message()
    # demo_dynamic_messages()
    # demo_fewshot_prompt_template()

    # demo_prompt_composition()
    demo_messages_placeholder()





