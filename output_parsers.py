"""
Output Parsers and Structured Output in LangChain V.1
"""

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def demo_str_parser():
    """Basic string output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Give me a one-word answer: What color is the sky?"
    )

    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({})

    print(f"Result: '{result}' (type: {type(result).__name__})")

def demo_json_parser():
    """JSON output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )

    parser = JsonOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"place": "Taj Mahal"})
    print(f"Result: {result}")
    print(f"City: {result['city']}, Country: {result['country']}")


def demo_pydantic_parser():
    """Pydantic output parser for type-safe structured data."""

    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")
        occupation: str = Field(description="The person's occupation")

    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    result = chain.invoke({"description": "A 30-year-old artist named Maria"})

    print(result) 


def demo_latest_pydantic_parser():
    # Structured Output
    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        review: str = Field(description="A brief review of the movie")
        rating: int = Field(description="The rating of the movie out of 10")

    # Bind the schema to the model
    structured_model = model.with_structured_output(MovieReview)

    result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
    print(result)
   

if __name__ == "__main__":
    print("=" * 50)
    print("Demo 1: String Parser")
    print("=" * 50)
    demo_str_parser()

    print("\n" + "=" * 50)
    print("Demo 2: JSON Parser")
    print("=" * 50)
    demo_json_parser()

    print("\n" + "=" * 50)
    print("Demo 3: Pydantic Parser")
    print("=" * 50)
    demo_pydantic_parser()


    print("\n" + "=" * 50)
    print("Demo 4: Latest Pydantic Parser")
    print("=" * 50)
    demo_latest_pydantic_parser()
