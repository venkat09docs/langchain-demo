# Model	                Dimensions	Cost per 1M tokens	    Best For
# text-embedding-3-small	1536	    $0.02	            General use
# text-embedding-3-large	3072	    $0.13	            High accuracy
# text-embedding-ada-002	1536	    $0.10	            Legacy

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# single text
text = "This is a sample text to be embedded."

embedding = embeddings.embed_query(text)

# print(f"Embedding for single text: {embedding}")

print(len(embedding))

# multiple texts
embeds = embeddings.embed_documents(
    ["This is the first document.", "This is the second document."]
)

# print(f"Embeddings for multiple texts: {embeds}")
print(f"Number of embeddings returned: {len(embeds)}")


"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions


# Ollama 
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama2-7b-embedding-q4_0")

"""

