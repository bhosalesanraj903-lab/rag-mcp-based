from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Initialize the embedding model used to embed user queries
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                               api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = FAISS.load_local("faiss_index", 
                               embeddings, 
                               allow_dangerous_deserialization=True)

mcp = FastMCP(name="RAG server",
              instructions="This server provides data retreival tool and addition tool."
              "Call add() for performing summation of two numbers."
              "Call get_data() for performing agentic rag.")

@mcp.tool(name="Addition")
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b

@mcp.tool(name="RAG")
def get_data(query: str):
    """
    Retrieve relevant documents from the FAISS vector store
    based on semantic similarity with the query.
    """
    return vectorstore.similarity_search(query, k=2)

@mcp.prompt(name="system prompt")
def get_system_prompt():
    """Get the system prompt."""
    return f"You are an helpful AI assistant."

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)