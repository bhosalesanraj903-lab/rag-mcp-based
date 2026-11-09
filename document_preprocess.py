from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

# Load the PDF file and convert it into a list of Document objects
documents = PyPDFLoader("sample_file.pdf").load() # ← add filename

# Create a text splitter:
# - chunk_size=1000 characters per chunk
# - chunk_overlap=100 characters shared between chunks (to preserve context)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Split the PDF documents into smaller text chunks
split_docs = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings model
# This model converts text into numerical vectors
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create a FAISS index using L2 (Euclidean) distance
# The dimension of the index must match the embedding vector size
# We get it by embedding a sample query ("Hello World")
index = faiss.IndexFlatL2(
    len(embeddings.embed_query("Hello World"))
)

# Create a FAISS vector store:
# - embeddings: how text is converted to vectors
# - index: FAISS similarity search index
# - docstore: stores original documents in memory
# - index_to_docstore_id: maps FAISS vectors to document IDs
vectorstore = FAISS(
    embeddings,
    index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add the split document chunks to the vector store
# Each chunk is embedded and indexed for similarity search
vectorstore.add_documents(split_docs)

# Save the FAISS index and metadata locally to disk
# This allows reuse without re-embedding everything
vectorstore.save_local("faiss_index")