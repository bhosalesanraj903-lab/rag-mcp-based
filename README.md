# MCP-based RAG System with LangChain & FAISS

This repository implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, **FAISS**, and **FastMCP**, exposed through a **Model Context Protocol (MCP)** server and consumed by an agent-based client.

The project demonstrates how to:

* Preprocess documents into vector embeddings
* Store and retrieve them using FAISS
* Expose retrieval as MCP tools
* Use an LLM agent to interact with those tools

---

## Project Structure

```
.
├── document_preprocess.py   # PDF ingestion + FAISS index creation
├── mcp_server.py            # FastMCP server exposing RAG + utility tools
├── mcp_client.py            # Agent client that consumes MCP tools
├── requirements.txt         # Python dependencies
├── .env_example.txt         # Environment variable template
└── README.md
```

---

## Architecture Overview

1. **Document Preprocessing**

   * PDFs are loaded and split into chunks
   * Each chunk is embedded using OpenAI embeddings
   * Vectors are stored in a FAISS index on disk

2. **MCP Server**

   * Loads the FAISS index
   * Exposes:

     * A **RAG tool** for semantic search
     * A simple **Addition tool** (demo utility)
   * Serves tools over HTTP using FastMCP

3. **MCP Client**

   * Connects to the MCP server
   * Retrieves tools and system prompt
   * Runs a LangChain agent backed by an OpenAI chat model
   * Accepts interactive user queries via terminal

---

## Requirements

* Python **3.9+**
* An **OpenAI API key**

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file based on the example:

```bash
cp .env_example.txt .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Step 1: Build the FAISS Vector Index

Run the document preprocessing script:

```bash
python document_preprocess.py
```

What this does:

* Loads `sample_file.pdf`
* Splits text into overlapping chunks
* Generates embeddings using `text-embedding-3-small`
* Saves the FAISS index locally in `faiss_index/`

> ⚠️ Ensure `sample_file.pdf` exists in the project root before running.

---

## Step 2: Start the MCP Server

Launch the MCP server:

```bash
python mcp_server.py
```

The server:

* Loads the FAISS index
* Starts an HTTP MCP endpoint at:

  ```
  http://localhost:8000/mcp
  ```

### Exposed Tools

| Tool Name  | Description                                      |
| ---------- | ------------------------------------------------ |
| `Addition` | Returns the sum of two integers                  |
| `RAG`      | Performs semantic document retrieval using FAISS |

---

## Step 3: Run the MCP Client (Agent)

In a new terminal:

```bash
python mcp_client.py
```

You can now interactively query the agent:

```text
User input: What is this document about?
User input: Add 5 and 7
User input: q
```

The agent:

* Decides when to call MCP tools
* Uses retrieved documents to answer questions (RAG)
* Responds in natural language

---

## Key Technologies

* **LangChain** – Agent orchestration and document processing
* **FAISS** – Vector similarity search
* **FastMCP** – Lightweight MCP server implementation
* **OpenAI Embeddings & Chat Models**
* **Python AsyncIO** – Non-blocking agent execution

---

## Security Notes

* Never commit your `.env` file
* The FAISS index is loaded with `allow_dangerous_deserialization=True`; only load trusted indexes

---

## Possible Extensions

* Add support for multiple documents or file types
* Persist vector stores per user or project
* Add authentication to MCP endpoints
* Replace OpenAI with local or open-source embedding models
* Add streaming responses to the client



