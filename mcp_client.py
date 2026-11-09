from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the chat-based LLM that will act as the agent's brain
llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=os.getenv("OPENAI_API_KEY"))

async def main():
    client = MultiServerMCPClient({ "RAG server":
                                {"transport": "http",
                                 "url":"http://localhost:8000/mcp"}
                                })
    tools = await client.get_tools()
    prompt = await client.get_prompt(server_name="RAG server", prompt_name="system prompt")
    
    agent = create_agent(llm,
                         tools,
                         system_prompt=prompt[0].content)
    while True:
        # Read user input from the terminal
        user_input = input("User input: ")
        if not user_input or user_input=="q":
            break

        # Invoke the agent with a chat-style message payload
        response = await agent.ainvoke({'messages':[{"role":"user", "content":user_input}]})

        # Print the final assistant response
        print(response["messages"][-1].content)

if __name__=="__main__":
    asyncio.run(main())