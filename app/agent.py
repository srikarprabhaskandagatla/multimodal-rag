import json
import os
from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

from app.retriever import get_retriever
from app.cache import get_cached, set_cached

SYSTEM_PROMPT = """You are a multimodal retrieval assistant with access to a 
50,000-document corpus containing both text documents and images.

Your job is to:
1. Analyze the user's query to determine if it is text-only, image-only, or multimodal.
2. Call the appropriate retrieval tool (text_retriever_tool, image_retriever_tool, 
   or multimodal_retriever_tool).
3. Synthesize the retrieved results into a concise, grounded answer.
4. Always cite the doc_id and source of documents you reference.

ROUTING RULES:
- Text-only query (no image) → text_retriever_tool
- Image provided, no text → image_retriever_tool  
- Both text AND image provided → multimodal_retriever_tool
- Ambiguous → prefer multimodal_retriever_tool

Never hallucinate information not present in the retrieved documents."""

TOOLS = [text_retriever_tool, image_retriever_tool, multimodal_retriever_tool]


@tool
async def text_retriever_tool(query: str) -> str:
    """Retrieve documents from the corpus using a text query."""
    cached = await get_cached(text=query, has_image=False)
    if cached:
        return json.dumps(cached)

    retriever = get_retriever()
    results = await retriever.retrieve(text=query, top_k=5)
    await set_cached(text=query, has_image=False, results=results)
    return json.dumps(results, default=str)


@tool
async def image_retriever_tool(image_description: str) -> str:
    """Retrieve documents from the corpus using an image description."""
    cached = await get_cached(text=image_description, has_image=True)
    if cached:
        return json.dumps(cached)

    retriever = get_retriever()
    results = await retriever.retrieve(text=image_description, top_k=5)
    await set_cached(text=image_description, has_image=True, results=results)
    return json.dumps(results, default=str)


@tool
async def multimodal_retriever_tool(query: str) -> str:
    """Retrieve documents from the corpus using both text and image context."""
    cached = await get_cached(text=f"multimodal:{query}", has_image=True)
    if cached:
        return json.dumps(cached)

    retriever = get_retriever()
    results = await retriever.retrieve(text=query, top_k=5)
    await set_cached(text=f"multimodal:{query}", has_image=True, results=results)
    return json.dumps(results, default=str)

def build_agent() -> AgentExecutor:
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],   
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],       
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,        
        max_tokens=1000,
        request_timeout=30,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=3,    
        return_intermediate_steps=True,
    )


async def run_agent(query: str, chat_history: list | None = None) -> dict[str, Any]:
    agent_executor = build_agent()
    result = await agent_executor.ainvoke({
        "input": query,
        "chat_history": chat_history or [],
    })
    return {
        "answer": result["output"],
        "tool_calls": [
            {
                "tool": step[0].tool,
                "input": step[0].tool_input,
            }
            for step in result.get("intermediate_steps", [])
        ],
    }