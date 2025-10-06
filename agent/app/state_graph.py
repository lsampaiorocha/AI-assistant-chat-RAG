from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage

from .services.llm_openai import OpenAIChat


class State(TypedDict):
    # This reducer appends any messages you return under the "messages" key
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = OpenAIChat()


async def chatbot(state: State):
    # No normalization needed—your wrapper accepts LC messages or dicts
    try:
        reply = await llm.complete(state["messages"])
        if not reply:
            reply = "(No response generated)"
    except Exception as e:
        reply = f"(Error calling LLM: {e})"
        print("LLM ERROR:", e)

    # IMPORTANT: return ONLY the new message; add_messages will append it
    return {"messages": [AIMessage(content=reply)]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(interrupt_after=["chatbot"])
