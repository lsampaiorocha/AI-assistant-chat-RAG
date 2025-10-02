from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .services.llm_openai import OpenAIChat


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = OpenAIChat()

async def chatbot(state: State):
    fixed_messages = []
    for m in state["messages"]:
        if isinstance(m, dict):
            fixed_messages.append({
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            })
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", str(m))
            fixed_messages.append({"role": role, "content": content})

    reply = await llm.complete(fixed_messages)
    return {"messages": [{"role": "assistant", "content": reply}]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(interrupt_after=["chatbot"])
