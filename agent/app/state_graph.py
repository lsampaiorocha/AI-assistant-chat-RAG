# agent/app/state_graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Our conversation state is a plain dict
State = Dict[str, Any]

# -------------------- Nodes --------------------

def start(state: State) -> State:
    """No-op node used only to choose the first real node via router."""
    # Ensure counters exist
    return {
        **state,
        "intro_done": state.get("intro_done", False),
        "tests_done": state.get("tests_done", 0),
        "general_done": state.get("general_done", 0),
    }

def introduction(state: State) -> State:
    """Step 1: ask about user's experience (runs once)."""
    return {
        **state,
        "messages": state["messages"] + [
            {
                "role": "assistant",
                "content": (
                    "Before we begin, could you tell me a bit about your current "
                    "experience and knowledge in AI engineering?"
                ),
            }
        ],
        "intro_done": True,
    }

def testing(state: State) -> State:
    """Step 2: ask 3 knowledge questions."""
    questions = [
        "Can you explain what embeddings are used for in AI?",
        "What’s the difference between supervised and unsupervised learning?",
        "How does a vector database like Chroma or FAISS help in a RAG pipeline?",
    ]
    done = min(state.get("tests_done", 0), len(questions))
    if done < len(questions):
        q = questions[done]
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": q}],
            "tests_done": done + 1,
        }
    return state  # safety

def exploration(state: State) -> State:
    """Step 3: ask 4 general questions."""
    questions = [
        "What do you think is the biggest challenge in applying AI in real-world systems?",
        "Can you think of a good example of when NOT to use deep learning?",
        "Why is evaluation important in AI systems?",
        "How do you see the role of AI evolving in the next 5 years?",
    ]
    done = min(state.get("general_done", 0), len(questions))
    if done < len(questions):
        q = questions[done]
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": q}],
            "general_done": done + 1,
        }
    return state  # safety

def feedback(state: State) -> State:
    """Step 4: finish with feedback."""
    return {
        **state,
        "messages": state["messages"] + [
            {
                "role": "assistant",
                "content": (
                    "Great work today! Based on this short interview, I recommend reviewing "
                    "some fundamentals of machine learning, vector databases, and evaluation "
                    "methods. Keep practicing, and I’ll be ready to test you again in the next session!"
                ),
            }
        ],
        "finished": True,
    }

# -------------------- Router --------------------

def route(state: State) -> str:
    """
    Decide the next node based on durable flags/counters in state.
    Return one of: 'introduction' | 'testing' | 'exploration' | 'feedback'
    """
    intro_done   = state.get("intro_done", False)
    tests_done   = state.get("tests_done", 0)
    general_done = state.get("general_done", 0)

    if not intro_done:
        return "introduction"
    if tests_done < 3:
        return "testing"
    if general_done < 4:
        return "exploration"
    return "feedback"

# -------------------- Build graph --------------------

graph = StateGraph(state_schema=State)

graph.add_node("start", start)
graph.add_node("introduction", introduction)
graph.add_node("testing", testing)
graph.add_node("exploration", exploration)
graph.add_node("feedback", feedback)

# Entry point must be a node name, so we land on 'start' and immediately route
graph.set_entry_point("start")

# Conditional routing from each step
graph.add_conditional_edges(
    "start",
    route,
    {
        "introduction": "introduction",
        "testing": "testing",
        "exploration": "exploration",
        "feedback": "feedback",
    },
)

graph.add_conditional_edges(
    "introduction",
    route,
    {
        "testing": "testing",
        "exploration": "exploration",   # (just in case counters were pre-set)
        "feedback": "feedback",
        "introduction": "introduction", # safety (shouldn't loop if intro_done=True)
    },
)

graph.add_conditional_edges(
    "testing",
    route,
    {
        "testing": "testing",
        "exploration": "exploration",
        "feedback": "feedback",
    },
)

graph.add_conditional_edges(
    "exploration",
    route,
    {
        "exploration": "exploration",
        "feedback": "feedback",
    },
)

# Feedback is terminal
graph.add_edge("feedback", END)

# Pause AFTER each node so only one step runs per request.
memory = MemorySaver()
conversation_app = graph.compile(
    checkpointer=memory,
    interrupt_after=["introduction", "testing", "exploration", "feedback"],
)