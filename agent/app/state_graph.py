# state_graph.py

from typing import Dict, List, Tuple
from typing_extensions import TypedDict

import json
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .services.llm_openai import OpenAIChat


# =========================
# STATE
# =========================
class State(TypedDict, total=False):
    messages: List[dict]
    personas: Dict[str, List[dict]]   # optional; per-persona threads (committee)
    phase: str                        # mentor | pm | cto | vc | committee


graph_builder = StateGraph(State)
llm = OpenAIChat()


# =========================
# Prompt loading utilities
# =========================
def _prompts_dir() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here / "prompts", here.parent / "prompts", Path.cwd() / "prompts"]:
        if p.exists() and p.is_dir():
            return p
    return here / "prompts"  # fallback


def _read_prompt_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Prompt file must be a JSON object.")
    persona = str(data.get("persona", "")).strip()
    style = str(data.get("style", "")).strip()
    domain = str(data.get("domain", "")).strip()
    lines = data.get("prompt", [])
    if isinstance(lines, list):
        lines = [str(x).strip() for x in lines if str(x).strip()]
    elif isinstance(lines, str):
        lines = [lines.strip()]
    else:
        lines = []
    return {"persona": persona, "style": style, "domain": domain, "lines": lines}


def _compose_system(spec: dict) -> str:
    head = []
    if spec.get("persona"):
        head.append(f"You are {spec['persona']}.")
    if spec.get("style"):
        head.append(f"Style: {spec['style']}.")
    if spec.get("domain"):
        head.append(f"Domain: {spec['domain']}.")
    rules = "\n".join(f"- {line}" for line in spec.get("lines", []))
    parts = [(" ".join(head)).strip()]
    if rules:
        parts.append("Behavioral guidelines:\n" + rules)
    return "\n\n".join(p for p in parts if p).strip()


def _load_role_specs() -> Dict[str, str]:
    """Return composed system prompts for MENTOR, PM, CTO, VC with fallbacks."""
    prompts = {}
    dirp = _prompts_dir()

    defaults = {
        "MENTOR": {
            "persona": "AI Mentor",
            "style": "visionary, provocative, focused on innovation and design",
            "domain": "startup mentorship, entrepreneurship, product development",
            "lines": [
                "You are an AI Mentor inspired at Steve Jobs, acting as a mentor for startup founders.",
                "Speak with sharp insight, challenge assumptions, and push people to think bigger.",
                "Always focus on product excellence, user experience, innovation, and building impactful companies.",
                "Keep answers short (2–4 sentences), practical, and inspiring.",
                "Prefer natural conversation: share one point, then ask a follow-up question to gather context.",
                "Do not drift into unrelated topics.",
                "Avoid long encyclopedic responses or lengthy bullet lists unless explicitly asked.",
            ],
        },
        "PM": {
            "persona": "Product Manager (PM)",
            "style": "direct, pragmatic, outcome-driven",
            "domain": "product strategy, discovery, prioritization, roadmap, validation",
            "lines": [
                "Act like a seasoned PM who prioritizes user value and measurable outcomes.",
                "Emphasize discovery, lean validation, MVP slicing, and clear success metrics.",
                "Translate ambiguous ideas into hypotheses, experiments, and iteration plans.",
                "Keep answers concise (2–4 sentences) and end with a clarifying question when needed.",
            ],
        },
        "CTO": {
            "persona": "Chief Technology Officer (CTO)",
            "style": "pragmatic, architecture-first, security- and cost-aware",
            "domain": "systems design, scalability, data pipelines, infra, security, build-vs-buy",
            "lines": [
                "Focus on practical architectures, data flows, and operational concerns.",
                "Call out trade-offs, risks, costs, and security implications explicitly.",
                "Prefer simple, reliable solutions before complex ones; note migration paths.",
                "Keep answers concise (2–4 sentences) and propose next technical steps.",
            ],
        },
        "VC": {
            "persona": "Venture Capital Partner (VC)",
            "style": "sharp, thesis-driven, risk-aware",
            "domain": "market sizing, defensibility, unit economics, traction, GTM",
            "lines": [
                "Evaluate markets, moats, economics, and evidence with discipline.",
                "Highlight fatal risks early; suggest ways to de-risk with minimal cost.",
                "Ask for the few metrics that truly matter at this stage.",
                "Keep answers concise (2–4 sentences) and end with a funding-related checkpoint.",
            ],
        },
    }

    for name in ("MENTOR", "PM", "CTO", "VC"):
        fp = dirp / f"{name}.json"
        try:
            if fp.exists():
                spec = _read_prompt_json(fp)
                prompts[name] = _compose_system(spec)
            else:
                prompts[name] = _compose_system(defaults[name])
        except Exception as e:
            print(f"[prompts] Using fallback for {name}: {e}")
            prompts[name] = _compose_system(defaults[name])

    return prompts


ROLE_SYSTEM: Dict[str, str] = _load_role_specs()


# =========================
# Router helpers
# =========================
LABELS = ("MENTOR", "PM", "CTO", "VC", "COMMITTEE")

_ROUTER_SYSTEM = (
    "You are a strict router.\n"
    "Look ONLY at the latest user message and decide the addressee.\n"
    "Output EXACTLY ONE label (uppercase, no punctuation):\n"
    "MENTOR | PM | CTO | VC | COMMITTEE\n\n"
    "Rules:\n"
    "- If the message asks for 'committee', 'panel', 'all of you', 'VC+PM+CTO', or similar, choose COMMITTEE.\n"
    "- If it explicitly or implicitly calls out a role (pm, vc, cto, product manager, venture capitalist), choose that role.\n"
    "- Otherwise choose MENTOR."
)

def _last_user(messages: List[dict]) -> dict | None:
    return next((m for m in reversed(messages) if m.get("role") == "user"), None)

def _committee_trigger(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\bcommittee\b|\bpanel\b|all of you|vc\W*pm\W*cto|pm\W*cto\W*vc", t))

def _heuristic_label(text: str) -> str:
    t = (text or "").lower()
    if _committee_trigger(t): return "COMMITTEE"
    if re.search(r"\bcto\b|chief technology officer|@cto", t): return "CTO"
    if re.search(r"\bpm\b|product manager|@pm", t): return "PM"
    if re.search(r"\bvc\b|investor|venture capital(ist)?|@vc", t): return "VC"
    return "MENTOR"

async def _classify(messages: List[dict]) -> str:
    last = _last_user(messages)
    if not last:
        return "MENTOR"
    text = str(last.get("content", ""))

    # Hard committee trigger
    if _committee_trigger(text):
        return "COMMITTEE"

    # Ask LLM to classify
    try:
        router_msgs = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": text},
        ]
        raw = (await llm.complete(router_msgs) or "").strip().upper()
        label = re.sub(r"[^A-Z]", "", raw)
        if label not in LABELS:
            raise ValueError(f"Unknown label: {raw!r} → {label!r}")
        return label
    except Exception as e:
        print(f"[router] LLM classify failed, using heuristic: {e}")
        return _heuristic_label(text)


# =========================
# Seeding / labeling helpers
# =========================
def _ensure_seed(history: List[dict], system_prompt: str) -> List[dict]:
    """Replace any old system message with the correct persona prompt."""
    new_history = [m for m in history if m.get("role") != "system"]
    return [{"role": "system", "content": system_prompt}] + new_history


LABEL_PREFIX = {
    "MENTOR": "**Mentor:** ",
    "PM":     "**PM:** ",
    "CTO":    "**CTO:** ",
    "VC":     "**VC:** ",
}

def _apply_label(label: str, reply: str) -> str:
    """
    Add a role prefix exactly once.
    If the model already started with the role name (with/without bold and colon),
    we don't add another prefix.
    """
    reply_clean = reply.strip()
    role = label.lower()
    # Matches: "mentor:", "**Mentor:**", " VC : " etc., case-insensitive
    pattern = rf"^\s*(?:\*\*\s*)?{role}\s*:\s*"
    if re.match(pattern, reply_clean, flags=re.IGNORECASE):
        return reply_clean
    return f"{LABEL_PREFIX[label]}{reply_clean}"


async def _run_persona(name: str, history: List[dict], user_msg: dict) -> Tuple[List[dict], str]:
    system_text = ROLE_SYSTEM[name]
    seeded = _ensure_seed(history, system_text) + [user_msg]
    try:
        reply_text = await llm.complete(seeded)
        reply_text = reply_text or "(No response generated)"
    except Exception as e:
        reply_text = f"(Error calling LLM for {name}: {e})"
        print(f"LLM ERROR ({name}):", e)
    return seeded + [{"role": "assistant", "content": reply_text}], reply_text


# =========================
# Nodes
# =========================
async def router(state: State):
    label = await _classify(state.get("messages", []))
    phase_map = {
        "MENTOR": "mentor",
        "PM": "pm",
        "CTO": "cto",
        "VC": "vc",
        "COMMITTEE": "committee",
    }
    return {"phase": phase_map.get(label, "mentor")}


async def mentor_node(state: State):
    seeded = _ensure_seed(state["messages"], ROLE_SYSTEM["MENTOR"])
    try:
        reply = await llm.complete(seeded)
        reply = reply or "(No response generated)"
    except Exception as e:
        reply = f"(Error calling LLM: {e})"
        print("LLM ERROR (MENTOR):", e)
    return {
        "messages": [{"role": "assistant", "content": _apply_label("MENTOR", reply)}],
        "phase": "mentor",
    }


async def pm_node(state: State):
    seeded = _ensure_seed(state["messages"], ROLE_SYSTEM["PM"])
    try:
        reply = await llm.complete(seeded)
        reply = reply or "(No response generated)"
    except Exception as e:
        reply = f"(Error calling LLM: {e})"
        print("LLM ERROR (PM):", e)
    personas = dict(state.get("personas", {}))
    personas["PM"] = personas.get("PM", []) + seeded[-2:]  # last user + this assistant
    return {
        "messages": [{"role": "assistant", "content": _apply_label("PM", reply)}],
        "personas": personas,
        "phase": "pm",
    }


async def cto_node(state: State):
    seeded = _ensure_seed(state["messages"], ROLE_SYSTEM["CTO"])
    try:
        reply = await llm.complete(seeded)
        reply = reply or "(No response generated)"
    except Exception as e:
        reply = f"(Error calling LLM: {e})"
        print("LLM ERROR (CTO):", e)
    personas = dict(state.get("personas", {}))
    personas["CTO"] = personas.get("CTO", []) + seeded[-2:]
    return {
        "messages": [{"role": "assistant", "content": _apply_label("CTO", reply)}],
        "personas": personas,
        "phase": "cto",
    }


async def vc_node(state: State):
    seeded = _ensure_seed(state["messages"], ROLE_SYSTEM["VC"])
    try:
        reply = await llm.complete(seeded)
        reply = reply or "(No response generated)"
    except Exception as e:
        reply = f"(Error calling LLM: {e})"
        print("LLM ERROR (VC):", e)
    personas = dict(state.get("personas", {}))
    personas["VC"] = personas.get("VC", []) + seeded[-2:]
    return {
        "messages": [{"role": "assistant", "content": _apply_label("VC", reply)}],
        "personas": personas,
        "phase": "vc",
    }


async def committee_node(state: State):
    personas_state = dict(state.get("personas", {}))
    last_user = _last_user(state.get("messages", []))
    if last_user is None:
        return await mentor_node(state)

    all_replies = {}
    for name in ("PM", "CTO", "VC"):
        history = personas_state.get(name, [])
        new_hist, text = await _run_persona(name, history, last_user)
        personas_state[name] = new_hist
        all_replies[name] = text

    combined = (
        "🧑‍⚖️ **Committee Response**\n\n"
        f"**PM:** {_apply_label('PM', all_replies['PM'])}\n\n"
        f"**CTO:** {_apply_label('CTO', all_replies['CTO'])}\n\n"
        f"**VC:** {_apply_label('VC', all_replies['VC'])}\n\n"
        "If you want a single, consolidated recommendation, say: `make a final decision`."
    )

    return {
        "messages": [{"role": "assistant", "content": combined}],
        "personas": personas_state,
        "phase": "committee",
    }


# =========================
# Graph wiring
# =========================
graph_builder.add_node("router", router)
graph_builder.add_node("mentor", mentor_node)
graph_builder.add_node("pm", pm_node)
graph_builder.add_node("cto", cto_node)
graph_builder.add_node("vc", vc_node)
graph_builder.add_node("committee", committee_node)

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges(
    "router",
    lambda s: s.get("phase", "mentor"),
    {
        "mentor": "mentor",
        "pm": "pm",
        "cto": "cto",
        "vc": "vc",
        "committee": "committee",
    },
)
graph_builder.add_edge("mentor", END)
graph_builder.add_edge("pm", END)
graph_builder.add_edge("cto", END)
graph_builder.add_edge("vc", END)
graph_builder.add_edge("committee", END)

graph = graph_builder.compile()
