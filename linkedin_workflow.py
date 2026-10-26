
import json
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# Fail early when required credentials are missing.
required_keys = ["CF_ACCOUNT_ID", "CF_AI_API_TOKEN",
                 "GROQ_API_KEY", "TAVILY_API_KEY"]
missing = [key for key in required_keys if not os.getenv(key)]
if missing:
    raise RuntimeError("Missing environment variables: " + ", ".join(missing))

# Tools and models: retain the existing integrations.
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

writer_llm = ChatCloudflareWorkersAI(
    model_name="@cf/mistralai/mistral-small-3.1-24b-instruct",
    temperature=0.7,
    max_tokens=800,
)
writer_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
)

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    reviewer_feedback: str
    is_approved: bool
    attempts: int
    scorecard: dict

WRITER_SYSTEM_PROMPT = """
You are a skilled LinkedIn content writer.

Write a complete, publish-ready LinkedIn post about the supplied topic.
Never ask questions. Never reveal internal reasoning, plans, or search notes.
Use web search when current facts are needed. Never invent statistics or claims.
Start with a strong hook. Give one valuable takeaway.
Use short paragraphs and a natural, professional, human tone.
Target 150–200 words. End with a relevant question or CTA.
Do not use hashtags. Return only the post text.

When revising, address the reviewer's feedback and fix specific issues.
"""

REVIEWER_SYSTEM_PROMPT = """
You are a strict LinkedIn content reviewer. Review the draft against these
seven criteria:
1. Strong hook in the first line
2. One clear, valuable takeaway
3. Short, skimmable paragraphs
4. Roughly 150–200 words
5. Ends with an engaging question or CTA
6. Professional, human tone
7. No hashtags

Return ONLY a valid JSON object in this format:
{
  "verdict": "APPROVED" or "REJECTED",
  "scores": {
    "Strong hook": "PASS" or "NEEDS WORK",
    "Clear takeaway": "PASS" or "NEEDS WORK",
    "Skimmable paragraphs": "PASS" or "NEEDS WORK",
    "Word count": "PASS" or "NEEDS WORK",
    "Ending CTA": "PASS" or "NEEDS WORK",
    "Human tone": "PASS" or "NEEDS WORK",
    "No hashtags": "PASS" or "NEEDS WORK"
  },
  "feedback": "One short, actionable paragraph"
}

Be strict. Approve only if all criteria are met. Check the actual word count.
Do not approve merely because the writing sounds polished.
"""

def content_to_text(content) -> str:
    """Normalize string or block-based model output."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content)

def writer_node(state: State) -> dict:
    messages = state.get("messages", [])
    last = messages[-1] if messages else None

    # A tool response means the current draft is still in progress.
    # Continue with the actual tool context; don't increment attempts.
    if isinstance(last, ToolMessage):
        response = writer_with_tools.invoke(
            [("system", WRITER_SYSTEM_PROMPT), *messages]
        )
        return {"messages": [response]}

    # A new attempt starts only on initial drafting or revision.
    attempt = state.get("attempts", 0) + 1
    topic = state["topic"]

    if attempt == 1:
        instruction = (
            f"Write a LinkedIn post about: {topic}. "
            "Use web search only if needed for current facts."
        )
    else:
        instruction = (
            f"Revise the LinkedIn post about: {topic}\n\n"
            f"Reviewer feedback:\n{state.get('reviewer_feedback', '')}\n\n"
            "Fix the issues and return a complete replacement post."
        )

    response = writer_with_tools.invoke([
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", instruction),
    ])
    return {
        "messages": [("human", instruction), response],
        "attempts": attempt,
    }

tool_node = ToolNode(tools)

def should_use_tool(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "extract_draft"

def extract_draft_node(state: State) -> dict:
    last = state["messages"][-1]
    draft = content_to_text(last.content).strip()
    return {"draft": draft}

def reviewer_node(state: State) -> dict:
    prompt = (
        "Evaluate this complete LinkedIn post. Return the requested JSON only.\n\n"
        f"TOPIC: {state['topic']}\n\n"
        f"DRAFT:\n{state['draft']}"
    )
    response = reviewer_llm.invoke([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", prompt),
    ])
    raw = content_to_text(response.content).strip()

    # Be tolerant of JSON fenced by a model.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)

    try:
        result = json.loads(raw)
        verdict = str(result.get("verdict", "")).upper()
        scores = result.get("scores", {})
        feedback = str(result.get("feedback", "")).strip()
        approved = verdict == "APPROVED"
        if verdict not in {"APPROVED", "REJECTED"}:
            raise ValueError("Unexpected verdict")
    except (ValueError, TypeError, json.JSONDecodeError):
        # Fail closed: malformed review output must not approve a draft.
        approved = False
        scores = {}
        feedback = "The reviewer returned an invalid response. Please retry."
    
    return {
        "reviewer_feedback": feedback,
        "is_approved": approved,
        "scorecard": scores,
    }

def should_stop_looping(state: State):
    if state.get("is_approved", False):
        return END
    if state.get("attempts", 0) >= 3:
        return END
    return "writer"

graph = StateGraph(State)
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "writer")
graph.add_conditional_edges("writer", should_use_tool)
graph.add_edge("tools", "writer")
graph.add_edge("extract_draft", "reviewer")
graph.add_conditional_edges("reviewer", should_stop_looping)

app = graph.compile()

def initial_state(topic: str) -> State:
    return {
        "topic": topic,
        "messages": [],
        "draft": "",
        "reviewer_feedback": "",
        "is_approved": False,
        "attempts": 0,
        "scorecard": {},
    }
