from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime
from urllib.parse import urlsplit

import streamlit as st
from langchain_core.messages import ToolMessage

from linkedin_workflow import app, initial_state


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LinkedIn Content Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logger = logging.getLogger("linkedin_content_studio")

MAX_TOPIC_LENGTH = 2_000
MAX_LOG_LINES = 100
MAX_SOURCES = 30
KNOWN_NODES = {
    "writer": "Cloudflare writer",
    "tools": "Tavily search",
    "extract_draft": "Draft extraction",
    "reviewer": "Groq reviewer",
}
GRAPH_NODES = [
    ("writer", "Writer"),
    ("tools", "Tavily Search"),
    ("extract_draft", "Extract Draft"),
    ("reviewer", "Reviewer"),
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {
    "page": "home",
    "submitted_topic": "",
    "topic_input": "",
    "last_topic": "",
    "status": "idle",  # idle | running | success | error
    "error_message": "",
    "history": [],  # revisions for the current run
    "post_history": [],  # completed posts from this browser session
    "logs": [],
    "sources": [],
    "draft": "",
    "feedback": "",
    "scorecard": {},
    "approved": False,
    "attempts": 0,
    "completed_nodes": [],
    "failed_node": None,
    "run_started_at": "",
    "run_finished_at": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --canvas: #F4F7FB;
        --surface: #FFFFFF;
        --ink: #183153;
        --muted: #60728A;
        --border: #DCE5EF;
        --accent: #167D78;
        --accent-hover: #116762;
        --accent-soft: #E5F4F1;
        --dark-surface: #FFFFFF;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--ink);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #EAF5F3 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: clamp(24px, 4vw, 42px);
        margin-bottom: 22px;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 12px;
        font-weight: 750;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: var(--ink);
        font-size: clamp(30px, 4vw, 44px);
        font-weight: 760;
        line-height: 1.12;
        letter-spacing: -1.2px;
        margin: 0 0 14px 0;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 16px;
        line-height: 1.7;
        max-width: 720px;
        margin: 0;
    }

    .metric {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 88px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: var(--ink);
        font-size: 22px;
        font-weight: 750;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .draft-box {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        color: #243B53;
        line-height: 1.8;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 10px !important;
        min-height: 44px;
        font-weight: 600;
        transition: background-color 120ms ease, border-color 120ms ease;
    }

    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="stDownloadButton"] button {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid #C9D6E4 !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] button:hover {
        background: #EDF5F8 !important;
        border-color: #9EB4C8 !important;
        color: var(--ink) !important;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid #C9D6E4 !important;
        border-radius: 10px !important;
    }

    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #8191A5 !important;
        -webkit-text-fill-color: #8191A5 !important;
    }

    [data-testid="stLinkButton"] a {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid #C9D6E4 !important;
        border-radius: 10px !important;
    }

    [data-testid="stLinkButton"] a:hover {
        background: #EDF5F8 !important;
        color: var(--ink) !important;
        border-color: #9EB4C8 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero {
            padding: 22px;
            border-radius: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_event(message: str) -> None:
    """Append a bounded, timestamped execution event."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"{timestamp}  {message}")
    st.session_state.logs = st.session_state.logs[-MAX_LOG_LINES:]


def safe_http_url(value: str) -> bool:
    """Accept only absolute HTTP(S) URLs with a hostname."""
    try:
        parsed = urlsplit(value.strip())
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
    except (ValueError, AttributeError):
        return False


def extract_urls(value) -> list[str]:
    """Extract valid HTTP(S) source URLs from tool output without inventing links."""
    found: set[str] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key.lower() in {"url", "link", "source_url"} and isinstance(val, str):
                    candidate = val.strip()
                    if safe_http_url(candidate):
                        found.add(candidate)
                else:
                    walk(val)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            try:
                walk(json.loads(obj))
            except (ValueError, TypeError):
                # URLs in free-form tool text are best-effort only.
                candidates = re.findall(r'https?://[^\s<>"\'\]]+', obj)
                for candidate in candidates:
                    candidate = candidate.rstrip(".,;:!?)")
                    if safe_http_url(candidate):
                        found.add(candidate)

    walk(value)
    return sorted(found)


def start_generation(topic: str) -> bool:
    """Validate and initialize one generation run."""
    topic = topic.strip()
    if not topic:
        st.session_state.error_message = "Enter a topic before generating."
        return False
    if len(topic) > MAX_TOPIC_LENGTH:
        st.session_state.error_message = (
            f"Your topic is too long. Keep it under {MAX_TOPIC_LENGTH:,} characters."
        )
        return False

    st.session_state.submitted_topic = topic
    st.session_state.last_topic = topic
    st.session_state.error_message = ""
    st.session_state.status = "running"
    st.session_state.error_message = ""
    st.session_state.history = []
    st.session_state.logs = []
    st.session_state.sources = []
    st.session_state.draft = ""
    st.session_state.feedback = ""
    st.session_state.scorecard = {}
    st.session_state.approved = False
    st.session_state.attempts = 0
    st.session_state.completed_nodes = []
    st.session_state.failed_node = None
    st.session_state.run_started_at = datetime.now().isoformat(timespec="seconds")
    st.session_state.run_finished_at = ""
    log_event("Generation started.")
    return True


def reset_current_run() -> None:
    """Clear the active run while preserving completed posts in session history."""
    st.session_state.page = "home"
    st.session_state.submitted_topic = ""
    st.session_state.last_topic = ""
    st.session_state.topic_input = ""
    st.session_state.status = "idle"
    st.session_state.error_message = ""
    st.session_state.history = []
    st.session_state.logs = []
    st.session_state.sources = []
    st.session_state.draft = ""
    st.session_state.feedback = ""
    st.session_state.scorecard = {}
    st.session_state.approved = False
    st.session_state.attempts = 0
    st.session_state.completed_nodes = []
    st.session_state.failed_node = None
    st.session_state.run_started_at = ""
    st.session_state.run_finished_at = ""


def graph_diagram(active_node: str | None, completed_nodes: set[str], approved: bool = False,
                  failed_node: str | None = None) -> str:
    """Render a graph whose states come from observed execution events."""
    ids = {node_id: label for node_id, label in GRAPH_NODES}
    lines = [
        "digraph Workflow {",
        'graph [rankdir=LR, bgcolor="transparent", nodesep="0.35", ranksep="0.5", pad="0.15"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, margin="0.16,0.12"];',
        'edge [color="#B8C4D3", penwidth=1.4, arrowsize=0.7];',
    ]

    for node_id, label in GRAPH_NODES:
        fill, border, font = "#FFFFFF", "#DFE5EE", "#172B4D"
        node_label = label
        if node_id == failed_node:
            fill, border = "#FCE8E6", "#C9372C"
            node_label += "\\nFAILED"
        elif node_id == active_node:
            fill, border = "#E7F4F2", "#176B67"
            node_label += "\\nIN PROGRESS"
        elif node_id in completed_nodes:
            fill, border = "#E8F5EC", "#438A5E"
            node_label += "\\nCOMPLETED"
        if node_id == "reviewer" and approved and node_id in completed_nodes:
            fill, border = "#DDF5E7", "#218653"
            node_label = "Reviewer\\nAPPROVED"

        lines.append(
            f'"{node_id}" [label="{node_label}", fillcolor="{fill}", '
            f'color="{border}", fontcolor="{font}", penwidth=1.5];'
        )

    lines.extend([
        '"writer" -> "tools" [label="if called", fontsize=8];',
        '"tools" -> "writer" [label="tool result", fontsize=8];',
        '"writer" -> "extract_draft" [label="draft", fontsize=8];',
        '"extract_draft" -> "reviewer";',
        '"reviewer" -> "writer" [label="revise if rejected", fontsize=8];',
        "}",
    ])
    return "\n".join(lines)


def render_scorecard(scores: dict) -> None:
    criteria = [
        "Strong hook",
        "Clear takeaway",
        "Skimmable paragraphs",
        "Word count",
        "Ending CTA",
        "Human tone",
        "No hashtags",
    ]
    if not scores:
        st.info("The reviewer scorecard will appear when review finishes.")
        return

    passed = 0
    for criterion in criteria:
        result = str(scores.get(criterion, "NOT SCORED")).strip().upper()
        if result == "PASS":
            passed += 1
            st.write(f"🟢 **{criterion}** — Pass")
        elif result in {"NEEDS WORK", "FAIL", "FAILED"}:
            st.write(f"🔴 **{criterion}** — Needs work")
        else:
            st.write(f"⚪ **{criterion}** — Not scored")
    st.progress(passed / len(criteria))
    st.caption(f"{passed} of {len(criteria)} criteria passed")


def render_draft(draft: str) -> None:
    """Render untrusted model output as escaped text inside a styled container."""
    if not draft:
        st.info("The draft will appear here when extraction completes.")
        return
    safe_draft = html.escape(draft)
    st.markdown(
        f'<div class="draft-box">{safe_draft}</div>',
        unsafe_allow_html=True,
    )


def render_logs(logs: list[str], height: int = 280) -> None:
    if not logs:
        st.caption("Execution events will appear here.")
        return
    with st.container(height=height, border=True):
        for line in logs[-MAX_LOG_LINES:]:
            # Use Streamlit's normal text rendering; do not inject log text as HTML.
            st.caption(line)


def render_sources(sources: list[str]) -> None:
    if not sources:
        st.caption("No source URLs were captured from tool results.")
        return
    for index, url in enumerate(sources[:MAX_SOURCES], start=1):
        source_col, link_col = st.columns([0.18, 0.82], vertical_alignment="center")
        with source_col:
            st.caption(f"Source {index}")
        with link_col:
            st.markdown(f"[Open source {index}]({url})")


def save_completed_post() -> None:
    """Save one completed post to in-session history, avoiding duplicate saves."""
    if not st.session_state.draft:
        return
    record = {
        "topic": st.session_state.last_topic,
        "draft": st.session_state.draft,
        "approved": st.session_state.approved,
        "scorecard": dict(st.session_state.scorecard),
        "feedback": st.session_state.feedback,
        "attempts": st.session_state.attempts,
        "finished_at": st.session_state.run_finished_at or datetime.now().isoformat(timespec="seconds"),
    }
    # A new run clears current-run history; append one completed record per run.
    if not st.session_state.post_history or st.session_state.post_history[-1] != record:
        st.session_state.post_history.append(record)
        st.session_state.post_history = st.session_state.post_history[-25:]


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-powered content creation</div>
            <h1 class="hero-title">Turn your ideas into<br>LinkedIn-ready posts.</h1>
            <p class="hero-copy">
                Start with a topic. Your AI writer drafts your post, researches when
                needed, and reviews the result before you decide what to publish.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 0.85], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("What would you like to write about?")
            st.caption("Enter a topic, project, professional experience, or an idea you want to share.")

            with st.form("topic_form", clear_on_submit=False):
                topic_input = st.text_area(
                    "LinkedIn post topic",
                    placeholder=(
                        "Example: What I learned while building my first "
                        "multi-agent AI research system..."
                    ),
                    height=145,
                    max_chars=MAX_TOPIC_LENGTH,
                    key="topic_input",
                )
                st.caption(f"Up to {MAX_TOPIC_LENGTH:,} characters.")
                submitted = st.form_submit_button(
                    "Generate LinkedIn post  →",
                    type="primary",
                    use_container_width=True,
                )

            if st.session_state.error_message:
                st.error(st.session_state.error_message)

            if submitted:
                if start_generation(topic_input):
                    st.session_state.page = "workspace"
                    st.rerun()
                else:
                    st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("#### Your post, step by step")
            st.markdown("**01 · Write**")
            st.caption("Turn your topic into a clear professional post.")
            st.divider()
            st.markdown("**02 · Research when needed**")
            st.caption("Search for supporting information when the workflow calls for it.")
            st.divider()
            st.markdown("**03 · Review and refine**")
            st.caption("Check the draft against writing criteria before using it.")

        if st.session_state.post_history:
            st.write("")
            with st.container(border=True):
                st.markdown("#### Recent posts")
                st.caption("Saved in this browser session only.")
                for index, record in enumerate(reversed(st.session_state.post_history[-3:]), start=1):
                    topic_label = record["topic"]
                    st.write(f"**{topic_label[:90]}{'…' if len(topic_label) > 90 else ''}**")
                    st.caption(
                        f"{'Approved' if record['approved'] else 'Not approved'} · "
                        f"{record['finished_at']}"
                    )

    st.stop()


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✍️ Content Studio")
    st.caption("AI writing · Research · Review")
    st.divider()

    if st.button("← Back to home", use_container_width=True):
        reset_current_run()
        st.rerun()

    st.markdown("#### Current topic")
    st.info(st.session_state.submitted_topic or "No topic submitted.")

    with st.expander("Change topic", expanded=False):
        replacement_topic = st.text_area(
            "Topic",
            value=st.session_state.submitted_topic,
            max_chars=MAX_TOPIC_LENGTH,
            height=120,
            key="replacement_topic",
        )
        if st.button("Generate from this topic", use_container_width=True):
            if start_generation(replacement_topic):
                st.rerun()
            else:
                st.error(st.session_state.error_message)

    st.divider()
    st.markdown("#### Pipeline")
    st.caption("1. Cloudflare writer")
    st.caption("2. Tavily search when needed")
    st.caption("3. Draft extraction")
    st.caption("4. Groq reviewer")
    st.caption("The graph/backend should enforce the maximum attempt count.")

    if st.button("Clear current run", use_container_width=True):
        reset_current_run()
        st.rerun()


nav_col, topic_col = st.columns([0.22, 0.78], vertical_alignment="center")
with nav_col:
    if st.button("← Back", key="workspace_back_button", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
with topic_col:
    st.caption("Current topic: " + (st.session_state.submitted_topic[:140] or "Untitled"))

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">LinkedIn content workspace</div>
        <h1 class="hero-title" style="font-size: clamp(27px, 3vw, 36px);">
            From topic to publish-ready post.
        </h1>
        <p class="hero-copy">
            Follow the workflow, inspect reviewer feedback, and copy or download
            your draft when it is ready.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_labels = {
    "idle": "Ready",
    "running": "Running",
    "success": "Completed",
    "error": "Failed",
}
metric_cols = st.columns(3)
metrics = [
    ("Draft attempts", f"{st.session_state.attempts}"),
    ("Status", status_labels.get(st.session_state.status, "Unknown")),
    ("Revisions this run", str(len(st.session_state.history))),
]
for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(value)}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")


# ---------------------------------------------------------------------------
# Execute the graph synchronously while showing observed updates
# ---------------------------------------------------------------------------
if st.session_state.status == "running":
    left, right = st.columns([1.0, 1.15], gap="large")
    with left:
        st.subheader("Workflow")
        graph_area = st.empty()
        st.subheader("Execution log")
        log_area = st.empty()
    with right:
        st.subheader("Live draft")
        draft_area = st.empty()
        review_area = st.empty()

    active_node = None
    completed_nodes = set(st.session_state.completed_nodes)
    failed_node = None
    current_state = initial_state(st.session_state.last_topic)

    try:
        graph_area.graphviz_chart(
            graph_diagram(active_node, completed_nodes),
            use_container_width=True,
        )

        saw_update = False
        for event in app.stream(current_state, stream_mode="updates"):
            if not isinstance(event, dict):
                continue

            for node, update in event.items():
                if not isinstance(update, dict):
                    continue
                if node not in KNOWN_NODES:
                    log_event(f"Received update from an unrecognized node: {node}.")
                    continue

                saw_update = True
                active_node = node
                log_event(f"{KNOWN_NODES[node]} update received.")

                # Merge node state updates except messages, which are reducer-managed.
                for key, value in update.items():
                    if key != "messages":
                        current_state[key] = value

                # Collect links only from actual tool messages.
                messages = update.get("messages", [])
                if not isinstance(messages, (list, tuple)):
                    messages = [messages]
                for message in messages:
                    if isinstance(message, ToolMessage):
                        for url in extract_urls(message.content):
                            if url not in st.session_state.sources:
                                st.session_state.sources.append(url)
                                st.session_state.sources = st.session_state.sources[:MAX_SOURCES]
                        log_event(f"Tool result received: {getattr(message, 'name', None) or 'search'}.")

                st.session_state.attempts = current_state.get(
                    "attempts", st.session_state.attempts
                )
                draft_value = current_state.get("draft", st.session_state.draft)
                st.session_state.draft = draft_value if isinstance(draft_value, str) else str(draft_value or "")
                feedback_value = current_state.get("reviewer_feedback", st.session_state.feedback)
                st.session_state.feedback = feedback_value if isinstance(feedback_value, str) else str(feedback_value or "")
                scorecard_value = current_state.get("scorecard", st.session_state.scorecard)
                st.session_state.scorecard = scorecard_value if isinstance(scorecard_value, dict) else {}
                st.session_state.approved = bool(current_state.get("is_approved", st.session_state.approved))

                # This event is an observed node update; record it as completed.
                completed_nodes.add(node)
                st.session_state.completed_nodes = list(completed_nodes)
                active_node = None

                if node == "extract_draft" and st.session_state.draft:
                    attempt_number = st.session_state.attempts
                    # Avoid duplicate revision entries for repeated updates in one attempt.
                    if not st.session_state.history or (
                        st.session_state.history[-1].get("attempt") != attempt_number
                        or st.session_state.history[-1].get("draft") != st.session_state.draft
                    ):
                        st.session_state.history.append({
                            "attempt": attempt_number,
                            "draft": st.session_state.draft,
                            "feedback": "",
                            "scorecard": {},
                        })
                    log_event(
                        f"Draft extracted ({len(st.session_state.draft.split())} words)."
                    )

                if node == "reviewer" and st.session_state.history:
                    st.session_state.history[-1]["feedback"] = st.session_state.feedback
                    st.session_state.history[-1]["scorecard"] = dict(st.session_state.scorecard)
                    verdict = "APPROVED" if st.session_state.approved else "NEEDS REVISION"
                    log_event(f"Reviewer verdict: {verdict}.")

                graph_area.graphviz_chart(
                    graph_diagram(None, completed_nodes, st.session_state.approved),
                    use_container_width=True,
                )
                with log_area.container():
                    render_logs(st.session_state.logs, height=260)
                with draft_area.container():
                    render_draft(st.session_state.draft)
                with review_area.container():
                    st.markdown("#### Reviewer scorecard")
                    render_scorecard(st.session_state.scorecard)
                    if st.session_state.feedback:
                        st.markdown("#### Reviewer feedback")
                        st.write(st.session_state.feedback)

        # A graph that returns without raising may still have produced no usable result.
        st.session_state.run_finished_at = datetime.now().isoformat(timespec="seconds")
        if st.session_state.draft:
            st.session_state.status = "success"
            log_event("Graph execution finished.")
            save_completed_post()
        else:
            st.session_state.status = "error"
            st.session_state.error_message = (
                "The workflow ended without producing a draft. Check the backend logs and graph state."
            )
            log_event("Graph finished without a draft.")
        st.rerun()

    except Exception as exc:
        st.session_state.status = "error"
        st.session_state.run_finished_at = datetime.now().isoformat(timespec="seconds")
        st.session_state.error_message = (
            "Generation failed. Check your API credentials, model access, and server logs, then try again."
        )
        failed_node = active_node
        st.session_state.failed_node = failed_node
        # Keep details in server logs, not in the public UI.
        logger.exception("LinkedIn post generation failed at node %r", failed_node)
        log_event(f"Execution failed ({type(exc).__name__}).")
        st.rerun()


# ---------------------------------------------------------------------------
# Results and recovery view
# ---------------------------------------------------------------------------
left, right = st.columns([1.0, 1.15], gap="large")

with left:
    st.subheader("Workflow")
    failed_node = st.session_state.failed_node
    st.graphviz_chart(
        graph_diagram(
            None,
            set(st.session_state.completed_nodes),
            st.session_state.approved,
            failed_node=failed_node,
        ),
        use_container_width=True,
    )

    st.subheader("Execution log")
    render_logs(st.session_state.logs, height=280)

    if st.session_state.status == "error":
        st.error(st.session_state.error_message or "Generation failed.")
        if st.button("Retry generation", type="primary", use_container_width=True):
            if start_generation(st.session_state.last_topic or st.session_state.submitted_topic):
                st.rerun()

with right:
    st.subheader("Generated post")
    if st.session_state.draft:
        render_draft(st.session_state.draft)
        st.caption(
            f"{len(st.session_state.draft.split())} words · "
            f"{'Reviewer approved' if st.session_state.approved else 'Not approved by reviewer'}"
        )

        download_cols = st.columns(2)
        with download_cols[0]:
            st.download_button(
                "Download TXT",
                data=st.session_state.draft,
                file_name="linkedin_post.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with download_cols[1]:
            st.download_button(
                "Download Markdown",
                data=f"# LinkedIn post\n\n{st.session_state.draft}",
                file_name="linkedin_post.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("Copy-ready plain text", expanded=False):
            st.code(st.session_state.draft, language=None)
    else:
        st.info("Enter a topic on the home page to generate a post.")

    st.subheader("Reviewer scorecard")
    render_scorecard(st.session_state.scorecard)
    if st.session_state.feedback:
        st.markdown("**Latest reviewer feedback**")
        st.write(st.session_state.feedback)

    st.subheader("Research sources")
    render_sources(st.session_state.sources)

with st.expander("Revision history for this run", expanded=False):
    if not st.session_state.history:
        st.caption("Draft revisions will appear here when extraction completes.")
    else:
        for item in reversed(st.session_state.history):
            attempt_label = item.get("attempt", "?")
            with st.expander(f"Attempt {attempt_label}", expanded=False):
                st.write(item.get("draft", ""))
                draft_text = item.get("draft", "")
                st.caption(f"{len(draft_text.split())} words")
                if item.get("feedback"):
                    st.markdown("**Reviewer feedback**")
                    st.write(item["feedback"])
                if item.get("scorecard"):
                    render_scorecard(item["scorecard"])

with st.expander("Recent completed posts in this session", expanded=False):
    if not st.session_state.post_history:
        st.caption("Completed posts will appear here. They are not stored permanently.")
    else:
        for index, record in enumerate(reversed(st.session_state.post_history), start=1):
            topic = record.get("topic", "Untitled topic")
            with st.expander(f"{index}. {topic[:70]}", expanded=False):
                st.caption(
                    f"{'Approved' if record.get('approved') else 'Not approved'} · "
                    f"{record.get('finished_at', '')}"
                )
                st.write(record.get("draft", ""))
                st.download_button(
                    "Download this post",
                    data=record.get("draft", ""),
                    file_name=f"linkedin_post_{index}.txt",
                    mime="text/plain",
                    key=f"history_download_{index}",
                )
