import os
import streamlit as st
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables (works locally with .env and on Streamlit Cloud via st.secrets)
load_dotenv()

# for key in ["CLOUDFLARE_ACCOUNT_ID", "CF_AI_API_TOKEN", "GROQ_API_KEY", "TAVILY_API_KEY"]:
#     if key in st.secrets and key not in os.environ:
#         os.environ[key] = st.secrets[key]

# ==========================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(
    page_title="AI LinkedIn Post Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .linkedin-card {
        background-color: #ffffff;
        color: #191919;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .author-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 12px;
    }
    .avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background-color: #0a66c2;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 20px;
        margin-right: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LANGGRAPH DEFINITION
# ==========================================
class State(TypedDict):
    topic: str
    audience: str
    tone: str
    goal: str
    messages: Annotated[list, add_messages]
    draft: str
    reviewer_feedback: str
    is_approved: bool
    attempts: int
    max_attempts: int

WRITER_SYSTEM_PROMPT = """You are a skilled LinkedIn content writer.
Your task is to produce a complete, publish-ready LinkedIn post about the supplied topic.

Rules:
- Always produce the actual LinkedIn post.
- Never ask clarifying questions.
- Never return internal reasoning, planning, or search notes.
- Use web search for current facts when necessary.
- Never invent statistics or unsupported claims.
- Start with a strong hook in the first line.
- Include one clear and valuable takeaway.
- Use short, readable paragraphs.
- Target 150–200 words.
- Match the requested Target Audience, Tone, and Goal.
- End with a relevant question or CTA.
- Do not use hashtags.
- When revising, directly address the reviewer's feedback.
- Return ONLY the complete post text."""

REVIEWER_SYSTEM_PROMPT = """You are a strict LinkedIn content reviewer. You judge whether a post is publish-ready.
Evaluate against these criteria:
1. Strong hook in the first line
2. One clear, valuable takeaway
3. Easy to skim - uses short paragraphs
4. Roughly 150-200 words
5. Ends with an engaging question or CTA
6. Matches the requested tone (not corporate-robotic)
7. No hashtags

Respond in EXACTLY this format:
VERDICT: APPROVED or REJECTED
FEEDBACK: <one short paragraph explaining why>

Be strict but fair. Approve only if the post genuinely meets all criteria."""

@st.cache_resource
def build_graph(use_groq_for_writer: bool = False):
    search_tool = TavilySearch(max_results=3)
    tools = [search_tool]

    if use_groq_for_writer:
        writer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    else:
        writer_llm = ChatCloudflareWorkersAI(
            model_name="@cf/deepseek-ai/deepseek-v4.1-flash-1017",
            temperature=0.7,
            max_tokens=800
        )
    
    writer_llm_with_tools = writer_llm.bind_tools(tools)
    reviewer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

    def writer_node(state: State) -> dict:
        messages = state.get("messages", [])
        
        # Check if we are returning from a Tool Call (Tavily Search)
        if messages and isinstance(messages[-1], ToolMessage):
            response = writer_llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Otherwise, this is a new draft attempt
        attempts = state.get("attempts", 0) + 1
        topic = state["topic"]
        audience = state["audience"]
        tone = state["tone"]
        goal = state["goal"]
        previous_feedback = state.get("reviewer_feedback", "")

        context_specs = f"Topic: {topic}\nTarget Audience: {audience}\nTone: {tone}\nGoal/CTA Focus: {goal}"

        if attempts == 1:
            user_msg = HumanMessage(content=(
                f"Write a LinkedIn post with these specifications:\n{context_specs}\n\n"
                f"If you need current facts or recent trends, search the web first."
            ))
            invoke_msgs = [SystemMessage(content=WRITER_SYSTEM_PROMPT), user_msg]
            response = writer_llm_with_tools.invoke(invoke_msgs)
            return {
                "messages": [SystemMessage(content=WRITER_SYSTEM_PROMPT), user_msg, response],
                "attempts": attempts
            }
        else:
            user_msg = HumanMessage(content=(
                f"Your previous draft on '{topic}' was REJECTED.\n"
                f"Here was your previous draft:\n---\n{state.get('draft', '')}\n---\n\n"
                f"Reviewer's feedback:\n{previous_feedback}\n\n"
                f"Write a new, improved draft that fixes these exact issues while respecting:\n{context_specs}"
            ))
            response = writer_llm_with_tools.invoke(messages + [user_msg])
            return {
                "messages": [user_msg, response],
                "attempts": attempts
            }

    def extract_draft_node(state: State) -> dict:
        last_message = state["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            draft = next((item["text"] for item in content if item.get("type") == "text"), str(content))
        else:
            draft = str(content)
        
        # Strip <think>...</think> tags if the model outputs raw thinking tags in string
        if "</think>" in draft:
            draft = draft.split("</think>")[-1].strip()
            
        return {"draft": draft}

    def reviewer_node(state: State) -> dict:
        draft = state["draft"]
        word_count = len(draft.split())
        prompt = (
            f"Requested Tone: {state['tone']} | Target Audience: {state['audience']}\n"
            f"Approximate Word Count: {word_count} words\n\n"
            f"Review this LinkedIn post draft:\n---\n{draft}\n---\n"
            "Review the draft against all the specified criteria."
        )
        response = reviewer_llm.invoke([
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        review_text = response.content.strip()
        is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

        if "FEEDBACK:" in review_text:
            feedback = review_text.split("FEEDBACK:", 1)[1].strip()
        else:
            feedback = review_text

        return {
            "reviewer_feedback": feedback,
            "is_approved": is_approved
        }

    def should_use_tool(state: State):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "extract_draft"

    def should_stop_looping(state: State):
        if state["is_approved"]:
            return END
        if state["attempts"] >= state.get("max_attempts", 3):
            return END
        return "writer"

    graph = StateGraph(State)
    graph.add_node("writer", writer_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "writer")
    graph.add_conditional_edges("writer", should_use_tool)
    graph.add_edge("tools", "writer")
    graph.add_edge("extract_draft", "reviewer")
    graph.add_conditional_edges("reviewer", should_stop_looping)

    return graph.compile()

# ==========================================
# 3. STREAMLIT FRONTEND & QUESTIONNAIRE
# ==========================================
st.title("✍️ Multi-Agent LinkedIn Post Studio")
st.caption("Powered by LangGraph, Tavily Web Search, Cloudflare Workers AI & Groq")

with st.sidebar:
    st.header("⚙️ Post Configuration")
    
    audience = st.selectbox(
        "🎯 Who is your target audience?",
        [
            "Software Engineers & Developers",
            "Data Scientists & AI/ML Engineers",
            "Tech Founders & Executives",
            "Recruiters & Hiring Managers",
            "General Professional Network"
        ]
    )
    
    tone = st.selectbox(
        "🗣️ What tone should the post have?",
        [
            "Insightful & Contrarian",
            "Practical & Action-Oriented",
            "Storytelling & Personal",
            "Analytical & Data-Driven"
        ]
    )
    
    goal = st.selectbox(
        "🚀 What is the primary goal?",
        [
            "Spark debate in the comments",
            "Share a practical technical lesson",
            "Highlight an industry trend",
            "Demonstrate domain expertise"
        ]
    )
    
    st.divider()
    st.subheader("🔧 Agent Settings")
    max_attempts = st.slider("Max Revision Cycles", min_value=1, max_value=5, value=3)
    use_groq_fallback = st.toggle(
        "Use Groq (Llama 3.3 70B) for Writer", 
        value=False,
        help="Enable this if you hit Cloudflare 429 Too Many Requests rate limits."
    )
    
    with st.expander("📊 View Agent Workflow Graph"):
        try:
            app_graph = build_graph(use_groq_fallback)
            st.image(app_graph.get_graph().draw_mermaid_png(), caption="LangGraph Architecture")
        except Exception:
            st.info("Workflow: Writer ⇄ Tavily Search → Extract → Reviewer ↺")

# Main Questionnaire Form
with st.form("post_generator_form"):
    topic = st.text_area(
        "💡 What topic or specific insight do you want to write about?",
        placeholder="e.g., Why most RAG pipelines fail in production due to poor chunking strategies, and how semantic chunking fixes it...",
        height=100
    )
    submitted = st.form_submit_button("✨ Generate & Review Post", type="primary", use_container_width=True)

# ==========================================
# 4. EXECUTION & LIVE STREAMING
# ==========================================
if submitted:
    if not topic.strip():
        st.warning("Please enter a topic before generating!")
    else:
        app = build_graph(use_groq_fallback)
        
        initial_state = {
            "topic": topic.strip(),
            "audience": audience,
            "tone": tone,
            "goal": goal,
            "messages": [],
            "draft": "",
            "reviewer_feedback": "",
            "is_approved": False,
            "attempts": 0,
            "max_attempts": max_attempts
        }

        col1, col2 = st.columns([1, 1.2], gap="large")
        
        iteration_history = []
        current_state = initial_state.copy()

        with col1:
            st.subheader("🔄 Live Agent Progress")
            status_box = st.status("Initializing Multi-Agent Pipeline...", expanded=True)
            
            try:
                for event in app.stream(initial_state):
                    for node_name, state_update in event.items():
                        current_state.update(state_update)
                        
                        if node_name == "writer":
                            last_msg = state_update["messages"][-1]
                            if getattr(last_msg, "tool_calls", None):
                                query = last_msg.tool_calls[0]["args"].get("query", topic)
                                status_box.write(f"🔍 **Writer Agent** requested web search: *'{query}'*")
                            else:
                                attempt_num = current_state.get("attempts", 1)
                                status_box.write(f"✍️ **Writer Agent** completed Draft #{attempt_num}")
                                
                        elif node_name == "tools":
                            status_box.write("🌐 **Tavily Search** fetched live web results.")
                            
                        elif node_name == "extract_draft":
                            draft_words = len(state_update["draft"].split())
                            status_box.write(f"📄 **Draft Extracted** ({draft_words} words). Sending to Reviewer...")
                            
                        elif node_name == "reviewer":
                            approved = state_update["is_approved"]
                            feedback = state_update["reviewer_feedback"]
                            attempt_num = current_state.get("attempts", 1)
                            
                            iteration_history.append({
                                "attempt": attempt_num,
                                "draft": current_state.get("draft", ""),
                                "approved": approved,
                                "feedback": feedback
                            })
                            
                            if approved:
                                status_box.write(f"✅ **Reviewer Agent** APPROVED Draft #{attempt_num}!")
                            else:
                                status_box.write(f"❌ **Reviewer Agent** REJECTED Draft #{attempt_num}: *{feedback}*")
                                
                status_box.update(label="Generation Complete!", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="Pipeline Error", state="error", expanded=True)
                st.error(f"Error during execution: {e}")
                st.stop()

            # Show Iteration History Accordion
            if iteration_history:
                st.subheader("🗂️ Revision & Feedback Log")
                for item in iteration_history:
                    badge = "✅ Approved" if item["approved"] else "❌ Rejected"
                    with st.expander(f"Attempt #{item['attempt']} — {badge}", expanded=not item["approved"]):
                        st.markdown(f"**Reviewer Feedback:**\n> {item['feedback']}")
                        st.markdown("**Draft Text:**")
                        st.text(item["draft"])

        with col2:
            st.subheader("📱 Final LinkedIn Preview")
            final_draft = current_state.get("draft", "")
            
            if final_draft:
                # Metric bar
                m1, m2, m3 = st.columns(3)
                m1.metric("Status", "Approved ✅" if current_state.get("is_approved") else "Max Attempts ⚠️")
                m2.metric("Total Attempts", f"{current_state.get('attempts', 1)} / {max_attempts}")
                m3.metric("Word Count", f"{len(final_draft.split())} words")

                # Styled LinkedIn Card Preview
                st.markdown(f"""
                <div class="linkedin-card">
                    <div class="author-header">
                        <div class="avatar">IN</div>
                        <div>
                            <div style="font-weight: 600; font-size: 15px;">Your LinkedIn Profile</div>
                            <div style="font-size: 12px; color: #666;">AI & Software Engineering • Just now • 🌐</div>
                        </div>
                    </div>
                    <div>{final_draft}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("**📋 One-Click Copy:**")
                st.code(final_draft, language=None)