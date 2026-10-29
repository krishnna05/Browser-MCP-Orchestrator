"""
Streamlit UI for Browser Multi-Agent Orchestration powered by LangGraph & MCP.
Handles UI rendering, telemetry visualization, and asynchronous graph dispatching.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import os
import streamlit as st
import yaml

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent

from orchestration.graph import (
    build_orchestrator_graph,
    stream_orchestrator_execution,
    load_openai_config,
)

# Page configuration
st.set_page_config(
    page_title="Browser Multi-Agent Orchestrator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern UI Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-supervisor { background-color: #EDE9FE; color: #6D28D9; border: 1px solid #DDD6FE; }
    .badge-navigator { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .badge-extractor { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-evaluator { background-color: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
    .trace-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Architecture Overview and System Telemetry
with st.sidebar:
    st.markdown("### 🏛️ Agent Architecture")
    st.markdown(
        """
        <div style="font-size: 0.88rem; line-height: 1.5; color: #475569;">
        <span class="agent-badge badge-supervisor">Supervisor</span> Planner & router<br/>
        <span class="agent-badge badge-navigator">Navigator</span> Goto, click, scroll, type<br/>
        <span class="agent-badge badge-extractor">Extractor</span> DOM parsing & screenshots<br/>
        <span class="agent-badge badge-evaluator">Evaluator</span> Sufficiency & synthesis
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 💡 Example Objectives")
    sample_1 = "Go to news.ycombinator.com and extract the top 5 articles."
    sample_2 = "Navigate to python.org, click on Downloads, and summarize latest version."
    sample_3 = "Go to quotes.toscrape.com and extract the first 3 quotes with their authors."

    if st.button("📰 Hacker News Top 5", use_container_width=True):
        st.session_state["user_input_prompt"] = sample_1
    if st.button("🐍 Python Downloads", use_container_width=True):
        st.session_state["user_input_prompt"] = sample_2
    if st.button("💬 Quotes to Scrape", use_container_width=True):
        st.session_state["user_input_prompt"] = sample_3

    st.divider()
    cfg = load_openai_config()
    st.markdown("### ⚙️ Runtime Settings")
    st.caption(f"**Model:** `{cfg['model']}`")
    st.caption(f"**API Key:** `{'Configured ✅' if cfg['api_key'] else 'Missing ⚠️'}`")
    st.caption("**Engine:** LangGraph StateGraph (async)")


# Application Header
st.markdown("<div class='main-header'>🌐 Browser-MCP Multi-Agent Orchestrator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Deterministic multi-agent web automation utilizing LangGraph state machines and Playwright MCP tool binding.</div>",
    unsafe_allow_html=True,
)

# Initialize Session State
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="browser_orchestrator_session")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.compiled_graph = build_orchestrator_graph()
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)
    st.session_state.is_processing = False
    st.session_state.last_result = None
    st.session_state.last_steps = []


async def prepare_agent():
    """Initializes Playwright MCP persistent connection once per session."""
    if not st.session_state.initialized:
        try:
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()

            st.session_state.browser_agent = Agent(
                name="browser_operator",
                instruction="Browser automation actor attached to Playwright MCP.",
                server_names=["playwright"],
            )
            await st.session_state.browser_agent.initialize()
            st.session_state.initialized = True
            return None
        except Exception as err:
            return f"MCP initialization failed: {str(err)}"
    return None


# Input area
default_prompt = st.session_state.get("user_input_prompt", "")
command_prompt = st.text_area(
    "Specify Automation Objective",
    value=default_prompt,
    placeholder="e.g., Navigate to news.ycombinator.com, find the first 3 links, and summarize their headlines.",
    height=110,
)


async def dispatch_command(target_message: str, status_ui: Any, log_container: Any) -> tuple[str, list]:
    """
    Executes the compiled multi-agent LangGraph workflow.
    Streams real-time transitions and telemetry directly to the Streamlit UI.
    """
    cfg = load_openai_config()
    if not cfg["api_key"]:
        return (
            "Authentication required: Please set OPENAI_API_KEY environment variable "
            "or configure `mcp_agent.secrets.yaml` with your OpenAI credentials.",
            [],
        )

    init_err = await prepare_agent()
    if init_err:
        return f"Initialization error: {init_err}", []

    final_report = "Execution completed without summary output."
    accumulated_steps = []

    try:
        stream = stream_orchestrator_execution(
            compiled_graph=st.session_state.compiled_graph,
            mcp_agent=st.session_state.browser_agent,
            user_objective=target_message,
        )

        async for event in stream:
            event_type = event.get("type")

            if event_type == "telemetry":
                msg = event.get("message", "")
                detail = event.get("detail", "")
                node = event.get("node", "workflow")

                status_ui.update(
                    label=f"Phase: {node.title()} | {msg}",
                    state="running",
                )

                with log_container:
                    if detail:
                        st.markdown(f"{msg}\n> *{detail}*")
                    else:
                        st.markdown(f"{msg}")

            elif event_type == "complete":
                final_report = event.get("final_report", final_report)
                accumulated_steps = event.get("steps", [])
                status_ui.update(
                    label="✅ Multi-Agent Workflow Execution Complete",
                    state="complete",
                    expanded=False,
                )

            elif event_type == "error":
                final_report = event.get("final_report", event.get("message"))
                accumulated_steps = event.get("steps", [])
                status_ui.update(
                    label="❌ Multi-Agent Workflow Error Encountered",
                    state="error",
                    expanded=True,
                )

        return final_report, accumulated_steps

    except Exception as err:
        return f"Graph execution error: {str(err)}", accumulated_steps


# Execute button
execute_btn = st.button(
    "🚀 Execute Multi-Agent Workflow",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_processing or not command_prompt.strip(),
)

if execute_btn:
    st.session_state.is_processing = True

    # Real-time Telemetry Display during execution
    st.markdown("### 📡 Live Execution Telemetry")
    status_box = st.status("Initializing Multi-Agent Workflow...", expanded=True)
    live_log = st.container()

    try:
        report, steps = st.session_state.loop.run_until_complete(
            dispatch_command(command_prompt, status_box, live_log)
        )
        st.session_state.last_result = report
        st.session_state.last_steps = steps
    finally:
        st.session_state.is_processing = False

# Summary and traces rendering
if st.session_state.last_result:
    st.markdown("---")
    st.markdown("### 📋 Final Execution Summary")
    st.markdown(st.session_state.last_result)

    if st.session_state.last_steps:
        with st.expander("🔍 Intermediate Execution Trace & Node History", expanded=False):
            st.markdown(f"**Total Steps Executed:** {len(st.session_state.last_steps)}")
            for idx, step in enumerate(st.session_state.last_steps, 1):
                node = step.get("node", "unknown").upper()
                action = step.get("action", "")
                status = step.get("status", "unknown")
                timestamp = step.get("timestamp", "")
                result = step.get("result", "")
                params = step.get("params")

                badge_color = {
                    "SUPERVISOR": "🟣",
                    "NAVIGATOR": "🔵",
                    "EXTRACTOR": "🟡",
                    "EVALUATOR": "🟢",
                }.get(node, "⚪")

                st.markdown(
                    f"**{idx}. {badge_color} [{node}] `{action}`** — *{status.upper()}* ({timestamp})"
                )
                if params:
                    st.code(f"Parameters: {params}", language="json")
                if result:
                    st.text(f"Output: {result[:500]}")
                st.markdown("---")

# Quick Start info guide
if not st.session_state.last_result:
    st.markdown(
        """
        <div style='padding: 20px; background-color: #F8FAFC; border-radius: 10px; border: 1px solid #E2E8F0; margin-top: 20px;'>
        <h4 style="margin-top:0;">Architecture Quick Start:</h4>
        <ol>
            <li>Provide OpenAI API credentials in <code>mcp_agent.secrets.yaml</code> or set the <code>OPENAI_API_KEY</code> environment variable.</li>
            <li>Specify an automation instruction (or pick an example from the sidebar).</li>
            <li>Click <strong>Execute Multi-Agent Workflow</strong> to watch the Supervisor route tasks dynamically between Navigator, Extractor, and Evaluator nodes.</li>
        </ol>
        <p><strong>Deterministic Multi-Agent Specialization:</strong></p>
        <ul>
            <li><strong>Supervisor (Planner):</strong> Controls state progression and conditional edges.</li>
            <li><strong>Navigator (Actor):</strong> Exclusively bound to navigation and browser interaction tools.</li>
            <li><strong>Extractor (Actor):</strong> Exclusively bound to DOM snapshot, screenshot, and parsing tools.</li>
            <li><strong>Evaluator (Summarizer):</strong> Validates data sufficiency and synthesizes the final report.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
