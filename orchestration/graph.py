"""
Deterministic Multi-Agent Browser Orchestration Workflow using LangGraph.
Implements Supervisor (Planner), Navigator (Actor), Extractor (Actor),
and Evaluator (Summarizer) nodes with strict tool partitioning and telemetry.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional
import yaml

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from orchestration.state import (
    AgentState,
    EvaluatorDecision,
    SupervisorDecision,
    create_initial_state,
)

# Maximum iterations to prevent unbounded agent loops
MAX_GRAPH_ITERATIONS = 8

# Network timeout for individual browser MCP tool executions (seconds)
TOOL_EXECUTION_TIMEOUT = 40.0

# Tool partitioning definitions
NAVIGATOR_TOOL_PATTERNS = {
    "navigate",
    "click",
    "type",
    "fill_form",
    "press_key",
    "hover",
    "select_option",
    "wait_for",
    "tabs",
    "drag",
}

EXTRACTOR_TOOL_PATTERNS = {
    "snapshot",
    "take_screenshot",
    "evaluate",
    "find",
    "console_messages",
    "network_requests",
    "network_request",
}


def load_openai_config() -> Dict[str, Any]:
    """Reads OpenAI configuration and API keys from secrets and config files."""
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL")

    # Load secrets if present
    secrets_file = os.path.join(os.getcwd(), "mcp_agent.secrets.yaml")
    if os.path.exists(secrets_file):
        try:
            with open(secrets_file, "r", encoding="utf-8") as f:
                sec_data = yaml.safe_load(f) or {}
                if "openai" in sec_data and isinstance(sec_data["openai"], dict):
                    if not api_key:
                        api_key = sec_data["openai"].get("api_key")
                    if "base_url" in sec_data["openai"]:
                        base_url = sec_data["openai"].get("base_url")
        except Exception:
            pass

    # Load config if present
    config_file = os.path.join(os.getcwd(), "mcp_agent.config.yaml")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
                if "openai" in cfg_data and isinstance(cfg_data["openai"], dict):
                    model_name = cfg_data["openai"].get("default_model", model_name)
        except Exception:
            pass

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    return {
        "api_key": api_key,
        "model": model_name,
        "base_url": base_url,
    }


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Returns an initialized ChatOpenAI instance with configured credentials."""
    cfg = load_openai_config()
    kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "temperature": temperature,
        "request_timeout": 60.0,
        "max_retries": 2,
    }
    if cfg["api_key"]:
        kwargs["api_key"] = cfg["api_key"]
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]

    return ChatOpenAI(**kwargs)


def partition_mcp_tools(all_tools: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Partitions MCP tools into Navigator-only and Extractor-only tool sets.
    Converts raw tool objects into OpenAI function schemas.
    """
    nav_schemas = []
    ext_schemas = []

    for tool in all_tools:
        name = getattr(tool, "name", "")
        desc = getattr(tool, "description", "") or ""
        schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}

        openai_func = {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema,
            },
        }

        # Check Navigator affinity
        if any(p in name.lower() for p in NAVIGATOR_TOOL_PATTERNS):
            nav_schemas.append(openai_func)

        # Check Extractor affinity
        if any(p in name.lower() for p in EXTRACTOR_TOOL_PATTERNS):
            ext_schemas.append(openai_func)

    return {
        "navigator": nav_schemas,
        "extractor": ext_schemas,
    }


def extract_url_from_text(text: str) -> Optional[str]:
    """Extracts Page URL reported in tool output."""
    match = re.search(r"Page URL:\s*(https?://[^\s\n\r]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


async def execute_mcp_tool_safely(
    mcp_agent: Any,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Invokes an MCP tool with robust timeout, error handling, and output sanitization.
    """
    try:
        call_res = await asyncio.wait_for(
            mcp_agent.call_tool(tool_name, tool_args),
            timeout=TOOL_EXECUTION_TIMEOUT,
        )

        content_parts = []
        if hasattr(call_res, "content") and isinstance(call_res.content, list):
            for part in call_res.content:
                if hasattr(part, "text") and part.text:
                    content_parts.append(part.text)
                elif hasattr(part, "data"):
                    mime = getattr(part, "mimeType", "binary/data")
                    content_parts.append(f"[Binary/Image Content: {mime}]")
                else:
                    content_parts.append(str(part))
        raw_result = "\n".join(content_parts) if content_parts else str(call_res)
        is_error = getattr(call_res, "isError", False)

        return {
            "status": "failure" if is_error else "success",
            "result": raw_result,
            "error": raw_result if is_error else None,
        }
    except asyncio.TimeoutError:
        return {
            "status": "failure",
            "result": f"Action timed out after {TOOL_EXECUTION_TIMEOUT}s.",
            "error": f"TimeoutError: Tool execution '{tool_name}' exceeded deadline.",
        }
    except Exception as exc:
        return {
            "status": "failure",
            "result": f"Execution failed: {str(exc)}",
            "error": str(exc),
        }


# ============================================================================
# AGENT NODES
# ============================================================================


async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor (Planner):
    Analyzes the user objective, browser location, past actions, and missing data.
    Outputs a deterministic routing command to Navigator, Extractor, or Evaluator.
    """
    objective = state.get("user_objective", "")
    current_url = state.get("current_url", "")
    steps = state.get("intermediate_steps", [])
    extracted = state.get("extracted_data", [])
    missing = state.get("missing_requirements")
    iterations = state.get("iteration_count", 0) + 1

    # Safety termination if loop count exceeds threshold
    if iterations >= MAX_GRAPH_ITERATIONS:
        return {
            "workflow_status": "planning",
            "next_node": "evaluator",
            "iteration_count": iterations,
            "last_instruction": "Maximum iteration budget reached. Summarize all available findings.",
            "intermediate_steps": [
                {
                    "node": "supervisor",
                    "action": "route",
                    "params": {"target": "evaluator", "reason": "iteration_limit"},
                    "status": "success",
                    "result": "Iteration budget reached, forcing evaluation.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

    steps_summary = (
        "\n".join([f"- [{s.get('node')}] {s.get('action')}: {s.get('status')}" for s in steps[-6:]])
        if steps
        else "No previous steps executed."
    )
    extracted_summary = f"{len(extracted)} items extracted so far."

    system_prompt = (
        "You are the Supervisor (Planner) of a deterministic multi-agent browser automation team.\n"
        "Your team consists of:\n"
        "1. 'navigator': Interacts with the browser (navigates to URL, clicks, types, scrolls, waits).\n"
        "2. 'extractor': Gathers DOM structure, snapshots, text, console logs, or screenshots.\n"
        "3. 'evaluator': Reviews extracted data against user objective and generates final reports.\n\n"
        "Routing Rules:\n"
        "- If browser has not navigated to target website yet, route to 'navigator'.\n"
        "- If page is loaded and data needs to be inspected or scraped, route to 'extractor'.\n"
        "- If actions on page (like clicking or searching) are required before extraction, route to 'navigator'.\n"
        "- If sufficient data has been collected or Evaluator is needed, route to 'evaluator'.\n"
        "- Output deterministic routing with clear instructions."
    )

    user_prompt = (
        f"User Objective: {objective}\n"
        f"Active Browser URL: {current_url or 'About:blank (not navigated)'}\n"
        f"Extracted Data Status: {extracted_summary}\n"
        f"Missing Requirements: {missing or 'None'}\n"
        f"Recent Execution Steps:\n{steps_summary}\n"
    )

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SupervisorDecision)

    try:
        decision: SupervisorDecision = await structured_llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        target_node = decision.next_node
        instruction = decision.instruction
    except Exception as exc:
        # Fallback heuristic if LLM fails (e.g. rate limit / network error)
        if not current_url:
            target_node = "navigator"
            instruction = f"Navigate to the relevant web page for: {objective}"
        elif len(extracted) == 0:
            target_node = "extractor"
            instruction = "Extract page snapshot and text content."
        else:
            target_node = "evaluator"
            instruction = "Evaluate collected information."

    return {
        "workflow_status": "planning",
        "next_node": target_node,
        "iteration_count": iterations,
        "last_instruction": instruction,
        "intermediate_steps": [
            {
                "node": "supervisor",
                "action": "route",
                "params": {"target": target_node},
                "status": "success",
                "result": f"Routing to {target_node.title()}: {instruction}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }


async def navigator_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Navigator (Actor):
    Exclusively bound to navigation tools (goto, click, scroll/press_key, type, hover).
    Interacts with the web page and updates current_url and intermediate_steps.
    """
    objective = state.get("user_objective", "")
    current_url = state.get("current_url", "")
    instruction = state.get("last_instruction", "Navigate to target web page.")

    # Retrieve MCP Agent and tool schemas from config
    configurable = (config or {}).get("configurable", {})
    mcp_agent = configurable.get("mcp_agent")
    tool_schemas = configurable.get("navigator_tool_schemas", [])

    if not mcp_agent:
        return {
            "workflow_status": "error",
            "error_message": "MCP Browser Agent is uninitialized in navigator node.",
            "intermediate_steps": [
                {
                    "node": "navigator",
                    "action": "error",
                    "status": "failure",
                    "result": "Missing mcp_agent context.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

    system_msg = (
        "You are the Navigator Agent responsible for controlling browser navigation and interactions.\n"
        "You have access ONLY to navigation tools (navigate, click, type, fill_form, press_key, hover, tabs).\n"
        "Call the appropriate navigation tool to execute the user instruction.\n"
        "Always ensure valid URL formats (e.g. https://...)."
    )

    user_msg = (
        f"Goal: {objective}\n"
        f"Current Browser Location: {current_url or 'Empty'}\n"
        f"Supervisor Instruction: {instruction}\n"
    )

    llm = get_llm(temperature=0.0)
    if tool_schemas:
        bound_llm = llm.bind_tools(tool_schemas)
    else:
        bound_llm = llm

    executed_steps: List[Dict[str, Any]] = []
    updated_url = current_url

    try:
        response = await bound_llm.ainvoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])

        if response.tool_calls:
            for tc in response.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})

                tool_exec = await execute_mcp_tool_safely(mcp_agent, t_name, t_args)

                # Attempt to extract new URL from output or arguments
                extracted_url = extract_url_from_text(tool_exec["result"])
                if extracted_url:
                    updated_url = extracted_url
                elif "navigate" in t_name.lower() and "url" in t_args:
                    updated_url = t_args["url"]

                executed_steps.append(
                    {
                        "node": "navigator",
                        "action": t_name,
                        "params": t_args,
                        "status": tool_exec["status"],
                        "result": tool_exec["result"][:800],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
        else:
            # Fallback if LLM responded with prose instead of tool call
            # Heuristic: if instruction mentions a URL or navigation and we don't have a URL
            url_match = re.search(r"https?://[^\s'\"]+", instruction)
            if url_match:
                target = url_match.group(0)
                tool_exec = await execute_mcp_tool_safely(
                    mcp_agent, "playwright_browser_navigate", {"url": target}
                )
                updated_url = target
                executed_steps.append(
                    {
                        "node": "navigator",
                        "action": "playwright_browser_navigate",
                        "params": {"url": target},
                        "status": tool_exec["status"],
                        "result": tool_exec["result"][:800],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                executed_steps.append(
                    {
                        "node": "navigator",
                        "action": "interpret",
                        "status": "success",
                        "result": response.content or "Completed navigation evaluation.",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
    except Exception as exc:
        executed_steps.append(
            {
                "node": "navigator",
                "action": "error",
                "status": "failure",
                "result": f"Navigator error: {str(exc)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return {
        "workflow_status": "navigating",
        "current_url": updated_url,
        "intermediate_steps": executed_steps,
    }


async def extractor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Extractor (Actor):
    Exclusively bound to DOM parsing, snapshot, and screenshot tools.
    Extracts webpage content and appends findings to extracted_data.
    """
    objective = state.get("user_objective", "")
    current_url = state.get("current_url", "")
    instruction = state.get("last_instruction", "Extract DOM snapshot and text.")

    configurable = (config or {}).get("configurable", {})
    mcp_agent = configurable.get("mcp_agent")
    tool_schemas = configurable.get("extractor_tool_schemas", [])

    if not mcp_agent:
        return {
            "workflow_status": "error",
            "error_message": "MCP Browser Agent is uninitialized in extractor node.",
            "intermediate_steps": [
                {
                    "node": "extractor",
                    "action": "error",
                    "status": "failure",
                    "result": "Missing mcp_agent context.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

    system_msg = (
        "You are the Extractor Agent specialized in extracting DOM snapshots, text, and visual screenshots.\n"
        "You have access ONLY to extraction tools (snapshot, take_screenshot, evaluate, find, network_requests).\n"
        "Invoke the extraction tool best suited to capture the requested data for the user's objective."
    )

    user_msg = (
        f"Goal: {objective}\n"
        f"Active URL: {current_url}\n"
        f"Supervisor Instruction: {instruction}\n"
    )

    llm = get_llm(temperature=0.0)
    if tool_schemas:
        bound_llm = llm.bind_tools(tool_schemas)
    else:
        bound_llm = llm

    executed_steps: List[Dict[str, Any]] = []
    extracted_items: List[Dict[str, Any]] = []

    try:
        response = await bound_llm.ainvoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])

        if response.tool_calls:
            for tc in response.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})

                tool_exec = await execute_mcp_tool_safely(mcp_agent, t_name, t_args)

                executed_steps.append(
                    {
                        "node": "extractor",
                        "action": t_name,
                        "params": t_args,
                        "status": tool_exec["status"],
                        "result": tool_exec["result"][:600],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                if tool_exec["status"] == "success":
                    extracted_items.append(
                        {
                            "type": "screenshot" if "screenshot" in t_name.lower() else "snapshot",
                            "source_url": current_url,
                            "content": tool_exec["result"],
                            "metadata": {"tool": t_name, "params": t_args},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
        else:
            # Deterministic fallback: execute snapshot to ensure DOM data is captured
            tool_exec = await execute_mcp_tool_safely(mcp_agent, "playwright_browser_snapshot", {})
            executed_steps.append(
                {
                    "node": "extractor",
                    "action": "playwright_browser_snapshot",
                    "params": {},
                    "status": tool_exec["status"],
                    "result": tool_exec["result"][:600],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            if tool_exec["status"] == "success":
                extracted_items.append(
                    {
                        "type": "snapshot",
                        "source_url": current_url,
                        "content": tool_exec["result"],
                        "metadata": {"tool": "playwright_browser_snapshot"},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
    except Exception as exc:
        executed_steps.append(
            {
                "node": "extractor",
                "action": "error",
                "status": "failure",
                "result": f"Extractor error: {str(exc)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return {
        "workflow_status": "extracting",
        "intermediate_steps": executed_steps,
        "extracted_data": extracted_items,
    }


async def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluator (Summarizer):
    Compares extracted_data against the user_objective.
    If data is insufficient: updates missing_requirements and routes back to Supervisor.
    If complete: generates the final markdown report and terminates workflow.
    """
    objective = state.get("user_objective", "")
    extracted = state.get("extracted_data", [])
    current_url = state.get("current_url", "")
    iterations = state.get("iteration_count", 0)

    # Compile data preview
    combined_content = []
    for idx, item in enumerate(extracted[-4:], 1):
        content_snippet = item.get("content", "")[:3500]
        combined_content.append(f"--- Extracted Item {idx} ({item.get('type')}) ---\n{content_snippet}")

    all_data_str = "\n\n".join(combined_content) if combined_content else "No data extracted."

    system_prompt = (
        "You are the Evaluator and Summarizer agent.\n"
        "Your task is twofold:\n"
        "1. Assess whether the extracted web data sufficiently answers/satisfies the User Objective.\n"
        "2. If incomplete, set is_complete=False and explain precisely what information is still missing.\n"
        "3. If complete (or if the iteration budget is exhausted), set is_complete=True and write a clear, "
        "well-organized Markdown summary report containing all relevant findings, URLs, and insights."
    )

    user_prompt = (
        f"User Objective: {objective}\n"
        f"Active URL: {current_url}\n"
        f"Iterations: {iterations}/{MAX_GRAPH_ITERATIONS}\n"
        f"Collected Web Content:\n{all_data_str}\n"
    )

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(EvaluatorDecision)

    try:
        eval_decision: EvaluatorDecision = await structured_llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        # Force completion if iterations reached limit
        if iterations >= MAX_GRAPH_ITERATIONS:
            is_done = True
            report = (
                eval_decision.summary_markdown
                or f"### Summary Report\n\n**Objective:** {objective}\n\n**Collected Content:**\n{all_data_str[:2000]}"
            )
        else:
            is_done = eval_decision.is_complete
            report = eval_decision.summary_markdown

        if is_done:
            return {
                "workflow_status": "complete",
                "next_node": "__end__",
                "final_report": report,
                "missing_requirements": None,
                "intermediate_steps": [
                    {
                        "node": "evaluator",
                        "action": "finalize",
                        "status": "success",
                        "result": "Objective successfully fulfilled. Markdown report generated.",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }
        else:
            return {
                "workflow_status": "evaluating",
                "next_node": "supervisor",
                "missing_requirements": eval_decision.missing_requirements or "Additional data required.",
                "intermediate_steps": [
                    {
                        "node": "evaluator",
                        "action": "critique",
                        "status": "in_progress",
                        "result": f"Incomplete data: {eval_decision.missing_requirements}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }
    except Exception as exc:
        # Fallback if LLM call fails
        fallback_report = (
            f"### Execution Summary\n\n"
            f"**Objective:** {objective}\n\n"
            f"**Active URL:** {current_url}\n\n"
            f"**Extracted Data Items:** {len(extracted)}\n\n"
            f"**Note:** Automated evaluation encountered error: `{str(exc)}`.\n\n"
            f"**Raw Data Preview:**\n```\n{all_data_str[:1500]}\n```"
        )
        return {
            "workflow_status": "complete",
            "next_node": "__end__",
            "final_report": fallback_report,
            "intermediate_steps": [
                {
                    "node": "evaluator",
                    "action": "fallback_summary",
                    "status": "success",
                    "result": f"Evaluator fallback due to: {str(exc)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

def route_from_supervisor(state: AgentState) -> str:
    """Evaluates supervisor routing decision."""
    target = state.get("next_node", "navigator")
    if target in ("navigator", "extractor", "evaluator"):
        return target
    return "evaluator"


def route_from_evaluator(state: AgentState) -> str:
    """Routes to END if complete, or back to Supervisor if missing data."""
    if state.get("workflow_status") == "complete" or state.get("next_node") == "__end__":
        return END
    return "supervisor"


def build_orchestrator_graph() -> Any:
    """
    Constructs and compiles the deterministic multi-agent LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Register agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("navigator", navigator_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("evaluator", evaluator_node)

    # Workflow entry point
    workflow.add_edge(START, "supervisor")

    # Conditional branching from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "navigator": "navigator",
            "extractor": "extractor",
            "evaluator": "evaluator",
        },
    )

    # Actor nodes transitions
    # Navigator routes back to Supervisor to assess new page state
    workflow.add_edge("navigator", "supervisor")

    # Extractor routes to Evaluator to evaluate newly acquired data
    workflow.add_edge("extractor", "evaluator")

    # Evaluator conditional edge
    workflow.add_conditional_edges(
        "evaluator",
        route_from_evaluator,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )

    return workflow.compile()


async def stream_orchestrator_execution(
    compiled_graph: Any,
    mcp_agent: Any,
    user_objective: str,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Streams the asynchronous execution of the multi-agent graph.
    Yields intermediate telemetry events capturing node transitions, tool invocations,
    and the final markdown report for the Streamlit UI.
    """
    # Discover and partition tools
    try:
        tools_result = await mcp_agent.list_tools()
        raw_tools = getattr(tools_result, "tools", [])
    except Exception as exc:
        raw_tools = []
        yield {
            "type": "telemetry",
            "node": "system",
            "message": f"⚠️ Failed to list MCP tools: {str(exc)}",
            "status": "warning",
        }

    partitioned = partition_mcp_tools(raw_tools)

    config = {
        "configurable": {
            "mcp_agent": mcp_agent,
            "navigator_tool_schemas": partitioned["navigator"],
            "extractor_tool_schemas": partitioned["extractor"],
        }
    }

    initial_state = create_initial_state(user_objective)

    yield {
        "type": "telemetry",
        "node": "supervisor",
        "message": "🧠 Supervisor analyzing user objective and formulating initial plan...",
        "status": "in_progress",
    }

    final_report = None
    accumulated_steps: List[Dict[str, Any]] = []

    try:
        async for output in compiled_graph.astream(initial_state, config=config):
            for node_name, node_state in output.items():
                status = node_state.get("workflow_status", "")
                steps = node_state.get("intermediate_steps", [])
                report = node_state.get("final_report")
                current_url = node_state.get("current_url")
                next_node = node_state.get("next_node")
                missing = node_state.get("missing_requirements")

                if steps:
                    accumulated_steps.extend(steps)
                    latest_step = steps[-1]
                    step_action = latest_step.get("action", "")
                    step_res = latest_step.get("result", "")
                    step_status = latest_step.get("status", "success")

                    if node_name == "supervisor":
                        yield {
                            "type": "telemetry",
                            "node": "supervisor",
                            "message": f"🧠 Supervisor routed workflow to: **{next_node.title() if next_node else 'Next'}**",
                            "detail": step_res,
                            "status": step_status,
                        }
                    elif node_name == "navigator":
                        yield {
                            "type": "telemetry",
                            "node": "navigator",
                            "message": f"🧭 Navigator executed: `{step_action}` (URL: {current_url or 'in-transit'})",
                            "detail": step_res[:300],
                            "status": step_status,
                        }
                    elif node_name == "extractor":
                        yield {
                            "type": "telemetry",
                            "node": "extractor",
                            "message": f"🔍 Extractor captured data via `{step_action}`",
                            "detail": step_res[:300],
                            "status": step_status,
                        }
                    elif node_name == "evaluator":
                        if missing:
                            yield {
                                "type": "telemetry",
                                "node": "evaluator",
                                "message": f"⚖️ Evaluator: Data insufficient ({missing}). Looping back to Supervisor.",
                                "detail": step_res,
                                "status": "in_progress",
                            }
                        else:
                            yield {
                                "type": "telemetry",
                                "node": "evaluator",
                                "message": "⚖️ Evaluator: Objective fulfilled. Compiling final markdown report.",
                                "detail": step_res,
                                "status": "success",
                            }

                if report:
                    final_report = report

        yield {
            "type": "complete",
            "final_report": final_report or "### Execution Complete\n\nWorkflow completed without final summary text.",
            "steps": accumulated_steps,
        }

    except Exception as exc:
        yield {
            "type": "error",
            "message": f"Workflow failed with exception: {str(exc)}",
            "final_report": f"### Execution Error\n\nAn unexpected error occurred during multi-agent graph execution:\n\n`{str(exc)}`",
            "steps": accumulated_steps,
        }
