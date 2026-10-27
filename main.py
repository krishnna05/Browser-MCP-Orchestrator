import asyncio
import os
import streamlit as st
from textwrap import dedent

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Setup the UI configuration and design
st.set_page_config(page_title="Browser Automation Agent", page_icon="🌐", layout="wide")

# Header section
st.markdown("<h1 class='main-header'>🌐 Web Browser Agent</h1>", unsafe_allow_html=True)
st.markdown("Automate web navigation, scraping, and interactive page workflows using the Model Context Protocol.")

# Sidebar helper instructions
with st.sidebar:
    st.markdown("### Suggested Operations")
    
    st.markdown("**Navigation**")
    st.markdown("- Go to codinzhub.com")
    
    st.markdown("**Interactions**")
    st.markdown("- Click on search input in top ")

# Text area for user instructions
command_prompt = st.text_area("Specify Actions", 
                    placeholder="Describe what you want the browser agent to do...")

# Initialize system state variables if not already set
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="browser_agent_session")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.llm = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)
    st.session_state.is_processing = False

# Setup background browser agent and connections
async def prepare_agent():
    if not st.session_state.initialized:
        try:
            # Instantiate application context
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()
            
            # Configure agent system instructions
            st.session_state.browser_agent = Agent(
                name="browser_operator",
                instruction="""You are a web automation assistant capable of controlling a browser via Playwright.
                    - Access websites and perform actions like click, scroll, fill, and type.
                    - Gather and parse data from web pages.
                    - Take screenshots of targeted elements when relevant.
                    - Present concise markdown summaries of content.
                    - Chain actions sequentially to complete tasks.
                    
                Provide a summary report once the instructions have been carried out.""",
                server_names=["playwright"],
            )
            
            # Start agent and hook LLM pipeline
            await st.session_state.browser_agent.initialize() if hasattr(st.session_state.browser_agent, "initialize") else None
            st.session_state.llm = await st.session_state.browser_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Retrieve and log tools
            logger = st.session_state.mcp_agent_app.logger
            available_tools = await st.session_state.browser_agent.list_tools()
            logger.info("Registered tools:", data=available_tools)
            
            st.session_state.initialized = True
        except Exception as err:
            return f"Agent preparation failed: {str(err)}"
    return None

# Execute the specified user command
async def dispatch_command(target_message):
    # Verify we have the secrets required for LLM connection
    has_secrets_file = os.path.exists(os.path.join(os.path.dirname(__file__), "mcp_agent.secrets.yaml"))
    if not os.getenv("OPENAI_API_KEY") and not has_secrets_file:
        return (
            "Authentication required: Set the OPENAI_API_KEY environment variable "
            "or populate mcp_agent.secrets.yaml in the project directory."
        )

    try:
        # Guarantee initialized state
        init_err = await prepare_agent()
        if init_err:
            return init_err
        
        # Query the language model integration
        completion = await st.session_state.llm.generate_str(
            message=target_message, 
            request_params=RequestParams(use_history=True, maxTokens=10000)
        )
        return completion
    except Exception as err:
        return f"Execution error: {str(err)}"

# Check status variables
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

def trigger_execution():
    st.session_state.is_processing = True

# Action button
st.button(
    "🚀 Execute Command",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_processing,
    on_click=trigger_execution,
)

# Handle running request inside streamlit loop
if st.session_state.is_processing:
    with st.spinner("Automating browser actions..."):
        response_data = st.session_state.loop.run_until_complete(dispatch_command(command_prompt))
    st.session_state.last_result = response_data
    st.session_state.is_processing = False
    st.rerun()

# Output section
if st.session_state.last_result:
    st.markdown("### Execution Summary")
    st.markdown(st.session_state.last_result)

# Instruction guide for new setups
if 'result' not in locals():
    st.markdown(
        """<div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h4>Quick Start Guide:</h4>
        <ol>
            <li>Provide OpenAI API credentials inside the <code>mcp_agent.secrets.yaml</code> file</li>
            <li>Input your instructions for the browsing agent</li>
            <li>Click 'Execute Command' to begin the workflow</li>
        </ol>
        <p><strong>System Features:</strong></p>
        <ul>
            <li>Active browser navigation via Playwright</li>
            <li>Interactive elements interaction (click, text inputs, scrolls)</li>
            <li>Specific area/element screenshot capabilities</li>
            <li>Web page parsing and text extraction</li>
            <li>Multi-step process orchestration</li>
        </ul>
        </div>""", 
        unsafe_allow_html=True
    )
