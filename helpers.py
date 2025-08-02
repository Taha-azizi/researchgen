import os
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from fpdf import FPDF
from ddgs import DDGS
import re
from langchain_core.agents import AgentAction, AgentFinish

# --- Configuration for your Local AI Agent ---
OLLAMA_MODEL_FOR_AGENT = "gemma3:27b"  # Changed to a generally capable model
WEB_CONTENT_MAX_CHARS = 4000
AGENT_MAX_ITERATIONS = 15
REPORT_MIN_SOURCES = 3

# ==============================================================================
# 1. TOOL AND HELPER FUNCTIONS
# ==============================================================================

def _fetch_web_content_for_tool(url: str) -> str:
    """
    [Internal Helper Function]
    Attempts to fetch the main textual content from a given URL.
    Returns the content or a specific error message.
    """
    print(f"\n[Tool Action]: Fetching content from URL: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, nav, header, footer tags for cleaner content
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.extract()

        # Get text, preferring 'article' or 'main' tags
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.body.get_text(separator='\n', strip=True)

        # Basic quality check
        if len(text.split()) < 100:
            return f"Error: Content from {url} is too short or lacks meaningful text. Please discard this source."

        return text[:WEB_CONTENT_MAX_CHARS] + "..." if len(text) > WEB_CONTENT_MAX_CHARS else text

    except requests.exceptions.HTTPError as http_err:
        return f"Error: URL {url} returned HTTP status {http_err.response.status_code}. This source is likely invalid."
    except requests.exceptions.RequestException as req_err:
        return f"Error: Network issue for {url}: {req_err}. The website may be down or the URL is incorrect."
    except Exception as e:
        return f"Error: An unexpected error occurred while processing {url}: {e}."

def _summarize_and_format_report(query: str, accumulated_data: dict) -> str:
    """
    Uses an LLM to summarize the accumulated data into a structured report.
    """
    print("\n[Report Synthesis]: Synthesizing final report with LLM...")
    llm = Ollama(model=OLLAMA_MODEL_FOR_AGENT, temperature=0.3)

    content_for_llm = ""
    sources_list = []
    for url, content in accumulated_data.items():
        content_for_llm += f"--- Content from {url} ---\n{content}\n\n"
        sources_list.append(url)

    if not content_for_llm:
        return f"## Research Report on {query}\n\nNo relevant content was gathered."

    report_prompt_template = """
    You are an expert research analyst. Create a comprehensive research report based on the provided content.
    The original research question was: "{query}"

    --- Provided Research Content ---
    {content_for_llm}
    --- End of Provided Research Content ---

    Synthesize this information into a well-structured report. Focus ONLY on answering the original research question. Do not introduce outside information.
    Your report MUST follow this exact format:

    ## Research Report on {query}

    [Your comprehensive summary of findings, organized into paragraphs. This must directly answer the original question and be based solely on the provided content.]

    ### Key Takeaways:
    - [Bullet point 1]
    - [Bullet point 2]
    - ... (Summarize the most important points.)

    ### Sources:
    - {sources_list}
    """
    
    prompt = report_prompt_template.format(
        query=query,
        content_for_llm=content_for_llm,
        sources_list='\n- '.join(sources_list)
    )

    try:
        report = llm.invoke(prompt)
        return report
    except Exception as e:
        print(f"Error during report synthesis: {e}")
        return f"## Error during Report Synthesis\n\nAn error occurred: {e}"

def _clean_filename(text: str) -> str:
    """Cleans text for use as a filename."""
    cleaned_text = re.sub(r'[^\w\s-]', '', text).strip()
    cleaned_text = re.sub(r'[-\s]+', '-', cleaned_text)
    return cleaned_text[:50]

def generate_pdf_report(report_content: str, query_topic: str):
    """Generates a PDF report from the agent's findings."""
    print("\n[PDF Generation]: Creating PDF report...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Encode title and content to handle special characters
    title_text = f"Research Report on {query_topic}".encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, title_text)
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    report_lines = report_content.split('\n')
    for line in report_lines:
        encoded_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, encoded_line)

    filename = f"AI_Research_Report_{_clean_filename(query_topic)}.pdf"
    pdf.output(filename)
    print(f"✅ [Success]: PDF report saved as '{filename}'")
    return filename

def ddgs_search_tool(query: str) -> str:
    print(f"[Custom Search]: Searching for '{query}' using ddgs...")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region="wt-wt",
                safesearch="moderate",
                timelimit="m",
                max_results=10
            )
            seen = set()
            unique_urls = []
            for result in results:
                url = result.get('href')
                if url and url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            if unique_urls:
                return "\n".join(unique_urls)
            else:
                return "No usable links found."
    except Exception as e:
        return f"Error during DDGS search: {e}"

# ==============================================================================
# 2. REVISED AGENT INITIALIZATION AND EXECUTION LOGIC
# ==============================================================================

def initialize_agent_and_tools():
    """
    Initializes the LangChain agent with a local LLM and custom tools.
    """
    print(f"\n[Setup]: Initializing AI Research Agent with model: {OLLAMA_MODEL_FOR_AGENT}...")

    llm = Ollama(model=OLLAMA_MODEL_FOR_AGENT, temperature=0.1)

    tools = [
        Tool(
            name="Search",
            func=ddgs_search_tool,
            description="Use this to find URLs on the internet. Input should be a short search query. It returns a list of real URLs."
        ),
        Tool(
            name="Scrape",
            func=_fetch_web_content_for_tool,
            description="Use this to fetch the text content of a URL. The input must be a valid URL. If the tool returns an error or unsuitable content, you MUST try a different URL from your search results or perform a new search."
        )
    ]

    # This is the new, more strategic prompt.
    # It takes the agent's state (history, visited URLs) as input.
    agent_prompt_template = """
    You are an autonomous AI research assistant. Your goal is to answer the user's question by gathering information from the web.
    You will work step-by-step, using one tool at a time.

    **Research State:**
    - Question: {input}
    - Sources Found: {sources_found_count} out of {min_sources_required}
    - Visited URLs: {visited_urls}
    
    **Your Task:**
    1.  Start with a 'Search' to find relevant articles and sources for the Question.
    2.  Review search results and use the 'Scrape' tool on the most promising URL.
    3.  **Critically analyze the result of the Scrape tool:**
        - If the content is good, you've found a source. Continue searching and scraping until you have enough sources.
        - If you get an error or the content is unsuitable, **immediately discard that URL**. Add it to your 'Visited URLs' list in your mind and **do not try it again**.
    4.  If your initial search yields no good URLs, **formulate a new, different search query** to find better sources.
    5.  Once you have gathered {min_sources_required} high-quality sources, you are done. Your final output must be `FINISH`.

    **Tools Available:**
    {tools}

    **Previous Steps (Log):**
    {agent_scratchpad}

    **Your Next Step:**
    Thought: Your reasoning for the next action.
    Action: The tool to use, must be one of [{tool_names}].
    Action Input: The input for the selected tool.
    """
    
    prompt = PromptTemplate.from_template(agent_prompt_template).partial(
        min_sources_required=REPORT_MIN_SOURCES,
        tool_names=", ".join([t.name for t in tools])
    )
    
    agent = create_react_agent(llm, tools, prompt)
    print("✅ [Setup]: AI Agent initialized successfully.")
    return agent, tools

# In helpers.py, replace the entire function with this one:

def run_research_task(query: str):
    """
    Main function to run the research process using a custom loop.
    This loop manages the agent's state and memory.
    """
    agent, tools = initialize_agent_and_tools()
    tool_map = {tool.name: tool for tool in tools}

    # State variables for our custom loop
    state = {
        "input": query,
        # This is the change: intermediate_steps will be a list of tuples
        "intermediate_steps": [], 
        "visited_urls": set(),
        "accumulated_data": {},
    }

    print(f"\n🚀 [Execution]: Starting research for: '{query}'")

    for i in range(AGENT_MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{AGENT_MAX_ITERATIONS} ---")
        
        # Format inputs for the agent, including the current state
        agent_input = {
            "input": state["input"],
            # Pass intermediate_steps here instead of agent_scratchpad
            "intermediate_steps": state["intermediate_steps"], 
            "sources_found_count": len(state["accumulated_data"]),
            "visited_urls": ", ".join(state["visited_urls"]) if state["visited_urls"] else "None",
            # No need to pass 'tools' here, as create_react_agent already has them.
            # If your prompt *explicitly* uses {tools}, it needs to be formatted.
            # However, the default create_react_agent often handles this internally
            # based on the `tools` passed during its creation.
        }
        
        # Run the agent to get the next action OBJECT
        # The agent.invoke will return an AgentAction or AgentFinish object
        agent_output = agent.invoke(agent_input)

        # Check if the agent has finished
        if isinstance(agent_output, AgentFinish):
            print(f"\n[Execution]: Agent has signaled FINISH. Final Answer: {agent_output.return_values['output']}")
            break # Exit the loop as the agent is done
            
        elif isinstance(agent_output, AgentAction):
            # We now have an AgentAction object
            agent_action = agent_output # Rename for clarity
            
            # Execute the chosen tool by accessing the object's attributes
            if agent_action.tool in tool_map:
                tool_to_use = tool_map[agent_action.tool]
                tool_input = agent_action.tool_input
                
                if tool_to_use.name == "Scrape":
                    state["visited_urls"].add(tool_input)

                observation = tool_to_use.run(tool_input)
                
                # Process the observation and update state
                if tool_to_use.name == "Scrape" and not observation.startswith("Error:"):
                    print(f"✅ [Success]: Scraped content from {tool_input}")
                    state["accumulated_data"][tool_input] = observation
                else:
                    print(f"⚪️ [Info]: Observation from '{tool_to_use.name}': {observation[:200]}...")

                # Append the action and observation to intermediate_steps
                state["intermediate_steps"].append((agent_action, observation))

            else:
                print(f"⚠️ [Warning]: Agent chose an invalid tool: {agent_action.tool}")
                # Append a dummy observation for the invalid tool to keep the history consistent
                state["intermediate_steps"].append((agent_action, f"Invalid tool '{agent_action.tool}' selected."))
        else:
            print(f"⚠️ [Warning]: Unexpected output type from agent.invoke: {type(agent_output)}")
            print(f"Output: {agent_output}")
            break # Break on unexpected output
            
        # Check if we have enough sources
        if len(state["accumulated_data"]) >= REPORT_MIN_SOURCES:
            print(f"\n[Execution]: Found {len(state['accumulated_data'])} sources, concluding research.")
            break # Exit the loop if enough sources are found
            
    # --- Post-Execution: Report Generation ---
    if len(state["accumulated_data"]) >= REPORT_MIN_SOURCES:
        final_report = _summarize_and_format_report(query, state["accumulated_data"])
        generate_pdf_report(final_report, query)
    else:
        print("\n [Failure]: Could not gather enough sources to generate a report.")
        print(f"   - Gathered: {len(state['accumulated_data'])}/{REPORT_MIN_SOURCES} sources.")
        print("   - Consider increasing AGENT_MAX_ITERATIONS or refining the query.")