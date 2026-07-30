import asyncio
from mcp.server.mcpserver import MCPServer
import ollama

# 1. Initialize the MCP Server
# We define a server that will act as the data layer for our agents.
mcp = MCPServer("ResearchServer")

# Expose a tool to the agents via MCP
@mcp.tool()
def read_company_report(section: str) -> str:
    """Simulates reading a specific section of a local financial report."""
    data_store = {
        "financials": "Q3 Revenue grew by 15% due to AI infrastructure investments.",
        "risks": "Supply chain delays in GPU procurement are causing project bottlenecks.",
        "strategy": "Shifting focus to edge AI computing and multi-agent workflows."
    }
    return data_store.get(section.lower(), "Section not found.")

# 2. Define the Multi-Agent Workflow
async def run_multi_agent_system():
    print("Starting Multi-Agent System via MCP...")
    
    # In a fully distributed system, the MCP server would run on a separate port.
    # For this tutorial, we are calling the tool directly to simulate the client-server interaction 
    # without needing complex asynchronous subprocess management in a single script.
    
    # Agent 1: The Researcher (Data Extraction)
    target_section = "financials"
    print(f"\n[Researcher Agent] Requesting data for: {target_section}")
    
    # The agent uses the standardized MCP tool to get context
    raw_data = read_company_report(target_section)
    print(f"[Researcher Agent] Retrieved Data: {raw_data}")
    
    # Agent 2: The Editor (Formatting and Synthesis)
    print("\n[Editor Agent] Processing raw data into a professional summary...")
    
    prompt = (
        f"You are an expert financial editor. Summarize the following raw data "
        f"into a single, professional sentence suitable for an executive brief.\n"
        f"Raw Data: {raw_data}"
    )
    
    # Using Ollama for a 100% free, local inference
    response = ollama.chat(model='mistral', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    final_output = response['message']['content']
    print(f"\n[Final Output]\n{final_output}")

if __name__ == "__main__":
    asyncio.run(run_multi_agent_system())