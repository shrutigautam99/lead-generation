from langchain_mcp_adapters.client import MultiServerMCPClient

def get_mcp_clients():
    """
    Initializes and returns a MultiServerMCPClient instance for interacting
    with multiple MCP servers, currently configured for Playwright automation.
    
    Returns:
        MultiServerMCPClient instance if successful, None otherwise.
    """
    try:
        # Create MCP client with a single Playwright server configuration
        mcp_clients = MultiServerMCPClient({
            "playwright": {
                "url": "http://localhost:8931/mcp",  # Local MCP server endpoint
                "transport": "streamable_http"       # Use streamable HTTP transport
            }
        })
        return mcp_clients
    except Exception as e:
        # Catch initialization errors and return None
        print(f"Error initializing MultiServerMCPClient: {e}")
        return None
