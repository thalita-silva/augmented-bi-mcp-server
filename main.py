from fastmcp import FastMCP

# Import the tools from your data-analytics-server.py
# Assuming that data-analytics-server.py is in the same folder as main.py or is in the PYTHONPATH.
import data_analytics_server

def main():
    print("Hello from augmented-bi-mcp!")

    # Run the FastMCP server (this will handle all tool requests)
    data_analytics_server.mcp.run()  # This runs the MCP server that listens for interactions

if __name__ == "__main__":
    main()


    