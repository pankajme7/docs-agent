"""MCP server exposing the same document retrieval function used by direct RAG."""
from __future__ import annotations

import json
import logging

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from grounded_docs.config import Settings
from grounded_docs.logging_utils import log_payload
from grounded_docs.retrieval import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = Settings.from_env()
retriever = Retriever(settings)
mcp = MCPServer(
    "GroundedDocs",
    instructions="Search the curated Kubernetes documentation corpus. Use search_docs before answering documentation questions.",
)


@mcp.tool(title="Search Kubernetes documentation")
def search_docs(query: str, top_k: int = 4) -> str:
    """Search indexed Kubernetes documentation and return ranked text chunks with source URLs.

    Args:
        query: Natural-language question or search phrase.
        top_k: Maximum number of chunks to return (1-8).
    """
    if not query.strip():
        raise ToolError("query must not be empty")
    if not 1 <= top_k <= 8:
        raise ToolError("top_k must be between 1 and 8")
    logger.info("MCP server received search_docs request")
    chunks = retriever.search(query, top_k)
    response = json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False, indent=2)
    log_payload(logger, "MCP server response to MCP client", response)
    return response


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
    )
