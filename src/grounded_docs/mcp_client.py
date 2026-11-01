"""MCP client helpers: discover tools and invoke them over Streamable HTTP."""
from __future__ import annotations

import json
import logging
from typing import Any

from mcp import Client

from grounded_docs.logging_utils import log_payload

logger = logging.getLogger(__name__)


async def discover_openai_tools(mcp_url: str) -> list[dict[str, Any]]:
    """Discover MCP tools and convert their JSON schemas to OpenAI function-tool format."""
    logger.info("MCP client connecting to discover tools at %s", mcp_url)
    async with Client(mcp_url) as client:
        logger.info("MCP client requesting tool list with list_tools()")
        result = await client.list_tools()
        log_payload(
            logger,
            "MCP server tools discovered",
            [
                {
                    "name": tool.name,
                    "description": tool.description or tool.title or tool.name,
                    "input_schema": tool.input_schema,
                }
                for tool in result.tools
            ],
        )
        converted = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or tool.title or tool.name,
                    "parameters": tool.input_schema,
                },
            }
            for tool in result.tools
        ]
        log_payload(logger, "Tool definitions prepared for the LLM", converted)
        return converted


async def invoke_mcp_tool(mcp_url: str, name: str, arguments: dict[str, Any]) -> str:
    logger.info("MCP client connecting to invoke tool %s at %s", name, mcp_url)
    log_payload(logger, "MCP client tool-call arguments", {"name": name, "arguments": arguments})
    async with Client(mcp_url) as client:
        result = await client.call_tool(name, arguments)
        if result.is_error:
            blocks = [getattr(block, "text", "") for block in result.content]
            response = "MCP tool error: " + "\n".join(block for block in blocks if block)
        else:
            blocks = [getattr(block, "text", "") for block in result.content]
            text_content = "\n".join(block for block in blocks if block)
            if text_content:
                response = text_content
            elif result.structured_content is not None:
                response = json.dumps(result.structured_content, ensure_ascii=False)
            else:
                response = ""
        log_payload(logger, "MCP client received tool response", response)
        return response
