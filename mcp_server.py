from typing import Dict, Any
from tool_registry import ToolRegistry


class SimpleMCPServer:
    def __init__(self):
        self.tool_registry = ToolRegistry()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests."""
        method = request.get("method")
        request_id = request.get("id")

        try:
            if method == "tools/list":
                return {
                    "id": request_id,
                    "result": {"tools": self.tool_registry.list_tools()},
                }

            elif method == "tools/call":
                tool_name = request["params"]["name"]
                arguments = request["params"].get("arguments", {})

                result = self.tool_registry.execute_tool(tool_name, arguments)

                return {
                    "id": request_id,
                    "result": {"content": result},
                }

            else:
                return {
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }

        except Exception as e:
            return {
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }
