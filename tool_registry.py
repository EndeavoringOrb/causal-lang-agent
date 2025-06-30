from typing import Dict, List, Any
from tools.base import BaseTool
from tools.time import TimeTool
from tools.python_interpreter import PythonInterpreterTool


class ToolRegistry:
    """Registry for managing all available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        default_tools = [TimeTool(), PythonInterpreterTool()]

        for tool in default_tools:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self._tools[tool.name] = tool

    def unregister_tool(self, tool_name: str):
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a tool by name."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self._tools[tool_name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools in MCP format."""
        return [tool.to_dict() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute a tool with given arguments."""
        tool = self.get_tool(tool_name)
        return tool.execute(arguments)
