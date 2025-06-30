from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseTool(ABC):
    """Base class for all MCP tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for tool input validation."""
        pass

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the tool with given arguments."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to MCP tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }