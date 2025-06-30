import datetime
from typing import Dict, Any, List
from tools.base import BaseTool


class TimeTool(BaseTool):
    """Tool for getting current date and time."""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "Get the current date and time"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Date format string (optional, defaults to '%Y-%m-%d %H:%M:%S')",
                    "default": "%Y-%m-%d %H:%M:%S",
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get current date and time."""
        date_format = arguments.get("format", "%Y-%m-%d %H:%M:%S")

        try:
            current_time = datetime.datetime.now().strftime(date_format)
            result = f"Current date and time: {current_time}"

            return [{"type": "text", "text": result}]

        except Exception as e:
            return [{"type": "text", "text": f"Error getting time: {str(e)}"}]
