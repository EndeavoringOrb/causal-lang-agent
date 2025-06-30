import json
import platform
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging
from mcp_server import SimpleMCPServer
from utils.utils import extract_code

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient:
    def __init__(self):
        """
        Initialize MCP client to communicate with an MCP server.

        :param server_command: Command to start the MCP server (e.g., ['python', 'mcp_server.py'])
        """
        self.server = SimpleMCPServer()
        self.tools = {}
        self.next_id = 1

        response = self.server.handle_request(
            {"id": self._get_next_id(), "method": "tools/list"}
        )
        for tool_info in response["result"]["tools"]:
            tool = MCPTool(
                name=tool_info["name"],
                description=tool_info["description"],
                input_schema=tool_info["inputSchema"],
            )
            self.tools[tool.name] = tool
        logger.info(f"Listed {len(self.tools)} tools from MCP server.")

    def _get_next_id(self) -> int:
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool with the given arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        logger.info(f"Calling tool '{tool_name}' with arguments: {arguments}")
        response = self.server.handle_request(
            {
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )

        if "error" in response:
            logger.error(f"Tool call '{tool_name}' failed: {response['error']}")
            raise RuntimeError(f"Tool call failed: {response['error']['message']}")

        result_content = response.get("result", {}).get("content", [])
        logger.info(f"Tool '{tool_name}' returned: {result_content}")
        return result_content

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools formatted for LLM function calling."""
        tools = []
        for tool in self.tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return tools


class EnhancedLlamaServerClient:
    def __init__(
        self,
        base_url: str = (
            "http://localhost:55552"
            if platform.system() == "Windows"
            else "http://10.255.255.254:55552"
        ),
    ):
        """
        Enhanced Llama server client with MCP tool support.

        :param base_url: The base URL of the llama.cpp server
        :param mcp_client: Optional MCP client for tool access
        """
        self.base_url = base_url.rstrip("/")
        self.mcp_client = MCPClient()

    def generate_tool_selection_grammar(self, valid_names: list[str]) -> str:
        """Generate grammar for tool selection step."""
        name_choices = " | ".join([f'"\\"{name}\\""' for name in valid_names])
        name_choices = f"({name_choices})"

        grammar = (
            r"""
root ::= ("{" "\"tool_call\": " tool-call "}")
tool-call ::= ("{" tool-call-name-kv "}")
tool-call-name-kv ::= "\"name\": " """
            + name_choices
            + r"""
"""
        )
        return grammar

    def generate_parameter_grammar(self, param_schema: Dict[str, Any]) -> Optional[str]:
        """Generate grammar for a specific parameter based on its schema."""
        param_type = param_schema.get("type", "string")

        if param_type == "string":
            # For string parameters, allow any text (no grammar constraint)
            return None
        elif param_type == "python_code":
            return r'root ::= "```python\n" ("`" | "``" | [^`])* "```"'
        elif param_type == "number":
            return r'root ::= ("-")? ([0-9]+ ("." [0-9]+)?)'
        elif param_type == "integer":
            return r'root ::= ("-")? [0-9]+'
        elif param_type == "boolean":
            return r'root ::= ("true" | "false")'
        else:
            # For complex types, don't use grammar
            return None

    def _extract_required_parameters(self, tool_schema: Dict[str, Any]) -> List[str]:
        """Extract required parameters from tool schema."""
        return tool_schema.get("required", [])

    def _extract_parameter_schema(
        self, tool_schema: Dict[str, Any], param_name: str
    ) -> Dict[str, Any]:
        """Extract schema for a specific parameter."""
        properties = tool_schema.get("properties", {})
        return properties.get(param_name, {"type": "string"})

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        max_tool_calls: int = 5,
        **kwargs,
    ) -> Generator:
        """
        Generate response with tool calling support.

        :param messages: Conversation messages
        :param stream: Whether to stream the final response
        :param max_tool_calls: Maximum number of tool calls to allow
        :param kwargs: Additional parameters for the LLM
        """
        current_messages = messages.copy()
        tool_calls_made = 0

        while tool_calls_made < max_tool_calls:
            if not self.mcp_client:
                logger.debug("No MCP client, breaking from tool loop.")
                break

            tools = self.mcp_client.get_tools_for_llm()
            respond_tool_name = "respond"
            valid_tool_names = [tool["function"]["name"] for tool in tools] + [
                respond_tool_name
            ]

            # Step 1: Tool selection
            messages_for_tool_selection = current_messages.copy()
            system_message = self._create_tool_selection_system_message(tools)

            if (
                messages_for_tool_selection
                and messages_for_tool_selection[0].get("role") == "system"
            ):
                messages_for_tool_selection[0]["content"] += "\n\n" + system_message
            else:
                messages_for_tool_selection.insert(
                    0, {"role": "system", "content": system_message}
                )

            logger.info("Step 1: Getting tool selection from LLM...")
            tool_selection_response = self._make_generate_request(
                messages_for_tool_selection,
                grammar=self.generate_tool_selection_grammar(valid_tool_names),
            )
            tool_selection_response = next(tool_selection_response)

            logger.debug(f"Tool selection response: {tool_selection_response}")
            tool_call_parsed = self._extract_tool_selection(tool_selection_response)

            if not tool_call_parsed:
                logger.warning(
                    "LLM response did not contain a valid tool selection. Breaking."
                )
                break

            tool_name = tool_call_parsed["name"]

            if tool_name == respond_tool_name:
                # Step 2: Get the response text
                messages_for_response = current_messages.copy()
                response_system_message = "Provide your conversational response directly, without any JSON formatting."

                if (
                    messages_for_response
                    and messages_for_response[0].get("role") == "system"
                ):
                    messages_for_response[0]["content"] = response_system_message
                else:
                    messages_for_response.insert(
                        0, {"role": "system", "content": response_system_message}
                    )

                final_response = self._make_generate_request(
                    messages_for_response,
                    stream=stream,
                    **kwargs,
                )

                if isinstance(final_response, str):
                    yield final_response
                else:
                    for chunk in final_response:
                        yield chunk
                return

            # Real tool call
            tool_calls_made += 1
            if tool_calls_made > max_tool_calls:
                logger.warning(f"Max tool calls ({max_tool_calls}) reached.")
                break

            # Add tool selection to conversation
            current_messages.append(
                {
                    "role": "assistant",
                    "content": f"Selected tool: {tool_name}",
                }
            )

            print(f"[Tool Selection] {tool_name}")

            # Step 2: Collect parameters for the selected tool
            selected_tool = None
            for tool in tools:
                if tool["function"]["name"] == tool_name:
                    selected_tool = tool
                    break

            if not selected_tool:
                logger.error(
                    f"Selected tool '{tool_name}' not found in available tools"
                )
                break

            tool_schema = selected_tool["function"]["parameters"]
            required_params = self._extract_required_parameters(tool_schema)
            all_properties = tool_schema.get("properties", {})

            # Collect all parameters (required and optional)
            parameters_to_collect = list(all_properties.keys())

            arguments = {}

            for param_name in parameters_to_collect:
                param_schema = self._extract_parameter_schema(tool_schema, param_name)
                is_required = param_name in required_params

                # Create messages for parameter collection
                messages_for_param = current_messages.copy()

                param_system_message = self._create_parameter_system_message(
                    tool_name, param_name, param_schema, is_required
                )

                if messages_for_param and messages_for_param[0].get("role") == "system":
                    messages_for_param[0]["content"] = param_system_message
                else:
                    messages_for_param.insert(
                        0, {"role": "system", "content": param_system_message}
                    )

                # Add a user message asking for the parameter
                messages_for_param.append(
                    {
                        "role": "user",
                        "content": f"Please provide the {param_name} parameter for the {tool_name} tool.",
                    }
                )

                logger.info(
                    f"Step 2.{len(arguments)+1}: Getting parameter '{param_name}' for tool '{tool_name}'"
                )

                # Generate parameter value
                param_grammar = self.generate_parameter_grammar(param_schema)
                param_response = self._make_generate_request(
                    messages_for_param,
                    grammar=param_grammar,
                    **kwargs,
                )
                param_response = next(param_response)

                param_value = (
                    param_response.strip() if isinstance(param_response, str) else ""
                )

                # Handle type conversion
                param_value = self._convert_parameter_value(param_value, param_schema)

                if param_value is not None or is_required:
                    arguments[param_name] = param_value
                    print(f"[Parameter] {param_name}: {param_value}")

            # Step 3: Execute the tool
            try:
                tool_result = self.mcp_client.call_tool(tool_name, arguments)
                print(
                    f"[Tool Result] {[item.get('text', '')[:100] for item in tool_result]}..."
                )

                # Add tool execution to conversation
                current_messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {"tool": tool_name, "result": tool_result}
                        ),
                    }
                )

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                current_messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"tool": tool_name, "error": str(e)}),
                    }
                )

        # Final conversational response if no explicit respond was called
        logger.info("Generating final conversational response.")

        final_messages = current_messages.copy()
        if (
            final_messages
            and final_messages[0].get("role") == "system"
            and (
                "Select a tool" in final_messages[0]["content"]
                or "Provide" in final_messages[0]["content"]
            )
        ):
            final_messages[0][
                "content"
            ] = "Provide a helpful conversational response based on the context."

        final_response = self._make_generate_request(
            final_messages,
            stream=stream,
            **kwargs,
        )

        if isinstance(final_response, str):
            yield final_response
        else:
            for chunk in final_response:
                yield chunk

    def _create_tool_selection_system_message(self, tools: List[Dict[str, Any]]) -> str:
        """Create system message for tool selection step."""
        tool_descriptions = []
        for tool in tools:
            func = tool["function"]
            tool_descriptions.append(f"- {func['name']}: {func['description']}")

        tool_descriptions.append(
            "- respond: Use this to respond to the user"
        )

        return f"""Select a tool to use from the following options:

{chr(10).join(tool_descriptions)}

Respond with JSON in this exact format:
{{"tool_call": {{"name": "tool_name"}}}}"""

    def _create_parameter_system_message(
        self,
        tool_name: str,
        param_name: str,
        param_schema: Dict[str, Any],
        is_required: bool,
    ) -> str:
        """Create system message for parameter collection step."""
        param_type = param_schema.get("type", "string")
        param_desc = param_schema.get("description", "")
        default_value = param_schema.get("default")

        message = f"""You are collecting the '{param_name}' parameter for the {tool_name} tool.

Parameter details:
- Name: {param_name}
- Type: {param_type}
- Description: {param_desc}
- Required: {is_required}"""

        if default_value is not None:
            message += f"\n- Default: {default_value}"

        if param_type == "string":
            message += "\n\nProvide the parameter value as plain text (no quotes or JSON formatting)."
        elif param_type in ["number", "integer"]:
            message += "\n\nProvide the parameter value as a number."
        elif param_type == "boolean":
            message += "\n\nProvide the parameter value as 'true' or 'false'."
        else:
            message += "\n\nProvide the parameter value directly."

        if not is_required:
            message += " If you don't want to provide this optional parameter, respond with 'null' or leave empty."

        return message

    def _convert_parameter_value(self, value: str, param_schema: Dict[str, Any]):
        """Convert parameter value to appropriate type based on schema."""
        if not value or value.lower() in ["null", "none", ""]:
            return param_schema.get("default")

        param_type = param_schema.get("type", "string")

        try:
            if param_type == "integer":
                return int(float(value))  # Handle cases like "5.0" -> 5
            elif param_type == "python_code":
                answer, code, _ = extract_code(value)
                return code
            elif param_type == "number":
                return float(value)
            elif param_type == "boolean":
                return value.lower() in ["true", "1", "yes", "on"]
            else:
                return value
        except (ValueError, TypeError):
            return value  # Return as string if conversion fails

    def _extract_tool_selection(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool selection from LLM response."""
        try:
            obj = json.loads(response)
            if isinstance(obj, dict) and "tool_call" in obj:
                tool_call = obj["tool_call"]
                if isinstance(tool_call, dict) and "name" in tool_call:
                    return {"name": tool_call["name"]}
        except json.JSONDecodeError:
            logger.debug(f"Tool selection response not valid JSON: {response}")
        return None

    def _make_generate_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        grammar: Optional[str] = None,
        **kwargs,
    ):
        """Make a generation request to the LLM server."""
        import requests

        # Apply template first
        response = requests.post(
            f"{self.base_url}/apply-template", json={"messages": messages}
        )
        response.raise_for_status()
        prompt = response.json()["prompt"]
        logger.debug(f"Applied template, prompt length: {len(prompt)}")

        with open("full_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        payload = {
            "prompt": prompt,
            "stream": stream,
            **kwargs,
        }
        if grammar:
            payload["grammar"] = grammar
            logger.debug("Using grammar for generation.")

        url = f"{self.base_url}/completion"
        logger.debug(f"Sending request to {url}")
        response = requests.post(url, json=payload, stream=payload["stream"])
        response.raise_for_status()

        if stream:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line == "data: [DONE]":
                        break
                    try:
                        line_data = json.loads(line[5:])  # Strip "data: " prefix
                        if "content" in line_data:
                            yield line_data["content"]
                        else:
                            print(line_data)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} - Line: {line}")
                        continue
        else:
            content: str = response.json()["content"]
            
            logger.debug(f"Response received, length: {len(content)}")
            yield content


if __name__ == "__main__":
    client = EnhancedLlamaServerClient()
    while True:
        text = input("Enter input: ")
        answer = ""
        for chunk in client.generate_with_tools([{"role": "user", "content": text}]):
            answer += chunk
            print(chunk, end="", flush=True)
        print("\n" * 3)
