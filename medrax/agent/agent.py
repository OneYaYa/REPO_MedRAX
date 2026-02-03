import json
import operator
import re
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        
        # Build message list with system prompt if needed
        final_messages = messages
        
        if self.system_prompt and messages and not isinstance(messages[0], SystemMessage):
            final_messages = [SystemMessage(content=self.system_prompt)] + messages
        elif self.system_prompt and not messages:
            final_messages = [SystemMessage(content=self.system_prompt)]
        
        # CRITICAL FIX: Ensure message roles alternate correctly for vLLM/OpenAI API
        # Remove consecutive messages with the same role
        cleaned_messages = []
        for msg in final_messages:
            # Map message types to OpenAI roles
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                # ToolMessage should be converted to user message for vLLM
                role = "user"
            else:
                role = "unknown"
            
            # Check if this role is different from the last added message
            if cleaned_messages:
                last_msg = cleaned_messages[-1]
                last_role = "system" if isinstance(last_msg, SystemMessage) else \
                           "user" if isinstance(last_msg, (HumanMessage, ToolMessage)) else \
                           "assistant" if isinstance(last_msg, AIMessage) else "unknown"
                
                # If same role, merge content
                if role == last_role and role == "user":
                    # Merge user messages
                    if isinstance(last_msg.content, str) and isinstance(msg.content, str):
                        cleaned_messages[-1] = HumanMessage(
                            content=f"{last_msg.content}\n\n{msg.content}"
                        )
                    continue
                elif role == last_role and role == "assistant":
                    # Skip duplicate assistant messages
                    continue
            
            cleaned_messages.append(msg)
        
        try:
            response = self.model.invoke(cleaned_messages)
        except Exception as e:
            print(f"Error invoking model: {e}")
            response = HumanMessage(content=f"Error: {str(e)}")
        
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.
        Supports both:
        1. Structured tool_calls (LangChain format, used by Qwen3)
        2. Text-based tool_code blocks (generated by Gemma-3)

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        print(f"[DEBUG has_tool_calls] Response type: {type(response)}")
        print(f"[DEBUG has_tool_calls] Response content type: {type(response.content) if hasattr(response, 'content') else 'N/A'}")
        
        # Check structured tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls and len(response.tool_calls) > 0:
            print(f"[DEBUG has_tool_calls] Found {len(response.tool_calls)} structured tool calls")
            return True
        
        # Check for text-based tool_code blocks (Gemma-3 format)
        if isinstance(response, AIMessage) and isinstance(response.content, str):
            has_tool_code = '```tool_code' in response.content
            print(f"[DEBUG has_tool_calls] Text contains tool_code: {has_tool_code}")
            if has_tool_code:
                print(f"[DEBUG has_tool_calls] Content preview:\n{response.content[:500]}")
                return True
        
        print(f"[DEBUG has_tool_calls] No tool calls found")
        return False

    # NEW: Parse text-based tool calls from ```tool_code ... ``` blocks
    def _parse_text_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text-based tool calls from ```tool_code ... ``` blocks.
        Supports formats like:
        ```tool_code
        tool_name: chest_xray_classifier
        input: {"image_path": "..."}
        ```
        """
        calls = []
        # More flexible regex: allow optional whitespace and different line endings
        pattern = r'```tool_code\s*\n(.*?)\n```'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        
        print(f"[DEBUG] Searching for tool_code blocks in text of length {len(text)}")
        num_matches = sum(1 for _ in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        print(f"[DEBUG] Found {num_matches} tool_code blocks")
        
        for match in matches:
            block = match.group(1).strip()
            print(f"[DEBUG] Parsing tool_code block:\n{block}")
            
            # Try JSON parsing
            try:
                obj = json.loads(block)
                tool_name = obj.get("tool") or obj.get("tool_name")
                args = obj.get("parameters") or obj.get("input") or obj.get("args") or {}
                print(f"[DEBUG] JSON parse success: tool={tool_name}, args={args}")
                calls.append({
                    "name": tool_name,
                    "args": args,
                    "id": f"text_tool_{len(calls)}",
                })
                continue
            except Exception as e:
                print(f"[DEBUG] JSON parse failed: {e}")
            
            # Try key: value parsing
            try:
                tool_name = None
                args = {}
                for line in block.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        key, val = line.split(':', 1)
                        key = key.strip()
                        val = val.strip()
                        
                        if key.lower() in ('tool_name', 'tool'):
                            tool_name = val.strip('"\'')
                            print(f"[DEBUG] Found tool_name: {tool_name}")
                        elif key.lower() in ('input', 'parameters', 'args', 'arguments'):
                            # Try to parse as JSON
                            try:
                                args = json.loads(val)
                                print(f"[DEBUG] Parsed {key} as JSON: {args}")
                            except Exception as json_err:
                                print(f"[DEBUG] Failed to parse {key} as JSON ({json_err}), trying alternatives")
                                # If not JSON, treat as single parameter value
                                if '=' in val:
                                    # Parse key=value pairs
                                    for pair in val.split(','):
                                        if '=' in pair:
                                            k, v = pair.split('=', 1)
                                            args[k.strip()] = v.strip('"\'')
                                    print(f"[DEBUG] Parsed as key=value pairs: {args}")
                                else:
                                    args = {"input": val.strip('"\'') }
                                    print(f"[DEBUG] Treated as single input: {args}")
                        else:
                            # Treat other keys as direct arguments
                            args[key] = val.strip('"\'')
                            print(f"[DEBUG] Added direct arg: {key}={args[key]}")
                
                if tool_name:
                    parsed_call = {
                        "name": tool_name,
                        "args": args or {},
                        "id": f"text_tool_{len(calls)}",
                    }
                    print(f"[DEBUG] Successfully parsed tool call: {parsed_call}")
                    calls.append(parsed_call)
                else:
                    print(f"[DEBUG] No tool_name found in block")
            except Exception as e:
                print(f"[DEBUG] Error parsing tool call: {e}")
        
        print(f"[DEBUG] Total tool calls parsed: {len(calls)}")
        return calls

    def _normalize_tool_name(self, tool_name: str) -> Optional[str]:
        """
        Normalize tool names to handle variations from different models.
        Gemma-3 may generate names like 'image_analysis' or 'chestXray_classifier'
        instead of the actual registered names like 'chest_xray_classifier'.
        
        Args:
            tool_name (str): The tool name generated by the model.
            
        Returns:
            Optional[str]: The actual registered tool name, or None if no match found.
        """
        # Direct match
        if tool_name in self.tools:
            return tool_name
        
        # Common mappings for Gemma-3 generated names
        name_mappings = {
            'image_analysis': 'chest_xray_classifier',
            'chestxray_classifier': 'chest_xray_classifier',
            'xray_classifier': 'chest_xray_classifier',
            'chest_xray_segmentation': 'chest_xray_segmentation',
            'xray_segmentation': 'chest_xray_segmentation',
            'llava_med': 'llava_med',
            'xray_vqa': 'xray_vqa',
            'report_generator': 'chest_xray_report_generator',
            'phrase_grounding': 'xray_phrase_grounding',
            'image_visualizer': 'image_visualizer',
            'dicom_processor': 'dicom_processor',
        }
        
        # Try case-insensitive lookup in mappings
        normalized = tool_name.lower().replace('-', '_')
        if normalized in name_mappings:
            return name_mappings[normalized]
        
        # Try fuzzy matching: case-insensitive, ignore underscores
        for actual_name in self.tools.keys():
            if actual_name.lower().replace('_', '') == normalized.replace('_', ''):
                return actual_name
        
        # Try partial match
        for actual_name in self.tools.keys():
            if normalized in actual_name.lower() or actual_name.lower() in normalized:
                return actual_name
        
        return None

    def _extract_image_path_mapping(self, messages: List[AnyMessage]) -> Dict[str, str]:
        """
        Extract image path mapping from user messages.
        Looks for patterns like:
        [0] /path/to/image1.jpg
        [1] /path/to/image2.jpg
        
        Returns:
            Dict mapping index strings to actual paths
        """
        mapping = {}
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                # Find patterns like [0] /path/to/file
                pattern = r'\[(\d+)\]\s+([^\n]+)'
                matches = re.finditer(pattern, content)
                for match in matches:
                    idx = match.group(1)
                    path = match.group(2).strip()
                    # Only add if it looks like a file path
                    if '/' in path or '\\' in path:
                        mapping[idx] = path
        return mapping

    def _resolve_image_path(self, path_or_index: str, image_mapping: Dict[str, str]) -> str:
        """
        Resolve image path from either an index (like "0") or a full path.
        
        Args:
            path_or_index: Either an index string or a full path
            image_mapping: Dictionary mapping indices to paths
            
        Returns:
            Resolved path, or original value if not found
        """
        # If it's just a digit, try to map it
        if path_or_index.isdigit() and path_or_index in image_mapping:
            return image_mapping[path_or_index]
        return path_or_index

    def _extract_call_args(self, call: Any) -> Dict[str, Any]:
        """Normalize tool call arguments from either LangChain objects or dicts.

        vLLM / open models sometimes return `arguments` (string JSON) instead of `args`.
        We standardize to a dict, falling back to an empty dict when missing/invalid.
        """
        # Prefer dict-style access
        raw_args = None
        if isinstance(call, dict):
            raw_args = call.get("args") or call.get("arguments")
        else:
            raw_args = getattr(call, "args", None) or getattr(call, "arguments", None)

        if raw_args is None:
            return {}

        # If arguments come as a JSON string, try to decode
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                # Fallback: treat as a single input field
                return {"input": raw_args}

        # Accept only dict; otherwise fallback to empty dict
        return raw_args if isinstance(raw_args, dict) else {}

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.
        Supports both structured tool_calls and text-based tool_code blocks.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        response = state["messages"][-1]
        tool_calls = []
        
        # Get structured tool_calls if available
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
        
        # Get text-based tool_code blocks if available
        elif isinstance(response, AIMessage) and isinstance(response.content, str):
            if '```tool_code' in response.content:
                tool_calls = self._parse_text_tool_calls(response.content)
        
        # Extract image path mapping from conversation history
        image_mapping = self._extract_image_path_mapping(state["messages"])
        
        results = []
        for call in tool_calls:
            # Handle both dict and object formats
            tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = self._extract_call_args(call)
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "unknown")
            
            # Resolve image_path if it's an index reference
            if isinstance(args, dict) and 'image_path' in args:
                args['image_path'] = self._resolve_image_path(
                    str(args['image_path']), 
                    image_mapping
                )
            
            print(f"Executing tool: {tool_name} with args: {args}")
            
            # Normalize tool name to handle model variations
            normalized_name = self._normalize_tool_name(tool_name)
            
            if normalized_name is None:
                print(f"....invalid tool: {tool_name} (no matching tool found)....")
                result = f"invalid tool '{tool_name}', available tools: {list(self.tools.keys())}"
            else:
                if normalized_name != tool_name:
                    print(f"....normalized '{tool_name}' to '{normalized_name}'....")
                try:
                    result = self.tools[normalized_name].invoke(args)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
                    print(f"Tool execution error: {e}")

            # Use ToolMessage for structured tool calls (Qwen3, GPT-4)
            # Use HumanMessage for text-based tool calls (Gemma-3) to avoid vLLM compatibility issues
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Structured tool calls - use ToolMessage
                results.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        name=tool_name,
                        content=str(result),
                    )
                )
            else:
                # Text-based tool calls (Gemma-3) - use HumanMessage to avoid vLLM API issues
                # Format the tool result as a user message
                tool_result_text = f"Tool '{tool_name}' execution result:\n{result}"
                results.append(
                    HumanMessage(content=tool_result_text)
                )

        # NEW: Fix for message role alternation
        # If no tools were called, still need to return something to maintain message order
        # But since has_tool_calls() should prevent us from entering here, this is a safety check
        if not results:
            # This shouldn't happen, but if it does, log it
            print("WARNING: execute_tools called but no tool calls found!")
            # Return empty list - workflow will skip this if has_tool_calls is working correctly
            pass
        
        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    # BACKUP: Original execute_tools method below (commented out for reference)
    # def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
    #     """
    #     Execute tool calls from the model's response.
    #
    #     Args:
    #         state (AgentState): The current state of the agent.
    #
    #     Returns:
    #         Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
    #     """
    #     tool_calls = state["messages"][-1].tool_calls
    #     results = []
    #
    #     for call in tool_calls:
    #         print(f"Executing tool: {call}")
    #         if call["name"] not in self.tools:
    #             print("\n....invalid tool....")
    #             result = "invalid tool, please retry"
    #         else:
    #             result = self.tools[call["name"]].invoke(call["args"])
    #
    #         results.append(
    #             ToolMessage(
    #                 tool_call_id=call["id"],
    #                 name=call["name"],
    #                 args=call["args"],
    #                 content=str(result),
    #             )
    #         )
    #
    #     self._save_tool_calls(results)
    #     print("Returning to model processing!")
    #
    #     return {"messages": results}

    def _save_tool_calls(self, messages: List[AnyMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.
        Handles both ToolMessage (structured) and HumanMessage (text-based tool results).

        Args:
            messages (List[AnyMessage]): List of messages containing tool results to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Structured tool call (Qwen3, GPT-4)
                log_entry = {
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "args": getattr(msg, "args", None),  # ToolMessage may not have args
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),
                }
                logs.append(log_entry)
            elif isinstance(msg, HumanMessage):
                # Text-based tool result (Gemma-3) - extract info from content
                log_entry = {
                    "tool_call_id": "text_tool_result",
                    "name": "text_tool_result",
                    "args": None,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),
                }
                logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
        
        # BACKUP: Original code below
        # log_entry = {
        #     "tool_call_id": call.tool_call_id,
        #     "name": call.name,
        #     "args": call.args,  # 问题：ToolMessage 没有 args 属性
        #     "content": call.content,
        #     "timestamp": datetime.now().isoformat(),
        # }
