# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections.abc import Sequence
from typing import Union, Optional

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ToolParserManager.register_module("gemma3_patch")
class Gemma3PatchToolParser(ToolParser):
    """
    Tool parser for Gemma 3 models using patched chat template.
    
    This parser handles tool calls injected via the user query approach,
    maintaining compatibility with the original Gemma conversation structure
    while adding tool calling capabilities.
    
    Expected format: {"name": "function_name", "arguments": {"param": "value"}}
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # Regex patterns for different JSON tool call formats
        self.json_patterns = [
            # Standard format: {"name": "func", "arguments": {...}}
            re.compile(
                r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}',
                re.DOTALL
            ),
            # Alternative format with quotes around arguments
            re.compile(
                r'\{"name":\s*"([^"]+)",\s*"arguments":\s*"(\{[^}]*\})"\}',
                re.DOTALL
            ),
            # Spaced format
            re.compile(
                r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
                re.DOTALL
            )
        ]
        
        # Pattern for streaming detection
        self.partial_pattern = re.compile(r'\{"name":', re.DOTALL)
    
    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model response."""
        
        tool_calls = []
        remaining_content = model_output
        found_matches = []
        
        # Try each pattern to find tool calls
        for pattern in self.json_patterns:
            matches = pattern.findall(model_output)
            
            for name, args_str in matches:
                try:
                    # Clean up arguments string if it has extra quotes
                    if args_str.startswith('"') and args_str.endswith('"'):
                        args_str = args_str[1:-1]
                    
                    # Validate arguments JSON
                    args_dict = json.loads(args_str)
                    
                    tool_call = ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(args_dict, ensure_ascii=False)
                        )
                    )
                    tool_calls.append(tool_call)
                    
                    # Track what we found for content cleaning
                    full_match = f'{{"name": "{name}", "arguments": {args_str}}}'
                    found_matches.append(full_match)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse tool arguments: {args_str}, error: {e}")
                    continue
        
        # Clean tool calls from content
        for match in found_matches:
            remaining_content = remaining_content.replace(match, "").strip()
        
        # Clean up any leftover JSON artifacts
        remaining_content = re.sub(r'\{\s*"name":\s*[^}]*\}', '', remaining_content).strip()
        remaining_content = re.sub(r'\n\s*\n', '\n', remaining_content).strip()
        
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=remaining_content if remaining_content else None
        )
    
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """Extract tool calls from streaming response."""
        
        # Check if we're starting or continuing a tool call
        if self.partial_pattern.search(current_text):
            # Try to find a complete tool call in current text
            for pattern in self.json_patterns:
                match = pattern.search(current_text)
                if match:
                    name, args_str = match.groups()
                    
                    try:
                        # Clean up arguments
                        if args_str.startswith('"') and args_str.endswith('"'):
                            args_str = args_str[1:-1]
                        
                        # Validate JSON
                        json.loads(args_str)
                        
                        # Handle streaming state
                        if self.current_tool_id == -1:
                            # New tool call starting
                            self.current_tool_id = 0
                            self.current_tool_name_sent = False
                            self.streamed_args_for_tool = [""]
                        
                        delta_tool_call = DeltaToolCall(index=0)
                        
                        if not self.current_tool_name_sent:
                            delta_tool_call.function = DeltaFunctionCall(name=name)
                            self.current_tool_name_sent = True
                            return DeltaMessage(tool_calls=[delta_tool_call])
                        
                        # Stream arguments
                        new_args = args_str[len(self.streamed_args_for_tool[0]):]
                        if new_args:
                            self.streamed_args_for_tool[0] += new_args
                            delta_tool_call.function = DeltaFunctionCall(arguments=new_args)
                            return DeltaMessage(tool_calls=[delta_tool_call])
                            
                    except json.JSONDecodeError:
                        continue
        
        # Reset tool call state if we don't see tool patterns
        if not self.partial_pattern.search(current_text) and self.current_tool_id >= 0:
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
        
        # Return regular content delta
        return DeltaMessage(content=delta_text)
    
    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Adjust request parameters for better tool calling performance.
        """
        # No specific adjustments needed for this parser
        return request