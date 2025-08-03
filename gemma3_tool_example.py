#!/usr/bin/env python3
"""
Example demonstrating Gemma 3 tool calling with patched chat template.

This example shows how to use Gemma 3 models with tool calling capabilities
by using a minimally modified chat template that preserves the original
conversation structure while adding tool support.
"""

import json
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Initialize Gemma 3 with patched tool template
llm = LLM(
    model="google/gemma-2-7b-it",  # or gemma-3-7b-instruct when available
    chat_template="/Users/tenzingayche/Desktop/vllm/gemma3_tool_patch.jinja",
    max_model_len=4096,
    temperature=0.1
)

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Tool implementations for demonstration
def get_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather function"""
    temp = "22°C" if unit == "celsius" else "72°F"
    return f"The weather in {location} is sunny with a temperature of {temp}."

def calculate(expression: str) -> str:
    """Safe calculation function"""
    try:
        # Only allow basic math operations for safety
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid expression: only basic math operations allowed"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def search_web(query: str, max_results: int = 5) -> str:
    """Mock web search function"""
    return f"Found {max_results} results for '{query}': [Mock search results would appear here]"

# Available functions mapping
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web
}

def run_conversation(messages, max_turns=5):
    """Run a conversation with tool calling support"""
    current_messages = messages.copy()
    
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")
        
        # Generate response
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            stop_token_ids=[107]  # Gemma's end token
        )
        
        outputs = llm.chat(current_messages, sampling_params, tools=tools)
        response_text = outputs[0].outputs[0].text.strip()
        
        print(f"Model response: {response_text}")
        
        # Check if response contains tool calls
        tool_calls = []
        
        # Simple regex to extract tool calls (in production, use the parser)
        import re
        pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}'
        matches = re.findall(pattern, response_text)
        
        if matches:
            print("Tool calls detected:")
            for name, args_str in matches:
                try:
                    args = json.loads(args_str)
                    print(f"  - {name}({json.dumps(args)})")
                    
                    # Execute function if available
                    if name in available_functions:
                        result = available_functions[name](**args)
                        print(f"    Result: {result}")
                        
                        # Add tool result to conversation
                        current_messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        current_messages.append({
                            "role": "tool",
                            "content": result,
                            "name": name
                        })
                        
                        tool_calls.append((name, args, result))
                    else:
                        print(f"    Function {name} not available")
                        
                except json.JSONDecodeError as e:
                    print(f"    Error parsing arguments: {e}")
        
        # If no tool calls or we got results, we can break
        if not matches:
            current_messages.append({
                "role": "assistant", 
                "content": response_text
            })
            break
    
    return current_messages

# Example conversations
def example_weather():
    """Example: Weather query"""
    print("=== Weather Query Example ===")
    messages = [
        {"role": "user", "content": "What's the weather like in Tokyo?"}
    ]
    run_conversation(messages)

def example_calculation():
    """Example: Math calculation"""
    print("\n=== Calculation Example ===") 
    messages = [
        {"role": "user", "content": "Can you calculate 15 * 23 + 47?"}
    ]
    run_conversation(messages)

def example_mixed():
    """Example: Mixed conversation with tools and regular chat"""
    print("\n=== Mixed Conversation Example ===")
    messages = [
        {"role": "user", "content": "Hi! Can you help me with some tasks?"},
        {"role": "assistant", "content": "Hello! I'd be happy to help you with various tasks. I can check weather, do calculations, search the web, or just chat. What would you like to do?"},
        {"role": "user", "content": "Great! First, what's the weather in London, and then calculate 25% of 200."}
    ]
    run_conversation(messages, max_turns=3)

if __name__ == "__main__":
    # Run examples
    example_weather()
    example_calculation() 
    example_mixed()
    
    print("\n=== Server Usage ===")
    print("To use with vLLM server:")
    print("vllm serve google/gemma-2-7b-it \\")
    print("    --chat-template gemma3_tool_patch.jinja \\")
    print("    --tool-call-parser gemma3_patch \\")
    print("    --enable-auto-tool-choice")