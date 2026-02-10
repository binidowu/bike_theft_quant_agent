import os
import json
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from openai import OpenAI
from backend.quant_tools import (
    load_csv,
    list_columns,
    calculate_stats,
    average_by_category,
    plot_distribution,
    plot_average_by_category,
    get_dataset_summary,
    QuantAgentError,
    ERR_TOOL_TIMEOUT,
    ERR_INTERNAL_ERROR,
    DEFAULT_DATASET_PATH
)

# --- Configuration ---
MAX_TOOL_CALLS = 6
TOOL_TIMEOUT_SECONDS = 10.0
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

# --- Client Setup ---
# We initialize the client lazily or globally. 
# For mocking purposes, the `call_llm` adapter will access this.
_OPENAI_CLIENT = None

def _get_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # We don't raise error here, but call_llm will fail if key is missing
            pass
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT

# --- Tool Registry ---

# Map of tool names to actual python functions
TOOL_REGISTRY: Dict[str, Callable] = {
    "load_csv": load_csv,
    "list_columns": list_columns,
    "calculate_stats": calculate_stats,
    "average_by_category": average_by_category,
    "plot_distribution": plot_distribution,
    "plot_average_by_category": plot_average_by_category,
    "get_dataset_summary": get_dataset_summary,
}

# OpenAI Tool Definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load the dataset and view its metadata (rows, columns, preview). Call this first if you need to understand the data structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the CSV file. Optional, defaults to project dataset."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List all columns and distinguish between numeric and categorical types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the CSV file."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_stats",
            "description": "Calculate descriptive statistics (mean, median, etc.) for a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Name of the numeric column."},
                    "path": {"type": "string", "description": "Path to the CSV file."}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "average_by_category",
            "description": "Compute average of a value column grouped by a category column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_col": {"type": "string", "description": "Numeric column to average."},
                    "category_col": {"type": "string", "description": "Categorical column to group by."},
                    "top_n": {"type": "integer", "description": "Number of top results to return (default 20)."},
                    "ascending": {"type": "boolean", "description": "Sort order (default False/Descending)."},
                    "path": {"type": "string", "description": "Path to the CSV file."}
                },
                "required": ["value_col", "category_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_distribution",
            "description": "Generate and save a histogram for a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Numeric column to plot."},
                    "bins": {"type": "integer", "description": "Number of bins (default 30)."},
                    "path": {"type": "string", "description": "Path to the CSV file."}
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_average_by_category",
            "description": "Generate and save a bar chart of averages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_col": {"type": "string", "description": "Numeric column to average."},
                    "category_col": {"type": "string", "description": "Categorical column to group by."},
                    "top_n": {"type": "integer", "description": "Number of top bars (default 20)."},
                    "path": {"type": "string", "description": "Path to the CSV file."}
                },
                "required": ["value_col", "category_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_summary",
            "description": "Get a high-level summary of the dataset (row count, missing values, data types).",
             "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the CSV file."}
                }
            }
        }
    }
]

# --- LLM Adapter ---

def call_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str = OPENAI_MODEL):
    """
    Adapter to call OpenAI API. Mock this function in tests to avoid real API usage.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        return response.choices[0].message
    except Exception as e:
        raise QuantAgentError(ERR_INTERNAL_ERROR, f"LLM Call failed: {str(e)}")

# --- Agent Runtime ---


def _get_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def _get_message_tool_calls(message: Any) -> List[Any]:
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", []) or []


def _tool_call_to_request_dict(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        return {
            "id": tool_call.get("id"),
            "type": tool_call.get("type", "function"),
            "function": {
                "name": (tool_call.get("function") or {}).get("name"),
                "arguments": (tool_call.get("function") or {}).get("arguments", "{}"),
            },
        }

    fn = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", None),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(fn, "name", None),
            "arguments": getattr(fn, "arguments", "{}"),
        },
    }


def _assistant_message_to_request_dict(message: Any) -> Dict[str, Any]:
    msg_dict: Dict[str, Any] = {
        "role": "assistant",
        "content": _get_message_content(message),
    }
    tool_calls = _get_message_tool_calls(message)
    if tool_calls:
        msg_dict["tool_calls"] = [_tool_call_to_request_dict(tc) for tc in tool_calls]
    return msg_dict


def _parse_tool_call(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        func = tool_call.get("function") or {}
        return {
            "id": tool_call.get("id"),
            "name": func.get("name"),
            "arguments": func.get("arguments", "{}"),
        }

    fn = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", None),
        "name": getattr(fn, "name", None),
        "arguments": getattr(fn, "arguments", "{}"),
    }


def _build_trace_summary(result: Dict[str, Any]) -> str:
    data = result.get("data", {})
    rows = data.get("rows")
    if isinstance(rows, list):
        return f"Success. Returned {len(rows)} rows."
    if isinstance(rows, int):
        return f"Success. Rows: {rows}."
    if "relative_path" in data or "file_path" in data:
        return "Success. Plot generated."
    return "Success."


def _execute_tool_with_timeout(func: Callable, args: Dict[str, Any], func_name: str) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **args)
        try:
            result = future.result(timeout=TOOL_TIMEOUT_SECONDS)
            return result
        except FuturesTimeoutError as exc:
            future.cancel()
            raise QuantAgentError(
                ERR_TOOL_TIMEOUT,
                f"Tool {func_name} exceeded {TOOL_TIMEOUT_SECONDS}s limit.",
            ) from exc


def run_agent_query(question: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for the agent.
    
    Args:
        question: The user's question.
        dataset_path: Optional path to the dataset to use.
        
    Returns:
        Dict containing the final answer, tool traces, plot paths, etc.
    """
    
    # 1. Setup Context
    target_path = dataset_path if dataset_path else DEFAULT_DATASET_PATH

    # Re-validate target dataset path early via tool layer before LLM loop.
    load_csv(path=target_path)
    
    messages = [
        {"role": "system", "content": (
            "You are a Quantitative Agent specializing in bicycle theft data analysis. "
            "You satisfy user requests by calling provided python tools. "
            "You must base your answer ONLY on the tool outputs. "
            "Do not guess. If you generate a plot, mention it in the text. "
            f"The default dataset is at '{DEFAULT_DATASET_PATH}'. "
            "Always pass the 'path' argument to tools if you are working with a specific file, "
            "otherwise the tool uses the default. "
            "If the tool output contains a file path, return it."
        )},
        {"role": "user", "content": question}
    ]
    
    tool_calls_trace = []
    generated_plots = []
    generated_tables = []
    
    # 2. Execution Loop
    tool_calls_used = 0
    while tool_calls_used < MAX_TOOL_CALLS:
        # A. Call LLM
        message = call_llm(messages, TOOL_DEFINITIONS, OPENAI_MODEL)
        tool_calls = _get_message_tool_calls(message)
        
        # B. Check for Tool Calls
        if not tool_calls:
            # No more tools, we have the final answer
            return {
                "ok": True,
                "answer": _get_message_content(message),
                "tool_calls": tool_calls_trace,
                "tables": generated_tables,
                "plot_files": generated_plots,
                "metadata": {
                    "dataset_path": target_path,
                    "tool_calls_used": tool_calls_used
                }
            }
        
        # C. Execute Tools
        messages.append(_assistant_message_to_request_dict(message))
        
        for tool_call in tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                break

            parsed_tool_call = _parse_tool_call(tool_call)
            func_name = parsed_tool_call["name"]
            tool_call_id = parsed_tool_call["id"]
            args_str = parsed_tool_call["arguments"]

            if not func_name:
                raise QuantAgentError(ERR_INTERNAL_ERROR, "Tool call missing function name.")
            
            try:
                args = json.loads(args_str)
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must deserialize to an object.")
            except Exception as exc:
                raise QuantAgentError(
                    ERR_INTERNAL_ERROR,
                    f"Invalid tool arguments for {func_name}: {str(exc)}",
                ) from exc

            # Inject path if not present (although typically model should pass it or tool defaults)
            args.setdefault("path", target_path)

            # Create trace entry
            trace_entry = {
                "tool": func_name,
                "arguments": args,
                "ok": False,
                "summary": ""
            }
            
            # Look up function
            if func_name not in TOOL_REGISTRY:
                raise QuantAgentError(
                    ERR_INTERNAL_ERROR,
                    f"Unknown tool requested by model: {func_name}",
                )
            else:
                func = TOOL_REGISTRY[func_name]
                try:
                    result = _execute_tool_with_timeout(func, args, func_name)
                except QuantAgentError:
                    # Locked contract: propagate coded tool/runtime errors to API boundary.
                    raise
                except Exception as e:
                    raise QuantAgentError(
                        ERR_INTERNAL_ERROR,
                        f"Tool {func_name} failed: {str(e)}",
                    ) from e

            # Update ID for OpenAI
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result, default=str)
            }
            messages.append(tool_msg)
            
            # Harvest artifacts for response
            if result.get("ok"):
                trace_entry["ok"] = True
                trace_entry["summary"] = _build_trace_summary(result)
                
                # Collect plots
                if "file_path" in result.get("data", {}):
                    # We want relative path for API
                    if "relative_path" in result["data"]:
                        generated_plots.append(result["data"]["relative_path"])
                    else:
                        generated_plots.append(result["data"]["file_path"])
                
                # Collect structured tables
                if "rows" in result.get("data", {}):
                    generated_tables.append({
                        "name": func_name,
                        "rows": result["data"]["rows"]
                    })
            else:
                trace_entry["summary"] = f"Error: {result.get('error')}"
            
            tool_calls_trace.append(trace_entry)
            tool_calls_used += 1
        
    # If we hit max steps
    # We can either return what we have or error. 
    # Usually returning the last thought is better than crashing, 
    # but strictly speaking we didn't finish.
    # Let's return a "best effort" answer.
    return {
         "ok": False,
         "answer": "Reached maximum tool call limit without a final answer.",
         "tool_calls": tool_calls_trace,
         "tables": generated_tables,
         "plot_files": generated_plots,
         "metadata": {
            "dataset_path": target_path,
            "tool_calls_used": tool_calls_used
        }
    }
