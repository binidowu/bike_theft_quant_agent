# Quantitative Agent Walkthrough

This document summarizes the quantitative agent implementation and how to verify it.

## What Was Added

The backend now supports a tool-calling quantitative agent endpoint:

- `POST /agent/query`

The endpoint orchestrates LLM tool use through:

- `backend/agent_runtime.py` (`call_llm`, tool registry, runtime loop)
- `backend/quant_tools.py` (data/stat/plot tools)

## Safety and Limits

- Strict path allowlisting to `data/` via resolved path checks.
- Locked error taxonomy using `QuantAgentError` and constants:
  - `UNSAFE_PATH`
  - `INVALID_COLUMN`
  - `BAD_REQUEST`
  - `TOOL_TIMEOUT`
  - `INTERNAL_ERROR`
- Runtime guardrails:
  - max tool calls = 6
  - per-tool timeout = 10 seconds
- API response table truncation to 20 rows.

## Behavior

`/agent/query` returns:

- `answer`
- `tool_calls` trace
- `tables`
- `plot_files`
- `metadata`

Client/domain errors return HTTP 400 with locked codes.
Runtime/system errors return HTTP 500 with locked codes.

## Testing

Automated tests are in:

- `tests/test_quant_tools.py`
- `tests/test_agent_query.py`

They cover:

- tool logic and path safety
- `/agent/query` orchestration with mocked LLM adapter
- timeout/error code mapping
- payload truncation

