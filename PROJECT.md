# Quantitative Agent Implementation Plan

This document captures the locked implementation plan for adding a quantitative agent to this project.

## Goal Description

Evolve the existing prediction app into a quantitative agent. The agent will answer user questions about the dataset by autonomously selecting and executing Python tools (`pandas`, `numpy`, `matplotlib`) and returning grounded answers with tool traces and plot files.

## Locked Constraints

No changes to the existing `/predict` endpoint behavior.  
Add a new endpoint: `POST /agent/query`.  
Save generated plots to `outputs/plots/`.  
Keep Flask as the backend framework (no FastAPI/Pydantic migration).  
Use manual JSON validation in Flask handlers.  
End-to-end tests must mock the LLM client for deterministic results.

## Key Refinements

Use a `call_llm` adapter pattern for clean test mocking.  
Use strict path safety with `pathlib.Path.resolve().is_relative_to()` so dataset files must be physically inside `data/`.  
Define a canonical `DEFAULT_DATASET_PATH` constant in `quant_tools.py`.  
Enforce row limits (`max 20`) in `quant_agent_api.py` response formatting (not in core quant tool logic).  
Use a locked error taxonomy based on constants plus a custom exception class.  
Use runtime limits: max 6 tool calls, 10 seconds timeout per tool call.  
Normalize plot filename components (alphanumeric + underscore) and generate timestamped filenames via a deterministic helper.

## Error Taxonomy (Locked)

Define constants and use them everywhere (no raw string literals in handlers/tests):

- `ERR_UNSAFE_PATH`
- `ERR_INVALID_COLUMN`
- `ERR_BAD_REQUEST`
- `ERR_TOOL_TIMEOUT`
- `ERR_INTERNAL_ERROR`

Define:

```python
class QuantAgentError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)
```

### Error Mapping to HTTP Status

`ERR_UNSAFE_PATH`, `ERR_INVALID_COLUMN`, `ERR_BAD_REQUEST` -> HTTP `400`  
`ERR_TOOL_TIMEOUT`, `ERR_INTERNAL_ERROR` -> HTTP `500`

Standard error response schema:

```json
{
  "ok": false,
  "error": "Message",
  "code": "ERROR_CODE"
}
```

## Proposed Changes

### Configuration and Dependencies

Modify `requirements.txt`:
- Add `openai>=1.0,<2.0`

Modify `.env.example`:
- Add `OPENAI_API_KEY=`
- Add optional `OPENAI_MODEL=`

### Backend

#### New: `backend/quant_tools.py`

Define canonical dataset constants:
- `DATA_DIR`
- `DEFAULT_DATASET_PATH` (env override or fallback to project dataset)

Define error constants:
- `ERR_UNSAFE_PATH`, `ERR_INVALID_COLUMN`, `ERR_BAD_REQUEST`, `ERR_TOOL_TIMEOUT`, `ERR_INTERNAL_ERROR`

Define `QuantAgentError`.

Implement pure tool functions:
- `load_csv`
  - Loads and caches dataframe.
  - Validates path via resolved-path allowlisting.
  - Raises `QuantAgentError(code=ERR_UNSAFE_PATH, ...)` when invalid.
- `list_columns`
  - Returns all columns and typed subsets.
- `calculate_stats`
  - Returns descriptive stats for numeric columns.
  - Raises `QuantAgentError(code=ERR_INVALID_COLUMN, ...)` when missing.
- `average_by_category`
  - Group-by mean summary with reasonable internal limit.
- `plot_distribution`
  - Saves histogram to `outputs/plots/plot_dist_{normalized_col}_{timestamp}.png`.
- `plot_average_by_category`
  - Saves grouped bar plot to `outputs/plots/plot_avg_{normalized_val}_{normalized_cat}_{timestamp}.png`.
- `get_dataset_summary`
  - High-level dataset metadata and quality overview.

#### New: `backend/agent_runtime.py`

Implement orchestration:
- `call_llm(messages, tools, model_name)` adapter around OpenAI SDK.
- `run_agent_query(...)` main entrypoint for tool-calling loop.
- Tool registry.
- Trace recording.
- Enforce:
  - `MAX_TOOL_CALLS = 6`
  - `TOOL_TIMEOUT = 10` seconds per tool call
- Raise `QuantAgentError(code=ERR_TOOL_TIMEOUT, ...)` for timeout failures.

#### Modify: `backend/quant_agent_api.py`

Add `POST /agent/query`:
- Manual request JSON validation.
- Raise `QuantAgentError(code=ERR_BAD_REQUEST, ...)` for malformed client input.
- Integrate `run_agent_query`.
- Truncate table payloads to max 20 rows in response formatting.

Error handling in route:
- Catch `QuantAgentError` and map by `.code`.
- Catch generic exceptions and return:
  - HTTP `500`
  - `code = ERR_INTERNAL_ERROR`

## Testing Plan

### New: `tests/test_quant_tools.py`

Cover:
- `load_csv` success and unsafe-path rejection.
- Path traversal attempts (e.g., `../`).
- `calculate_stats` invalid column behavior (`ERR_INVALID_COLUMN`).
- Plot filename normalization rules.
- `QuantAgentError` code/message behavior.

### New: `tests/test_agent_query.py`

Cover:
- End-to-end `/agent/query` behavior with mocked LLM adapter.
- Mock `agent_runtime.call_llm` (not SDK internals).
- Assert HTTP code mapping:
  - 400 for client-side errors.
  - 500 for runtime/internal errors.
- Assert returned `code` values (`ERR_INVALID_COLUMN`, `ERR_UNSAFE_PATH`, etc.).
- Assert row truncation to max 20 in API responses.

## Documentation Updates

Modify `README.md`:
- Add quantitative-agent architecture.
- Add tool signatures.
- Add sample query/response pairs.
- Add explicit error schema and HTTP code mapping.
- Explain that `/predict` remains unchanged while `/agent/query` is the new agent path.

## Verification Plan

### Automated Verification

Run test suite (`pytest`) and verify:
- Safe dataset loading + unsafe path rejection.
- Plot generation with normalized timestamped filenames.
- Deterministic mocked agent responses.
- Correct error codes and HTTP statuses.
- Table row truncation at 20 rows in API responses.

### Manual Verification

Set `OPENAI_API_KEY`, start API, and send live requests to `POST /agent/query`.

Confirm:
- Valid quantitative queries return grounded answers + tool trace.
- Plot-producing queries return real file paths under `outputs/plots/`.
- Large result tables are truncated in response payload.
- Malformed input produces expected `BAD_REQUEST` schema.

## Implementation Task List

- Create `backend/quant_tools.py`.
- Define dataset constants and canonical default dataset path.
- Define error constants and `QuantAgentError`.
- Implement `load_csv`, `list_columns`, `calculate_stats`, `average_by_category`, `plot_distribution`, `plot_average_by_category`, `get_dataset_summary`.
- Create `backend/agent_runtime.py`.
- Implement `call_llm` adapter and tool registry.
- Implement agent loop with max-call and timeout enforcement.
- Update `backend/quant_agent_api.py`:
  - Add `POST /agent/query`.
  - Add manual validation.
  - Add error mapping handler.
  - Add response row truncation.
- Create tests:
  - `tests/test_quant_tools.py`
  - `tests/test_agent_query.py`
- Update `README.md` with new architecture and error schema.

## Status

Planning complete and locked.  
Implementation has not started from this document.
