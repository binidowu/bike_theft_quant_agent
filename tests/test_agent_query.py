import json
import pytest
from unittest.mock import MagicMock, patch
from backend.quant_agent_api import app
from backend.quant_tools import (
    ERR_UNSAFE_PATH,
    ERR_INVALID_COLUMN,
    ERR_BAD_REQUEST,
    ERR_TOOL_TIMEOUT,
    ERR_INTERNAL_ERROR
)

# --- Fixtures ---

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- Mock Response Helpers ---

def mock_llm_tool_call(name, args_dict, call_id="call_123"):
    """Helper to create a mocked tool call response."""
    mock_msg = MagicMock()
    mock_msg.content = None
    
    mock_tool_call = MagicMock()
    mock_tool_call.id = call_id
    mock_tool_call.type = "function"
    mock_tool_call.function.name = name
    mock_tool_call.function.arguments = json.dumps(args_dict)
    
    mock_msg.tool_calls = [mock_tool_call]
    return mock_msg

def mock_llm_final_answer(content):
    """Helper to create a mocked final answer response."""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_msg.tool_calls = None
    return mock_msg

# --- Tests ---

def test_agent_query_bad_request(client):
    """Test validation of missing question."""
    response = client.post('/agent/query', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == ERR_BAD_REQUEST

@patch("backend.agent_runtime.load_csv")
@patch("backend.agent_runtime.call_llm")
def test_agent_query_happy_path(mock_call, mock_load, client):
    """Test a simple tool usage -> answer flow."""
    mock_load.return_value = {"ok": True}
    # Sequence of mocks: 1 tool call, 1 final answer
    mock_call.side_effect = [
        mock_llm_tool_call("load_csv", {}),
        mock_llm_final_answer("The dataset is loaded and has 1000 rows.")
    ]
    
    response = client.post('/agent/query', json={"question": "Load the data."})
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert "dataset is loaded" in data["answer"]
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool"] == "load_csv"

@patch("backend.agent_runtime.call_llm")
def test_agent_query_unsafe_path(mock_call, client):
    """Unsafe tool path must surface as API-level client error."""
    mock_call.side_effect = [
        mock_llm_tool_call("load_csv", {"path": "../../etc/passwd"}),
    ]
    
    response = client.post('/agent/query', json={"question": "Load secret."})
    assert response.status_code == 400
    data = response.get_json()
    assert data["ok"] is False
    assert data["code"] == ERR_UNSAFE_PATH
    assert "not allowed" in data["error"]

@patch("backend.agent_runtime.load_csv")
@patch("backend.agent_runtime._execute_tool_with_timeout")
@patch("backend.agent_runtime.call_llm")
def test_agent_query_timeout(mock_call, mock_exec, mock_load, client):
    """Test tool timeout raising 500."""
    from backend.quant_tools import QuantAgentError
    mock_load.return_value = {"ok": True}
    mock_call.side_effect = [mock_llm_tool_call("load_csv", {})]
    mock_exec.side_effect = QuantAgentError(ERR_TOOL_TIMEOUT, "Timed out")
    
    response = client.post('/agent/query', json={"question": "Big task."})
    assert response.status_code == 500
    data = response.get_json()
    assert data["code"] == ERR_TOOL_TIMEOUT

@patch("backend.agent_runtime.load_csv")
@patch("backend.agent_runtime.call_llm")
def test_agent_query_truncation(mock_call, mock_load, client):
    """Test that response payload is truncated to 20 rows."""
    mock_load.return_value = {"ok": True}
    # Mock a tool returning 50 rows
    rows_50 = [{"id": i} for i in range(50)]
    mock_tool = MagicMock()
    mock_tool.return_value = {"ok": True, "data": {"rows": rows_50}}
    
    # Use patch.dict for dictionary mocking
    with patch.dict("backend.agent_runtime.TOOL_REGISTRY", {"average_by_category": mock_tool}):
        mock_call.side_effect = [
            mock_llm_tool_call("average_by_category", {"value_col": "x", "category_col": "y"}),
            mock_llm_final_answer("Done.")
        ]
        
        response = client.post('/agent/query', json={"question": "Lots of data."})
        assert response.status_code == 200
        data = response.get_json()
        assert len(data["tables"][0]["rows"]) == 20
        assert data["tables"][0]["truncated"] is True
        assert data["tables"][0]["total_rows"] == 50

@patch("backend.agent_runtime.load_csv")
@patch("backend.agent_runtime.call_llm")
def test_agent_query_max_calls(mock_call, mock_load, client):
    """Test limit of 6 tool calls."""
    mock_load.return_value = {"ok": True}
    # Mock 7 tool calls
    mock_call.side_effect = [mock_llm_tool_call("load_csv", {}) for _ in range(7)]
    
    response = client.post('/agent/query', json={"question": "Keep going."})
    # Our API now returns 500 if ok is False. Max cells results in ok=False.
    assert response.status_code == 500
    data = response.get_json()
    assert data["ok"] is False
    assert "maximum tool call limit" in data["error"]
    assert data["metadata"]["tool_calls_used"] == 6
