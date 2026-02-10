import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.quant_tools import (
    load_csv,
    list_columns,
    calculate_stats,
    average_by_category,
    plot_distribution,
    plot_average_by_category,
    get_dataset_summary,
    QuantAgentError,
    ERR_UNSAFE_PATH,
    ERR_INVALID_COLUMN,
    DEFAULT_DATASET_PATH,
    PLOT_OUTPUT_DIR
)

# --- Fixtures ---

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a temporary sample CSV file for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "test_data.csv"
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["x", "y", "x", "y", "z"],
        "C": [10.5, 20.0, 15.2, 5.0, 100.0]
    })
    df.to_csv(file_path, index=False)
    return str(file_path)

# --- Tests ---

def test_load_csv_default():
    """Test load_csv with the default dataset path."""
    # This assumes the default data/bicycle_thefts_open_data.csv exists
    # or it will fail if the file is missing (ERR_BAD_REQUEST).
    try:
        result = load_csv()
        assert result["ok"] is True
        assert "rows" in result["data"]
    except QuantAgentError as e:
        # If file missing, it's not a logic error in the tool
        if "File not found" in e.message:
            pytest.skip("Default dataset not found, skipping default test.")
        raise e

def test_path_safety_traversal():
    """Test that path traversal outside data/ is rejected."""
    with pytest.raises(QuantAgentError) as excinfo:
        # Attempt to access something outside 'data/'
        load_csv(path="../../etc/passwd")
    assert excinfo.value.code == ERR_UNSAFE_PATH

def test_list_columns(sample_csv, monkeypatch):
    """Test list_columns returns expected columns."""
    # Monkeypatch Project Root to use tmp_path
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    
    # We must use relative path to tmp_path/data/test_data.csv
    rel_path = f"data/{Path(sample_csv).name}"
    
    result = list_columns(path=rel_path)
    assert result["ok"] is True
    assert set(result["data"]["all_columns"]) == {"A", "B", "C"}
    assert "A" in result["data"]["numeric_columns"]
    assert "B" in result["data"]["categorical_columns"]

def test_calculate_stats_numeric(sample_csv, monkeypatch):
    """Test calculate_stats on a numeric column."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    result = calculate_stats(column="A", path=rel_path)
    assert result["ok"] is True
    assert result["data"]["stats"]["mean"] == 3.0
    assert result["data"]["stats"]["missing"] == 0

def test_calculate_stats_invalid_column(sample_csv, monkeypatch):
    """Test calculate_stats with a missing column."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    with pytest.raises(QuantAgentError) as excinfo:
        calculate_stats(column="NON_EXISTENT", path=rel_path)
    assert excinfo.value.code == ERR_INVALID_COLUMN

def test_calculate_stats_non_numeric(sample_csv, monkeypatch):
    """Test calculate_stats on a categorical column."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    with pytest.raises(QuantAgentError) as excinfo:
        calculate_stats(column="B", path=rel_path)
    assert excinfo.value.code == ERR_INVALID_COLUMN

def test_average_by_category(sample_csv, monkeypatch):
    """Test average_by_category logic."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    result = average_by_category(value_col="A", category_col="B", path=rel_path)
    assert result["ok"] is True
    rows = result["data"]["rows"]
    # x average (1+3)/2 = 2
    # y average (2+4)/2 = 3
    # z average (5)/1 = 5
    # Default is descending: z(5), y(3), x(2)
    assert rows[0]["B"] == "z"
    assert rows[0]["mean"] == 5.0

def test_plot_distribution(sample_csv, monkeypatch):
    """Test plot_distribution saves a file and returns path."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    result = plot_distribution(column="A", path=rel_path)
    assert result["ok"] is True
    assert "relative_path" in result["data"]
    
    # Check if file exists
    full_path = Path(result["data"]["file_path"])
    assert full_path.exists()
    assert full_path.suffix == ".png"

def test_plot_average_by_category(sample_csv, monkeypatch):
    """Test plot_average_by_category saves a file."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    result = plot_average_by_category(value_col="C", category_col="B", path=rel_path)
    assert result["ok"] is True
    assert Path(result["data"]["file_path"]).exists()

def test_get_dataset_summary(sample_csv, monkeypatch):
    """Test get_dataset_summary returns overview."""
    monkeypatch.setattr("backend.quant_tools._get_project_root", lambda: Path(sample_csv).parent.parent)
    rel_path = f"data/{Path(sample_csv).name}"
    
    result = get_dataset_summary(path=rel_path)
    assert result["ok"] is True
    assert result["data"]["rows"] == 5
    assert result["data"]["columns"] == 3
