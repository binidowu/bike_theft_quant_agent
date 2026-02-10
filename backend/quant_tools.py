import os
import re
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for non-GUI plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# --- Constants ---
DEFAULT_DATASET_PATH = os.getenv("AGENT_DEFAULT_DATASET", "data/bicycle_thefts_open_data.csv")
PLOT_OUTPUT_DIR = "outputs/plots"
MAX_TOOL_TOP_N = 100

# --- Error Codes ---
ERR_UNSAFE_PATH = "UNSAFE_PATH"
ERR_INVALID_COLUMN = "INVALID_COLUMN"
ERR_BAD_REQUEST = "BAD_REQUEST"
ERR_TOOL_TIMEOUT = "TOOL_TIMEOUT"
ERR_INTERNAL_ERROR = "INTERNAL_ERROR"

# --- Custom Exception ---
class QuantAgentError(Exception):
    """Custom exception for Quantitative Agent errors."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

# --- Global Cache ---
# Simple in-memory cache for loaded dataframes to avoid re-reading CSVs constantly
_DF_CACHE: Dict[str, pd.DataFrame] = {}

# --- Helper Functions ---

def _get_project_root() -> Path:
    """Returns the project root directory."""
    # Assumes this file is in backend/ and project root is one level up
    return Path(__file__).resolve().parent.parent

def _validate_path(file_path: Optional[str]) -> Path:
    """
    Validates and resolves a file path.
    Ensures the path is within the 'data' directory of the project.
    Raises QuantAgentError if path is unsafe or invalid.
    """
    project_root = _get_project_root()
    data_dir = project_root / "data"
    
    # Use default if not provided
    if not file_path:
        target_path = project_root / DEFAULT_DATASET_PATH
    else:
        # Handle both absolute and relative paths
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = project_root / target_path
    
    try:
        resolved_path = target_path.resolve()
        resolved_data_dir = data_dir.resolve()
    except Exception as e:
         raise QuantAgentError(ERR_BAD_REQUEST, f"Invalid path format: {str(e)}")

    # Strict check: must be inside resolved data directory
    if not resolved_path.is_relative_to(resolved_data_dir):
        raise QuantAgentError(ERR_UNSAFE_PATH, f"Path '{file_path}' is not allowed. Must be within 'data/' directory.")
    
    if not resolved_path.exists():
         raise QuantAgentError(ERR_BAD_REQUEST, f"File not found: {resolved_path}")

    return resolved_path

def _get_timestamp() -> str:
    """Returns a deterministic timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _normalize_filename(text: str) -> str:
    """Normalizes a string for use in filenames (alphanumeric + underscore)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', text)

def _get_dataframe(file_path: Optional[str] = None) -> pd.DataFrame:
    """Helper to load dataframe with caching and validation."""
    validated_path = _validate_path(file_path)
    path_str = str(validated_path)
    
    if path_str in _DF_CACHE:
        return _DF_CACHE[path_str]
    
    try:
        df = pd.read_csv(validated_path)
        _DF_CACHE[path_str] = df
        return df
    except Exception as e:
        raise QuantAgentError(ERR_INTERNAL_ERROR, f"Failed to load CSV: {str(e)}")


def _validate_top_n(top_n: int) -> int:
    """Validate top_n and cap to tool-level safety limit."""
    if not isinstance(top_n, int):
        raise QuantAgentError(ERR_BAD_REQUEST, "top_n must be an integer.")
    if top_n < 1:
        raise QuantAgentError(ERR_BAD_REQUEST, "top_n must be >= 1.")
    return min(top_n, MAX_TOOL_TOP_N)

def _ensure_plot_dir():
    """Ensures the plot output directory exists."""
    project_root = _get_project_root()
    plot_dir = project_root / PLOT_OUTPUT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

# --- Tool Functions ---

def load_csv(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a CSV file and returns metadata.
    
    Args:
        path: Path to the CSV file (relative to project root or within data/).
              Defaults to DEFAULT_DATASET_PATH.
              
    Returns:
        Dictionary with dataset metadata (rows, columns, preview).
    """
    try:
        df = _get_dataframe(path)
        return {
            "ok": True,
            "tool": "load_csv",
            "data": {
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records")
            }
        }
    except QuantAgentError as e:
        raise e
    except Exception as e:
        raise QuantAgentError(ERR_INTERNAL_ERROR, str(e))

def list_columns(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Lists all columns in the dataset and determines their types (numeric vs categorical).
    """
    df = _get_dataframe(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return {
        "ok": True,
        "tool": "list_columns",
        "data": {
            "all_columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols
        }
    }

def calculate_stats(column: str, path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculates descriptive statistics for a specific numeric column.
    """
    df = _get_dataframe(path)
    
    if column not in df.columns:
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Column '{column}' not found in dataset.")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
         # We can still get some stats for non-numeric, but let's stick to the spec's implication
         # or just return what describe() gives. The spec said "numeric requirement"
         # but pandas describe handles both. Let's be safe and allow describe for all 
         # but warn if it's not what expected. Actually spec said "Enforce numeric requirement for stats/averages on value columns".
         # So for `calculate_stats`, let's check.
         raise QuantAgentError(ERR_INVALID_COLUMN, f"Column '{column}' is not numeric.")

    stats = df[column].describe().to_dict()
    # Add extra info like missing count
    stats["missing"] = int(df[column].isnull().sum())
    
    return {
        "ok": True,
        "tool": "calculate_stats",
        "data": {
            "column": column,
            "stats": stats
        }
    }

def average_by_category(value_col: str, category_col: str, top_n: int = 20, ascending: bool = False, path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculates the average of a value column grouped by a category column.
    """
    df = _get_dataframe(path)
    
    if value_col not in df.columns:
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Value column '{value_col}' not found.")
    if category_col not in df.columns:
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Category column '{category_col}' not found.")
        
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Value column '{value_col}' must be numeric.")

    top_n = _validate_top_n(top_n)

    # Group by and mean
    grouped = df.groupby(category_col)[value_col].agg(['mean', 'count']).reset_index()
    
    # Sort
    grouped = grouped.sort_values(by='mean', ascending=ascending)
    
    # Apply limit logic (internal limit, but we allow user to specify top_n too)
    # Spec says "return structured rows". 
    result_df = grouped.head(top_n)
    
    rows = result_df.to_dict(orient='records')
    
    return {
        "ok": True,
        "tool": "average_by_category",
        "data": {
            "value_col": value_col,
            "category_col": category_col,
            "rows": rows,
            "summary": f"Computed mean of {value_col} by {category_col}, top {top_n} results."
        }
    }

def plot_distribution(column: str, bins: int = 30, path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a histogram for a numeric column and saves it to a file.
    """
    df = _get_dataframe(path)
    
    if column not in df.columns:
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Column '{column}' must be numeric for histogram.")

    fig = None
    try:
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column].dropna(), bins=bins, edgecolor='k', alpha=0.7)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Save file
        timestamp = _get_timestamp()
        safe_col = _normalize_filename(column)
        filename = f"plot_dist_{safe_col}_{timestamp}.png"

        plot_dir = _ensure_plot_dir()
        output_path = plot_dir / filename

        fig.savefig(output_path)
    finally:
        if fig is not None:
            plt.close(fig)
    
    return {
        "ok": True,
        "tool": "plot_distribution",
        "data": {
            "file_path": str(output_path),
            "relative_path": f"{PLOT_OUTPUT_DIR}/{filename}",
            "column": column,
            "bins": bins
        }
    }

def plot_average_by_category(value_col: str, category_col: str, top_n: int = 20, ascending: bool = False, path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a bar chart for average of value_col by category_col.
    """
    # Reuse calculation logic
    # We call the function directly to get the data, but we need to handle the dict return
    # Alternatively, just redo the groupby to be cleaner and self-contained
    df = _get_dataframe(path)
    
    if value_col not in df.columns:
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Value column '{value_col}' not found.")
    if category_col not in df.columns:
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Category column '{category_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise QuantAgentError(ERR_INVALID_COLUMN, f"Value column '{value_col}' must be numeric.")

    top_n = _validate_top_n(top_n)

    grouped = df.groupby(category_col)[value_col].mean().reset_index()
    grouped = grouped.sort_values(by=value_col, ascending=ascending).head(top_n)
    
    fig = None
    try:
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(grouped[category_col].astype(str), grouped[value_col], alpha=0.7)
        ax.set_title(f"Average {value_col} by {category_col} (Top {top_n})")
        ax.set_xlabel(category_col)
        ax.set_ylabel(f"Average {value_col}")
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()

        # Save
        timestamp = _get_timestamp()
        safe_val = _normalize_filename(value_col)
        safe_cat = _normalize_filename(category_col)
        filename = f"plot_avg_{safe_val}_{safe_cat}_{timestamp}.png"

        plot_dir = _ensure_plot_dir()
        output_path = plot_dir / filename

        fig.savefig(output_path)
    finally:
        if fig is not None:
            plt.close(fig)
    
    return {
        "ok": True,
        "tool": "plot_average_by_category",
        "data": {
            "file_path": str(output_path),
            "relative_path": f"{PLOT_OUTPUT_DIR}/{filename}",
            "rows_used": len(grouped)
        }
    }

def get_dataset_summary(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns high-level summary of the dataset.
    """
    df = _get_dataframe(path)
    
    return {
        "ok": True,
        "tool": "get_dataset_summary",
        "data": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
    }
