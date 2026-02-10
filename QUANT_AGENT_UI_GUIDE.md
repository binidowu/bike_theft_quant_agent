# Quantitative Agent UI Guide

This guide explains how to use the **Quantitative Agent** section in the frontend and what kinds of results to expect.

## What the Quantitative Agent Does

The Quantitative Agent answers dataset questions by calling backend data tools (`pandas`/`numpy`/`matplotlib`) through `POST /agent/query`.

In the UI, you can expect:
- A natural-language answer
- Tool trace (which tools were used)
- Tables (when applicable)
- Plot file paths (when plots are generated)

## Before You Start

1. Activate your environment:

```bash
. .venv/bin/activate
```

2. Make sure `.env` contains an OpenAI key:

```env
OPENAI_API_KEY=your_key_here
```

3. Start backend:

```bash
python3 -m backend.quant_agent_api
```

4. Start frontend:

```bash
python3 -m http.server 5500 --directory frontend
```

5. Open:

`http://127.0.0.1:5500`

## How to Use the UI

Go to the **Quantitative Agent** card on the page.

Enter:
- **Question** (required)
- **Dataset path** (optional, default dataset is used if empty)

Then click **Ask Quant Agent**.

## Sample Questions by Expected Result Type

### 1) Answer only (no plot expected)

Question:

`What is the average BIKE_COST by NEIGHBOURHOOD_158?`

Expected result:
- You get a text answer summarizing top categories
- You get a table with grouped mean values
- Plot files section is usually empty

Question:

`Show descriptive stats for BIKE_COST.`

Expected result:
- Text summary of stats (mean, min, max, etc.)
- Table/stat payload may appear
- No plot unless the model decides to call a plotting tool

More answer-only prompts you can use:

- `List the columns in this dataset and separate numeric vs categorical columns.`
  Expected result: answer with column overview, usually no plots.

- `Give me a high-level summary of the dataset (rows, columns, missing values).`
  Expected result: dataset summary answer; may include a structured summary table.

- `What is the average BIKE_SPEED by NEIGHBOURHOOD_158?`
  Expected result: grouped average answer + table, no plot.

- `Show descriptive statistics for BIKE_SPEED.`
  Expected result: stats answer for BIKE_SPEED, no plot.

- `What are the top 10 NEIGHBOURHOOD_158 values by average BIKE_COST?`
  Expected result: ranked grouped averages + table, no plot.

- `What are the bottom 10 NEIGHBOURHOOD_158 values by average BIKE_COST?`
  Expected result: lower-ranked grouped averages + table, no plot.

- `Compare OCC_HOUR and REPORT_HOUR with descriptive stats.`
  Expected result: answer-only stats comparison (if both columns are available).

### 2) Answer with plot/graph expected

Question:

`Plot the distribution of BIKE_COST and summarize what it shows.`

Expected result:
- Text interpretation
- Plot file path appears under **Plot Files** (for example: `outputs/plots/plot_dist_BIKE_COST_YYYYMMDD_HHMMSS.png`)

Question:

`Create a bar chart of average BIKE_COST by NEIGHBOURHOOD_158 for the top 15 groups.`

Expected result:
- Text summary + grouped result
- Plot file path appears (for example: `outputs/plots/plot_avg_BIKE_COST_NEIGHBOURHOOD_158_YYYYMMDD_HHMMSS.png`)

More plot-generating prompts you can use:

- `Plot the distribution of BIKE_SPEED and summarize key observations.`
  Expected result: summary + histogram plot path.

- `Create a histogram for OCC_HOUR and explain the pattern.`
  Expected result: summary + distribution plot path (if OCC_HOUR exists).

- `Create a bar chart of average BIKE_SPEED by NEIGHBOURHOOD_158 for top 20 categories.`
  Expected result: summary + grouped bar chart path.

- `Plot average BIKE_COST by NEIGHBOURHOOD_158 for top 10 and explain the top/bottom values.`
  Expected result: summary + bar chart path + likely grouped table.

- `Generate a plot for the distribution of LAT_WGS84.`
  Expected result: summary + distribution plot path (if LAT_WGS84 exists).

- `Generate a plot for the distribution of LONG_WGS84.`
  Expected result: summary + distribution plot path (if LONG_WGS84 exists).

## 2.5) “Safe starter” prompt sequence (high success rate)

Use this sequence when testing from scratch:

1. `List all columns and identify numeric columns.`
2. Pick one numeric column returned by step 1.
3. Ask: `Show descriptive stats for <NUMERIC_COLUMN>.`
4. Ask: `Plot the distribution of <NUMERIC_COLUMN> and summarize it.`

This reduces invalid-column errors and gives you reliable successful responses.

## 3) Error examples (expected failure)

Question:

`Show stats for column DOES_NOT_EXIST.`

Expected result:
- Error shown in UI
- Typical code: `INVALID_COLUMN`

Dataset path (optional field):

`../secrets.csv`

Question:

`What is the average BIKE_COST by NEIGHBOURHOOD_158?`

Expected result:
- Error shown in UI
- Typical code: `UNSAFE_PATH` (path is outside `data/`)

Question field left empty and click submit.

Expected result:
- Client-side validation message in UI (no backend call)

## Output Notes

- Tables are truncated to max 20 rows at API response level for safety.
- Plot files are saved to `outputs/plots/`.
- The UI currently shows plot file paths; image preview is not automatic.

## Error Codes You May See

- `BAD_REQUEST`: invalid request format or missing required fields
- `INVALID_COLUMN`: column name is not in dataset or not valid for operation
- `UNSAFE_PATH`: dataset path is outside allowed `data/` directory
- `TOOL_TIMEOUT`: tool exceeded execution limit
- `INTERNAL_ERROR`: unexpected server/runtime issue

## Quick Validation Flow

1. Run a no-plot question (average/grouped query).
2. Run a plot question (distribution or bar chart).
3. Run an invalid-column question to confirm error handling.
4. Confirm tool trace/tables/plot path sections update correctly in UI.
