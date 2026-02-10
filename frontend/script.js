// ==================== CONFIG ====================
const API_BASE_URL = "http://127.0.0.1:5005";

let googleMapsApiKey = "";
let latitude = null;
let longitude = null;

// ==================== GOOGLE PLACES ====================

function setLocationModeHint(message, isError = false) {
  const hint = document.getElementById("locationModeHint");
  if (!hint) return;
  hint.textContent = message;
  hint.style.color = isError ? "#b42318" : "";
}

async function fetchPublicConfig() {
  try {
    const res = await fetch(`${API_BASE_URL}/config`);
    if (!res.ok) {
      throw new Error(`Config endpoint returned ${res.status}`);
    }
    const data = await res.json();
    googleMapsApiKey = (data.google_maps_api_key || "").trim();
  } catch (_err) {
    googleMapsApiKey = "";
    setLocationModeHint(
      "Could not load backend config. You can still enter coordinates manually.",
      true
    );
  }
}

function loadGooglePlacesScript() {
  if (!googleMapsApiKey) {
    setLocationModeHint(
      "Google autocomplete is disabled. Enter latitude and longitude manually below, or set GOOGLE_MAPS_API_KEY in backend .env."
    );
    return;
  }

  setLocationModeHint("Loading Google autocomplete...");

  const script = document.createElement("script");
  script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(
    googleMapsApiKey
  )}&libraries=places&callback=initAutocomplete`;
  script.async = true;
  script.defer = true;
  script.onerror = () => {
    setLocationModeHint(
      "Google autocomplete failed to load. You can still enter coordinates manually.",
      true
    );
  };
  document.head.appendChild(script);
}

window.initAutocomplete = function initAutocomplete() {
  const input = document.getElementById("locationInput");
  if (!input) return;

  if (!window.google || !google.maps || !google.maps.places) {
    setLocationModeHint(
      "Google autocomplete is unavailable. You can still enter coordinates manually.",
      true
    );
    return;
  }

  const autocomplete = new google.maps.places.Autocomplete(input);
  setLocationModeHint(
    "Google autocomplete is ready. Pick a suggestion, or enter coordinates manually."
  );

  autocomplete.addListener("place_changed", () => {
    const place = autocomplete.getPlace();
    if (place && place.geometry) {
      latitude = place.geometry.location.lat();
      longitude = place.geometry.location.lng();

      const manualLatInput = document.getElementById("manualLat");
      const manualLonInput = document.getElementById("manualLon");
      if (manualLatInput) manualLatInput.value = latitude.toFixed(6);
      if (manualLonInput) manualLonInput.value = longitude.toFixed(6);
    }
  });
};

// ==================== HELPERS ====================

function extractDateFeatures(dateString) {
  if (!dateString) return null;

  const date = new Date(dateString);
  if (isNaN(date.getTime())) return null;

  const startOfYear = new Date(date.getFullYear(), 0, 0);
  const diff = date - startOfYear;
  const oneDay = 1000 * 60 * 60 * 24;
  const doy = Math.floor(diff / oneDay);

  return {
    day: date.getDate(),
    doy,
    hour: date.getHours(),
    year: date.getFullYear(),
  };
}

function showError(message) {
  const errorBox = document.getElementById("errorBox");
  if (!errorBox) return;
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  const errorBox = document.getElementById("errorBox");
  if (!errorBox) return;
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function setLoading(isLoading) {
  const loadingBox = document.getElementById("loadingBox");
  const predictBtn = document.getElementById("predictBtn");

  if (loadingBox) {
    loadingBox.classList.toggle("hidden", !isLoading);
  }
  if (predictBtn) {
    predictBtn.disabled = isLoading;
    predictBtn.textContent = isLoading ? "Predicting..." : "🔍 Predict Theft Risk";
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatInlineMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function parseCoordinate(value) {
  if (value === null || value === undefined) return null;
  const trimmed = String(value).trim();
  if (!trimmed) return null;
  const parsed = Number.parseFloat(trimmed);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function resolveCoordinates() {
  const manualLat = parseCoordinate(document.getElementById("manualLat")?.value);
  const manualLon = parseCoordinate(document.getElementById("manualLon")?.value);

  const hasManualLat = manualLat !== null;
  const hasManualLon = manualLon !== null;

  if (hasManualLat !== hasManualLon) {
    throw new Error("Enter both latitude and longitude, or leave both blank.");
  }

  if (hasManualLat && hasManualLon) {
    if (Number.isNaN(manualLat) || Number.isNaN(manualLon)) {
      throw new Error("Latitude and longitude must be valid numbers.");
    }
    if (manualLat < -90 || manualLat > 90) {
      throw new Error("Latitude must be between -90 and 90.");
    }
    if (manualLon < -180 || manualLon > 180) {
      throw new Error("Longitude must be between -180 and 180.");
    }
    return { latitude: manualLat, longitude: manualLon };
  }

  if (latitude !== null && longitude !== null) {
    return { latitude, longitude };
  }

  throw new Error(
    "Provide a location by selecting a Google suggestion or entering latitude and longitude manually."
  );
}

function buildPayload() {
  clearError();

  const occDateTime = document.getElementById("occDateTime").value;
  const reportDateTime = document.getElementById("reportDateTime").value;
  const bikeCostStr = document.getElementById("bikeCost").value;
  const bikeSpeedStr = document.getElementById("bikeSpeed").value;

  if (!occDateTime || !reportDateTime) {
    throw new Error("Please provide both occurrence and report date/time.");
  }

  const occ = extractDateFeatures(occDateTime);
  const rep = extractDateFeatures(reportDateTime);

  if (!occ || !rep) {
    throw new Error("One of the dates is invalid. Please re-select the date and time.");
  }

  const bikeCost = Number.parseFloat(bikeCostStr);
  const bikeSpeed = Number.parseFloat(bikeSpeedStr);

  if (Number.isNaN(bikeCost) || bikeCost <= 0) {
    throw new Error("Please enter a valid positive value for bike cost.");
  }

  if (Number.isNaN(bikeSpeed) || bikeSpeed <= 0) {
    throw new Error("Please enter a valid positive value for bike speed.");
  }

  const coords = resolveCoordinates();

  return {
    OCC_YEAR: occ.year,
    OCC_DAY: occ.day,
    OCC_DOY: occ.doy,
    OCC_HOUR: occ.hour,
    REPORT_YEAR: rep.year,
    REPORT_DAY: rep.day,
    REPORT_DOY: rep.doy,
    REPORT_HOUR: rep.hour,
    BIKE_SPEED: bikeSpeed,
    BIKE_COST: bikeCost,
    LONG_WGS84: coords.longitude,
    LAT_WGS84: coords.latitude,
  };
}

function renderResult(data) {
  const resultPanel = document.getElementById("result");
  if (!resultPanel) return;

  if (data.error) {
    resultPanel.className = "result-panel error-state";
    resultPanel.innerHTML = `
      <div class="result-content">
        <h2>⚠️ Error</h2>
        <p>${data.error}</p>
      </div>
    `;
    return;
  }

  const cls = data.prediction_class;
  const prob = typeof data.probability_stolen === "number" ? data.probability_stolen : null;
  const probPercent = prob !== null ? Number((prob * 100).toFixed(2)) : null;
  const baselineRate =
    typeof data.baseline_positive_rate === "number" ? data.baseline_positive_rate : null;
  const baselinePercent = baselineRate !== null ? Number((baselineRate * 100).toFixed(2)) : null;
  const deltaPercent =
    probPercent !== null && baselinePercent !== null
      ? Number((probPercent - baselinePercent).toFixed(2))
      : null;

  let riskLabel = "";
  let statusText = "";
  let panelClass = "";

  if (cls === 1) {
    riskLabel = "Higher Non-Recovery Risk";
    statusText = "This theft report is estimated to be more likely labeled STOLEN (not recovered).";
    panelClass = "result-panel risk-high";
  } else {
    riskLabel = "Lower Non-Recovery Risk";
    statusText = "This theft report is estimated to be relatively more likely labeled RECOVERED.";
    panelClass = "result-panel risk-low";
  }

  let riskCategoryText = "";
  if (probPercent !== null) {
    if (probPercent >= 75) riskCategoryText = "Very high risk";
    else if (probPercent >= 50) riskCategoryText = "Moderate to high risk";
    else if (probPercent >= 25) riskCategoryText = "Low to moderate risk";
    else riskCategoryText = "Relatively low risk";
  }

  resultPanel.className = panelClass;
  resultPanel.innerHTML = `
    <div class="result-content">
      <h2>${riskLabel}</h2>
      <p>${statusText}</p>

      ${
        probPercent !== null
          ? `
          <div class="probability-block">
            <div class="prob-header">
              <span>Estimated theft risk score (model output)</span>
              <strong>${probPercent}%</strong>
            </div>
            <div class="prob-bar">
              <div class="prob-bar-fill" style="width: ${probPercent}%;"></div>
            </div>
            <p class="prob-caption">${riskCategoryText}</p>
            ${
              baselinePercent !== null
                ? `
                  <div class="baseline-block">
                    <span class="baseline-title">Dataset baseline</span>
                    <div class="baseline-chips">
                      <span class="chip baseline-chip">STOLEN / not recovered: <strong>${baselinePercent}%</strong></span>
                      ${
                        deltaPercent !== null
                          ? `<span class="chip baseline-chip baseline-delta ${
                              deltaPercent >= 0 ? "delta-up" : "delta-down"
                            }">${deltaPercent >= 0 ? "+" : ""}${deltaPercent}% vs baseline</span>`
                          : ""
                      }
                    </div>
                  </div>
                `
                : ""
            }
          </div>
        `
          : `<p class="muted">Probability not available for this model type.</p>`
      }

      <div class="chips-row">
        <span class="chip">Prediction label: <strong>${data.prediction_label}</strong></span>
        <span class="chip">Model: <strong>${data.model_type || "Trained classifier"}</strong></span>
      </div>
    </div>
  `;
}

// ==================== MAIN ACTION ====================

async function sendPrediction() {
  try {
    clearError();
    setLoading(true);

    const payload = buildPayload();

    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "An error occurred while calling the API.");
    }

    renderResult(data);
  } catch (err) {
    showError(err.message || "Unexpected error. Please try again.");
  } finally {
    setLoading(false);
  }
}

// ==================== SAMPLE DATA BUTTON ====================

function fillSampleData() {
  const now = new Date();
  const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

  const occInput = document.getElementById("occDateTime");
  const repInput = document.getElementById("reportDateTime");

  if (occInput) occInput.value = oneHourAgo.toISOString().slice(0, 16);
  if (repInput) repInput.value = now.toISOString().slice(0, 16);

  const bikeCostInput = document.getElementById("bikeCost");
  const bikeSpeedInput = document.getElementById("bikeSpeed");

  if (bikeCostInput) bikeCostInput.value = 800;
  if (bikeSpeedInput) bikeSpeedInput.value = 18;

  const manualLatInput = document.getElementById("manualLat");
  const manualLonInput = document.getElementById("manualLon");

  if (manualLatInput) manualLatInput.value = 43.6532;
  if (manualLonInput) manualLonInput.value = -79.3832;
}

// ==================== QUANT AGENT ====================

function showAgentError(message) {
  const errorBox = document.getElementById("agentErrorBox");
  if (!errorBox) return;
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearAgentError() {
  const errorBox = document.getElementById("agentErrorBox");
  if (!errorBox) return;
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function setAgentLoading(isLoading) {
  const loadingBox = document.getElementById("agentLoadingBox");
  const queryBtn = document.getElementById("agentQueryBtn");

  if (loadingBox) {
    loadingBox.classList.toggle("hidden", !isLoading);
  }
  if (queryBtn) {
    queryBtn.disabled = isLoading;
    queryBtn.textContent = isLoading ? "Running..." : "Ask Quant Agent";
  }
}

function buildAgentPayload() {
  clearAgentError();
  const questionValue = document.getElementById("agentQuestion")?.value ?? "";
  const datasetPathValue = document.getElementById("agentDatasetPath")?.value ?? "";

  const question = questionValue.trim();
  if (!question) {
    throw new Error("Please enter a question for the quantitative agent.");
  }

  const payload = { question };
  const datasetPath = datasetPathValue.trim();
  if (datasetPath) {
    payload.dataset_path = datasetPath;
  }

  return payload;
}

function normalizeAgentAnswerText(answer) {
  return String(answer ?? "")
    .replace(/\\\$/g, "$")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\r/g, "")
    .trim();
}

function stripMarkdownTokens(text) {
  return String(text ?? "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`/g, "")
    .trim();
}

function formatCellValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }
    return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return String(value);
}

function toTitleCase(text) {
  return String(text)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function renderAgentMetaChips({ ok, code, metadata, tables, plotFiles }) {
  const chips = [];
  const datasetPath = metadata.dataset_path || "default dataset";
  const toolCallsUsed = Number.isFinite(metadata.tool_calls_used) ? String(metadata.tool_calls_used) : "0";

  if (ok) {
    chips.push(`<span class="chip chip-success">Status: <strong>SUCCESS</strong></span>`);
  } else {
    chips.push(`<span class="chip chip-danger">Status: <strong>ERROR</strong></span>`);
    if (code) {
      chips.push(`<span class="chip chip-danger">Error code: <strong>${escapeHtml(code)}</strong></span>`);
    }
  }

  chips.push(`<span class="chip">Dataset: <strong>${escapeHtml(datasetPath)}</strong></span>`);
  chips.push(`<span class="chip">Tool calls used: <strong>${escapeHtml(toolCallsUsed)}</strong></span>`);
  chips.push(`<span class="chip">Tables: <strong>${tables.length}</strong></span>`);
  chips.push(`<span class="chip">Plots: <strong>${plotFiles.length}</strong></span>`);

  return chips.join("");
}

function tryBuildMetricList(answerText) {
  const compact = answerText.replace(/\n+/g, " ").replace(/\s+/g, " ").trim();
  const parts = compact.split(/\s+-\s+/).map((part) => part.trim()).filter(Boolean);
  if (parts.length < 4) return null;

  const intro = parts.shift() || "";
  const items = [];
  for (const part of parts) {
    const idx = part.indexOf(":");
    if (idx < 1) continue;
    const label = stripMarkdownTokens(part.slice(0, idx));
    const value = stripMarkdownTokens(part.slice(idx + 1));
    if (!label || !value) continue;
    items.push({ label, value });
  }

  if (items.length < 3) return null;
  return { intro, items };
}

function renderAnswerSummary(answer) {
  const normalized = normalizeAgentAnswerText(answer);
  if (!normalized) {
    return `<p class="agent-answer-paragraph">No answer returned.</p>`;
  }

  const metricList = tryBuildMetricList(normalized);
  if (metricList) {
    return `
      <p class="agent-answer-paragraph">${formatInlineMarkdown(metricList.intro)}</p>
      <ul class="agent-metric-list">
        ${metricList.items
          .map(
            (item) => `
              <li class="agent-metric-item">
                <span class="agent-metric-label">${escapeHtml(item.label)}</span>
                <span class="agent-metric-value">${escapeHtml(item.value)}</span>
              </li>
            `
          )
          .join("")}
      </ul>
    `;
  }

  const paragraphs = normalized
    .split(/\n{2,}|\n/)
    .map((part) => part.trim())
    .filter(Boolean);

  return paragraphs
    .map((paragraph) => `<p class="agent-answer-paragraph">${formatInlineMarkdown(paragraph)}</p>`)
    .join("");
}

function renderToolCalls(toolCalls) {
  if (!toolCalls.length) {
    return `<p class="muted">No tool calls were returned.</p>`;
  }

  return `
    <ul class="agent-list">
      ${toolCalls
        .map((entry) => {
          const args = escapeHtml(JSON.stringify(entry.arguments || {}, null, 2));
          return `
            <li class="agent-list-item">
              <div class="agent-line">
                <strong>${escapeHtml(entry.tool || "unknown_tool")}</strong>
                <span class="chip ${entry.ok ? "agent-chip-ok" : "agent-chip-err"}">
                  ${entry.ok ? "OK" : "FAILED"}
                </span>
              </div>
              <p class="muted">${escapeHtml(entry.summary || "")}</p>
              <details class="agent-details">
                <summary>Arguments</summary>
                <pre class="agent-code">${args}</pre>
              </details>
            </li>
          `;
        })
        .join("")}
    </ul>
  `;
}

function renderToolTraceSection(toolCalls) {
  return `
    <details class="agent-section agent-trace">
      <summary class="agent-section-title">Tool Trace (${toolCalls.length})</summary>
      <div class="agent-section-body">
        ${renderToolCalls(toolCalls)}
      </div>
    </details>
  `;
}

function renderTables(tables) {
  if (!tables.length) {
    return `<p class="muted">No structured tables returned for this query.</p>`;
  }

  return tables
    .map((table, index) => {
      const rows = Array.isArray(table.rows) ? table.rows : [];
      const tableName = toTitleCase(table.name || `table_${index + 1}`);

      if (!rows.length) {
        return `
          <section class="agent-table-block">
            <h4>${escapeHtml(tableName)}</h4>
            <p class="muted">No rows returned.</p>
          </section>
        `;
      }

      const columns = Array.from(
        rows.reduce((set, row) => {
          Object.keys(row || {}).forEach((key) => set.add(key));
          return set;
        }, new Set())
      );

      const headerHtml = [`<th>#</th>`, ...columns.map((col) => `<th>${escapeHtml(col)}</th>`)].join("");
      const bodyHtml = rows
        .map((row, rowIndex) => {
          const cells = columns
            .map((col) => `<td>${escapeHtml(formatCellValue(row[col]))}</td>`)
            .join("");
          return `<tr><td>${rowIndex + 1}</td>${cells}</tr>`;
        })
        .join("");

      return `
        <section class="agent-table-block">
          <div class="agent-table-header">
            <h4>${escapeHtml(tableName)}</h4>
            <span class="chip">Rows: <strong>${rows.length}</strong></span>
          </div>
          <div class="agent-table-wrap">
            <table class="agent-table">
              <thead><tr>${headerHtml}</tr></thead>
              <tbody>${bodyHtml}</tbody>
            </table>
          </div>
          ${table.truncated ? `<p class="muted">Showing first ${rows.length} rows only.</p>` : ""}
        </section>
      `;
    })
    .join("");
}

function renderPlotFiles(plotFiles) {
  if (!plotFiles.length) {
    return `<p class="muted">No plot files generated for this query.</p>`;
  }

  return `
    <ul class="agent-list compact">
      ${plotFiles
        .map(
          (path) => `
            <li class="agent-list-item">
              <span class="chip">Plot</span>
              <code>${escapeHtml(path)}</code>
            </li>
          `
        )
        .join("")}
    </ul>
  `;
}

function renderAgentResult(data) {
  const panel = document.getElementById("agentResult");
  if (!panel) return;

  const toolCalls = Array.isArray(data.tool_calls) ? data.tool_calls : [];
  const tables = Array.isArray(data.tables) ? data.tables : [];
  const plotFiles = Array.isArray(data.plot_files) ? data.plot_files : [];
  const metadata = data.metadata && typeof data.metadata === "object" ? data.metadata : {};

  if (!data.ok) {
    panel.className = "result-panel error-state";
    panel.innerHTML = `
      <div class="result-content">
        <h2>Agent Error</h2>
        <p class="agent-answer-paragraph">${escapeHtml(data.error || "The query failed.")}</p>
        <div class="chips-row">
          ${renderAgentMetaChips({ ok: false, code: data.code, metadata, tables, plotFiles })}
        </div>
      </div>
      ${renderToolTraceSection(toolCalls)}
    `;
    return;
  }

  panel.className = "result-panel agent-success";
  panel.innerHTML = `
    <div class="result-content">
      <h2>Quantitative Answer</h2>
      <div class="chips-row">
        ${renderAgentMetaChips({ ok: true, code: "", metadata, tables, plotFiles })}
      </div>
    </div>

    <section class="agent-section">
      <h3 class="agent-section-title">Executive Summary</h3>
      <div class="agent-section-body">
        ${renderAnswerSummary(data.answer)}
      </div>
    </section>

    <section class="agent-section">
      <h3 class="agent-section-title">Tables</h3>
      <div class="agent-section-body">
        ${renderTables(tables)}
      </div>
    </section>

    <section class="agent-section">
      <h3 class="agent-section-title">Plot Files</h3>
      <div class="agent-section-body">
        ${renderPlotFiles(plotFiles)}
      </div>
    </section>

    ${renderToolTraceSection(toolCalls)}
  `;
}

async function sendAgentQuery() {
  clearAgentError();

  let payload;
  try {
    payload = buildAgentPayload();
  } catch (validationErr) {
    showAgentError(validationErr.message || "Please check your input.");
    return;
  }

  try {
    setAgentLoading(true);
    const res = await fetch(`${API_BASE_URL}/agent/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data = {};
    try {
      data = await res.json();
    } catch (_err) {
      data = {};
    }

    if (!res.ok || !data.ok) {
      const fallbackMessage = `Agent request failed (${res.status}).`;
      const errorPayload = {
        ok: false,
        error: data.error || fallbackMessage,
        code: data.code || "",
        tool_calls: data.tool_calls || [],
        tables: data.tables || [],
        plot_files: data.plot_files || [],
        metadata: data.metadata || {},
      };
      renderAgentResult(errorPayload);
      return;
    }

    renderAgentResult(data);
  } catch (err) {
    const message = err.message || "Unexpected error while querying the agent.";
    renderAgentResult({
      ok: false,
      error: message,
      code: "INTERNAL_ERROR",
      tool_calls: [],
      tables: [],
      plot_files: [],
      metadata: {},
    });
  } finally {
    setAgentLoading(false);
  }
}

function fillAgentSampleQuestion() {
  const questionInput = document.getElementById("agentQuestion");
  if (!questionInput) return;
  questionInput.value = "What is the average BIKE_COST by NEIGHBOURHOOD_158?";
}

async function initApp() {
  if (!document.getElementById("locationModeHint")) return;
  await fetchPublicConfig();
  loadGooglePlacesScript();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
