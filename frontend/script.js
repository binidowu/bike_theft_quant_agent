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
