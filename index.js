/* index.js
   - ML predict (/predict)
   - NLP advisory auto-generation (/generate_from_risk)
*/

// ---------------- CONFIG ----------------
const API_BARANGAYS = "http://127.0.0.1:8000/api/barangays";
const API_PREDICT  = "http://127.0.0.1:8000/predict";
const API_NLP_RISK = "http://127.0.0.1:8001/generate_from_risk";
const CSV_FALLBACK = "/mnt/data/floodwatch_MLdataset.csv";

// ---------------- ELEMENTS ----------------
const locationInput = document.getElementById("location");
const datalist = document.getElementById("barangayList");
const elevationInput = document.getElementById("elevation_m");

const durationInput = document.getElementById("duration");
const rainfallInput = document.getElementById("rainfall");

const form = document.getElementById("floodForm");

const rectYellow = document.querySelector(".rectangle-alert.yellow");
const rectOrange = document.querySelector(".rectangle-alert.orange");
const rectRed = document.querySelector(".rectangle-alert.red");

const advisoryContent = document.getElementById("advisoryContent");

// ---------------- state ----------------
let barangays = [];
let barangayLookup = new Map();
let lastRiskLabel = null; // ⭐ store last risk label

// ---------------- helpers ----------------
function normalize(s) {
  return (s || "").toString().trim().replace(/\s+/g, " ").normalize("NFKD").toLowerCase();
}

function clearActive() {
  [rectYellow, rectOrange, rectRed].forEach(r => r && r.classList.remove("active"));
}

// ---------------- load barangays ----------------
async function loadBarangays() {
  try {
    const resp = await fetch(API_BARANGAYS);
    if (!resp.ok) throw new Error("API not available");
    const json = await resp.json();
    barangays = json.map(it => {
      if (it.elevations && Array.isArray(it.elevations))
        return { barangay: it.barangay, elevations: it.elevations };
      if (it.elevation !== undefined)
        return { barangay: it.barangay, elevations: [it.elevation] };
      return { barangay: it.barangay, elevations: [] };
    });
  } catch (err) {
    console.warn("Failed to fetch API barangays - trying CSV fallback", err);
    try {
      const r = await fetch(CSV_FALLBACK);
      if (!r.ok) throw new Error("CSV fallback not available");
      const text = await r.text();
      barangays = parseCsvToBarangays(text);
    } catch (err2) {
      console.error("Failed to load fallback CSV", err2);
      barangays = [];
    }
  }
  buildLookup();
  populateDatalist();
}

function parseCsvToBarangays(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines.shift().split(",").map(h => h.trim().replace(/^"|"$/g, ""));
  const bIdx = headers.indexOf("Barangay");
  const eIdx = headers.indexOf("Elevation_m");
  const groups = new Map();
  for (const L of lines) {
    const cols = L.split(",").map(c => c.trim().replace(/^"|"$/g, ""));
    const name = cols[bIdx] || "";
    const elev = parseFloat(cols[eIdx] || "NaN");
    if (!name) continue;
    if (!Number.isNaN(elev)) {
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(elev);
    }
  }
  return Array.from(groups.entries()).map(([barangay, elevations]) => ({ barangay, elevations }));
}

function buildLookup() {
  barangayLookup.clear();
  for (const b of barangays) {
    barangayLookup.set(normalize(b.barangay), b);
  }
}

function populateDatalist() {
  datalist.innerHTML = "";
  barangays.forEach(b => {
    const opt = document.createElement("option");
    opt.value = b.barangay;
    datalist.appendChild(opt);
  });
}

// ---------------- input change ----------------
locationInput.addEventListener("change", () => {
  const norm = normalize(locationInput.value);
  const found = barangayLookup.get(norm);
  const elevPreviewEl = document.getElementById("elevPreviewValue");

  if (found) {
    locationInput.dataset.valid = "true";
    locationInput.dataset.barangay = found.barangay;

    elevationInput.value = found.elevations.length
      ? String(found.elevations[0])
      : "";

    elevPreviewEl.textContent = found.elevations.length
      ? found.elevations[Math.floor(Math.random() * found.elevations.length)].toFixed(3)
      : "—";
  } else {
    locationInput.dataset.valid = "false";
    locationInput.dataset.barangay = "";
    elevationInput.value = "";
    elevPreviewEl.textContent = "—";
  }
});

// ---------------- FORM SUBMIT ----------------
form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  await predictFlood();
});

// ---------------- MAIN FUNCTION ----------------
async function predictFlood() {
  clearActive();
  
  // Clear previous advisory first
  if (advisoryContent) {
    advisoryContent.innerText = "Loading advisory...";
  }

  const raw = locationInput.value.trim();
  const norm = normalize(raw);
  const found = barangayLookup.get(norm);

  if (!found) {
    alert("Please choose a Barangay from the suggestions (exact match).");
    return locationInput.focus();
  }

  const duration = Number(durationInput.value);
  const rainfall = Number(rainfallInput.value);

  if (!(duration > 0)) {
    alert("Enter a valid duration (e.g., 1,3,6).");
    return durationInput.focus();
  }
  if (!(rainfall >= 0)) {
    alert("Enter valid rainfall intensity (>= 0).");
    return rainfallInput.focus();
  }

  const payload = {
    Barangay: found.barangay,
    Duration_hr: duration,
    Rainfall_mm: rainfall
  };

  try {
    // ---------------- ML PREDICTION ----------------
    console.log("Sending ML prediction request...", payload);
    const response = await fetch(API_PREDICT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const txt = await response.text().catch(() => "Unknown error from ML API");
      console.error("ML API error:", txt);
      throw new Error(`ML API error: ${txt}`);
    }

    const result = await response.json();
    console.log("ML API response:", result);

    // ---------------- COLOR HIGHLIGHT ----------------
    clearActive();
    if (result.numeric_label === 0) rectYellow.classList.add("active");
    if (result.numeric_label === 1) rectOrange.classList.add("active");
    if (result.numeric_label === 2) rectRed.classList.add("active");

    // ---------------- NLP CALL ----------------
    const riskLabel = result.risk_label || result.label || (
      result.numeric_label === 0 ? "Low" :
      result.numeric_label === 1 ? "Moderate" : "High"
    );

    console.log("Risk label determined:", riskLabel);
    lastRiskLabel = riskLabel; // ⭐ store latest risk

    let nlp = { advisory: "No advisory available." };
    try {
      console.log("Calling NLP API with risk:", riskLabel);
      const nlpResponse = await fetch(API_NLP_RISK, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ risk: riskLabel })
      });

      console.log("NLP API response status:", nlpResponse.status);
      
      if (nlpResponse.ok) {
        nlp = await nlpResponse.json();
        console.log("NLP API response data:", nlp);
      } else {
        const errorText = await nlpResponse.text().catch(() => "Unknown error");
        console.warn("NLP API returned bad status:", nlpResponse.status, errorText);
        nlp.advisory = `NLP API error: ${nlpResponse.status}. ${errorText}`;
      }
    } catch (nlpErr) {
      console.warn("NLP service unreachable:", nlpErr);
      nlp.advisory = `NLP service error: ${nlpErr.message}. Please check if NLP API is running on port 8001.`;
    }

    // ---------------- SHOW ADVISORY ----------------
    if (advisoryContent) {
      advisoryContent.innerText = nlp.advisory || "No advisory available.";
    }

    // ---------------- UPDATE SAFETY GUIDELINES ----------------
    let riskLevel = null;
    if (result.numeric_label === 0) riskLevel = 0;
    else if (result.numeric_label === 1) riskLevel = 1;
    else if (result.numeric_label === 2) riskLevel = 2;
    
    if (riskLevel !== null && typeof updateSafetyGuidelines === 'function') {
      updateSafetyGuidelines(riskLevel);
    } else {
      console.warn("updateSafetyGuidelines function not found");
    }

  } catch (error) {
    console.error("Error in predictFlood:", error);
    alert("Error connecting to backend: " + (error.message || error));
    
    // Reset safety guidelines on error
    if (typeof resetSafetyGuidelines === 'function') {
      resetSafetyGuidelines();
    }
  }
}

// ---------------- OPTIONAL: Refresh Advisory Only ----------------
async function refreshAdvisory() {
  if (!lastRiskLabel) return;

  try {
    const res = await fetch(API_NLP_RISK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ risk: lastRiskLabel })
    });
    if (!res.ok) throw new Error("Failed to fetch advisory");

    const data = await res.json();
    advisoryContent.innerText = data.advisory;
  } catch (err) {
    console.error(err);
    advisoryContent.innerText = "No advisory available.";
  }
}

// ---------------- init ----------------
loadBarangays();

// Add console log to check if script loaded
console.log("FloodWatch index.js loaded successfully");