/* index.js
   - ML predict (/predict)
   - NLP advisory auto-generation (/generate_from_risk)
*/

// ---------------- CONFIG ----------------
const API_BARANGAYS = "http://127.0.0.1:8000/api/barangays";
const API_PREDICT  = "http://127.0.0.1:8000/predict";     
const API_NLP_RISK = "http://127.0.0.1:8001/generate_from_risk";  // ⭐ NLP service
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
    if (!name || Number.isNaN(elev)) continue;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(elev);
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
    const response = await fetch(API_PREDICT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();

    // ---------------- COLOR HIGHLIGHT ----------------
    clearActive();
    if (result.numeric_label === 0) rectYellow.classList.add("active");
    if (result.numeric_label === 1) rectOrange.classList.add("active");
    if (result.numeric_label === 2) rectRed.classList.add("active");

    // ---------------- NLP CALL (NEW) ----------------
    const riskLabel = result.risk_label || result.label || (
      result.numeric_label === 0 ? "Low" :
      result.numeric_label === 1 ? "Moderate" : "High"
    );

    const nlpResponse = await fetch(API_NLP_RISK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ risk: riskLabel })
    });

    const nlp = await nlpResponse.json();

    // ---------------- SHOW ADVISORY ----------------
    if (advisoryContent) {
      advisoryContent.innerText = nlp.advisory || "No advisory available.";
    }

    document.getElementById("advisoryBox")?.scrollIntoView({ behavior: "smooth" });

    // optional popup
    if (typeof showAdvisoryPopup === "function") showAdvisoryPopup(nlp);

  } catch (error) {
    alert("Error connecting to backend: " + error.message);
    console.error(error);
  }
}

// ---------------- init ----------------
loadBarangays();
