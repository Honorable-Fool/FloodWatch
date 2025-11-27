/* index.js
   - populates datalist from /api/barangays (fallback to local CSV)
   - submits form to POST /predict
   - receives response and highlights only the rectangle-alert box (0/1/2)
*/

// ---------------- CONFIG ----------------
// uses your uploaded CSV fallback path so datalist can load if /api/barangays is down
const API_BARANGAYS = "http://127.0.0.1:8000/api/barangays";
const API_PREDICT = "http://127.0.0.1:8000/predict";
// Local uploaded CSV path (developer-provided). This will be used as fallback when /api/barangays fails.
const CSV_FALLBACK = "/mnt/data/floodwatch_MLdataset.csv";

// ---------------- ELEMENTS ----------------
const locationInput = document.getElementById("location");
const datalist = document.getElementById("barangayList");
const elevationInput = document.getElementById("elevation_m");

const durationInput = document.getElementById("duration");
const rainfallInput = document.getElementById("rainfall");

const form = document.getElementById("floodForm");

// rectangle elements (only these will be used to show prediction)
const rectYellow = document.querySelector(".rectangle-alert.yellow");
const rectOrange = document.querySelector(".rectangle-alert.orange");
const rectRed = document.querySelector(".rectangle-alert.red");

// ---------------- state ----------------
let barangays = []; // [{barangay, elevations: []}, ...]
let barangayLookup = new Map(); // normalized -> object

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
      if (it.elevations && Array.isArray(it.elevations)) return { barangay: it.barangay, elevations: it.elevations };
      if (it.elevation !== undefined) return { barangay: it.barangay, elevations: [it.elevation] };
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

// ---------------- input change handling ----------------
locationInput.addEventListener("change", () => {
  const norm = normalize(locationInput.value);
  const found = barangayLookup.get(norm);
  if (found) {
    locationInput.dataset.valid = "true";
    locationInput.dataset.barangay = found.barangay;
    if (found.elevations && found.elevations.length) elevationInput.value = String(found.elevations[0]);
    else elevationInput.value = "";
  } else {
    locationInput.dataset.valid = "false";
    locationInput.dataset.barangay = "";
    elevationInput.value = "";
  }
  // inside your existing locationInput change handler, after finding 'found'.
// set preview:
  const elevPreviewEl = document.getElementById("elevPreviewValue");
  if (found && found.elevations && found.elevations.length) {
    const sampleLocal = found.elevations[Math.floor(Math.random()*found.elevations.length)];
    elevPreviewEl.textContent = sampleLocal.toFixed(3);
} else {
    elevPreviewEl.textContent = "—";
}

});

// ---------------- form submit -> call predict ----------------
form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  clearActive();

  const raw = locationInput.value.trim();
  const norm = normalize(raw);
  const found = barangayLookup.get(norm);

  if (!found) {
    alert("Please choose a Barangay from the suggestions (exact match).");
    locationInput.focus();
    return;
  }

  const duration = Number(durationInput.value);
  const rainfall = Number(rainfallInput.value);

  if (!(duration > 0)) { alert("Enter a valid duration (e.g., 1,3,6)."); durationInput.focus(); return; }
  if (!(rainfall >= 0)) { alert("Enter a valid rainfall intensity (>= 0)."); rainfallInput.focus(); return; }

  const payload = {
    Barangay: found.barangay,
    Duration_hr: duration,
    Rainfall_mm: rainfall
  };

  try {
    const res = await fetch(API_PREDICT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || "Prediction API error");
    }

    const data = await res.json();
    // Highlight only the rectangle alert corresponding to numeric_label (0/1/2)
    clearActive();
    if (data.numeric_label === 0 && rectYellow) rectYellow.classList.add("active");
    if (data.numeric_label === 1 && rectOrange) rectOrange.classList.add("active");
    if (data.numeric_label === 2 && rectRed) rectRed.classList.add("active");

    // We DO NOT use an advisory box to display ML results per your request.
    // (If you later want to show sampled elevation, you can do so separately.)

  } catch (err) {
    console.error("Prediction failed:", err);
    alert("Prediction failed. Check console for details.");
  }
});

// ---------------- init ----------------
loadBarangays();
