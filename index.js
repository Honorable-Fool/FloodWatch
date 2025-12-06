/* index.js
   - Handles Map initialization (Satellite + Labels)
   - Manages User Input & Autocomplete
   - Orchestrates the Prediction Flow (ML -> Map -> NLP -> History)
   - Handles Geolocation ("Locate Me")
*/

// ---------------- CONFIGURATION ----------------
const API_BARANGAYS = "http://127.0.0.1:8000/api/barangays";
const API_PREDICT  = "http://127.0.0.1:8000/predict";
const API_NLP_RISK = "http://127.0.0.1:8001/generate_from_risk";
const CSV_FALLBACK = "floodwatch-ml/data/floodwatch_MLdataset.csv";

// ---------------- DOM ELEMENTS ----------------
const locationInput = document.getElementById("location");
const datalist = document.getElementById("barangayList");
const elevationInput = document.getElementById("elevation_m");
const durationInput = document.getElementById("duration");
const rainfallInput = document.getElementById("rainfall");
const form = document.getElementById("floodForm");
const advisoryContent = document.getElementById("advisoryContent");

// Risk Indicators
const rectYellow = document.querySelector(".rectangle-alert.yellow");
const rectOrange = document.querySelector(".rectangle-alert.orange");
const rectRed = document.querySelector(".rectangle-alert.red");

// ---------------- STATE MANAGEMENT ----------------
let barangays = [];
let barangayLookup = new Map(); // Optimization: Map for O(1) lookups instead of looping arrays
let map;

const riskColors = {
    0: 'rgba(255, 255, 0, 1)', // Yellow
    1: 'rgba(255, 165, 0, 1)', // Orange
    2: 'rgba(255, 0, 0, 1)'    // Red
};

// ---------------- INITIALIZATION ----------------
document.addEventListener('DOMContentLoaded', () => {
    // Center map on Dasmariñas, Cavite
    map = L.map('map').setView([14.3293, 120.9367], 13);

    // Layer 1: Satellite Imagery (Base)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri'
    }).addTo(map);

    // Layer 2: Labels Overlay (Roads/Names on top)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    // Create Legend Control
    const legend = L.control({position: 'bottomright'});
    legend.onAdd = function () {
        const div = L.DomUtil.create('div', 'legend');
        div.style.backgroundColor = "white"; 
        div.style.padding = "10px";
        div.style.borderRadius = "5px";
        div.innerHTML += '<i style="background: rgba(255, 255, 0, 1)"></i> Low Risk<br>';
        div.innerHTML += '<i style="background: rgba(255, 165, 0, 1)"></i> Moderate Risk<br>';
        div.innerHTML += '<i style="background: rgba(255, 0, 0, 1)"></i> High Risk<br>';
        return div;
    };
    legend.addTo(map);

    addCityBoundary();
    loadBarangays(); // Load data for the dropdown
});

// ---------------- MAP FUNCTIONS ----------------

// Fetches Dasmariñas boundary geometry and draws it on the map
async function addCityBoundary() {
    if (!map) return;
    
    // Strict search query with viewbox to avoid finding other "Dasmariñas" locations
    const query = encodeURIComponent("Dasmariñas, Cavite, Philippines");
    const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&polygon_geojson=1&limit=1&viewbox=120.90,14.40,121.05,14.20&bounded=1`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (data && data.length > 0 && data[0].geojson) {
            L.geoJSON(data[0].geojson, {
                className: 'city-boundary',
                style: {
                    color: "#7be1e1", // Cyan outline
                    weight: 3,
                    fillOpacity: 0.1
                }
            }).addTo(map);
        }
    } catch (err) {
        console.warn("Could not load city boundary:", err);
    }
}

// Adds a marker to the map with a "Jitter" effect to prevent stacking
async function highlightLocation(locationName, riskLevel, elevation) {
    if (!map || !locationName) return;

    // Search for coordinates
    const query = encodeURIComponent(`${locationName}, Dasmariñas Cavite, Philippines`);
    const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&limit=1&viewbox=120.90,14.40,121.05,14.20&bounded=1`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (data && data.length > 0) {
            let lat = parseFloat(data[0].lat);
            let lon = parseFloat(data[0].lon);

            // --- JITTER LOGIC ---
            // Adds a random offset (approx 80m radius) so points don't overlap perfectly
            const maxRadius = 0.0008; 
            const angle = Math.random() * 2 * Math.PI;
            const r = Math.sqrt(Math.random()) * maxRadius;
            
            lat += r * Math.cos(angle);
            lon += r * Math.sin(angle);

            const color = riskColors[riskLevel] || '#3388ff';

            // Create Marker
            const newMarker = L.circleMarker([lat, lon], {
                radius: 12,          
                fillColor: color,    
                color: "#fff",       
                weight: 2,           
                opacity: 1,
                fillOpacity: 0.9     
            }).addTo(map);

            map.flyTo([lat, lon], 15);
            
            // Custom Popup with "Remove" Button
            const popupContainer = document.createElement('div');
            popupContainer.innerHTML = `
                <strong>${locationName}</strong><br>
                Risk: ${riskLevel === 0 ? "Low" : riskLevel === 1 ? "Moderate" : "High"}<br>
                <small>Elev: ${elevation ? elevation.toFixed(2) + 'm' : 'N/A'}</small>
            `;
            
            const deleteBtn = document.createElement('button');
            deleteBtn.innerText = "Remove Marker";
            deleteBtn.style.cssText = "margin-top:8px; background:#ff4444; color:white; border:none; padding:5px 10px; border-radius:4px; cursor:pointer; width:100%; font-size:11px;";
            deleteBtn.onclick = () => map.removeLayer(newMarker);
            
            popupContainer.appendChild(deleteBtn);
            newMarker.bindPopup(popupContainer).openPopup();

        } else {
            console.warn("Location not found:", locationName);
        }
    } catch (err) {
        console.error("Geocoding error:", err);
    }
}

// ---------------- DATA LOADING & UTILS ----------------

function normalize(s) {
    // Normalizes text (lowercase, removes accents) for fuzzy matching
    return (s || "").toString().trim().replace(/\s+/g, " ").normalize("NFKD").toLowerCase();
}

function clearActive() {
    // Resets the CSS class that makes the risk boxes glow
    [rectYellow, rectOrange, rectRed].forEach(r => r && r.classList.remove("active"));
}

async function loadBarangays() {
    try {
        const resp = await fetch(API_BARANGAYS);
        if (!resp.ok) throw new Error("API not available");
        const json = await resp.json();
        
        // Standardize data structure
        barangays = json.map(it => ({
            barangay: it.barangay, 
            elevations: Array.isArray(it.elevations) ? it.elevations : (it.elevation ? [it.elevation] : [])
        }));
    } catch (err) {
        // Fallback to local CSV if API is down
        console.warn("API fail, using CSV fallback");
        try {
            const r = await fetch(CSV_FALLBACK);
            if (!r.ok) throw new Error("CSV fallback failed");
            barangays = parseCsvToBarangays(await r.text());
        } catch (err2) { 
            console.error("Data load failed", err2); 
            barangays = []; 
        }
    }
    
    // Build Map for O(1) Access
    barangayLookup.clear();
    for (const b of barangays) barangayLookup.set(normalize(b.barangay), b);
    
    // Populate Dropdown
    datalist.innerHTML = "";
    barangays.forEach(b => {
        const opt = document.createElement("option");
        opt.value = b.barangay;
        datalist.appendChild(opt);
    });
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
        const name = cols[bIdx];
        const elev = parseFloat(cols[eIdx]);
        
        if (name && !isNaN(elev)) {
            if (!groups.has(name)) groups.set(name, []);
            groups.get(name).push(elev);
        }
    }
    return Array.from(groups.entries()).map(([barangay, elevations]) => ({ barangay, elevations }));
}

// ---------------- EVENT LISTENERS ----------------

// 1. Location Input Change (Auto-fill elevation)
locationInput.addEventListener("change", () => {
    const found = barangayLookup.get(normalize(locationInput.value));
    const elevPreviewEl = document.getElementById("elevPreviewValue");

    if (found) {
        locationInput.dataset.valid = "true";
        // Fill hidden input with first elevation found (ML backend handles sampling)
        elevationInput.value = found.elevations.length ? String(found.elevations[0]) : "";
        // Show random elevation sample to user
        elevPreviewEl.textContent = found.elevations.length 
            ? found.elevations[Math.floor(Math.random() * found.elevations.length)].toFixed(3) 
            : "—";
    } else {
        locationInput.dataset.valid = "false";
        elevationInput.value = "";
        elevPreviewEl.textContent = "—";
    }
});

// 2. Form Submit
form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await predictFlood();
});

// ---------------- PREDICTION LOGIC ----------------
async function predictFlood() {
    clearActive();
    if (advisoryContent) advisoryContent.innerText = "Loading advisory...";

    // Validation
    const found = barangayLookup.get(normalize(locationInput.value));
    if (!found) return alert("Please choose a valid Barangay.");

    const duration = Number(durationInput.value);
    const rainfall = Number(rainfallInput.value);
    if (duration <= 0 || rainfall < 0) return alert("Invalid rainfall parameters.");

    const payload = {
        Barangay: found.barangay,
        Duration_hr: duration,
        Rainfall_mm: rainfall
    };

    try {
        // A. Call ML Backend
        const response = await fetch(API_PREDICT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();

        // B. Update UI Visuals
        clearActive();
        const colors = [rectYellow, rectOrange, rectRed];
        if (colors[result.numeric_label]) colors[result.numeric_label].classList.add("active");

        highlightLocation(found.barangay, result.numeric_label, result.chosen_elevation);

        // C. Call NLP Backend for Text
        const riskLabel = result.risk_label || (["Low", "Moderate", "High"][result.numeric_label]);
        let advisoryText = "No advisory available.";
        
        try {
            const nlpResp = await fetch(API_NLP_RISK, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ risk: riskLabel })
            });
            const nlpData = await nlpResp.json();
            advisoryText = nlpData.advisory;
        } catch (e) {
            console.error("NLP Error", e);
        }

        if (advisoryContent) advisoryContent.innerText = advisoryText;

        // D. Update Safety Guidelines Panel
        if (typeof updateSafetyGuidelines === 'function') {
            updateSafetyGuidelines(result.numeric_label);
        }

        // E. Save to LocalStorage History
        const historyRecord = {
            id: Date.now(),
            timestamp: new Date().toLocaleString(),
            location: found.barangay,
            elevation: result.chosen_elevation,
            rainfall: rainfall,
            duration: duration,
            riskLevel: result.numeric_label,
            riskLabel: riskLabel,
            advisory: advisoryText
        };
        saveHistory(historyRecord);

    } catch (error) {
        console.error(error);
        alert("Prediction failed: " + error.message);
        if (typeof resetSafetyGuidelines === 'function') resetSafetyGuidelines();
    }
}

// ---------------- GEOLOCATION LOGIC ----------------
const locateBtn = document.getElementById("locateBtn");
let userMarker = null; 

if (locateBtn) {
    locateBtn.addEventListener("click", () => {
        if (!navigator.geolocation) return alert("Geolocation not supported.");

        const originalIcon = locateBtn.innerHTML;
        locateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; // Loading state

        navigator.geolocation.getCurrentPosition(async (pos) => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;

            if (!map) return;

            // Update user marker
            if (userMarker) map.removeLayer(userMarker);
            userMarker = L.marker([lat, lon]).addTo(map).bindPopup("<strong>You are here</strong>").openPopup();
            map.flyTo([lat, lon], 16);

            // Reverse Geocoding to find address
            try {
                const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`;
                const response = await fetch(url);
                const data = await response.json();
                
                if (data.address) {
                    const city = data.address.city || data.address.town || "";
                    if (!city.includes("Dasmariñas")) {
                        alert(`Located in ${city}, outside supported area.`);
                    } else {
                        // Extract and normalize Barangay name
                        let detected = data.address.quarter || data.address.neighbourhood || data.address.suburb || "";
                        detected = detected.replace(/^Barangay\s+/i, "");
                        
                        const norm = normalize(detected);
                        let match = barangayLookup.get(norm);

                        // Fuzzy match fallback
                        if (!match) {
                            for (const [key, val] of barangayLookup.entries()) {
                                if (key.includes(norm) || norm.includes(key)) {
                                    match = val;
                                    break;
                                }
                            }
                        }

                        if (match) {
                            locationInput.value = match.barangay;
                            locationInput.dispatchEvent(new Event("change"));
                        } else {
                            alert("Detected location but couldn't match specific Barangay.");
                        }
                    }
                }
            } catch (e) { console.error(e); }
            
            locateBtn.innerHTML = originalIcon;
        }, () => {
            alert("Permission denied.");
            locateBtn.innerHTML = originalIcon;
        });
    });
}

function saveHistory(record) {
    let history = JSON.parse(localStorage.getItem("floodHistory") || "[]");
    history.unshift(record);
    if (history.length > 50) history = history.slice(0, 50); // Limit to last 50
    localStorage.setItem("floodHistory", JSON.stringify(history));
}