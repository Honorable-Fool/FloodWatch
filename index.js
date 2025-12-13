/* index.js
   - Handles Map initialization (Satellite + Labels)
   - Manages User Input with Searchable Datalist
   - Orchestrates the Prediction Flow (ML -> Map -> NLP -> History)
   - Handles Geolocation ("Locate Me")
   - FIX: Uses Session Storage to persist map markers across page navigation
*/

// ---------------- CONFIGURATION ----------------
const API_BARANGAYS = "http://127.0.0.1:8000/api/barangays";
const API_PREDICT  = "http://127.0.0.1:8000/predict";
const API_NLP_RISK = "http://127.0.0.1:8001/generate_from_risk";
const CSV_FALLBACK = "/mnt/data/floodwatch_MLdataset.csv";

// ---------------- DOM ELEMENTS ----------------
const locationInput = document.getElementById("location"); // The text input box
const datalist = document.getElementById("barangayList");  // The hidden list of options
const elevationInput = document.getElementById("elevation_m");
const durationInput = document.getElementById("duration");
const rainfallInput = document.getElementById("rainfall");
const form = document.getElementById("floodForm");
const advisoryContent = document.getElementById("advisoryContent");

// Risk Indicators (Colored Boxes)
const rectYellow = document.querySelector(".rectangle-alert.yellow");
const rectOrange = document.querySelector(".rectangle-alert.orange");
const rectRed = document.querySelector(".rectangle-alert.red");

// ---------------- STATE MANAGEMENT ----------------
let barangays = [];
let barangayLookup = new Map(); // Map for fast O(1) lookup
let map;

const riskColors = {
    0: 'rgba(255, 255, 0, 1)', // Yellow
    1: 'rgba(255, 165, 0, 1)', // Orange
    2: 'rgba(255, 0, 0, 1)'    // Red
};

// Array to hold the Leaflet markers and their data
let currentMarkersData = []; 

// ---------------- INITIALIZATION ----------------
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Map centered on Dasmariñas, Cavite
    map = L.map('map').setView([14.3293, 120.9367], 13);

    // 2. Layer 1: Satellite Imagery (Base Layer)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri'
    }).addTo(map);

    // 3. Layer 2: Labels Overlay (Roads/Names on top of satellite)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    // 4. Create Legend Control
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

    // 5. Load Data and previous session markers
    addCityBoundary();
    loadBarangays(); 
    loadMarkersFromSession(); // NEW: Load markers stored in the current session
});

// ---------------- SESSION STORAGE FUNCTIONS ----------------

function saveMarkersToSession() {
    // Saves the currentMarkersData array to Session Storage
    sessionStorage.setItem('floodMarkers', JSON.stringify(currentMarkersData));
}

function loadMarkersFromSession() {
    // Loads and draws markers from Session Storage when the page loads
    const savedData = sessionStorage.getItem('floodMarkers');
    if (!savedData) return;

    currentMarkersData = JSON.parse(savedData);
    
    // Iterate over saved data and render each marker
    currentMarkersData.forEach(data => {
        // Use a modified version of highlightLocation that just draws the marker
        // without doing a full geocoding lookup
        drawMarker(data.lat, data.lon, data.locationName, data.riskLevel, data.elevation);
    });
}

// ---------------- MAP FUNCTIONS ----------------

// Fetches Dasmariñas boundary geometry and draws it on the map
async function addCityBoundary() {
    if (!map) return;
    
    const query = encodeURIComponent("Dasmariñas, Cavite, Philippines");
    const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&polygon_geojson=1&limit=1&viewbox=120.90,14.40,121.05,14.20&bounded=1`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (data && data.length > 0 && data[0].geojson) {
            L.geoJSON(data[0].geojson, {
                className: 'city-boundary',
                style: {
                    color: "#7be1e1", // Cyan outline for visibility
                    weight: 3,
                    fillOpacity: 0.1
                }
            }).addTo(map);
        }
    } catch (err) {
        console.warn("Could not load city boundary:", err);
    }
}

// Low-level function to draw a single circle marker (used by loadMarkersFromSession)
function drawMarker(lat, lon, locationName, riskLevel, elevation) {
    const color = riskColors[riskLevel] || '#3388ff';

    const newMarker = L.circleMarker([lat, lon], {
        radius: 12,          
        fillColor: color,    
        color: "#fff",       
        weight: 2,           
        opacity: 1,
        fillOpacity: 0.9     
    }).addTo(map);

    // Function to remove the marker and update the session storage array
    const removeMarker = () => {
        map.removeLayer(newMarker);
        
        // Remove the corresponding entry from the global data array
        currentMarkersData = currentMarkersData.filter(m => m.id !== newMarker._leaflet_id);
        saveMarkersToSession();
    };

    // --- CUSTOM POPUP WITH DELETE BUTTON ---
    const popupContainer = document.createElement('div');
    popupContainer.innerHTML = `
        <strong>${locationName}</strong><br>
        Risk: ${riskLevel === 0 ? "Low" : riskLevel === 1 ? "Moderate" : "High"}<br>
        <small>Elev: ${elevation ? elevation.toFixed(2) + 'm' : 'N/A'}</small>
    `;
    
    const deleteBtn = document.createElement('button');
    deleteBtn.innerText = "Remove Marker";
    deleteBtn.style.cssText = "margin-top:8px; background:#ff4444; color:white; border:none; padding:5px 10px; border-radius:4px; cursor:pointer; width:100%; font-size:11px;";
    deleteBtn.onclick = removeMarker;
    
    popupContainer.appendChild(deleteBtn);
    newMarker.bindPopup(popupContainer);

    // Assign a unique ID to the Leaflet object and store it in our data model
    newMarker._leaflet_id = newMarker._leaflet_id || Date.now() + Math.random(); 

    return newMarker;
}

// High-level function called after a new prediction
async function highlightLocation(locationName, riskLevel, elevation) {
    if (!map || !locationName) return;

    // 1. Geocode the location
    const query = encodeURIComponent(`${locationName}, Dasmariñas Cavite, Philippines`);
    const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&limit=1&viewbox=120.90,14.40,121.05,14.20&bounded=1`;

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (data && data.length > 0) {
            let lat = parseFloat(data[0].lat);
            let lon = parseFloat(data[0].lon);

            // JITTER LOGIC: Add random offset
            const maxRadius = 0.0008; 
            const angle = Math.random() * 2 * Math.PI;
            const r = Math.sqrt(Math.random()) * maxRadius;
            
            lat += r * Math.cos(angle);
            lon += r * Math.sin(angle);

            // 2. Draw the marker using the specialized function
            const newMarker = drawMarker(lat, lon, locationName, riskLevel, elevation);
            map.flyTo([lat, lon], 15);
            newMarker.openPopup();

            // 3. Save the new marker data to session storage
            currentMarkersData.push({
                id: newMarker._leaflet_id,
                lat: lat,
                lon: lon,
                locationName: locationName,
                riskLevel: riskLevel,
                elevation: elevation
            });
            saveMarkersToSession();
            
        } else {
            console.warn("Location not found:", locationName);
        }
    } catch (err) {
        console.error("Geocoding error:", err);
    }
}


// ---------------- DATA LOADING & UTILS ----------------

// Fixes common encoding issues (e.g., Santo NiÃ±o -> Santo Niño)
function fixEncoding(str) {
    if (!str) return "";
    return str.replace(/Ã±/g, "ñ").replace(/Ã‘/g, "Ñ");
}

// Normalizes text for internal comparison (lowercase, removes accents)
function normalize(s) {
    return (s || "").toString().trim().replace(/\s+/g, " ").normalize("NFKD").toLowerCase();
}

function clearActive() {
    // Resets the visual glow of the alert boxes
    [rectYellow, rectOrange, rectRed].forEach(r => r && r.classList.remove("active"));
}

async function loadBarangays() {
    try {
        const resp = await fetch(API_BARANGAYS);
        if (!resp.ok) throw new Error("API not available");
        const json = await resp.json();
        
        // Map API data to our internal structure and fix names
        barangays = json.map(it => {
            const cleanName = fixEncoding(it.barangay);
            const elevs = Array.isArray(it.elevations) ? it.elevations : (it.elevation ? [it.elevation] : []);
            return { barangay: cleanName, elevations: elevs };
        });
    } catch (err) {
        // Fallback to local CSV if API is unreachable
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
    
    // Populate the search suggestions
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
        const name = fixEncoding(cols[bIdx]); // Apply encoding fix here too
        const elev = parseFloat(cols[eIdx]);
        
        if (name && !isNaN(elev)) {
            if (!groups.has(name)) groups.set(name, []);
            groups.get(name).push(elev);
        }
    }
    return Array.from(groups.entries()).map(([barangay, elevations]) => ({ barangay, elevations }));
}

// Populates the <datalist> for the searchable input field
function populateDatalist() {
    datalist.innerHTML = "";
    barangays.forEach(b => {
        const opt = document.createElement("option");
        opt.value = b.barangay; // Value shown in the list
        datalist.appendChild(opt);
    });
}

// ---------------- EVENT LISTENERS ----------------

// 1. Location Input Change (Auto-fill elevation)
locationInput.addEventListener("change", () => {
    const found = barangayLookup.get(normalize(locationInput.value));
    const elevPreviewEl = document.getElementById("elevPreviewValue");

    if (found) {
        locationInput.dataset.valid = "true";
        locationInput.dataset.barangay = found.barangay;
        
        elevationInput.value = found.elevations.length ? String(found.elevations[0]) : "";
        
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
    const raw = locationInput.value.trim();
    const norm = normalize(raw);
    const found = barangayLookup.get(norm);

    if (!found) {
        alert("Please choose a Barangay from the suggestions (exact match).");
        return locationInput.focus();
    }

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

        if (!response.ok) {
            const txt = await response.text().catch(() => "Unknown error");
            throw new Error(`ML API error: ${txt}`);
        }
        const result = await response.json();

        // B. Update UI Visuals
        clearActive();
        const colors = [rectYellow, rectOrange, rectRed];
        if (colors[result.numeric_label]) colors[result.numeric_label].classList.add("active");

        // Highlight map location (which also saves the marker to session storage)
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
            advisoryText = "Could not fetch advisory text.";
        }

        if (advisoryContent) advisoryContent.innerText = advisoryText;

        // D. Update Safety Guidelines Panel
        if (typeof updateSafetyGuidelines === 'function') {
            updateSafetyGuidelines(result.numeric_label);
        }

        // E. Save to LocalStorage History (for the Analytics Dashboard)
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

// ---------------- GEOLOCATION LOGIC (LOCATE ME) ----------------
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

            // Reverse Geocoding
            try {
                const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`;
                const response = await fetch(url);
                const data = await response.json();
                
                if (data.address) {
                    const city = data.address.city || data.address.town || "";
                    if (!city.includes("Dasmariñas")) {
                        alert(`You are in ${city}, outside supported area.`);
                    } else {
                        // ROBUST DETECTION: Check multiple fields for the location name
                        let detectedRaw = data.address.quarter || 
                                          data.address.neighbourhood || 
                                          data.address.suburb || 
                                          data.address.village || 
                                          data.address.city_district || "";
                        
                        let detectedBarangay = detectedRaw.replace(/^Barangay\s+/i, "").trim();
                        
                        if (detectedBarangay) {
                            const norm = normalize(detectedBarangay);
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
                                locationInput.value = match.barangay; // Auto-fill the input
                                locationInput.dispatchEvent(new Event("change")); // Trigger updates
                            } else {
                                alert(`Located near "${detectedBarangay}", but it doesn't match our barangay list.`);
                            }
                        } else {
                            alert("Could not detect specific Barangay name from this location.");
                        }
                    }
                }
            } catch (e) { console.error(e); }
            
            locateBtn.innerHTML = originalIcon;
        }, () => {
            alert("Permission denied or location unavailable.");
            locateBtn.innerHTML = originalIcon;
        });
    });
}

// Saves marker data to Session Storage
function saveMarkersToSession() {
    sessionStorage.setItem('floodMarkers', JSON.stringify(currentMarkersData));
}

// Loads and draws markers from Session Storage
function loadMarkersFromSession() {
    const savedData = sessionStorage.getItem('floodMarkers');
    if (!savedData) return;

    currentMarkersData = JSON.parse(savedData);
    
    // Draw each marker saved in the session
    currentMarkersData.forEach(data => {
        drawMarker(data.lat, data.lon, data.locationName, data.riskLevel, data.elevation);
    });
}


function saveHistory(record) {
    let history = JSON.parse(localStorage.getItem("floodHistory") || "[]");
    history.unshift(record);
    if (history.length > 50) history = history.slice(0, 50); 
    localStorage.setItem("floodHistory", JSON.stringify(history));
}

// Low-level function to draw a single circle marker (called by load/highlight)
function drawMarker(lat, lon, locationName, riskLevel, elevation) {
    const color = riskColors[riskLevel] || '#3388ff';

    const newMarker = L.circleMarker([lat, lon], {
        radius: 12,          
        fillColor: color,    
        color: "#fff",       
        weight: 2,           
        opacity: 1,
        fillOpacity: 0.9     
    }).addTo(map);

    // Function to remove the marker and update the session storage array
    const removeMarker = () => {
        map.removeLayer(newMarker);
        
        // Remove the corresponding entry from the global data array
        currentMarkersData = currentMarkersData.filter(m => m.id !== newMarker._leaflet_id);
        saveMarkersToSession();
    };

    // --- CUSTOM POPUP WITH DELETE BUTTON ---
    const popupContainer = document.createElement('div');
    popupContainer.innerHTML = `
        <strong>${locationName}</strong><br>
        Risk: ${riskLevel === 0 ? "Low" : riskLevel === 1 ? "Moderate" : "High"}<br>
        <small>Elev: ${elevation ? elevation.toFixed(2) + 'm' : 'N/A'}</small>
    `;
    
    const deleteBtn = document.createElement('button');
    deleteBtn.innerText = "Remove Marker";
    deleteBtn.style.cssText = "margin-top:8px; background:#ff4444; color:white; border:none; padding:5px 10px; border-radius:4px; cursor:pointer; width:100%; font-size:11px;";
    deleteBtn.onclick = removeMarker;
    
    popupContainer.appendChild(deleteBtn);
    newMarker.bindPopup(popupContainer);

    // Assign a unique ID to the Leaflet object and store it in our data model
    newMarker._leaflet_id = newMarker._leaflet_id || Date.now() + Math.random(); 

    return newMarker;
}
