/* analytics.js
   - Loads prediction history from LocalStorage
   - Renders Statistics (Total, High Risk Count)
   - Renders Charts using Chart.js (Pie, Scatter)
   - Renders the History Table
*/

document.addEventListener("DOMContentLoaded", () => {
    loadAnalytics();

    // Clear history button logic
    document.getElementById("clearHistoryBtn").addEventListener("click", () => {
        if (confirm("Are you sure you want to clear all history?")) {
            localStorage.removeItem("floodHistory");
            location.reload(); // Refresh to clear view
        }
    });
});

function loadAnalytics() {
    const history = JSON.parse(localStorage.getItem("floodHistory") || "[]");
    
    updateStats(history);
    renderTable(history);
    renderCharts(history);
}

// Calculates summary statistics for the sidebar
function updateStats(history) {
    document.getElementById("totalCount").innerText = history.length;
    
    // Count predictions where riskLevel is 2 (High/Red)
    const highRisk = history.filter(h => h.riskLevel === 2).length;
    document.getElementById("highCount").innerText = highRisk;

    // Calculate most frequent location
    if (history.length > 0) {
        const counts = {};
        history.forEach(h => counts[h.location] = (counts[h.location] || 0) + 1);
        // Reduce to find the key with maximum value
        const topLoc = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        document.getElementById("topLocation").innerText = topLoc;
    }
}

// Renders the HTML table rows
function renderTable(history) {
    const tbody = document.getElementById("historyTableBody");
    tbody.innerHTML = "";

    if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;">No predictions yet. Go to Dashboard to make one!</td></tr>`;
        return;
    }

    history.forEach(item => {
        const row = document.createElement("tr");
        
        // Determine badge color class based on risk level
        const riskClass = item.riskLevel === 0 ? "badge-yellow" : item.riskLevel === 1 ? "badge-orange" : "badge-red";
        
        row.innerHTML = `
            <td>${item.timestamp}</td>
            <td><strong>${item.location}</strong></td>
            <td>${item.rainfall}</td>
            <td>${item.elevation.toFixed(1)}</td>
            <td><span class="risk-badge ${riskClass}">${item.riskLabel}</span></td>
            <td><small>Input: ${item.rainfall}mm / ${item.duration}hr</small></td>
            <td title="${item.advisory}" style="max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: help;">
                ${item.advisory}
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Renders Chart.js graphs
function renderCharts(history) {
    if (history.length === 0) return;

    // 1. Pie Chart: Distribution of Risk Levels
    const counts = [0, 0, 0]; // Index 0=Low, 1=Mod, 2=High
    history.forEach(h => counts[h.riskLevel]++);

    new Chart(document.getElementById("riskChart"), {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
            datasets: [{
                data: counts,
                backgroundColor: ['#FFC107', '#FF9800', '#F44336'],
                borderWidth: 1
            }]
        }
    });

    // 2. Scatter Chart: Correlation between Rainfall/Duration and Risk
    const scatterData = history.map(h => ({
        x: h.duration,
        y: h.rainfall,
        // Set point color based on the resulting risk
        backgroundColor: h.riskLevel === 0 ? '#FFC107' : h.riskLevel === 1 ? '#FF9800' : '#F44336',
        r: h.riskLevel === 2 ? 8 : 5 // Make High risk points slightly larger
    }));

    new Chart(document.getElementById("scatterChart"), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Predictions',
                data: scatterData,
                backgroundColor: '#007bff', // Fallback color
                pointBackgroundColor: scatterData.map(d => d.backgroundColor), // Custom color per point
                pointRadius: scatterData.map(d => d.r)
            }]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Duration (hours)' } },
                y: { title: { display: true, text: 'Rainfall (mm/hr)' } }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Rain: ${context.raw.y}mm, Dur: ${context.raw.x}hr`;
                        }
                    }
                }
            }
        }
    });
}