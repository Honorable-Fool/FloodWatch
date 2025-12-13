/* safety_guidelines.js
   - Contains PAGASA safety guidelines for each alert level
   - Called from index.js after prediction to update the UI
*/

// Dictionary mapping Risk Level (0, 1, 2) to display properties
const SAFETY_GUIDELINES = {
  0: { // LOW (Yellow)
    levelText: "LOW RISK - Stay Alert",
    levelClass: "safety-yellow",
    iconClass: "warning-yellow",
    iconText: "!",
    tips: [
      "Monitor weather updates from PAGASA regularly.",
      "Clear drainage canals and gutters around your home.",
      "Prepare emergency kits with food, water, flashlight, and first aid.",
      "Secure important documents in waterproof containers.",
      "Check on neighbors, especially elderly and PWDs.",
      "Avoid unnecessary travel during rainfall.",
      "Charge mobile phones and power banks.",
      "Stay updated on river water level announcements."
    ]
  },
  1: { // MODERATE (Orange)
    levelText: "MODERATE RISK - Be Prepared",
    levelClass: "safety-orange",
    iconClass: "warning-orange",
    iconText: "!!",
    tips: [
      "Prepare to evacuate if you live in flood-prone areas.",
      "Move household items to higher ground.",
      "Disconnect electrical appliances to prevent short circuits.",
      "Prepare vehicles for possible evacuation.",
      "Avoid crossing rivers, creeks, or flooded areas.",
      "Coordinate with barangay officials for evacuation plans.",
      "Keep children indoors and away from floodwaters.",
      "If advised to evacuate, do so immediately to designated centers.",
      "Avoid walking or driving through floodwaters."
    ]
  },
  2: { // HIGH (Red)
    levelText: "HIGH RISK - Take Immediate Action",
    levelClass: "safety-red",
    iconClass: "warning-red",
    iconText: "!!!",
    tips: [
      "EVACUATE IMMEDIATELY if in high-risk areas.",
      "Follow evacuation orders from authorities without delay.",
      "Bring only essential items and emergency kits.",
      "Turn off main power switch and gas valves before leaving.",
      "Do not wait for floodwaters to enter your home.",
      "Avoid rivers, creeks, and landslide-prone slopes.",
      "Use designated evacuation routes - do not take shortcuts.",
      "If trapped, move to the highest point and call for help.",
      "Do not attempt to cross flowing water above ankle depth.",
      "Stay away from fallen power lines and electrical wires.",
      "Listen to battery-powered radio for official instructions."
    ]
  }
};

// Function to update the safety UI based on the predicted risk level
function updateSafetyGuidelines(riskLevel) {
  const guidelines = SAFETY_GUIDELINES[riskLevel];
  
  // Get DOM elements
  const safetyTips = document.getElementById("safetyTips");
  const safetyLevelIndicator = document.getElementById("safetyLevelIndicator");
  const safetyLevelText = document.getElementById("safetyLevelText");
  const warningIcon = document.getElementById("warningIcon");
  
  // 1. Update warning icon in advisory header
  warningIcon.className = `warning-icon ${guidelines.iconClass}`;
  warningIcon.textContent = guidelines.iconText;
  warningIcon.style.display = "inline-block";
  
  // 2. Update safety level indicator banner
  safetyLevelIndicator.className = `safety-level ${guidelines.levelClass}`;
  safetyLevelIndicator.style.display = "block";
  safetyLevelText.textContent = guidelines.levelText;
  
  // 3. Populate safety tips list
  safetyTips.innerHTML = "";
  guidelines.tips.forEach(tip => {
    const tipElement = document.createElement("div");
    tipElement.className = "safety-tip";
    tipElement.innerHTML = `<i class="fas fa-exclamation-circle"></i><span>${tip}</span>`;
    safetyTips.appendChild(tipElement);
  });
}

// Function to reset safety guidelines to default (e.g. on load or error)
function resetSafetyGuidelines() {
  const safetyTips = document.getElementById("safetyTips");
  const safetyLevelIndicator = document.getElementById("safetyLevelIndicator");
  const warningIcon = document.getElementById("warningIcon");
  
  // Default tips explaining how to use the system
  safetyTips.innerHTML = `
    <div class="safety-tip">
      <i class="fas fa-info-circle"></i>
      <span>Select a barangay and click "Predict Flood Risk" to see specific safety guidelines for the current alert level.</span>
    </div>
    <div class="safety-tip">
      <i class="fas fa-volume-up"></i>
      <span>Always monitor official PAGASA bulletins through radio, TV, or social media.</span>
    </div>
    <div class="safety-tip">
      <i class="fas fa-phone-alt"></i>
      <span>Save emergency hotlines: NDRRMC (911-1406), PAGASA (433-8526).</span>
    </div>
  `;
  
  // Hide specific risk indicators
  safetyLevelIndicator.style.display = "none";
  warningIcon.style.display = "none";
}