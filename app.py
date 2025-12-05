""" FloodWatch NLP API Service FastAPI endpoint for flood advisory classification """
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
from typing import List
import random

# ---------------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------------
app = FastAPI(title="FloodWatch NLP API", version="1.2")

# Enable CORS for the frontend (port 5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Model Loading (Correct Paths)
# ---------------------------------------------------------
MODEL_PATH = Path("models_nlp/nb_model.pkl")
VECTORIZER_PATH = Path("models_nlp/tfidf_vectorizer.pkl")
model = None
vectorizer = None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ NLP Model and Vectorizer loaded successfully")
except Exception as e:
    print("\n⚠ WARNING: Could not load model or vectorizer.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")
    print(f"Error: {e}")
    print("Run: python train_nlp.py to regenerate the model.\n")

# ---------------------------------------------------------
# Static Advisory Messages (Used for Risk-Based NLP)
# ---------------------------------------------------------
ADVISORY_MESSAGES = {
    0: {
        "level_name": "Low Chance (Yellow Warning)",
        "advisory": [
            "Based on the latest weather models, Light to Moderate Rainfall is the dominant precipitation expected across the affected areas in the coming hours, indicating a generally manageable level of rain that may cause minor inconvenience but is not immediately anticipated to trigger severe widespread flooding.",
            "Minor Flooding is Possible in low-lying and flood-prone areas, especially in settlements with poor drainage systems. All residents, particularly those in susceptible barangays, are strongly advised to stay vigilant and maintain a high level of preparedness for potential changes in local conditions.",
            "Several provinces and their component local government units are expected to experience Light to Moderate Rain over the forecast period. The public is urged to exercise caution, avoid unnecessary travel, and monitor official weather updates diligently for safety.",
            "Current observations indicate that Light to Moderate Rain is Presently Affecting the region, and this weather disturbance has the potential to expand its coverage and reach adjacent cities and municipalities within the next one to two hours.",
            "Weather radar and satellite imagery analysis suggest that the rain-bearing clouds will propagate, resulting in an expansion of the Light to Moderate Rain coverage to more areas across the region within the critical timeframe of the next two to three hours.",
            "Over the span of the next twelve (12) hours, residents should prepare for periods of Light to Moderate Rainfall, coupled with the Possibility of Isolated Thunderstorms, which could lead to localized, brief heavy downpours and lightning hazards.",
            "The prevailing atmospheric conditions indicate that Intermittent Light to Occasionally Moderate Rain, accompanied by Thunderstorms, is highly probable across the area over the next twelve (12) hours; citizens are reminded to secure properties and avoid outdoor activities during lightning events.",
            "Specifically, the Northern and Central sections of the affected region are forecast to receive Light to Moderate Rainfall within the immediate two-to-three-hour window. Local disaster risk reduction and management offices in these areas are advised to activate monitoring protocols.",
            "Due to the persistent rainfall and the potential for slow or inefficient drainage systems, Minor Flooding in urban centers and low-lying areas remains a possibility; it is imperative for the populace to stay continuously updated through official channels regarding water level status.",
            "The latest hydrological reports confirm that River Water Levels are Exhibiting a Continuous Receding Trend, indicating a decrease in immediate flash flood risk from major river systems; however, localized stream flooding remains a concern in smaller tributaries.",
            "While the morning hours were characterized by relatively Light to Moderate Rainfall, an intensification of precipitation is anticipated, with Heavier Rain Showers Expected to Develop Later Today, urging a reassessment of preparedness levels.",
            "Light to Moderate Rain is expected to cover multiple provinces, potentially impacting numerous communities and barangays. Provincial and municipal DRRMOs are directed to disseminate advisories to ensure community safety and readiness.",
            "As of the latest observation, Light to Moderate Rain is Currently Underway over the main area of concern, and cloud movement suggests a high probability of this rain band spreading to nearby, adjacent geographical areas in the immediate future.",
            "The current weather system is projected to sustain Light to Moderate Rain, interspaced with Sporadic Heavy Bursts of Rainfall. These brief but intense downpours may locally exacerbate surface runoff and drainage issues.",
            "Light to Moderate Rain with Occasional Heavy Showers is presently being experienced, and this precipitation pattern is expected to propagate, potentially reaching and affecting other currently dry locations in the region.",
            "Forecasts indicate that the prevailing weather, characterized by Light to Moderate Rain interspersed with Short-Duration Heavy Rainfall Events, will continue to persist over the area, demanding continued monitoring for localized flood development.",
            "The area is presently under the influence of Light to Moderate Rain, containing instances of Heavier Rainfall, which is assessed to have the capability to spread to a wider geographical scope, necessitating an expansion of the advisory coverage.",
            "Following the dissipation of the main rain mass and the subsequent recording of only Light to Moderate Rain Earlier, the standing official weather warnings for the previously affected areas have been formally lifted, indicating a significant de-escalation of the immediate threat.",
            "Despite any immediate lulls, the overall forecast trajectory indicates that Light to Moderate Rain is Still Expected to Materialize or continue intermittently within the next few hours, urging residents to maintain situational awareness.",
            "Light to Moderate Rain with Intermittent Heavy Bursts is presently ongoing and is projected to continue for a duration of several hours, potentially leading to saturated ground conditions and increased surface runoff.",
            "Multiple areas are currently under the influence of Light to Moderate Rain, featuring embedded periods of Heavy Showers. Residents in low-lying and coastal barangays are especially reminded to prepare for localized pooling of water.",
            "The current precipitation, marked by Light to Moderate Rain with Heavy Moments, is ongoing and its structure suggests it may persist for an extended period, requiring continuous monitoring of local drainage systems.",
            "Light to Moderate Rain with Occasional Heavy Rainfall continues to be observed across several key areas. Local government units are advised to ensure all waterways and drainage canals are clear to mitigate potential flooding.",
            "Minor Flooding remains a potential hazard in localized areas, particularly where water is not draining effectively. Citizens are urged to avoid walking or driving through floodwaters, even if they appear shallow.",
            "The possibility of Scattered Rain exists, but the overall intensity and distribution are such that they are not anticipated to cause major, widespread flooding at this time.",
            "All concerned agencies and the general public are advised to Keep Checking Official Channels for Updated Thunderstorm and Rainfall Advisories, as conditions can change rapidly and unexpectedly.",
            "The expectation is for More Scattered Rains to develop, but the analysis indicates that the current system is not strong enough to cause Major Flooding in the forecast area.",
            "The influence of the Southwest Monsoon (Habagat) has significantly weakened, and it is no longer expected to be a direct cause of flooding in the region, shifting the focus to localized weather phenomena.",
            "Observations confirm that the day has seen Light to Short Moderate Rains, and in conjunction with this, River Levels are Continuing to Drop, leading to a reduction in the overall hydrological threat.",
            "Looking ahead to the next few days, the forecast suggests the prevalence of Light to Occasionally Moderate Rain, along with the possibility of Thunderstorms, underscoring the need for sustained vigilance.",
            "There is a specific concern for Small Coastal Flooding or 'King Tide' phenomenon, which may be exacerbated during periods of high tide, particularly in the early morning and late afternoon hours; coastal residents must be prepared.",
            "The public, alongside local chief executives and disaster officials, is strictly advised to Stay Alert and Maintain Continuous Monitoring of Official Weather Updates to facilitate timely responses and decision-making.",
            "The weather is expected to feature Isolated Light Rains under Cloudy to Overcast Skies, a condition typically associated with a minimal threat level, yet still requiring caution.",
            "The forecast calls for Light Rains, with the possibility of Minor Flooding in susceptible areas. Residents of affected barangays must maintain vigilance and prepare for minor disturbances.",
            "Moderate Rain Showers are deemed likely in the forecast area. While there is no immediate significant risk to life or property, staying alert to all weather updates is crucial for safety preparations.",
            "A period of Light to Moderate Rainfall is Expected This Afternoon. Current projections show no significant flooding is anticipated from this event, but localized monitoring is still recommended.",
            "Currently, there is No Official Rainfall Warning in Effect. However, residents are strongly advised to stay updated with continuous weather bulletins in case conditions necessitate a swift change in warning status.",
            "The Flood Risk is Assessed as Minimal given the current Light Rain Forecast. Nevertheless, the public should continue monitoring local drainage and river conditions as a proactive measure.",
            "Occasional Light Rain is expected to affect parts of the area; residents are advised to monitor closely for any sudden, intense heavy bursts that may indicate the formation of localized thunderstorms.",
            "Light to Moderate Rain Showers are likely to impact low-lying barangays within the afternoon period. Local officials in these zones should ensure that pre-emptive measures are in place.",
            "While brief periods of heavy rain may occur due to localized convection, the overall forecast suggests that widespread, catastrophic flooding is highly unlikely at this time.",
            "The Light Rain is Currently Ongoing and is expected to continue for the next few hours. Motorists should drive with extra caution due to potentially wet and slippery roads.",
            "Intermittent Light to Moderate Rain Showers are possible, and these conditions could impact visibility on major thoroughfares and roads. Drivers are urged to reduce speed and use headlights.",
            "The area of Light to Moderate Rain has the potential to extend its reach to adjacent provinces by the time evening sets in, requiring an expansion of the precautionary advisory perimeter.",
            "Light Rains, accompanied by Isolated Thunderstorms, are likely to form specifically over mountain areas. Hikers and residents in mountainous barangays should take immediate precautions against landslides and lightning.",
            "Light to Moderate Rainfall may lead to slight water buildup or ponding in identified poor drainage zones. Citizens should ensure nearby storm drains are free of debris to aid flow.",
            "Scattered Light Rains are officially forecast to continue until tonight. The overall impact remains low, but preparedness for a damp evening is advised.",
            "Residents in various districts should Expect Light to Moderate Showers, with occasional thunder and lightning. Outdoor activities should be planned with this potential for electrical activity in mind.",
            "The Light Rains may persist throughout the entire night, though current models indicate minimal flooding concerns for the duration of this period.",
            "Light Rain Showers in urban and metropolitan locations may predictably slow down traffic flow. Commuters are advised to allocate extra time for travel this period.",
            "Light to Moderate Rainfall is a possibility this afternoon, which may affect riverside settlements. Communities adjacent to waterways are advised to monitor water level changes actively.",
            "Brief Moderate Rain Showers have the potential to rapidly develop due to localized thunderstorm activity. These short, intense events can cause temporary poor visibility.",
            "Light Rain is Likely to be the predominant weather across most barangays. The overall impacts are expected to remain low, provided drainage systems function normally.",
            "Light to Moderate Rain with isolated spots of heavier precipitation, is expected to continue intermittently. This requires a constant watch for localized areas receiving above-average rainfall.",
            "The Light Rainfalls observed may expand into nearby municipalities later tonight, suggesting a movement of the rain-bearing cloud system in that direction.",
            "Intermittent Rain Showers are anticipated, but based on current meteorological data, no major, widespread flooding is foreseen at this time.",
            "Light to Moderate Rain could linger and persist as the general cloud cover over the region continues to thicken and maintain atmospheric moisture.",
            "Short Moderate Showers are a possibility, which could lead to minor pooling of water on road surfaces, urging drivers to exercise vigilance against hydroplaning.",
            "Periods of Light Rain are expected. Residents must keep monitoring official channels for updated Thunderstorm Advisories, as these pose the main weather-related risk.",
            "Light to Moderate Showers may specifically hit elevated areas and valley communities. These locations need to monitor for quick runoff and flash flooding potential, respectively.",
            "Light Rainfall is currently forecast and may affect morning activities. The overall impact is low, and no major disruptions are expected to routine schedules.",
            "The predicted Light to Moderate Rains may be sufficient to trigger slight rises in small creeks and minor tributaries. Residents near these smaller waterways should be informed.",
            "Localized heavy rain is a distinct possibility, but these events are expected to be short-lived and quickly dissipate. The public is still advised to stay cautious during these brief heavy bursts.",
            "Light Rains with occasional stronger showers are possible as prevailing winds shift their direction, which can often introduce new moisture-laden air masses.",
            "Intermittent Light to Moderate Rains are expected to persist due to lingering atmospheric moisture and local convergence zones over the area.",
            "The Light to Moderate Rainfall predicted could usher in cooler conditions and a refreshing dip in temperature throughout the day.",
            "Light Rain is forecast to continue, with a low probability or chance of significant thunderstorm development accompanying the precipitation.",
            "Light to Occasionally Moderate Showers are projected to continue across scattered zones and localized regions, maintaining the need for umbrellas and light rain gear.",
            "The area of Light Rainfall may spread farther inland as the existing cloud systems continue their general movement eastward across the landmass.",
            "The potential for Minor Flooding still exists in established flood-prone areas, especially those with clogged or compromised drainage systems, even with the forecast of only light rain.",
            "Light to Moderate Rains are possible later in the forecast cycle, but the overall assessment is that the flood threat remains low, maintaining a generally safe status."
        ],
        "color": "#FFC107"
    },
    1: {
        "level_name": "Moderate Chance (Orange Warning)",
        "advisory": [
            "Localized Flooding is Highly Probable in areas identified as low-lying, urbanized, or situated in close proximity to river systems due to the anticipated volume of precipitation.",
            "Motorists and pedestrians are warned that Street Flooding may occur in various locations caused by the convergence of trapped surface water, potential drainage blockages, and the effects of high tide.",
            "Residents should anticipate that Local Flooding is likely to occur, particularly in built-up urban zones, depressions or low-lying areas, and communities situated adjacent to riverbanks.",
            "Precautions are strictly advised for individuals residing near mountain slopes and low-lying areas along the specific river systems mentioned in this advisory due to the risk of hazards.",
            "Light to Moderate, and at times Heavy Rain is currently being experienced over several areas; this weather condition is expected to continue and may affect adjacent localities.",
            "The General Public and Disaster Risk Reduction and Management Offices (DRRMOs) are advised to continuously monitor weather updates and await further advisories from official channels.",
            "Residents and Local Disaster Councils within the mentioned areas are advised to maintain vigilance, monitor river levels, and take appropriate precautionary actions against potential adversity.",
            "People living near rivers, mountain slopes, and low-lying areas are Strongly Advised to Take Preemptive Action as the threat of flooding remains possible and significant.",
            "The public and local disaster offices are directed to Prepare for River Flooding, Water Buildup, and Tide-Affected Hazards, ensuring all safety protocols are in place.",
            "Residents in Mountain-Slope and Low-Lying Areas are strictly advised to stay alert for the possibility of Flashfloods and Landslides triggered by continuous rainfall.",
            "Communities situated in Mountainous and Low-Lying Zones along the identified river systems are advised to remain on high alert for rapid changes in water levels.",
            "The public and Local Disaster Offices are advised to Take Necessary Precautions to mitigate risks associated with river flooding, flashfloods, and rain-induced landslides.",
            "Local Disaster Councils and Residents living near mountain slopes and low-lying areas of the mentioned river systems are urged to monitor their immediate surroundings for signs of danger.",
            "The Public and Involved Disaster Councils are advised to remain in a state of alert, execute needed precautions, and constantly monitor official weather updates.",
            "Forecasts indicate that Light to Moderate, and at times Heavy Rains are expected to prevail over the area, potentially causing water levels to rise.",
            "Heavy Rainfall is Expected due to the prevalence of the Shear Line, with an accumulated rainfall of 50–100 mm projected from today until tomorrow.",
            "A significant volume of Heavy Rainfall (approx. 50–100 mm) attributed to the Shear Line is expected to occur from tomorrow noon extending to the following day.",
            "Another episode of Heavy Rainfall (50–100 mm) driven by the Shear Line system is anticipated to affect the region the day after the current forecast period.",
            "Moderate to Heavy Rain, with estimated accumulation of 50–100 mm, is expected to affect several areas from today until noon tomorrow; residents must prepare.",
            "Moderate to Heavy Rain (50–100 mm) is projected to persist from tomorrow noon to the next day in several key areas, increasing the risk of saturation.",
            "Rainfall Accumulation may be Higher in mountainous terrains and elevated places due to orographic lifting, increasing the risk of slope failure.",
            "Hazard Impacts may Worsen in localities that have already experienced heavy rain beforehand, as soil saturation levels are likely critical.",
            "Light to Moderate, and sometimes Heavy Rain is currently being recorded in some areas and is forecast to persist, requiring continuous monitoring.",
            "Waterways and Tributaries listed in the bulletin may still be affected by rising waters; communities along these banks should remain vigilant.",
            "Residents near Mountain Slopes and Low-Lying Areas of the mentioned river systems are still advised to take specific precautionary measures against hydro-meteorological hazards.",
            "Flooding is Expected in susceptible places, including significant Road Flooding and Water Buildup in low-lying residential areas and subdivisions.",
            "River Systems and Tributaries listed in the bulletin may continue to be affected by surface runoff; immediate awareness is required for nearby residents.",
            "The weather outlook indicates Significant Cloud Cover, Scattered Rain, and Isolated Thunderstorms, which may bring sudden moderate to heavy downpours.",
            "The official forecast calls for Mostly Cloudy Skies with Scattered Rain and Isolated Thunderstorms, necessitating readiness for sudden weather changes.",
            "Rainfall Intensity is expected to range from light to moderate, becoming Occasionally Heavy, which may trigger rapid localized flooding.",
            "Heavy Rainfall is Expected in Dasmariñas City; residents specifically in low-lying and flood-prone barangays should closely monitor flooding conditions.",
            "Floodwaters may reach Critical Points by tonight; Community Response Teams have been activated to assist in potential emergency situations.",
            "Extreme Caution is Advised due to intermittent heavy rains; urban Drainage Systems may be Overwhelmed Temporarily, leading to street inundation.",
            "Riverbanks are Nearing Overflow Levels; continuous rainfall may trigger localized flooding in adjacent settlements.",
            "Emergency Response Teams have been mobilized as reports indicate Water Levels are Rising in critical low-lying areas.",
            "Thunderstorm Activity is Increasing; residents are advised to take precautions, especially those residing in known flood-prone zones.",
            "Flood Checkpoints have been established by authorities; Local Officials are Monitoring Weather Developments closely to ensure public safety.",
            "Persistent Rains are Expected Overnight, keeping the risk of Minor Landslides and Flooding possible in vulnerable terrain.",
            "Community Assistance Teams are ready for immediate deployment, and Evacuation Centers have been prepared for potential displacement of families.",
            "The Local Government Advises Caution and strongly suggests avoiding non-essential outdoor activity during these rainy hours to ensure safety.",
            "Low-Lying and Riverside Communities may experience Rising Water Levels as rains persist; immediate evacuation preparations should be made if necessary.",
            "Urban Areas with Poor Drainage Systems remain at high risk of Quick Water Buildup during episodes of heavy showers.",
            "Residents living near Creeks and Tributaries are advised to stay alert for Sudden Water Surges caused by upstream downpours.",
            "Flooding may still occur in areas that have already received significant rainfall earlier today due to delayed runoff and soil saturation.",
            "Roads in some Barangays may become slippery or flooded due to continuous rain; Motorists should Slow Down and exercise maximum driving caution.",
            "Light to Moderate Rain may Intensify, thereby increasing the chances of Localized Flooding in identified hazard zones.",
            "Mountain Communities are advised to strictly watch for possible Soil Movement or Landslides due to the highly saturated ground conditions.",
            "Drainage-Challenged Areas may experience Recurring Water Accumulation as rainbands continue to move across the region.",
            "Nearby Rivers may Swell rapidly from surface runoff; residents should monitor local advisories and water level markers.",
            "Heavy Rainfall over the Next Few Hours has the potential to trigger Flashfloods in identified vulnerable zones and catchment areas.",
            "Communities Affected by Earlier Flooding should prepare for the possibility of Rising Water Again as new rain bands develop.",
            "Weather Conditions may Worsen in areas directly impacted by the Shear Line, leading to prolonged periods of precipitation.",
            "Preemptive Evacuation may be Necessary in places experiencing persistent water accumulation to ensure the safety of residents.",
            "Landslide-Prone Areas are urged to remain vigilant and observant of their surroundings due to the effects of Continuous Rainfall.",
            "Water Levels in Small Rivers and Streams may rise rapidly with incoming heavy rain; immediate awareness is required for nearby households.",
            "Local Authorities Advise families to prepare Emergency Kits as rainfall intensity may increase significantly overnight.",
            "Flood-Prone Communities should expect possible Road Closures and impassable routes if rains intensify later in the forecast period.",
            "Residents in Coastal Areas should watch for Tide-Assisted Flooding, particularly during peak tide hours which may aggravate drainage outflow.",
            "Continuous Moderate to Heavy Rains may strain local drainage capacities, resulting in prolonged street flooding.",
            "People Living in Hazard Zones are advised to Coordinate Closely with Local Disaster Councils for timely evacuation and safety instructions.",
            "Moderate to Heavy Rainfall may worsen existing flood conditions in some barangays, delaying the subsidence of floodwaters.",
            "Nearby Watercourses may Overflow if the rain persists beyond the next few hours; residents should be ready to move to higher ground.",
            "The Public is Advised to Avoid Crossing swollen rivers and flooded pathways to prevent accidents and drowning incidents.",
            "Emergency Services are on standby and may respond to reports of Rising Water in low-lying communities.",
            "The Flashflood Risk remains High in mountainous regions with steep slopes; residents should observe for changes in river color or flow.",
            "Intermittent Heavy Rains may cause Sudden Water Surges in urban drainage systems, posing risks to pedestrians and vehicles.",
            "Visibility on Major Roads may be significantly reduced due to heavy downpours; Motorists should Slow Down and use headlights.",
            "Water Levels may Fluctuate Quickly depending on rainfall intensity and the efficiency of local drainage flow.",
            "Strong Rainfall could trigger Localized Inundation in residential zones with poor runoff or clogged waterways.",
            "Local Government Units Advise Preemptive Evacuation if conditions continue to worsen to ensure zero casualties.",
            "Isolated Thunderstorms may enhance rainfall totals in already wet areas, further increasing the risk of flooding and landslides."
        ],
        "color": "#FF9800"
    },
    2: {
        "level_name": "High Chance (Red Warning)",
        "advisory": [
            "Severe flooding and landslides are predicted by PAGASA to occur frequently throughout the warned regions due to persistent heavy rainfall and unstable soil conditions.",
            "Flooding is likely to occur frequently, particularly in identified low-lying areas, urbanized centers with poor drainage, or river-front places susceptible to overflow.",
            "In places that are officially classified as moderately to very sensitive to ground movement, landslides are likely and pose a significant threat to communities.",
            "In extremely vulnerable places with steep slopes and loose soil, landslides may occur with little warning during periods of intense precipitation.",
            "Potential landslides or flash floods may be brought on by the forecasted moderate to severe rains expected over the next 24 to 48 hours across the affected provinces.",
            "Potential landslides or flash floods are expected during strong thunderstorms that may develop locally, bringing intense, short-duration rainfall.",
            "A Flood Warning has been issued by authorities due to continuous rains and rapidly rising river levels; immediate evacuation of all vulnerable areas is mandatory for resident safety.",
            "Water levels are rising dangerously near riverbanks across multiple watersheds; local authorities strongly advise preparedness and possible evacuation for riverside communities.",
            "Severe flooding is expected across low-lying regions; all residents are urgently advised to follow official evacuation orders and established safety protocols without delay.",
            "Persistent heavy rains are forecast to continue across the region over the next several hours; all identified flood-prone zones should prepare immediately for possible evacuation.",
            "A Flash Flood Alert is currently in effect for vulnerable watersheds; the public must avoid all travel and stay indoors in secure locations until further official notice is given.",
            "A Heavy Rainfall Warning has been issued by PAGASA; the risk of flash floods is now high in all vulnerable zones, especially near mountains and rivers.",
            "The Flood Alert has been officially raised to Red status; immediate evacuation is strongly recommended for all residents in high-risk barangays to ensure their safety.",
            "A Flash Flood Warning remains in effect for the area; avoid all travel, remain indoors, and monitor official updates until this warning is lifted by authorities.",
            "A heavy downpour is expected within the forecast period; prepare for potential disruptions in transportation services, power supply, and communication networks.",
            "The Landslide Risk Warning has been activated for susceptible areas; residents living in or near steep slope areas are urgently advised to evacuate to safer ground.",
            "Major flooding is likely in all low-lying areas and river basins; people in these locations should prepare emergency kits and be ready to evacuate if instructed.",
            "Many places across the warned regions may experience strong floods and landslides as a result of the ongoing severe weather system affecting the country.",
            "Flooding is very likely, especially in low-lying or crowded residential areas and in all places situated near rivers and major waterways that may overflow.",
            "Landslides are expected in areas where the ground is already unstable due to previous rainfall and existing geological conditions.",
            "Landslides may happen in places identified with very soft soil composition or on steep ground that is prone to erosion and slope failure.",
            "Heavy rain is expected to continue from today until tomorrow due to the enhanced shear line affecting the eastern and southern portions of the country.",
            "Heavy rain is expected to persist from tomorrow until the next day because of the continued influence of the shear line across Luzon and Visayas.",
            "Heavy rain is expected midweek due to the projected reactivation of the shear line, bringing another round of significant rainfall to already saturated areas.",
            "Very strong rain, with intensities that may exceed warning thresholds, is expected today until tomorrow in many areas of Luzon, particularly the eastern seaboard.",
            "Very strong rain is expected tomorrow until midweek in several regions of Visayas and Mindanao as the weather disturbance moves across the archipelago.",
            "Disaster officials at all levels of government should take all necessary steps and mobilize resources to protect people and property from the impending hazards.",
            "Communities in coastal and low-lying areas should prepare for combined hazards including strong winds, heavy rains, and potentially destructive high tide effects.",
            "Flooding continues to be reported in some areas as large rivers and major tributaries continue to rise and approach critical levels.",
            "Heavy rain is forecast in several regions until Wednesday because of the combined effects of a tropical disturbance and the prevailing shear line.",
            "Strong thunderstorms with heavy rain and gusty winds are happening now over parts of Metro Manila and Southern Luzon and may last for a few more hours.",
            "Communities situated near mountains, rivers, and coastlines should prepare comprehensively for possible flash floods, landslides, and river flooding.",
            "People in affected areas should stay alert for all hazards associated with the storm, including flooding and strong winds, and keep monitoring PAGASA updates.",
            "Flooding along rivers and in naturally low places is possible as water accumulates; take necessary precautions and avoid these areas.",
            "Coastal and riverside communities should prepare for the possibility of sudden flooding aggravated by the coincidence of heavy rain and high-tide effects.",
            "Coastal areas in the warned regions have been experiencing recurring flooding for several weeks because of seasonal high tides, which may worsen with additional rain.",
            "People living in mountainous regions and low-lying areas adjacent to rivers should maintain heightened vigilance and watch continuously for signs of flash floods and landslides.",
            "All residents are advised to take appropriate precautions against the dual threats of flooding and high tide effects directly linked to the current storm system.",
            "Communities located near watersheds and river systems should begin preparing now for potential flash floods and river flooding by securing belongings and planning evacuation routes.",
            "People in the warned areas should prepare for the likelihood of rain-triggered flooding and landslides by reviewing family emergency plans and heeding local advisories.",
            "The prevailing Southwest Monsoon will continue to bring light to heavy rain across western sections, and flooding in low-lying places and near rivers is possible.",
            "Those residing near mountain slopes and along riverbanks should stay alert for possible landslides and floods by monitoring their surroundings and official warnings.",
            "Roads in affected areas may become blocked by floodwater or debris, classes at all levels may be suspended, and agricultural crops may get damaged by the excessive rainfall.",
            "Low-lying areas, particularly urban centers with inadequate drainage, may experience significant flooding because of the combined effects of poor drainage capacity and high tide.",
            "Communities throughout the warned provinces should prepare for the possibility of sudden flooding, landslides in unstable slopes, and river overflow from swollen waterways.",
            "People living near all major and minor rivers should stay alert for flash floods and landslides by monitoring river levels and observing for signs of soil movement.",
            "Moderate to heavy rain is currently falling across the entire river basin and may continue for several more hours due to the active Intertropical Convergence Zone (ITCZ).",
            "Continuous torrential rain over saturated watersheds may trigger severe flash floods in highly exposed communities with little time for warning and evacuation.",
            "Widespread flooding is expected across multiple provinces as major river systems continue to rise rapidly from the heavy rainfall recorded over the past 24 hours.",
            "Landslides are highly possible in areas characterized by steep slopes and heavily saturated mountain soil that has reached its water-holding capacity.",
            "Preemptive evacuation is strongly advised by authorities in all villages and sitios located near unstable hillsides identified as high-risk for slope failure.",
            "Severe storm conditions associated with this weather disturbance may produce rapid-onset flash floods within minutes in identified low-lying zones and near small creeks.",
            "Intense thunderstorms embedded within the larger weather system may cause sudden landslides in vulnerable settlements located on or at the base of unstable slopes.",
            "Authorities from PAGASA and the Office of Civil Defense warn that overflowing rivers may inundate and isolate nearby barangays, cutting off access routes.",
            "Heavy rainfall over urban centers may overwhelm existing drainage systems, causing widespread urban flooding and significant disruption to daily activities.",
            "National and local emergency response teams are on high alert status due to the increasing risk of massive landslides in geologically hazardous areas.",
            "Residents in all officially identified flood-prone districts and municipalities are urged to move to higher ground immediately as a life-saving precaution.",
            "Rapid water rise is expected in small creeks and tributaries throughout the warned areas; avoid attempting to cross any flooded areas on foot or by vehicle.",
            "Communities identified as landslide-prone by MGB hazard maps should evacuate early, before nightfall, to avoid the increased hazards and complications of nighttime evacuation.",
            "Severe rainfall over the next several hours may cause significant soil erosion and slope collapse in multiple regions, particularly in mining and upland areas.",
            "Floodwaters in affected communities may reach dangerous depth levels capable of sweeping away persons and vehicles if the current downpours continue for several more hours.",
            "Strong rain bands associated with the prevailing shear line are expected to affect the region and may cause region-wide flooding across multiple provinces and cities.",
            "After the initial period of heavy rain, the ground remains critically unstable; landslides may occur without further warning even during periods of reduced rainfall.",
            "Flash floods may develop within just a few minutes during peak thunderstorm activity, especially in small catchment basins and urban waterways.",
            "Major rivers in the affected basins may reach critical water levels; continuous monitoring by authorities and the public is essential for timely response.",
            "Communities situated along steep terrain and landslide-prone zones are advised by local officials to enforce preemptive evacuation protocols for all residents.",
            "Multiple barangays across several municipalities may experience both dangerous river overflow from main channels and extensive rain-induced flooding from surface runoff.",
            "Continuous strong rains over mountainous areas may isolate some interior communities and sitios due to roads being blocked by landslides and fallen debris.",
            "All designated emergency shelters and evacuation centers should prepare for a possible influx of evacuees due to the rising flood and landslide hazards in surrounding areas.",
            "The combination of seasonal high tide and forecast heavy rain may significantly worsen the severity and duration of flooding in all coastal neighborhoods and towns.",
            "Swift and dangerous water currents may form unexpectedly in low-lying areas and drainage channels; keep all children indoors and safe from moving floodwaters."
        ],
        "color": "#F44336"
    }
}

# ---------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------
class AdvisoryRequest(BaseModel):
    text: str

class AdvisoryResponse(BaseModel):
    level: int
    level_name: str
    advisory: str
    color: str
    confidence: dict

class RiskRequest(BaseModel):
    risk: str

# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "FloodWatch NLP API",
        "model_loaded": model is not None and vectorizer is not None,
        "endpoints": ["/predict", "/batch-predict", "/generate_from_risk"]
    }

# ---------------------------------------------------------
# Predict Advisory from Full Text
# ---------------------------------------------------------
@app.post("/predict", response_model=AdvisoryResponse)
def predict_flood_level(request: AdvisoryRequest):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_nlp.py first."
        )
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        text_vector = vectorizer.transform([request.text])
        prediction = int(model.predict(text_vector)[0])
        probabilities = model.predict_proba(text_vector)[0]
        confidence = {f"level_{i}": float(prob) for i, prob in enumerate(probabilities)}
        
        info = ADVISORY_MESSAGES[prediction]
        random_advisory = random.choice(info["advisory"])

        return AdvisoryResponse(
            level=prediction,
            level_name=info["level_name"],
            advisory=random_advisory,  # ← Use the variable already created
            color=info["color"],
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------------------------------------------------------
# Batch Prediction (For Testing)
# ---------------------------------------------------------
@app.post("/batch-predict")
def batch_predict(texts: List[str]):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        results = []
        for text in texts:
            v = vectorizer.transform([text])
            p = int(model.predict(v)[0])
            probs = model.predict_proba(v)[0]
            results.append({
                "text": text,
                "level": p,
                "level_name": ADVISORY_MESSAGES[p]["level_name"],
                "confidence": {f"level_{i}": float(prob) for i, prob in enumerate(probs)}
            })
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ---------------------------------------------------------
# Risk Level → Advisory Message
# (Used by main FloodWatch frontend)
# ---------------------------------------------------------
@app.post("/generate_from_risk")
def generate_from_risk(req: RiskRequest):
    risk_map = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }
    
    if req.risk not in risk_map:
        raise HTTPException(status_code=400, detail="Risk must be: Low, Moderate, or High")

    level = risk_map[req.risk]
    info = ADVISORY_MESSAGES[level]

    return {
        "level": level,
        "level_name": info["level_name"],
        "advisory": random.choice(info["advisory"]),  # ← FIXED: Pick ONE random advisory
        "color": info["color"]
    }

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)