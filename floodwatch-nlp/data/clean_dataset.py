import pandas as pd
import re
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Nlp dataset.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "Nlp dataset_CLEANED.xlsx"

# 1. COMPREHENSIVE List of Banned Locations
# Key = Specific Location
# Value = Generic Replacement
REPLACEMENTS = {
    # --- METRO MANILA & CENTRAL LUZON ---
    "Metro Manila": "the metropolitan area",
    "Manila": "the city",
    "Rizal": "the area",
    "Batangas": "nearby provinces",
    "Cavite": "the province",
    "Laguna": "nearby areas",
    "Quezon": "the vicinity",
    "Bulacan": "low-lying areas",
    "Pampanga": "the region",
    "Nueva Ecija": "northern areas",
    "Tarlac": "central plains",
    "Zambales": "coastal areas",
    "Bataan": "the peninsula",
    "Dasmariñas": "the city",

    # --- NORTHERN LUZON (Cordillera/Ilocos/Cagayan) ---
    "Nueva Vizcaya": "upland areas",
    "Isabela": "the valley",
    "Quirino": "mountainous areas",
    "Aurora": "eastern areas",
    "Cagayan": "northern provinces",
    "La Union": "northern coastal areas",
    "Benguet": "highland areas",
    "Pangasinan": "the plains",
    "Ifugao": "highland areas",
    "Apayao": "northern highlands",
    "Kalinga": "northern highlands",
    "Mountain Province": "mountainous zones",

    # --- SOUTHERN LUZON & BICOL ---
    "Occidental Mindoro": "western islands",
    "Oriental Mindoro": "eastern islands",
    "Marinduque": "island provinces",
    "Romblon": "island provinces",
    "Palawan": "western islands",
    "Camarines Norte": "southern areas",
    "Camarines Sur": "southern areas",
    "Catanduanes": "eastern islands",
    "Albay": "volcanic areas",
    "Sorsogon": "southern tips",
    "Masbate": "island areas",
    "Bicol": "southern regions",

    # --- VISAYAS ---
    "Antique": "western coastal areas",
    "Aklan": "western areas",
    "Capiz": "central areas",
    "Iloilo": "the province",
    "Guimaras": "the island",
    "Negros Occidental": "western plains",
    "Negros Oriental": "eastern plains",
    "Cebu": "central islands",
    "Bohol": "island provinces",
    "Siquijor": "small islands",
    "Biliran": "island areas",
    "Leyte": "eastern provinces",
    "Southern Leyte": "southern provinces",
    "Northern Samar": "northern coastal areas",
    "Eastern Samar": "eastern coastal areas",
    "Samar": "the region",
    "Visayas": "central islands",

    # --- MINDANAO ---
    "Agusan del Norte": "northern valleys",
    "Agusan del Sur": "southern valleys",
    "Surigao del Norte": "northern tips",
    "Surigao del Sur": "eastern coasts",
    "Dinagat Islands": "island groups",
    "Misamis Oriental": "coastal regions",
    "Misamis Occidental": "western coastal regions",
    "Bukidnon": "plateau areas",
    "Lanao del Norte": "interior areas",
    "Camiguin": "island provinces",
    "Davao Region": "southern regions",
    "Caraga": "eastern regions",
    "General Santos City": "the city",
    "Alabel": "nearby towns",
    "Mindanao": "southern islands",

    # --- RIVER BASINS & SPECIFIC AREAS ---
    "Bicol River Basin": "river basins",
    "Buayan – Malungon River Basin": "river basins",
    "Buayan–Malungon River Basin": "river basins",
    "Abra River Basin": "river basins",
    "Agus-Mandulog-Iligan River Basin": "river basins",
    "Agus-Mandulog and Iligan River Basin": "river basins",
    "Agusan River Basin": "river basins",
    "Tagum–Libuganon River Basin": "river basins",
    "Pampanga River Basin": "major river basins",
    "Pampanga Delta": "delta areas",
    "Candaba Swamp": "swampy areas",
    "Sarangani bay": "coastal bays",
    "Agusan River": "the river",
}

# 2. Regex Patterns to Delete
REMOVE_PATTERNS = [
    r"\(.*?\)",                 # Removes lists inside parentheses like (San Juan, Lobo...)
    r"\#\w+",                   # Removes hashtags like #Isabela
    r"Mindanao Pagasa.*",       # Removes specific agency mentions
    r"NL-PRSD",                 # Removes region specific acronyms
]

def clean_text(text):
    if not isinstance(text, str):
        return text
    
    # A. Apply Regex Removals
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # B. Apply Location Replacements
    # Sort keys by length (descending) to replace "Occidental Mindoro" before "Mindoro"
    for loc in sorted(REPLACEMENTS.keys(), key=len, reverse=True):
        replacement = REPLACEMENTS[loc]
        # Regex: \b ensures we match whole words (so "Samar" doesn't match inside "Samaritans")
        text = re.sub(r"(?i)\b" + re.escape(loc) + r"\b", replacement, text)
        
    # C. Clean up extra spaces/punctuation left behind
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s,\s", ", ", text) # Fix dangling commas
    text = re.sub(r"\s\.", ".", text)   # Fix dangling periods
    return text

def main():
    print(f"Opening {DATA_FILE}...")
    
    try:
        # Load the Excel File
        xls = pd.ExcelFile(DATA_FILE)
        sheets = {}
        
        # Process each sheet
        for sheet_name in xls.sheet_names:
            print(f"Processing sheet: {sheet_name}...")
            df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
            
            if 'Advisory Text' in df.columns:
                # Apply the cleaning function
                df['Advisory Text'] = df['Advisory Text'].apply(clean_text)
                
                # Remove rows that became empty or too short
                df = df[df['Advisory Text'].str.len() > 10]
            
            sheets[sheet_name] = df
            
        # Save back to a NEW Excel file
        with pd.ExcelWriter(OUTPUT_FILE) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print("\n" + "="*50)
        print("✓ SUCCESS! Dataset cleaned.")
        print(f"✓ Saved to: {OUTPUT_FILE}")
        print("="*50)
        print("ACTION REQUIRED: Rename this file to 'Nlp dataset.xlsx' to use it.")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()