import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv(r"data\FloodWatch_MLDataset.csv", encoding='ISO-8859-1')

# Columns to diversify
columns_to_diversify = ['Elevation_m', 'Duration_hr', 'Rainfall_mm']

# Function to add random noise
def add_noise(value, noise_factor=0.05):
    if pd.notnull(value):  # Only apply to non-null values
        noise = np.random.uniform(-noise_factor, noise_factor) * value
        return value + noise
    return value

# Apply noise to each specified column
for col in columns_to_diversify:
    df[col] = df[col].apply(lambda x: add_noise(x))

# Clip to realistic bounds
df['Duration_hr'] = df['Duration_hr'].clip(lower=0)  # No negative durations
df['Rainfall_mm'] = df['Rainfall_mm'].clip(lower=0)  # No negative rainfall
df['Elevation_m'] = df['Elevation_m'].clip(lower=0)  # Elevations shouldn't be negative

# Save the diversified dataset
df.to_csv('flood_dataset_diversified.csv', index=False)
print("Diversified dataset saved as 'flood_dataset_diversified.csv'")
print(f"Original rows: {len(df)}")
print("Sample of diversified data:")
print(df.head(10))  # Print first 10 rows for verification
