import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("ipl_data.csv")

# Replace 'No stats' with NaN
df.replace("No stats", np.nan, inplace=True)

# Display the first few rows
print(df.head())
