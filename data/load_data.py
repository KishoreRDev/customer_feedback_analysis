# Lifecycle Stage 3 — Data Understanding

import pandas as pd

# Function to load data from a CSV file
def load_data(path):
    # Read the CSV file and return the data as a DataFrame
    return pd.read_csv(path)