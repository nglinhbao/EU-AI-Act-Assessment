import pandas as pd

# Path to the CSV file
file_path = './datasets/results.csv'

# Read and display the CSV file
try:
    df = pd.read_csv(file_path, sep=';')  # Use semicolon as the delimiter
    print(df)
except Exception as e:
    print(f"Error reading the file: {e}")