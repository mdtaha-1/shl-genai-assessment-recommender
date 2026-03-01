import pandas as pd

# Load Excel file
df = pd.read_excel("Gen_AI Dataset.xlsx")

print("Dataset loaded successfully!")
print(df.head())