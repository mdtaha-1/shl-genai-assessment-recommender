import pandas as pd

df = pd.read_excel("Gen_AI Dataset.xlsx")

print("Rows:", len(df))
print("Unique URLs:", df["Assessment_url"].nunique())

print("\nSample URLs:")
print(df["Assessment_url"].drop_duplicates().head(10))