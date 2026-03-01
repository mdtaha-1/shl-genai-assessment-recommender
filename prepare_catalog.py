import pandas as pd

df = pd.read_excel("Gen_AI Dataset.xlsx")

catalog = df.drop_duplicates(subset=["Assessment_url"]).copy()

catalog["name"] = catalog["Assessment_url"].apply(
    lambda x: x.split("/")[-2].replace("-", " ").title()
)

catalog["text"] = catalog["name"] + " " + catalog["Query"]

catalog = catalog[["name", "Assessment_url", "text"]]

catalog.to_csv("catalog_prepared.csv", index=False)

print("Prepared catalog:", len(catalog))