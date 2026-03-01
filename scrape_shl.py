import requests
import pandas as pd

API_URL = "https://www.shl.com/api/products"

# pretend to be a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
}

print("Fetching SHL data...")

res = requests.get(API_URL, headers=headers)

print("Status code:", res.status_code)

if res.status_code != 200:
    print("Failed to fetch data")
    exit()

data = res.json()

rows = []

for item in data:
    url = item.get("url", "")
    name = item.get("title", "")
    category = item.get("category", "")

    if "individual" in category.lower():
        rows.append({
            "name": name,
            "url": "https://www.shl.com" + url
        })

df = pd.DataFrame(rows).drop_duplicates()

print("Found", len(df), "assessments")

df.to_csv("shl_catalog.csv", index=False)

print("Saved -> shl_catalog.csv")