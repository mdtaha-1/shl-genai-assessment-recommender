from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import time

print("Opening browser...")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://www.shl.com/solutions/products/product-catalog/")

time.sleep(5)

# auto scroll (load more results)
print("Scrolling page...")

last_height = driver.execute_script("return document.body.scrollHeight")

for _ in range(15):  # scroll multiple times
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

print("Collecting links...")

links = driver.find_elements(By.TAG_NAME, "a")

data = []

for link in links:
    href = link.get_attribute("href")
    text = link.text

    if href and "/product-catalog/view/" in href:
        data.append({
            "name": text,
            "url": href
        })

driver.quit()

df = pd.DataFrame(data).drop_duplicates()

print("Found:", len(df))

df.to_csv("shl_catalog.csv", index=False)

print("Saved -> shl_catalog.csv")