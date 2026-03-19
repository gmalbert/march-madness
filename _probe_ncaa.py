import requests
from bs4 import BeautifulSoup
import re

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
r = requests.get(
    "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings",
    headers=headers, timeout=20
)
print("Status:", r.status_code)
soup = BeautifulSoup(r.text, "html.parser")
tables = soup.find_all("table")
print("Tables found:", len(tables))
if tables:
    ths = [th.get_text(strip=True) for th in tables[0].find_all("th")]
    print("Headers:", ths[:15])
    rows = tables[0].find_all("tr")
    print("Rows:", len(rows))
    for i, row in enumerate(rows[1:4]):
        tds = [td.get_text(strip=True) for td in row.find_all("td")]
        print(f"  Row {i+1}:", tds[:10])

# Look for any JSON/API data embedded
scripts = soup.find_all("script")
for s in scripts:
    text = s.get_text() if s.string else ""
    if "ranking" in text.lower() and len(text) > 200:
        print("Found script with rankings data, first 300 chars:", text[:300])
        break

# Check for data attributes
print("\nPage title:", soup.title.text if soup.title else "none")
print("Content length:", len(r.text))
