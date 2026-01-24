from bs4 import BeautifulSoup
import re
import json

with open('data_files/kenpom_page.html', 'r', encoding='utf-8') as f:
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')

# Find ALL tr elements with td cells (data rows, not header rows)
all_trs = soup.find_all('tr')
print(f"Total <tr> elements: {len(all_trs)}")

# Extract all data rows
data_rows = []
for tr in all_trs:
    tds = tr.find_all('td')
    if tds and len(tds) > 3:  # Has data cells
        row_data = [td.get_text(strip=True) for td in tds]
        # Skip header rows
        if row_data[0] != 'Rk' and row_data[1] != 'Team':
            data_rows.append(row_data)

print(f"\nData rows found: {len(data_rows)}")
print(f"\nFirst 5 rows:")
for row in data_rows[:5]:
    print(f"  {row[:5]}")
print(f"\nLast 5 rows:")
for row in data_rows[-5:]:
    print(f"  {row[:5]}")