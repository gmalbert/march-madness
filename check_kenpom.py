from bs4 import BeautifulSoup

with open('data_files/kenpom_page.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

tables = soup.find_all('table')
print(f'Found {len(tables)} tables')

if tables:
    for i, table in enumerate(tables):
        tbody = table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            print(f'\nTable {i}: {len(rows)} rows')
            
            # Show first few and last few rows
            for j, row in enumerate(rows[:5]):
                cells = row.find_all('td')
                row_text = [cell.get_text(strip=True) for cell in cells]
                print(f"  Row {j}: {row_text[:4]}")
            print("  ...")
            for j, row in enumerate(rows[-5:], start=len(rows)-5):
                cells = row.find_all('td')
                row_text = [cell.get_text(strip=True) for cell in cells]
                print(f"  Row {j}: {row_text[:4]}")
