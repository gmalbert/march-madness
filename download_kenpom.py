from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
from bs4 import BeautifulSoup

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Initialize the driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Execute script to hide webdriver
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

import os

# Check if cached page exists
cache_file = 'data_files/kenpom_page.html'
if os.path.exists(cache_file):
    print("Loading from cache...")
    with open(cache_file, 'r', encoding='utf-8') as f:
        page_source = f.read()
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Set custom headers (NetRtg has no rank; subsequent metrics have value then rank)
    headers = ['Rk', 'Team', 'Conf', 'W-L', 'NetRtg', 'ORtg', 'ORtg_Rank', 'DRtg', 'DRtg_Rank',
               'AdjT', 'AdjT_Rank', 'Luck', 'Luck_Rank', 'SOS_NetRtg', 'SOS_NetRtg_Rank',
               'SOS_ORtg', 'SOS_ORtg_Rank', 'SOS_DRtg', 'SOS_DRtg_Rank', 'NCSOS_NetRtg', 'NCSOS_NetRtg_Rank']
    
    # Find ALL tr elements in the entire page (not just tbody)
    all_trs = soup.find_all('tr')
    
    # Extract all data rows
    data = []
    for tr in all_trs:
        tds = tr.find_all('td')
        if tds and len(tds) > 3:  # Has data cells
            row_data = [td.get_text(strip=True) for td in tds]  # Take all columns
            # Skip header rows (where first cell is "Rk" and second is "Team")
            if row_data[0] != 'Rk' and row_data[1] != 'Team':
                data.append(row_data)
    
    print(f"Found {len(data)} teams in cached page")
else:
    # Fetch with Selenium
    driver.get("https://kenpom.com/")
    
    # Wait for page to fully load (all data is in the DOM even if hidden)
    time.sleep(5)
    print(f"Page title: {driver.title}")
    
    print("Finished loading all content")
    page_source = driver.page_source
    # Save to cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(page_source)
    print("Page cached.")
    driver.quit()
    
    # Parse the saved page with BeautifulSoup to get ALL rows
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Set custom headers (NetRtg has no rank; subsequent metrics have value then rank)
    headers = ['Rk', 'Team', 'Conf', 'W-L', 'NetRtg', 'ORtg', 'ORtg_Rank', 'DRtg', 'DRtg_Rank',
               'AdjT', 'AdjT_Rank', 'Luck', 'Luck_Rank', 'SOS_NetRtg', 'SOS_NetRtg_Rank',
               'SOS_ORtg', 'SOS_ORtg_Rank', 'SOS_DRtg', 'SOS_DRtg_Rank', 'NCSOS_NetRtg', 'NCSOS_NetRtg_Rank']
    
    # Find ALL tr elements in the entire page (not just tbody)
    all_trs = soup.find_all('tr')
    
    # Extract all data rows
    data = []
    for tr in all_trs:
        tds = tr.find_all('td')
        if tds and len(tds) > 3:  # Has data cells
            row_data = [td.get_text(strip=True) for td in tds]  # Take all columns
            # Skip header rows (where first cell is "Rk" and second is "Team")
            if row_data[0] != 'Rk' and row_data[1] != 'Team':
                data.append(row_data)
    
    print(f"Found {len(data)} teams in page")

# Create DataFrame
if data:
    # Ensure all rows have the same number of columns (pad with empty strings if needed)
    max_cols = max(len(row) for row in data)
    print(f"Data has {max_cols} columns")
    
    # Create headers based on actual column count
    if max_cols == 21:
        headers = ['Rk', 'Team', 'Conf', 'W-L', 'NetRtg', 'ORtg', 'ORtg_Rank', 'DRtg', 'DRtg_Rank',
                   'AdjT', 'AdjT_Rank', 'Luck', 'Luck_Rank', 'SOS_NetRtg', 'SOS_NetRtg_Rank',
                   'SOS_ORtg', 'SOS_ORtg_Rank', 'SOS_DRtg', 'SOS_DRtg_Rank', 'NCSOS_NetRtg', 'NCSOS_NetRtg_Rank']
    else:
        # Generate generic headers if column count doesn't match
        headers = [f'Col_{i}' for i in range(max_cols)]
    
    df = pd.DataFrame(data, columns=headers)
else:
    df = pd.DataFrame()

# Save to CSV
df.to_csv('data_files/kenpom_ratings.csv', index=False)

print(f"Extracted {len(df)} teams from KenPom and saved to data_files/kenpom_ratings.csv")