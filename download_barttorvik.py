from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# Set up Chrome options for headless and download
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": os.path.join(os.getcwd(), "data_files"),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True
})

# Initialize the driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Navigate to the URL
driver.get("https://barttorvik.com/team-tables_each.php?csv=1")

# Wait for download to complete (adjust time if needed)
time.sleep(5)

# Close the driver
driver.quit()

print("Download attempted. Check data_files/ for the CSV file.")