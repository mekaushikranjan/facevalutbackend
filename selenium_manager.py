from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

class SeleniumManager:
    def __init__(self):
        self.setup_dirs()
    
    def setup_dirs(self):
        # Create directories if they don't exist
        os.makedirs("downloads", exist_ok=True)
    
    def get_browser(self):
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Set download preferences
        prefs = {
            "download.default_directory": os.path.abspath("downloads"),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Initialize the Chrome driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        return driver
    
    def extract_images(self, driver, min_width=100, min_height=100):
        # Wait for page to load
        time.sleep(2)
        
        # Find all image elements
        image_elements = driver.find_elements(By.TAG_NAME, "img")
        
        # Extract image URLs
        image_urls = []
        for img in image_elements:
            try:
                # Get image dimensions
                width = img.get_attribute("width")
                height = img.get_attribute("height")
                
                # Convert to integers if possible
                try:
                    width = int(width) if width else 0
                    height = int(height) if height else 0
                except ValueError:
                    width = 0
                    height = 0
                
                # Skip small images (likely icons, etc.)
                if width < min_width or height < min_height:
                    continue
                
                # Get image URL
                src = img.get_attribute("src")
                if src and src.startswith(("http://", "https://")):
                    image_urls.append(src)
            except Exception as e:
                print(f"Error processing image element: {str(e)}")
        
        return image_urls
    
    def close_browser(self, driver):
        driver.quit()
