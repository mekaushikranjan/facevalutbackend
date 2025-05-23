import os
import requests
import uuid
from PIL import Image
from io import BytesIO
import time

class ImageProcessor:
    def __init__(self):
        self.setup_dirs()
    
    def setup_dirs(self):
        # Create directories if they don't exist
        os.makedirs("images", exist_ok=True)
    
    def download_image(self, url):
        try:
            # Generate a unique filename
            filename = f"{str(uuid.uuid4())}.jpg"
            filepath = os.path.join("images", filename)
            
            # Download the image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the image
            img = Image.open(BytesIO(response.content))
            img.save(filepath)
            
            return filepath
        
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
            raise
    
    def process_image(self, image_path):
        # This method can be expanded to include additional processing
        # such as resizing, format conversion, etc.
        try:
            img = Image.open(image_path)
            
            # Example: resize large images to save space
            max_size = 1920
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size))
                img.save(image_path)
            
            return image_path
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise
