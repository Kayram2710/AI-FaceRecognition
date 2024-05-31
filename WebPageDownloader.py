# A big portion of the main function was generated using chat gpt when prompted to explain what could be 
# used as way to mass download images from a site
# Disclaimer: The links provided were all for fecthing royalty free images 

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os

def download_images_selenium(url, folder="downloaded_images", start_image_id=1, min_width=300, min_height=300):
    # Setup Chrome and WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    driver.get(url)
    time.sleep(15) 
    
    # Find all images
    images = driver.find_elements("tag name", 'img')
    
    # Create folder for images
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    image_count = start_image_id
    # Download each image
    for img in images:
        # Check image dimensions before downloading
        width = img.get_property('naturalWidth')
        height = img.get_property('naturalHeight')
        if width < min_width or height < min_height:
            continue  # Skip small images
        
        src = img.get_attribute('src')
        try:
            img_response = requests.get(src, stream=True)
            image_path = os.path.join(folder, f"image_{image_count}.jpg")
            with open(image_path, 'wb') as file:
                for chunk in img_response.iter_content(chunk_size=1024):
                    file.write(chunk)
            image_count += 1
        except requests.exceptions.MissingSchema:
            print(f"Skipping URL {src} because it's not a valid URL")

    driver.quit()
    return image_count

#Manually Setting fetch
image_id = 528
i = 12

#link = "https://www.freepik.com/search?ai=excluded&format=search&last_filter=people_range&last_value=1&people=include&people_range=1&query=focused+people&selection=1&type=photo"
link = f"https://www.freepik.com/search?ai=excluded&format=search&last_filter=page&last_value={i}&page={i}&people=include&people_range=1&query=focused+people&selection=1&type=photo"

download_images_selenium(link, start_image_id=image_id)
