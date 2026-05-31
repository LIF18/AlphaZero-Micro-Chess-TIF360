import os
import urllib.request
from urllib.parse import urljoin
from bs4 import BeautifulSoup

BASE_URL = "https://database.nikonoel.fr/"
SAVE_DIR = "lichess_elite_dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Connecting to target website and resolving archive list: {BASE_URL} ...")

try:
    # Add basic request headers to defend against anti-scraping strategies
    req = urllib.request.Request(BASE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        html = response.read()
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all hyperlinks on the page
    links = soup.find_all('a')
    target_exts = ('.zip', '.pgn', '.zst', '.bz2', '.gz')
    
    downloaded_count = 0
    for link in links:
        href = link.get('href')
        if href and any(href.endswith(ext) for ext in target_exts):
            file_url = urljoin(BASE_URL, href)
            filename = os.path.basename(file_url)
            save_path = os.path.join(SAVE_DIR, filename)
            
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f"File already exists, skipping download: {filename}")
                continue
                
            print(f"Downloading: {filename} ...")
            try:
                # Execute network stream write to local disk
                urllib.request.urlretrieve(file_url, save_path)
                print(f"File saved successfully to: {save_path}")
                downloaded_count += 1
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

    print(f"Automated download pipeline completed! New files downloaded: {downloaded_count}")

except Exception as e:
    print(f"Failed to access or parse website: {e}")