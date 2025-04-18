import os
import csv
import requests
import re
import time
import random
from urllib.parse import urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from datetime import datetime

# Configuration
DOWNLOAD_DIR = r"C:\Users\Khwaish\.vscode\BeatIt\backend\training\training_data\musicgen"
CSV_FILE = f"udio_songs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
MAX_SONGS = 100  # Stop after downloading 100 songs
BASE_URL = "https://www.udio.com"
SEED_URLS = [
    "https://www.udio.com/home",
    "https://www.udio.com/tags/music",
    "https://www.udio.com/tags/pop",
    "https://www.udio.com/tags/rock",
    "https://www.udio.com/tags/rap",
    "https://www.udio.com/tags/electronic",
    "https://www.udio.com/trending"
]
MAX_PAGES_PER_SECTION = 3  # Maximum pages to crawl in each section
SCROLL_PAUSE_TIME = 2  # Time to pause between scrolls

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def setup_driver():
    """Set up and return a configured Chrome WebDriver"""
    options = Options()
    options.add_argument("--headless")  # Remove this line if you want to see the browser
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    return driver

def is_valid_url(url):
    """Check if URL is valid and belongs to the target website"""
    if not url or not isinstance(url, str):
        return False
    
    parsed = urlparse(url)
    return bool(parsed.netloc) and "udio.com" in parsed.netloc and parsed.scheme in ["http", "https"]

def scroll_page(driver, max_scrolls=5):
    """Scroll down the page to load more content"""
    scrolls = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            break
        
        last_height = new_height
        scrolls += 1

def get_all_links(driver):
    """Extract all links from the current page"""
    links = set()
    try:
        elements = driver.find_elements(By.TAG_NAME, 'a')
        for element in elements:
            try:
                href = element.get_attribute("href")
                if href and is_valid_url(href):
                    links.add(href)
            except StaleElementReferenceException:
                continue
    except Exception as e:
        print(f"Error extracting links: {e}")
    
    return links

def filter_song_links(links):
    """Filter links to keep only song URLs"""
    return {link for link in links if "/songs/" in link}

def filter_navigation_links(links):
    """Filter links to keep only navigation URLs (not song URLs)"""
    nav_links = set()
    for link in links:
        # Skip song links
        if "/songs/" in link:
            continue
        
        # Keep only links to sections we're interested in
        if any(section in link for section in ["/home", "/tags/", "/trending", "/discover", "/artists/"]):
            nav_links.add(link)
    
    return nav_links

def download_song(mp4_url, filepath):
    """Download a song from the given URL to the specified filepath"""
    try:
        mp4_data = requests.get(mp4_url, timeout=30)
        with open(filepath, "wb") as f:
            f.write(mp4_data.content)
        return True
    except Exception as e:
        print(f"Error downloading MP4: {e}")
        return False

def extract_song_details(driver, song_url):
    """Extract song details from a song page"""
    details = {
        "name": "Title not found",
        "prompt": "Prompt not found",
        "lyrics": "Lyrics not found",
        "tags": [],
        "mp4_url": None
    }
    
    try:
        # Get song name
        try:
            name_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//h1[contains(@class, "truncate-2-lines")]'))
            )
            details["name"] = name_element.text
        except (TimeoutException, NoSuchElementException):
            pass

        # Get prompt
        try:
            prompt_element = driver.find_element(By.XPATH, '//span[contains(@class, "hidden text-sm md:block")]')
            details["prompt"] = prompt_element.text.strip()
        except NoSuchElementException:
            pass

        # Get tags
        try:
            tags_elements = driver.find_elements(By.XPATH, '//a[contains(@class, "text-nowrap")]//span')
            details["tags"] = [t.text.strip() for t in tags_elements if t.text.strip()]
        except NoSuchElementException:
            pass

        # Get lyrics
        try:
            lyrics_elem = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//pre[contains(@class, "whitespace-pre-wrap")]'))
            )
            details["lyrics"] = lyrics_elem.text.strip()
        except (TimeoutException, NoSuchElementException):
            pass

        # Extract MP4 URL from page source using regex
        page_source = driver.page_source
        mp4_match = re.search(r'(https?://[^\s]+?\.(mp4|mp3))', page_source)
        if mp4_match:
            details["mp4_url"] = mp4_match.group(0)
    
    except Exception as e:
        print(f"Error extracting song details: {e}")
    
    return details

def main():
    driver = setup_driver()
    
    # Sets to keep track of visited URLs and found song links
    visited_urls = set()
    song_links = set()
    urls_to_visit = SEED_URLS.copy()
    
    # Create CSV file to record the song details
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Prompt", "Lyrics", "Tags", "Local_File_Path"])
        
        downloaded_count = 0
        pages_visited = 0
        
        # Continue until we've downloaded enough songs or run out of URLs to visit
        while urls_to_visit and downloaded_count < MAX_SONGS and pages_visited < len(SEED_URLS) * MAX_PAGES_PER_SECTION:
            # Get the next URL to visit
            current_url = urls_to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if current_url in visited_urls:
                continue
            
            print(f"\nVisiting page: {current_url}")
            visited_urls.add(current_url)
            pages_visited += 1
            
            try:
                # Navigate to the URL
                driver.get(current_url)
                time.sleep(2)  # Wait for page to load
                
                # Scroll to load more content
                scroll_page(driver)
                
                # Get all links from the page
                all_links = get_all_links(driver)
                
                # Extract song links
                new_song_links = filter_song_links(all_links)
                song_links.update(new_song_links)
                print(f"Found {len(new_song_links)} new song links on this page. Total: {len(song_links)}")
                
                # Extract navigation links for further crawling
                nav_links = filter_navigation_links(all_links)
                
                # Add new navigation links to our queue
                for link in nav_links:
                    if link not in visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)
                
                # Process song links if we're on a page with songs
                if "/songs/" in current_url:
                    # This is a song page, process it directly
                    details = extract_song_details(driver, current_url)
                    
                    if details["mp4_url"]:
                        # Prepare a safe filename for the song
                        safe_name = re.sub(r'[\\/*?:"<>|]', "_", details["name"]) + ".mp4"
                        filepath = os.path.join(DOWNLOAD_DIR, safe_name)
                        
                        # Check if the file has already been downloaded
                        if os.path.exists(filepath):
                            print(f"File already exists: {filepath}. Skipping download.")
                        else:
                            # Download the MP4 file
                            if download_song(details["mp4_url"], filepath):
                                print(f"Downloaded: {filepath}")
                                downloaded_count += 1
                            else:
                                print("Failed to download MP4.")
                                continue
                        
                        # Write the collected details to the CSV file
                        writer.writerow([
                            details["name"],
                            details["prompt"],
                            details["lyrics"],
                            ", ".join(details["tags"]),
                            filepath
                        ])
                        csvfile.flush()  # Ensure data is written to file
            
            except Exception as e:
                print(f"Error processing page {current_url}: {e}")
        
        # Process individual song links if we haven't reached MAX_SONGS yet
        song_links_list = list(song_links)
        random.shuffle(song_links_list)  # Randomize to get a diverse set
        
        for song_url in song_links_list:
            # Stop if we've downloaded enough songs
            if downloaded_count >= MAX_SONGS:
                break
            
            # Skip if we've already visited this URL
            if song_url in visited_urls:
                continue
            
            print(f"\nProcessing song: {song_url}")
            visited_urls.add(song_url)
            
            try:
                # Navigate to the song page
                driver.get(song_url)
                time.sleep(2)  # Wait for page to load
                
                # Extract song details
                details = extract_song_details(driver, song_url)
                
                if details["mp4_url"]:
                    # Prepare a safe filename for the song
                    safe_name = re.sub(r'[\\/*?:"<>|]', "_", details["name"]) + ".mp4"
                    filepath = os.path.join(DOWNLOAD_DIR, safe_name)
                    
                    # Check if the file has already been downloaded
                    if os.path.exists(filepath):
                        print(f"File already exists: {filepath}. Skipping download.")
                    else:
                        # Download the MP4 file
                        if download_song(details["mp4_url"], filepath):
                            print(f"Downloaded: {filepath}")
                            downloaded_count += 1
                        else:
                            print("Failed to download MP4.")
                            continue
                    
                    # Write the collected details to the CSV file
                    writer.writerow([
                        details["name"],
                        details["prompt"],
                        details["lyrics"],
                        ", ".join(details["tags"]),
                        filepath
                    ])
                    csvfile.flush()  # Ensure data is written to file
            
            except Exception as e:
                print(f"Error processing song {song_url}: {e}")
    
    print(f"\nDone! {downloaded_count} songs scraped and saved.")
    print(f"Visited {len(visited_urls)} unique URLs.")
    driver.quit()

if __name__ == "__main__":
    main()
