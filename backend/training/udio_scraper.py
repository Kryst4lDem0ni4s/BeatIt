'''import requests
from lxml import html
import re

url = "https://www.udio.com/songs/cnuS9UPwsmiz84SKaGrqcX"
response = requests.get(url)
tree = html.fromstring(response.content)

# Extract name (likely in <h1> or title area)
name_elements = tree.xpath('//h1/text()')
name = name_elements[0].strip() if name_elements else "Title not found"

# Extract prompt (description area)
prompt = tree.xpath('//div[contains(@class, "description") or contains(@class, "prompt")]/text()')[0].strip()

# Extract lyrics (from lyrics section)
lyrics = tree.xpath('//div[contains(@class, "lyrics")]//text()')[0].strip()

# Extract tags (from tags section)
tags = [tag.strip() for tag in tree.xpath('//div[contains(@class, "tags")]//span/text()')]

# Extract MP4 URL (check <video> or <audio> tags)
mp4_url = None
video_source = tree.xpath('//video/source/@src | //audio/source/@src')
if video_source and (video_source[0].endswith('.mp4') or video_source[0].endswith('.mp3')):
    mp4_url = video_source[0]
else:
    # Fallback: Search for .mp4 or .mp3 links in the HTML
    mp4_match = re.search(r'(https?://[^\s]+?\.(mp4|mp3))', response.text)
    mp4_url = mp4_match.group(0) if mp4_match else None

# Print results
print(f"Name: {name}")
print(f"Prompt: {prompt}")
print(f"Lyrics: {lyrics}")
print(f"Tags: {tags}")
print(f"MP4 URL: {mp4_url}")'''


'''
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

# Set up Selenium with Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in background (remove for visible browser)
driver = webdriver.Chrome(options=chrome_options)

# Open the webpage
url = "https://www.udio.com/songs/cnuS9UPwsmiz84SKaGrqcX"
driver.get(url)
time.sleep(3)  # Initial wait for page load

# Extract name (working)
try:
    name = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//h1[contains(@class, "truncate-2-lines")]'))
    ).text
except:
    name = "Title not found"

# Extract prompt (refined to target description)
try:
    prompt = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//span[contains(@class, "text-sm") and contains(text(), "Prompt:")]'))
    ).text.strip().replace("Prompt: ", "")
except:
    prompt = "Prompt not found"

# Extract tags (updated with class="text-[13px]")
try:
    tags_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@class, "w-full")]//a[contains(@class, "text-nowrap")]//span[contains(@class, "text-[13px]")]'))
    )
    tags = [tag.text.strip() for tag in tags_elements if tag.text.strip()]
except:
    tags = []

# Click the "Lyrics" button to open the drawer
try:
    lyrics = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//pre[contains(@class, "whitespace-pre-wrap")]'))
    ).text.strip()
except:
    lyrics = "Lyrics not found"


# Extract MP4 URL (working)
mp4_url = "MP4 URL not found"
page_source = driver.page_source
mp4_match = re.search(r'(https?://[^\s]+?\.(mp4|mp3))', page_source)
if mp4_match:
    mp4_url = mp4_match.group(0)

# Close the browser
driver.quit()

# Print results
print(f"Name: {name}")
print(f"Prompt: {prompt}")
print(f"Lyrics: {lyrics}")
print(f"Tags: {tags}")
print(f"MP4 URL: {mp4_url}")'''





'''import os
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # Remove for visible window
driver = webdriver.Chrome(options=chrome_options)

# Your target URL
url = "https://www.udio.com/songs/cnuS9UPwsmiz84SKaGrqcX"
driver.get(url)

# Title
try:
    name = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//h1[contains(@class, "truncate-2-lines")]'))
    ).text
except:
    name = "Title not found"

# Prompt
try:
    prompt = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//span[contains(@class, "hidden text-sm md:block")]'))
    ).text.strip()
except:
    prompt = "Prompt not found"

# Tags
try:
    tags_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, '//a[contains(@class, "text-nowrap")]//span'))
    )
    tags = [tag.text.strip() for tag in tags_elements if tag.text.strip()]
except:
    tags = []

# Lyrics
try:
    lyrics = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//pre[contains(@class, "whitespace-pre-wrap")]'))
    ).text.strip()
except:
    lyrics = "Lyrics not found"

# MP4 URL extraction from page source
mp4_url = "MP4 URL not found"
page_source = driver.page_source
mp4_match = re.search(r'(https?://[^\s]+?\.(mp4|mp3))', page_source)
if mp4_match:
    mp4_url = mp4_match.group(0)

driver.quit()  # Close browser

# Create download folder if not exists
download_dir = r"D:\Data\mp4_downloads"
os.makedirs(download_dir, exist_ok=True)

# Download the MP4
if "http" in mp4_url:
    try:
        mp4_response = requests.get(mp4_url)
        filename = f"{name}.mp4".replace(" ", "_").replace("/", "_")
        filepath = os.path.join(download_dir, filename)
        with open(filepath, "wb") as f:
            f.write(mp4_response.content)
        print(f"✅ MP4 downloaded to: {filepath}")
    except Exception as e:
        print(f"❌ Error downloading MP4: {e}")
else:
    print("❌ No valid MP4 URL found.")

# Print info
print(f"\n🎵 Name: {name}")
print(f"📝 Prompt: {prompt}")
print(f"🎶 Lyrics: {lyrics[:100]}...")  # Truncated for preview
print(f"🏷️ Tags: {tags}")
print(f"🔗 MP4 URL: {mp4_url}")



'''



import os
import csv
import requests
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
DOWNLOAD_DIR = r"D:\Data\mp4_downloads"
CSV_FILE = "udio_songs.csv"
MAX_SONGS = 10  # ⬅️ Only collect 10 songs for testing

# Setup headless Chrome
options = Options()
options.add_argument("--headless")  # Comment this out to see the browser
driver = webdriver.Chrome(options=options)

# Open trending page
driver.get("https://www.udio.com/home")
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//a[contains(@href, "/songs/")]'))
)


# Scroll and collect song links
print("🔍 Scrolling to collect song links...")
song_links = set()
last_height = driver.execute_script("return document.body.scrollHeight")

while len(song_links) < MAX_SONGS:
    links = driver.find_elements(By.XPATH, '//a[contains(@href, "/songs/")]')
    for link in links:
        href = link.get_attribute("href")
        if href and "/songs/" in href:
            song_links.add(href)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

print(f"✅ Collected {len(song_links)} song links.\n")

# Create CSV and output folder
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Name", "Prompt", "Lyrics", "Tags", "MP4_URL"])

    for index, song_url in enumerate(list(song_links)[:MAX_SONGS]):
        print(f"🎵 ({index+1}/{MAX_SONGS}) Scraping: {song_url}")
        try:
            driver.get(song_url)
            time.sleep(2)

            # Song name
            try:
                name = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//h1[contains(@class, "truncate-2-lines")]'))
                ).text
            except:
                name = "Title not found"

            # Prompt
            try:
                prompt = driver.find_element(By.XPATH, '//span[contains(@class, "hidden text-sm md:block")]').text.strip()
            except:
                prompt = "Prompt not found"

            # Tags
            try:
                tags_elements = driver.find_elements(By.XPATH, '//a[contains(@class, "text-nowrap")]//span')
                tags = [t.text.strip() for t in tags_elements if t.text.strip()]
            except:
                tags = []

            # Lyrics
            try:
                lyrics_elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, '//pre[contains(@class, "whitespace-pre-wrap")]'))
                )
                lyrics = lyrics_elem.text.strip()
            except:
                lyrics = "Lyrics not found"

            # MP4 Link
            mp4_url = "Not found"
            page_source = driver.page_source
            mp4_match = re.search(r'(https?://[^\s]+?\.(mp4|mp3))', page_source)
            if mp4_match:
                mp4_url = mp4_match.group(0)

            # Download MP4
            if "http" in mp4_url:
                try:
                    mp4_data = requests.get(mp4_url)
                    safe_name = re.sub(r'[\\/*?:"<>|]', "_", name)
                    filename = f"{safe_name}.mp4"
                    filepath = os.path.join(DOWNLOAD_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(mp4_data.content)
                    print(f"✅ Downloaded: {filepath}")
                except Exception as e:
                    print(f"❌ Error downloading MP4: {e}")
            else:
                print("❌ MP4 URL not found.")

            # Save to CSV
            writer.writerow([name, prompt, lyrics, ", ".join(tags), mp4_url])

        except Exception as e:
            print(f"⚠️ Error scraping {song_url}: {e}")

print("\n✅ Done! 10 songs scraped and saved.")
driver.quit()

