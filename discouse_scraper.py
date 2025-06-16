from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os
import time
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin

load_dotenv()

EMAIL = os.getenv("DISCOURSE_EMAIL")
PASSWORD = os.getenv("DISCOURSE_PASSWORD")

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
SEARCH_URL = (
        f"{BASE_URL}/search?q=after%3A2025-01-01%20before%3A2025-04-14%20%23courses%3A"
        f"tds-kb%20order%3Alatest_topic"
    )


def login_and_save_state():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"{BASE_URL}/login")

        page.fill('input[type="email"]', EMAIL)
        page.fill('input[type="password"]', PASSWORD)
        page.click('button#login-button')
        page.wait_for_url(BASE_URL + "/", timeout=20000)

        context.storage_state(path="discourse_state.json")
        browser.close()
        print("✅ Login complete and state saved.")


def extract_images(cooked_div):
    image_tags = cooked_div.select("a.lightbox")
    image_urls = [urljoin(BASE_URL, img["href"]) for img in image_tags if img.get("href")]
    return image_urls


def extract_all_hrefs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="discourse_state.json")
        page = context.new_page()
        
        page.goto("https://discourse.onlinedegree.iitm.ac.in/search?q=after%3A2025-01-01%20before%3A2025-04-14%20%23courses%3Atds-kb%20order%3Alatest_topic")
        page.wait_for_selector("div.search-results", timeout=10000)
        
        a_tags = page.locator("div.search-results a")
        count = a_tags.count()
        hrefs = []

        for i in range(count):
            href = a_tags.nth(i).get_attribute("href")
            if href:
                hrefs.append(urljoin(BASE_URL, href))

        print(f"🔗 Total <a> hrefs found in .search-results: {len(hrefs)}")
        print("🧪 Sample links:", hrefs[:5])

        browser.close()

def scroll_to_load_all(page, wait_time=1.5, max_scrolls=50):
    previous_height = 0
    for i in range(max_scrolls):
        page.mouse.wheel(0, 10000)
        time.sleep(wait_time)
        current_height = page.evaluate("() => document.body.scrollHeight")
        if current_height == previous_height:
            print(f"✅ Finished scrolling after {i+1} scrolls")
            break
        previous_height = current_height

def scrape_tds_qa():


    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="discourse_state.json")
        page = context.new_page()
        page.goto(SEARCH_URL)
        time.sleep(3)

        scroll_to_load_all(page)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        results = soup.select("div.search-results div.fps-result-entries div.fps-topic div.topic a.search-link")

        post_links = [urljoin(BASE_URL, a["href"]) for a in results]
        print(f"🔗 Found {len(post_links)} links")

        qa_data = []

        for post_url in post_links:
            page.goto(post_url)
            time.sleep(2)
            soup = BeautifulSoup(page.content(), "html.parser")

            cooked_divs = soup.select("div.topic-body div.cooked")
            if not cooked_divs:
                continue

            question_text = cooked_divs[0].get_text("\n", strip=True)
            question_images = extract_images(cooked_divs[0])

            replies = [
                {
                    "text": cooked.get_text("\n", strip=True),
                    "images": extract_images(cooked),
                    "url": post_url
                }
                for cooked in cooked_divs[1:]
            ]

            if replies:
                qa_data.append({
                    "question": question_text,
                    "images": question_images,
                    "url": post_url,
                    "answers": replies
                })

        with open("tds_discourse_scraped.json", "w") as f:
            json.dump(qa_data, f, indent=2)

        print(f"✅ Scraped {len(qa_data)} valid Q&A threads.")
        browser.close()


if __name__ == "__main__":
    login_and_save_state()
    scrape_tds_qa()
    #xtract_all_hrefs()
