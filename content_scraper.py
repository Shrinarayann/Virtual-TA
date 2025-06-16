import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
import re
import os
from datetime import datetime

class TDSCourseScraper:
    def __init__(self):
        self.base_url = "https://tds.s-anand.net/#/"
        self.scraped_data = []
        self.browser = None
        self.page = None
        
    async def setup_playwright(self):
        """Setup Playwright browser for JavaScript-heavy content"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        
    async def close_playwright(self):
        """Close Playwright browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
            
    async def explore_site_structure(self):
        """Explore the TDS site structure to understand navigation"""
        await self.page.goto(self.base_url)
        
        # Wait for the page to load
        await self.page.wait_for_load_state('networkidle')
        
        # Get page title and content
        title = await self.page.title()
        print(f"Page title: {title}")
        
        # Look for navigation elements, links, and content structure
        # Common patterns: menu items, sidebar navigation, content areas
        nav_selectors = [
            'nav', '.nav', '.navigation', '.menu', '.sidebar',
            'ul li a', '.toc', '.table-of-contents', 
            '[href*="#"]', 'a[href^="#"]'
        ]
        
        navigation_links = []
        for selector in nav_selectors:
            try:
                elements = await self.page.query_selector_all(selector)
                for element in elements:
                    href = await element.get_attribute('href')
                    text = await element.inner_text()
                    if href and text.strip():
                        navigation_links.append({
                            'href': href,
                            'text': text.strip(),
                            'selector': selector
                        })
            except Exception as e:
                continue
                
        return navigation_links
    
    async def scrape_page_content(self, url_fragment=""):
        """Scrape content from a specific page/section"""
        full_url = self.base_url + url_fragment if url_fragment else self.base_url
        
        await self.page.goto(full_url)
        await self.page.wait_for_load_state('networkidle')
        
        # Extract main content
        content = {
            'url': full_url,
            'fragment': url_fragment,
            'title': await self.page.title(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try different content selectors
        content_selectors = [
            'main', '.main', '.content', '.main-content',
            'article', '.article', 'section', '.section',
            '#content', '.container', '.wrapper'
        ]
        
        for selector in content_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    content['html'] = await element.inner_html()
                    content['text'] = await element.inner_text()
                    break
            except:
                continue
        
        # If no specific content area found, get body content
        if 'html' not in content:
            body = await self.page.query_selector('body')
            if body:
                content['html'] = await body.inner_html()
                content['text'] = await body.inner_text()
        
        return content
    
    async def scrape_all_content(self):
        """Main method to scrape all TDS course content"""
        await self.setup_playwright()
        
        try:
            print("Exploring site structure...")
            navigation_links = await self.explore_site_structure()
            
            print(f"Found {len(navigation_links)} navigation elements")
            
            # Scrape main page
            print("Scraping main page...")
            main_content = await self.scrape_page_content()
            self.scraped_data.append(main_content)
            
            # Extract unique hash fragments from navigation
            hash_fragments = set()
            for link in navigation_links:
                if link['href'].startswith('#/'):
                    hash_fragments.add(link['href'][2:])  # Remove #/ prefix
                elif link['href'].startswith('#'):
                    hash_fragments.add(link['href'][1:])   # Remove # prefix
            
            print(f"Found {len(hash_fragments)} unique sections to scrape")
            
            # Scrape each section
            for fragment in hash_fragments:
                if fragment:  # Skip empty fragments
                    print(f"Scraping section: {fragment}")
                    try:
                        section_content = await self.scrape_page_content(fragment)
                        self.scraped_data.append(section_content)
                        
                        # Be respectful - add small delay
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        print(f"Error scraping {fragment}: {str(e)}")
                        continue
            
            print(f"Successfully scraped {len(self.scraped_data)} pages/sections")
            
        finally:
            await self.close_playwright()
        
        return self.scraped_data
    
    def save_scraped_data(self, filename="tds_course_content.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
    
    def clean_and_structure_data(self):
        """Clean and structure the scraped data for better processing"""
        cleaned_data = []
        
        for item in self.scraped_data:
            if 'text' in item and item['text'].strip():
                # Clean text content
                text_content = re.sub(r'\s+', ' ', item['text']).strip()
                
                # Extract meaningful sections
                cleaned_item = {
                    'id': f"tds_{hash(item['url'] + item.get('fragment', ''))}",
                    'source': 'tds_course',
                    'url': item['url'],
                    'section': item.get('fragment', 'main'),
                    'title': item.get('title', ''),
                    'content': text_content,
                    'timestamp': item['timestamp'],
                    'type': 'course_content'
                }
                
                # Only add if content is substantial
                if len(text_content) > 100:  # Minimum content threshold
                    cleaned_data.append(cleaned_item)
        
        return cleaned_data

# Usage example
async def main():
    scraper = TDSCourseScraper()
    
    # Scrape all content
    scraped_data = await scraper.scrape_all_content()
    
    # Save raw data
    scraper.save_scraped_data("tds_raw_content.json")
    
    # Clean and structure data
    cleaned_data = scraper.clean_and_structure_data()
    
    # Save cleaned data
    with open("tds_cleaned_content.json", 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Scraped and cleaned {len(cleaned_data)} content sections")
    
    return cleaned_data

# Run the scraper
if __name__ == "__main__":
    asyncio.run(main())