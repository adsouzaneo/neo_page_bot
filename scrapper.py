import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import os
from pathlib import Path


class WebScraper:
    """Scrapes content from specified web pages"""
    
    def __init__(self, urls: List[str], save_dir: str = "scraped_pages"):
        self.urls = urls
        self.scraped_data = []
        self.save_dir = save_dir
        # Create directory if it doesn't exist
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    
    def scrape_page(self, url: str) -> Dict[str, str]:
        """Scrape a single web page and extract text content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Get title
            title = soup.title.string if soup.title else url
            
            return {
                'url': url,
                'title': title,
                'content': text
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': url,
                'content': f"Failed to scrape: {str(e)}"
            }
    
    def save_to_file(self, data: Dict[str, str], index: int):
        """Save scraped content to a text file"""
        filename = f"page_{index + 1}.txt"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"URL: {data['url']}\n")
            f.write(f"Title: {data['title']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(data['content'])
        
        print(f"  → Saved to {filepath}")
        return filepath
    
    def scrape_all(self, save_files: bool = True) -> List[Dict[str, str]]:
        """Scrape all specified URLs and optionally save to files"""
        self.scraped_data = []
        for idx, url in enumerate(self.urls):
            print(f"Scraping {idx + 1}/{len(self.urls)}: {url}")
            data = self.scrape_page(url)
            self.scraped_data.append(data)
            
            # Save to file
            if save_files:
                self.save_to_file(data, idx)
            
            time.sleep(1)  # Be polite to servers
        
        return self.scraped_data


if __name__ == "__main__":
    # Read URLs from pagesurl.txt
    urls = []
    with open('pagesurl.txt', 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(urls)} URLs to scrape\n")
    
    scraper = WebScraper(urls, save_dir="scraped_pages")
    data = scraper.scrape_all(save_files=True)
    
    print("\n" + "=" * 80)
    print("SCRAPING SUMMARY")
    print("=" * 80)
    for idx, item in enumerate(data, 1):
        print(f"\n{idx}. {item['title']}")
        print(f"   URL: {item['url']}")
        print(f"   Content: {len(item['content'])} characters")
