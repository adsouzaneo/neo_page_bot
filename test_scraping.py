"""
Quick test script to scrape URLs and save as text files
Run this to verify scraping works before running the full pipeline
"""

from scrapper import WebScraper

def main():
    # Read URLs from pagesurl.txt
    urls = []
    with open('pagesurl.txt', 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print("=" * 80)
    print(f"SCRAPING {len(urls)} PAGES")
    print("=" * 80)
    print()
    
    # Create scraper and scrape pages
    scraper = WebScraper(urls, save_dir="scraped_pages")
    documents = scraper.scrape_all(save_files=True)
    
    # Summary
    print()
    print("=" * 80)
    print("SCRAPING COMPLETE!")
    print("=" * 80)
    print(f"\nSaved {len(documents)} files to 'scraped_pages/' directory:")
    for idx in range(len(documents)):
        print(f"  \u2713 scraped_pages/page_{idx + 1}.txt")
    
    print(f"\nTotal content: {sum(len(doc['content']) for doc in documents):,} characters")
    print("\nYou can now run 'python main.py' to store in Milvus")

if __name__ == "__main__":
    main()
