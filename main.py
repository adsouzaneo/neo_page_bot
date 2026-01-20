from scrapper import WebScraper
from text_processor import TextChunker
from milvus_manager import MilvusManager
from dotenv import load_dotenv
import os


def main():
    """Main pipeline to scrape, process, and store documents in Milvus"""
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Read URLs from pagesurl.txt
    print("=" * 70)
    print("STEP 1: Web Scraping")
    print("=" * 70)
    
    # Read URLs from file
    urls = []
    with open('pagesurl.txt', 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"\nLoaded {len(urls)} URLs from pagesurl.txt:")
    for idx, url in enumerate(urls, 1):
        print(f"  {idx}. {url}")
    
    print(f"\nScraping {len(urls)} pages...\n")
    scraper = WebScraper(urls, save_dir="scraped_pages")
    documents = scraper.scrape_all(save_files=True)
    
    print(f"\n" + "=" * 70)
    print(f"Successfully scraped {len(documents)} documents")
    print("=" * 70)
    for idx, doc in enumerate(documents, 1):
        print(f"  {idx}. {doc['title']}")
        print(f"     Content: {len(doc['content'])} characters")
        print(f"     Saved: scraped_pages/page_{idx}.txt")
    
    # Step 2: Process and chunk the documents
    print("\n" + "=" * 70)
    print("STEP 2: Text Processing and Chunking")
    print("=" * 70)
    
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.process_documents(documents)
    
    print(f"\nCreated {len(chunks)} text chunks")
    
    # Step 3: Set up Milvus and store embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Milvus Setup and Data Insertion")
    print("=" * 70)
    
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    
    milvus = MilvusManager(
        collection_name="rag_documents",
        host=milvus_host,
        port=milvus_port
    )
    
    # Connect to Milvus
    milvus.connect()
    
    # Create collection
    milvus.create_collection()
    
    # Insert documents
    milvus.insert_documents(chunks)
    
    # Load collection
    milvus.load_collection()
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print(f"\nSuccessfully stored {len(chunks)} chunks in Milvus")
    print("\nYou can now run the chatbot with:")
    print("  python chatbot.py")
    
    # Test search
    print("\n" + "=" * 70)
    print("Testing search functionality...")
    print("=" * 70)
    
    test_query = "What is artificial intelligence?"
    print(f"\nTest query: '{test_query}'")
    results = milvus.search(test_query, top_k=3)
    
    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['title']}")
        print(f"   Text preview: {result['text'][:200]}...")
    
    # Disconnect
    milvus.disconnect()


if __name__ == "__main__":
    main()
