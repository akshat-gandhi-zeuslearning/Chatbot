import scrapy
from scrapy.crawler import CrawlerProcess
import os
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class BatchSpider(scrapy.Spider):
    name = "batch_spider"
    allowed_domains = ['staging.annotate.net']
    start_urls = ['https://staging.annotate.net/'] 
    visited_urls = set()
    batch_size = 10  # Number of pages to process in one batch
    batch_count = 0
    pages_in_current_batch = 0
    max_batches = 1 # Maximum number of batches to process

    # Filenames for storing results
    urls_anchors_filename = "output/urls_and_anchors.txt"
    text_filename = "output/text_content.txt"

    def start_requests(self):
        # Ensure the output directory exists
        os.makedirs('output', exist_ok=True)
        
        # Clear existing files if they exist
        with open(self.urls_anchors_filename, 'w', encoding='utf-8') as f:
            f.write('')
        with open(self.text_filename, 'w', encoding='utf-8') as f:
            f.write('')

        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        if response.url in self.visited_urls:
            return
        self.visited_urls.add(response.url)
        
        # Save the page content
        page_url = response.url
        self.save_anchors_and_text(page_url, response)

        # Increment page count for batch processing
        self.pages_in_current_batch += 1

        # If batch size is reached, wait before continuing
        if self.pages_in_current_batch >= self.batch_size:
            self.pages_in_current_batch = 0
            self.batch_count += 1
            print(f"Batch {self.batch_count} processed. Waiting for next batch...")

            # Stop after processing max_batches batches
            if self.batch_count >= self.max_batches:
                print(f"Reached maximum batch limit of {self.max_batches}. Stopping the crawler.")
                self.crawler.engine.close_spider(self, 'max_batches_reached')
                return
            
            time.sleep(5)  # Wait for 5 seconds before continuing
        
        # Extract all links and follow them
        for link in response.css('a::attr(href)').getall():
            if link.startswith('/'):
                link = response.urljoin(link)
            if self.is_valid_url(link):
                yield scrapy.Request(link, callback=self.parse)

    def save_anchors_and_text(self, url, response):
        soup = BeautifulSoup(response.body, 'html.parser')
        anchors = soup.find_all('a')
        text_content = soup.get_text(separator="\n", strip=True)

        # Save anchors
        with open(self.urls_anchors_filename, 'a', encoding='utf-8') as f:
            f.write(f"Page URL: {url}\n")
            
            f.write("\n")

        # Save text content
        with open(self.text_filename, 'a', encoding='utf-8') as f:
            f.write(f"Page URL: {url}\n")
            f.write(text_content)
            f.write("\n\n")

    def is_valid_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.scheme in ('http', 'https')

if __name__ == "__main__":
    # Create a folder to store the files if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Start the crawling process
    process = CrawlerProcess()
    process.crawl(BatchSpider)
    process.start()

    print("Crawling and saving data completed.")
