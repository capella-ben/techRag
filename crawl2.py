import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import csv
import argparse
from collections import deque

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_website_links(url):
    urls = set()
    domain_name = urlparse(url).netloc
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"})
        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not is_valid_url(href):
                continue
            if domain_name not in href:
                continue
            urls.add(href)
        return urls, soup.title.string if soup.title else "No Title"
    except requests.RequestException:
        return set(), "Error fetching page"

def crawl(start_url, max_depth=1):
    queue = deque([(start_url, 0)])
    seen_urls = set([start_url])

    while queue:
        url, depth = queue.popleft()
        if depth > max_depth:
            continue

        links, title = get_all_website_links(url)
        yield url, title
        print(f"Crawling: {url}\tTitle: {title}")

        for link in links:
            if link not in seen_urls:
                seen_urls.add(link)
                queue.append((link, depth + 1))

def sort_csv_file(filename):
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header
        sorted_rows = sorted(reader, key=lambda row: row[0])  # Sort by URL

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        writer.writerows(sorted_rows)  # Write the sorted rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web crawler script")
    parser.add_argument("url", help="URL to start crawling from")
    parser.add_argument("--max_depth", type=int, default=1, help="Maximum depth to crawl (default: 1)")
    parser.add_argument("--output", default="crawl_results.csv", help="Output file name (default: crawl_results.csv)")
    args = parser.parse_args()

    # Crawl and write results to file
    with open(args.output, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Title"])

        for url, title in crawl(args.url, max_depth=args.max_depth):
            writer.writerow([url, title])

    print("Crawl completed. Sorting results...")

    # Sort the CSV file
    sort_csv_file(args.output)

    print(f"Sorted results saved to {args.output}")

    # Print sorted URLs
    with open(args.output, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            print(row[0])  # Print URL
            