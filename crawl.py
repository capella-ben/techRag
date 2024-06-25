import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import re
import csv


def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_website_links(url):
    urls = set()
    domain_name = urlparse(url).netloc
    session = requests.Session()
    session.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = session.get(url)
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
        if href in urls:
            continue
        if domain_name not in href:
            continue
        urls.add(href)
    return urls, soup.title.string if soup.title else "No Title"

def crawl(url, max_depth=1):
    seen_urls = set()
    crawled_data = []
    def _crawl(urls, depth):
        if depth > max_depth:
            return
        new_urls = set()
        for url in urls:
            if url not in seen_urls:
                seen_urls.add(url)
                links, title = get_all_website_links(url)
                crawled_data.append((url, title))
                print(f"Crawling: {url}\tTitle: {title}") 
                new_urls.update(links)
        _crawl(new_urls, depth + 1)
    _crawl([url], 0)
    return crawled_data


if __name__ == "__main__":
    result = crawl("https://neo4j.com/docs/operations-manual/current/", max_depth=1)

    print()
    print('----------------------------------------------------------------------')
    result.sort(key=lambda x: x[0])
    for url, title in result:
        print(url)
    print()
    print()


    with open("crawl_results.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Title"])
        for url, title in result:
            writer.writerow([url, title])
    
    print("Results saved to crawl_results.csv")
    
