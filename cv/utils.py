import requests
from bs4 import BeautifulSoup

def scrape_wordlist(url):
    # url = "https://www.enchantedlearning.com/wordlist/food.shtml"
    response = requests.get(url)

    # using beautifulsoup, we get all text with div class wordlist-item
    soup = BeautifulSoup(response.text, "html.parser")
    words = soup.find_all("div", class_="wordlist-item")
    words = [word.text for word in words]

    return words