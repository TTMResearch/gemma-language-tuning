import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
from loguru import logger
import time
from fake_useragent import UserAgent
from tqdm import tqdm

class AfrikaansNewsScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'af,en-US;q=0.7,en;q=0.3',
        }
        
    def scrape_wikipedia_category(self, limit: int = 100) -> List[Dict]:
        """Scrape Afrikaans Wikipedia articles from a specific category"""
        articles = []
        # Using multiple categories to ensure good content
        categories = [
            "Suid-Afrika",
            "Afrikaanse_literatuur",
            "Afrikaanse_kultuur",
            "Geskiedenis_van_Suid-Afrika",
            "Suid-Afrikaanse_politiek",
            "Suid-Afrikaanse_musiek",
            "Afrikaanse_taal"
        ]
        
        for category in categories:
            if len(articles) >= limit:
                break
                
            base_url = f"https://af.wikipedia.org/wiki/Kategorie:{category}"
            
            try:
                response = requests.get(base_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look specifically in the mw-category-group div
                category_groups = soup.find_all('div', class_='mw-category-group')
                
                for group in category_groups:
                    links = group.find_all('a')
                    for link in links:
                        href = link.get('href', '')
                        # Skip special pages and categories
                        if (href.startswith('/wiki/') and 
                            not any(x in href for x in ['Kategorie:', 'Spesiaal:', 'Wikipedia:', 'Hulp:', 'Portaal:'])):
                            try:
                                article_url = f"https://af.wikipedia.org{href}"
                                article_data = self._get_wikipedia_article(article_url)
                                
                                if article_data:
                                    articles.append(article_data)
                                    logger.debug(f"Collected article: {article_data['title']}")
                                    
                                if len(articles) >= limit:
                                    break
                                    
                                time.sleep(1)  # Be nice to Wikipedia
                                
                            except Exception as e:
                                logger.error(f"Error processing article {href}: {str(e)}")
                                continue
                                
                    if len(articles) >= limit:
                        break
                        
            except Exception as e:
                logger.error(f"Error accessing Wikipedia category {category}: {str(e)}")
                continue
                
        logger.info(f"Collected {len(articles)} Wikipedia articles")
        return articles
    
    def _get_wikipedia_article(self, url: str) -> Dict:
        """Extract content from a Wikipedia article"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Skip if we can't find the main content
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if not content_div:
                return None
                
            title = soup.find('h1', id='firstHeading')
            if not title:
                return None
                
            title = title.text.strip()
            
            # Get all paragraphs
            paragraphs = []
            for p in content_div.find_all('p', recursive=False):
                if p.text.strip() and not p.find_parent('table'):
                    paragraphs.append(p.text.strip())
            
            content = ' '.join(paragraphs)
            
            # Only return if we have substantial content
            if len(content) > 200:
                return {
                    'title': title,
                    'content': content,
                    'source': 'wikipedia',
                    'url': url
                }
                
        except Exception as e:
            logger.error(f"Error extracting article content from {url}: {str(e)}")
            
        return None