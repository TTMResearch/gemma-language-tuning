from typing import List, Dict
import pandas as pd
from pathlib import Path
from loguru import logger
from .scrapers.news_scraper import AfrikaansNewsScraper

class AfrikaansDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.scraper = AfrikaansNewsScraper()
        
    def collect_data(self, num_pages: int = 100) -> pd.DataFrame:
        """Collect data from various sources"""
        logger.info("Starting data collection...")
        
        articles = []
        
        # Collect Wikipedia articles from category
        wiki_articles = self.scraper.scrape_wikipedia_category(limit=num_pages)
        articles.extend(wiki_articles)
        
        # Save raw data
        df = pd.DataFrame(articles)
        if not df.empty:
            raw_data_path = self.data_dir / "raw_articles.csv"
            df.to_csv(raw_data_path, index=False)
            logger.info(f"Saved {len(df)} articles to {raw_data_path}")
        else:
            logger.warning("No articles collected!")
        
        return df
    
    def create_instruction_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """Convert articles into instruction-response pairs with quality checks"""
        instruction_pairs = []
        
        for _, row in df.iterrows():
            # Skip if content is too short
            if len(row["content"]) < 200:
                continue
                
            # Create various instruction types
            pairs = [
                # Summarization
                {
                    "instruction": f"Vat hierdie artikel in Afrikaans saam:",
                    "input": row["content"],
                    "output": row["title"]
                },
                # Content Generation
                {
                    "instruction": "Skryf 'n kort paragraaf oor:",
                    "input": row["title"],
                    "output": row["content"][:500]  # First 500 chars as response
                },
                # Question Answering
                {
                    "instruction": "Beantwoord hierdie vraag oor die artikel:",
                    "input": f"Waaroor handel hierdie artikel? {row['content']}",
                    "output": row["title"]
                }
            ]
            instruction_pairs.extend(pairs)
        
        return instruction_pairs
    
    def save_processed_data(self, pairs: List[Dict], filename: str = "instruction_pairs.json"):
        """Save processed instruction pairs"""
        output_path = self.data_dir / filename
        pd.DataFrame(pairs).to_json(output_path, orient="records", lines=True)
        logger.info(f"Saved {len(pairs)} instruction pairs to {output_path}")