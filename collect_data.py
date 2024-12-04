from src.data.data_processing import AfrikaansDataProcessor
import argparse

def main(num_pages: int):
    processor = AfrikaansDataProcessor()
    
    # Collect articles
    print(f"Collecting {num_pages} articles...")
    df = processor.collect_data(num_pages=num_pages)
    print(f'Collected {len(df)} articles')
    
    # Create instruction pairs
    pairs = processor.create_instruction_pairs(df)
    processor.save_processed_data(pairs)
    print(f'Created {len(pairs)} instruction pairs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Afrikaans training data')
    parser.add_argument('--num_pages', type=int, default=50, help='Number of pages to collect')
    args = parser.parse_args()
    main(args.num_pages)