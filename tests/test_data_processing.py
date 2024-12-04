import pytest
from src.data.data_processing import AfrikaansDataProcessor

def test_data_collection():
    processor = AfrikaansDataProcessor()
    df = processor.collect_data(num_pages=1)  # Test with one page
    assert len(df) > 0
    assert "title" in df.columns
    assert "content" in df.columns