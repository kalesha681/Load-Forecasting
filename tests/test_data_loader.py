
import pytest
import pandas as pd
import numpy as np
from src.data_loader import parse_yearly_datetime, train_test_split

class TestDataLoader:
    def test_parse_yearly_datetime_standard(self):
        """Test parsing of standard format."""
        # Setup similar to what's in data_loader
        # date_col elements are strings like "01-JAN" or "01-Jan"
        # year_col elements are integers or strings like 2024
        
        years = pd.Series([2024, 2024])
        dates = pd.Series(["01-JAN", "02-FEB"])
        
        # Test just the parsing logic isolated? 
        # The function expects Series inputs
        
        result = parse_yearly_datetime(years, dates)
        
        assert len(result) == 2
        assert result[0] == pd.Timestamp("2024-01-01")
        assert result[1] == pd.Timestamp("2024-02-02")

    def test_parse_yearly_datetime_fallback(self):
        """Test fallback parsing if primary format fails slightly."""
        # If strict format fails, it tries fallback
        years = pd.Series([2024])
        dates = pd.Series(["01/01"]) # Might fail strict "%d-%b" but pass generic
        
        # However, the code does a replace '.', '' and upper().
        # "2024 01/01" might work with dayfirst=True
        
        result = parse_yearly_datetime(years, dates)
        assert result[0] == pd.Timestamp("2024-01-01")

    def test_parse_yearly_datetime_failure(self):
        """Test completely invalid date raises error."""
        years = pd.Series([2024])
        dates = pd.Series(["NOT-A-DATE"])
        
        with pytest.raises(Exception):
            parse_yearly_datetime(years, dates)

    def test_train_test_split(self):
        """Test simple split logic."""
        df = pd.DataFrame({'data': range(100)})
        train, test = train_test_split(df, test_days=2) # 2 days * 24 hours = 48 hours
        
        # Original logic: split_idx = len(df) - (test_days * 24)
        # 100 - 48 = 52
        
        assert len(train) == 52
        assert len(test) == 48
        
        # Concatenation should equal original
        assert len(pd.concat([train, test])) == 100

    def test_train_test_split_short_data(self):
        """Test split when data is shorter than test window."""
        df = pd.DataFrame({'data': range(10)})
        train, test = train_test_split(df, test_days=2) # 48
        
        # split_idx = 10 - 48 = -38 -> 0
        assert len(train) == 0
        assert len(test) == 10
