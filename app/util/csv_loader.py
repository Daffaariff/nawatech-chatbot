import pandas as pd
from typing import List, Dict

def load_faqs(file_path: str) -> List[Dict]:
    """Load FAQs from CSV file"""
    df = pd.read_csv(file_path)
    return df.to_dict('records')