import pandas as pd
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.append(path_root)
sys.path.append(path_project)
sys.path.append(path_this)

from util import logging

def load_faq_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load FAQ data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing the FAQ data or None if loading fails
    """
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['question', 'answer']
        
        # Validate that required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"CSV is missing required columns: {missing_columns}")
            return None
            
        # Check for empty values
        empty_rows = df[df[required_columns].isna().any(axis=1)]
        if not empty_rows.empty:
            logger.warning(f"Found {len(empty_rows)} rows with missing values")
            # Remove rows with missing values
            df = df.dropna(subset=required_columns)
            
        logger.info(f"Loaded {len(df)} FAQs from CSV")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None

def create_documents_from_faqs(df: pd.DataFrame) -> List[str]:
    """
    Convert FAQ DataFrame to a list of document strings.
    
    Args:
        df: DataFrame containing FAQ data with 'question' and 'answer' columns
        
    Returns:
        List of document strings
    """
    documents = []
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Create a document string
        document = f"Q: {question}\nA: {answer}"
        documents.append(document)
        
    logger.info(f"Created {len(documents)} documents from FAQs")
    return documents