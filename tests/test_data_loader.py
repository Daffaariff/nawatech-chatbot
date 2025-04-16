# tests/test_data_loader.py
import unittest
import pandas as pd
from pathlib import Path
from app.database.data_loader import load_faq_data, create_documents_from_faqs

class TestDataLoader(unittest.TestCase):
    def test_load_faq_data(self):
        # Create a sample CSV for testing
        test_csv = Path("test_faqs.csv")
        test_data = pd.DataFrame({
            "question": ["Test question?"],
            "answer": ["Test answer."]
        })
        test_data.to_csv(test_csv, index=False)
        
        # Test loading
        result = load_faq_data(test_csv)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["question"], "Test question?")
        
        # Clean up
        test_csv.unlink()
        
    def test_create_documents(self):
        # Test document creation
        test_data = pd.DataFrame({
            "question": ["Q1?", "Q2?"],
            "answer": ["A1.", "A2."]
        })
        
        docs = create_documents_from_faqs(test_data)
        self.assertEqual(len(docs), 2)
        self.assertTrue("Q1?" in docs[0])
        self.assertTrue("A1." in docs[0])