import unittest
import pandas as pd
from pathlib import Path
from app.database.vector_store import VectorStore
from app.models.qa_chain import QAChain

class TestChatbot(unittest.TestCase):
    def setUp(self):
        # Create test FAQ data
        self.test_csv = Path("test_faqs.csv")
        test_data = pd.DataFrame({
            "question": ["Apa itu Nawatech?", "Siapa CEO Nawatech?"],
            "answer": ["Perusahaan teknologi", "Arfan Arlanda"]
        })
        test_data.to_csv(self.test_csv, index=False)
        
        # Create documents for vector store
        documents = [
            f"Q: {row['question']}\nA: {row['answer']}" 
            for _, row in test_data.iterrows()
        ]
        
        # Initialize vector store
        self.vector_store = VectorStore()
        self.vector_store.initialize(documents)
        
        # Initialize QA chain
        retriever = self.vector_store.get_retriever()
        self.qa_chain = QAChain(retriever)
        
    def tearDown(self):
        if self.test_csv.exists():
            self.test_csv.unlink()
        
    def test_chatbot_response(self):
        # Test with an exact question from the data
        response = self.qa_chain.generate_response("Apa itu Nawatech?")
        self.assertIn("answer", response)
        self.assertTrue("Perusahaan teknologi" in response["answer"])
        
        # Test with a similar but not exact question
        response = self.qa_chain.generate_response("Tolong jelaskan tentang Nawatech")
        self.assertIn("answer", response)
        # Should still retrieve the relevant information
        self.assertTrue("Perusahaan" in response["answer"])