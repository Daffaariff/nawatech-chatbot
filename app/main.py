import streamlit as st
import os
import sys
from loguru import logger

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.append(path_root)
sys.path.append(path_project)
sys.path.append(path_this)

from config import FAQ_FILE
from database import (
    load_faq_data, 
    create_documents_from_faqs, 
    VectorStore
    )
from models import QAChain
from ui import ChatInterface
from util import logging

@st.cache_resource
def initialize_vector_store():
    """
    Initialize the vector store.
    
    Returns:
        Initialized vector store or None if initialization fails
    """
    # Load FAQ data
    df = load_faq_data(FAQ_FILE)
    if df is None:
        st.error("Failed to load FAQ data. Please check the CSV file.")
        return None
        
    # Create documents
    documents = create_documents_from_faqs(df)
    
    # Initialize vector store
    vector_store = VectorStore()
    if not vector_store.initialize(documents):
        st.error("Failed to initialize vector store.")
        return None
        
    return vector_store

def main():
    """Main application entry point."""
    # Initialize chat interface
    chat_interface = ChatInterface()
    chat_interface.display_header()
    chat_interface.display_messages()
    
    # Initialize vector store
    vector_store = initialize_vector_store()
    if vector_store is None:
        st.stop()
        
    # Get retriever
    retriever = vector_store.get_retriever()
    logger.debug(f"Retriever: {retriever}")
    if retriever is None:
        st.error("Failed to create retriever.")
        st.stop()
        
    # Initialize QA chain
    qa_chain = QAChain(retriever)
    
    # Get user input
    user_query = chat_interface.get_user_input()
    
    # Process user input
    if user_query:
        chat_interface.process_user_input(
            user_query,
            qa_chain.generate_response
        )

if __name__ == "__main__":
    main()
    
    # streamlit run app/main.py --server.address=10.12.1.160 --server.port=10020