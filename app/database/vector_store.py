from typing import List, Optional
import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema.retriever import BaseRetriever
from loguru import logger

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.append(path_root)
sys.path.append(path_project)
sys.path.append(path_this)

from config import TOP_K
from models import CustomLocalEmbeddings
from util import logging
class VectorStore:
    """
    Manages the InMemoryVectorStore for FAQ embeddings.
    """
    
    def __init__(self):
        """
        Initialize the vector store.
        """
        base_url = os.getenv("EMBEDDING_BASE_URL")
        api_key = os.getenv("EMBEDDING_API_KEY")
        model_name = os.getenv("EMBEDDING_MODEL", "ebbge-v2")
        
        if base_url and api_key:
            logger.info(f"Using custom local embedding model at {base_url} with model {model_name}")
            try:
                self.embeddings = CustomLocalEmbeddings(
                    base_url=base_url,
                    api_key=api_key,
                    model=model_name
                )
            except Exception as e:
                logger.error(f"Error configuring local embeddings: {e}")
                logger.info("Falling back to OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings()
        else:
            # Use standard OpenAI embeddings
            logger.info("Using standard OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings()
            
        self.vectorstore = None
        
    def initialize(self, documents: List[str]) -> bool:
        """
        Initialize the in-memory vector store with documents.
        
        Args:
            documents: List of document strings to embed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            try:
                test_embedding = self.embeddings.embed_query("Test embedding")
                logger.info(f"Successfully tested embedding connection. Vector dimensions: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"Error testing embedding connection: {e}")
                logger.error(f"Error details: {str(e)}")
                return False
            
            self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vectorstore.add_texts(documents)
            
            logger.info(f"Successfully initialized InMemoryVectorStore with {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error initializing InMemoryVectorStore: {e}")
            logger.error(f"Error details: {str(e)}")
            return False
    
    def get_retriever(self) -> Optional[BaseRetriever]:
        """
        Get a retriever for the vector store.
        
        Returns:
            Retriever or None if the vector store is not initialized
        """
        # result = self.vectorstore.as_retriever(
        #         search_kwargs={"k": TOP_K}
        #     )
        if self.vectorstore:
            logger.debug
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 10}
            )
        return None