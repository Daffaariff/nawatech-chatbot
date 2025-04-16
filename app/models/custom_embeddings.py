import requests
import logging
from typing import List, Dict, Any, Optional
import os
import sys
from langchain.embeddings.base import Embeddings
from loguru import logger

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.append(path_root)
sys.path.append(path_project)
sys.path.append(path_this)

from util import logging

class CustomLocalEmbeddings(Embeddings):
    """
    Custom embeddings class for the local embedding model.
    This class directly uses requests to interact with the API instead of
    relying on langchain_openai which might convert text to tokens.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "ebbge-v2",
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
    ):
        """
        Initialize the custom embeddings class.
        
        Args:
            base_url: Base URL for the API
            api_key: API key
            model: Model name
            dimensions: Dimensions for the embeddings (optional)
            encoding_format: Encoding format for the embeddings
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        self.embeddings_url = f"{self.base_url}/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    

    def _create_embedding_request(self, texts: List[str]) -> Dict[str, Any]:
        """
        Create the embedding request.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Request payload
        """
        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": self.encoding_format,
        }
        
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
            
        return payload
        
    def _make_embedding_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the embedding request.
        
        Args:
            payload: Request payload
            
        Returns:
            Response from the API
            
        Raises:
            ValueError: If the request fails
        """
        try:
            response = requests.post(
                self.embeddings_url,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response is not None:
                logger.error(f"Error from API: {response.status_code} - {response.text}")
                raise ValueError(f"API request failed: {response.status_code} - {response.text}")
            else:
                logger.error(f"Request error: {e}")
                raise ValueError(f"Request error: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
            
        # Process in batches to avoid overwhelming the API
        batch_size = 3
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Embedding batch of {len(batch)} documents")
            
            try:
                payload = self._create_embedding_request(batch)
                response = self._make_embedding_request(payload)
                
                # Extract embeddings from response
                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                all_embeddings.extend([[0.0] * (self.dimensions or 384)] * len(batch))
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding
        """
        payload = self._create_embedding_request([text])
        response = self._make_embedding_request(payload)
        
        # Extract embedding from response
        return response["data"][0]["embedding"]