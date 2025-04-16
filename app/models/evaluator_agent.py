"""
Response quality evaluator for the FAQ chatbot.
"""

import logging
from typing import Dict, Any, List, Tuple
import os
import re

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """
    Evaluates the quality of responses from the LLM.
    """
    
    def __init__(self):
        """
        Initialize the response evaluator.
        """
        self.use_model = os.getenv("USE_MODEL_EVALUATOR", "False").lower() == "true"
        
        if self.use_model:
            try:
                # Get API details from environment
                llm_base_url = os.getenv("LLM_BASE_URL")
                llm_api_key = os.getenv("LLM_API_KEY")
                llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
                
                if llm_base_url and llm_api_key:
                    self.eval_llm = ChatOpenAI(
                        base_url=llm_base_url,
                        api_key=llm_api_key,
                        model=llm_model,
                        temperature=0.0  # Use 0 temperature for consistent evaluation
                    )
                else:
                    # Use OpenAI API
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    self.eval_llm = ChatOpenAI(
                        api_key=openai_api_key,
                        model="gpt-3.5-turbo",
                        temperature=0.0
                    )
                
                logger.info("Initialized model-based response evaluator")
            except Exception as e:
                logger.error(f"Error initializing model-based evaluator: {e}")
                self.use_model = False
        
    def evaluate_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a response.
        
        Args:
            query: The user's query
            response: The LLM's response
            context: The retrieved context used to generate the response
            
        Returns:
            Dictionary with quality scores and evaluation details
        """
        if self.use_model:
            return self._evaluate_with_model(query, response, context)
        else:
            return self._evaluate_heuristic(query, response, context)
    
    def _evaluate_heuristic(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """
        Evaluate response quality using heuristic rules.
        
        Args:
            query: The user's query
            response: The LLM's response
            context: The retrieved context used to generate the response
            
        Returns:
            Dictionary with quality scores and evaluation details
        """
        # Initialize scores
        scores = {
            "relevance": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "accuracy": 0.0,
            "overall": 0.0
        }
        
        # Reasons for scores
        reasons = {
            "relevance": [],
            "completeness": [],
            "clarity": [],
            "accuracy": []
        }
        
        # 1. Relevance: Check if response contains key terms from the query
        query_terms = set(self._extract_keywords(query))
        response_terms = set(self._extract_keywords(response))
        common_terms = query_terms.intersection(response_terms)
        
        relevance_score = len(common_terms) / max(len(query_terms), 1)
        scores["relevance"] = min(relevance_score * 5, 5.0)  # Scale to 0-5
        
        if scores["relevance"] < 2.5:
            reasons["relevance"].append("Response doesn't contain many key terms from the query")
        else:
            reasons["relevance"].append("Response contains key terms from the query")
        
        # 2. Completeness: Check response length and structure
        word_count = len(response.split())
        if word_count < 10:
            scores["completeness"] = 1.0
            reasons["completeness"].append("Response is too short")
        elif word_count < 30:
            scores["completeness"] = 3.0
            reasons["completeness"].append("Response is of medium length")
        else:
            scores["completeness"] = 5.0
            reasons["completeness"].append("Response has good length")
        
        # 3. Clarity: Check for complex sentences and readability
        sentences = response.split('.')
        avg_words_per_sentence = word_count / max(len(sentences), 1)
        
        if avg_words_per_sentence > 25:
            scores["clarity"] = 2.0
            reasons["clarity"].append("Sentences are too long")
        elif avg_words_per_sentence > 15:
            scores["clarity"] = 3.5
            reasons["clarity"].append("Sentences are of moderate length")
        else:
            scores["clarity"] = 5.0
            reasons["clarity"].append("Sentences are concise")
        
        # 4. Accuracy: Check if response information is present in the context
        context_terms = set(self._extract_keywords(context))
        response_unique_terms = response_terms - query_terms
        context_overlap = response_unique_terms.intersection(context_terms)
        
        accuracy_score = len(context_overlap) / max(len(response_unique_terms), 1)
        scores["accuracy"] = min(accuracy_score * 5, 5.0)  # Scale to 0-5
        
        if scores["accuracy"] < 2.5:
            reasons["accuracy"].append("Response contains information not in the context")
        else:
            reasons["accuracy"].append("Response information is present in the context")
        
        # Calculate overall score (weighted average)
        weights = {"relevance": 0.3, "completeness": 0.2, "clarity": 0.2, "accuracy": 0.3}
        overall_score = sum(scores[metric] * weights[metric] for metric in weights)
        scores["overall"] = round(overall_score, 2)
        
        return {
            "scores": scores,
            "reasons": reasons,
            "method": "heuristic"
        }
    
    def _evaluate_with_model(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """
        Evaluate response quality using a model.
        
        Args:
            query: The user's query
            response: The LLM's response
            context: The retrieved context used to generate the response
            
        Returns:
            Dictionary with quality scores and evaluation details
        """
        try:
            # Create the evaluation prompt
            eval_prompt = f"""
            You are an expert evaluator of chatbot responses. Please evaluate the following:
            
            USER QUERY: {query}
            
            RETRIEVED CONTEXT: {context}
            
            CHATBOT RESPONSE: {response}
            
            Evaluate the response on the following criteria on a scale of 1-5 (5 being best):
            1. Relevance: How well does the response address the user's query?
            2. Completeness: How complete is the response?
            3. Clarity: How clear and easy to understand is the response?
            4. Accuracy: How accurately does the response use information from the context?
            
            For each criterion, provide a score and a brief reason.
            
            FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
            Relevance: [score]
            Reason: [reason]
            Completeness: [score]
            Reason: [reason]
            Clarity: [score]
            Reason: [reason]
            Accuracy: [score]
            Reason: [reason]
            Overall: [average_score]
            """
            
            # Get evaluation from LLM
            eval_response = self.eval_llm.invoke(eval_prompt)
            eval_text = eval_response.content
            
            # Parse the evaluation response
            scores, reasons = self._parse_evaluation(eval_text)
            
            return {
                "scores": scores,
                "reasons": reasons,
                "method": "model",
                "raw_evaluation": eval_text
            }
        except Exception as e:
            logger.error(f"Error during model-based evaluation: {e}")
            # Fall back to heuristic evaluation
            result = self._evaluate_heuristic(query, response, context)
            result["error"] = str(e)
            return result
    
    def _parse_evaluation(self, eval_text: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Parse the evaluation response from the LLM.
        
        Args:
            eval_text: The raw evaluation text from the LLM
            
        Returns:
            Tuple of (scores dict, reasons dict)
        """
        scores = {}
        reasons = {}
        
        # Parse scores and reasons
        for line in eval_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Parse scores
            score_match = re.match(r'(Relevance|Completeness|Clarity|Accuracy|Overall):\s*(\d+\.?\d*)', line)
            if score_match:
                metric, score = score_match.groups()
                scores[metric.lower()] = float(score)
                continue
                
            # Parse reasons
            reason_match = re.match(r'Reason:\s*(.*)', line)
            if reason_match:
                # Find which metric this reason belongs to
                for metric in ["relevance", "completeness", "clarity", "accuracy"]:
                    if metric.capitalize() in eval_text.split(line)[0].split("Reason:")[-2]:
                        reasons[metric] = [reason_match.group(1)]
                        break
        
        return scores, reasons
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        
        stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'untuk', 'pada', 'adalah', 'ini', 'itu',
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between', 'into', 'through',
            'this', 'that', 'these', 'those', 'of', 'from'
        }
        
        # Filter out stopwords and short words
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords