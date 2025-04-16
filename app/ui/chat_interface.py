import logging
import streamlit as st
import os
import sys
from typing import Dict, Any, List, Callable

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, '..'))
path_root = os.path.dirname(os.path.join(path_this, '../..'))
sys.path.append(path_root)
sys.path.append(path_project)
sys.path.append(path_this)

from app.config import APP_TITLE, APP_ICON, APP_DESCRIPTION, MAX_HISTORY_LENGTH
from app.util import logging

class ChatInterface:
    """
    Streamlit chat interface for the FAQ chatbot.
    """
    
    def __init__(self):
        """
        Initialize the chat interface.
        """
        self._configure_page()
        self._initialize_session_state()
        
    def _configure_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=APP_ICON
        )
        
        # Add custom CSS for better appearance
        st.markdown("""
        <style>
        .stApp {
            max-width: 1000px;
            margin: 0 auto;
        }
        .quality-score {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
            padding: 5px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .score-high {
            background-color: #d4edda;
            color: #155724;
        }
        .score-medium {
            background-color: #fff3cd;
            color: #856404;
        }
        .score-low {
            background-color: #f8d7da;
            color: #721c24;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        .metric {
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 10px;
        }
        .debug-info {
            font-size: 12px;
            margin-top: 10px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "show_scores" not in st.session_state:
            st.session_state.show_scores = True
            
        if "show_debug" not in st.session_state:
            st.session_state.show_debug = False
            
    def display_header(self):
        """Display the chat header."""
        st.title(f"{APP_TITLE} {APP_ICON}")
        st.markdown(APP_DESCRIPTION)
        
        # Add settings in a sidebar
        with st.sidebar:
            st.title("Settings")
            st.session_state.show_scores = st.checkbox("Show Quality Scores", value=st.session_state.show_scores)
            st.session_state.show_debug = st.checkbox("Show Debug Info", value=st.session_state.show_debug)
            
            # Add information about quality scores
            st.markdown("---")
            st.markdown("### Quality Metrics")
            st.markdown("""
            - **Relevance**: How relevant is the answer to the question
            - **Completeness**: How complete is the answer
            - **Clarity**: How clear and understandable is the answer
            - **Accuracy**: How accurately the answer uses information from the knowledge base
            """)
        
    def display_messages(self):
        """Display chat message history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display quality scores for assistant messages if enabled
                if message["role"] == "assistant" and "evaluation" in message and st.session_state.show_scores:
                    evaluation = message["evaluation"]
                    scores = evaluation.get("scores", {})
                    reasons = evaluation.get("reasons", {})
                    
                    # Get overall score
                    overall_score = scores.get("overall", 0)
                    
                    # Determine score class
                    score_class = "score-low"
                    if overall_score >= 4.0:
                        score_class = "score-high"
                    elif overall_score >= 3.0:
                        score_class = "score-medium"
                    
                    # Display overall score
                    st.markdown(f"""
                    <div class="quality-score {score_class}">
                        <strong>Quality Score:</strong> {overall_score}/5.0
                        <div class="metrics-container">
                            <span class="metric">Relevance: {scores.get('relevance', 0)}</span>
                            <span class="metric">Completeness: {scores.get('completeness', 0)}</span>
                            <span class="metric">Clarity: {scores.get('clarity', 0)}</span>
                            <span class="metric">Accuracy: {scores.get('accuracy', 0)}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display debug info if enabled
                    if st.session_state.show_debug:
                        with st.expander("Debug Info"):
                            st.write("Evaluation Method:", evaluation.get("method", "unknown"))
                            st.write("Reasons:")
                            for metric, metric_reasons in reasons.items():
                                st.write(f"- {metric.capitalize()}: {', '.join(metric_reasons)}")
                
    def add_message(self, role: str, content: str, **kwargs):
        """
        Add a message to the chat history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            **kwargs: Additional data to store with the message
        """
        message = {"role": role, "content": content, **kwargs}
        st.session_state.messages.append(message)
        
        # Limit history length
        if len(st.session_state.messages) > MAX_HISTORY_LENGTH:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY_LENGTH:]
            
    def get_user_input(self) -> str:
        """
        Get user input from the chat input.
        
        Returns:
            User input string or empty string if no input
        """
        return st.chat_input("Ask a question") or ""
        
    def process_user_input(self, query: str, response_generator: Callable[[str], Dict[str, Any]]):
        """
        Process user input and generate a response.
        
        Args:
            query: User query
            response_generator: Function to generate a response
        """
        # Add user message to chat history
        self.add_message("user", query)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                response = response_generator(query)
                answer = response.get("answer", "I'm sorry, I couldn't find an answer.")
                
                # Display response
                message_placeholder.markdown(answer)
                
                # Add assistant response to chat history with evaluation
                self.add_message(
                    "assistant", 
                    answer, 
                    evaluation=response.get("evaluation", {})
                )
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.markdown(error_message)
                logger.error(f"Error in chat interface: {e}")
                self.add_message("assistant", error_message)