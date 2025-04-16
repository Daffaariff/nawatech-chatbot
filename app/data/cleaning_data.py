import pandas as pd
import os
import sys
import re
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_faq_text_data(text_data):
    """
    Parse raw FAQ data, handling the specific format of Nawatech's data.
    
    Args:
        text_data: Raw text data containing question-answer pairs
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    text_data = text_data.replace("QuestionAnswer", "")
    qa_pairs = []
    
    parts = re.split(r'([^.!?]*\?)', text_data)
    parts = [p for p in parts if p.strip()]
    
    for i in range(0, len(parts)-1, 2):
        if i+1 < len(parts):
            question = parts[i].strip()
            answer = parts[i+1].strip()
            
            # Replace "layanan kami" (case-insensitive) with the desired question
            if re.search(r"layanan\s+kami", question, re.IGNORECASE):
                question = "Layanan apa saja yang diberikan Nawatech?"
            
            print(f"Question: {question}")
            qa_pairs.append({"question": question, "answer": answer})
    
    logger.info(f"Parsed {len(qa_pairs)} question-answer pairs from text data")
    return qa_pairs


def extract_text_from_excel(excel_path):
    """
    Extract text content from an Excel file.
    
    Args:
        excel_path: Path to the Excel file
        
    Returns:
        Concatenated text from all cells or None if failed
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Convert all columns to string and concatenate all cells
        text = ""
        for _, row in df.iterrows():
            for cell in row:
                if pd.notna(cell):
                    text += str(cell) + " "
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Excel: {e}")
        return None

def convert_excel_to_csv(excel_path, csv_path):
    """
    Convert Excel FAQ file to CSV format.
    
    Args:
        excel_path: Path to the Excel file
        csv_path: Path to save the CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Reading Excel file: {excel_path}")
        
        try:
            df = pd.read_excel(excel_path)
            
            logger.info(f"Excel file contains {len(df)} rows and {len(df.columns)} columns")
            
            if len(df.columns) >= 2 and len(df) > 0:
                df = df.iloc[:, :2]
                df.columns = ["question", "answer"]

                # Replace "layanan kami" questions with standardized version
                df["question"] = df["question"].apply(
                    lambda q: "Layanan apa saja yang diberikan Nawatech?" if isinstance(q, str) and re.search(r"layanan\s+kami", q, re.IGNORECASE) else q
                )

                
                empty_rows = df[df[["question", "answer"]].isna().any(axis=1)]
                if not empty_rows.empty:
                    logger.warning(f"Found {len(empty_rows)} rows with missing values, removing them")
                    df = df.dropna(subset=["question", "answer"])
                
                logger.info(f"Successfully processed Excel with structured data: {len(df)} FAQ pairs")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                
                # Save to CSV
                df.to_csv(csv_path, index=False)
                logger.info(f"Successfully saved {len(df)} FAQ pairs to {csv_path}")
                
                return True
            else:
                logger.warning("Excel structure not suitable for direct conversion, trying text extraction")
        except Exception as e:
            logger.warning(f"Could not process Excel as structured data: {e}")
        
        # Try text-based approach (extract text and parse)
        text_data = extract_text_from_excel(excel_path)
        if text_data:
            qa_pairs = parse_faq_text_data(text_data)
            
            if qa_pairs:
                # Create DataFrame from parsed data
                df = pd.DataFrame(qa_pairs)
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                
                # Save to CSV
                df.to_csv(csv_path, index=False)
                logger.info(f"Successfully saved {len(qa_pairs)} FAQ pairs to {csv_path} using text parsing")
                
                return True
            else:
                logger.error("Failed to parse FAQ text data from Excel")
                return False
        else:
            logger.error("Failed to extract text from Excel")
            return False
            
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {e}")
        return False

def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    data_dir = base_dir / "data"
    excel_path = data_dir / "FAQ_Nawa.xlsx"
    csv_path = data_dir / "faqs.csv"
    
    data_dir.mkdir(exist_ok=True)
    
    if excel_path.exists():
        logger.info(f"Found Excel file: {excel_path}")
        success = convert_excel_to_csv(excel_path, csv_path)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())