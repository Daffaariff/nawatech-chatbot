# Nawatech FAQ Chatbot

A chatbot for Nawatech that answers customer inquiries using large language model.

## Features

- Uses local language model for privacy and efficiency
- Integrates with custom embedding model 
- Easy FAQ management through Excel or CSV
- Response quality scoring system
- Multilingual support (Based on the LLM Capability)
- Comprehensive logging for debugging
- Docker deployment support
- Security protections

## Project Structure
nawatech-chatbot/
├── app                             
│   ├── config.py                   
│   ├── data                        
│   ├── database            
│   ├── models              
│   ├── ui                  
│   └── util                
├── data                    
├── docker-compose.yml      
├── Dockerfile              
├── README.MD               
├── requirements.txt        
└── tests                   

## Getting Started

### Setup

1. Clone this repository
2. Create a `.env` file from the example:
cp .env.example .env
3. Update the `.env` file with your API keys and endpoints

### Running Locally

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application on specific host and port
streamlit run app/main.py --server.address=<host> --server.port=<port>
The chatbot will be accessible at http://<host>:<port>
Running with Docker
bash# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f
FAQ Management
Update the Excel file and run the cleaning script:
bashpython -m app.data.cleaning_data
Or directly edit app/data/faqs.csv (requires question and answer columns).
Customization
Edit app/config.py to customize:

Model temperature for creativity level
Number of retrieved documents
Logging levels

Troubleshooting
Common issues:

Connection Errors: Check embedding model API accessibility
Missing FAQ Data: Verify CSV format
Docker Issues: Check logs with docker-compose logs -f