from setuptools import setup, find_packages

setup(
    name="nawatech-faq-chatbot",
    version="0.1.0",
    description="FAQ Chatbot using OpenAI, LangChain, and Streamlit",
    author="Daffa Arifadilah",
    author_email="daffaarifadilah@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.31.0",
        "langchain>=0.2.15",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.16",
        "langchain-core>=0.2.15",
        "pandas>=2.1.3",
        "python-dotenv>=1.0.0",
        "openai>=1.12.0",
    ],
    python_requires=">=3.9",
)