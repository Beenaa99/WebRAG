# WebRAG

A Streamlit application for crawling websites or processing PDFs, then answering questions about their content using retrieval-augmented generation.

## Features
- Crawl websites and index their content
- Process PDF documents
- Ask questions using similarity search
- Generate AI-powered answers via OpenAI
- Save and manage multiple indices

## Quick Start

```bash
# Install dependencies
pip install streamlit faiss-cpu numpy sentence-transformers openai pdfplumber beautifulsoup4 requests

# Run the app
streamlit run webrag-streamlit.py
```

## Usage
1. Crawl a website or upload a PDF
2. Ask questions about the content
3. Get relevant passages and AI-generated answers

## Requirements
- Python 3.7+
- OpenAI API key (for AI answer generation)
