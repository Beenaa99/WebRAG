import os
import json
import pickle
import requests
import tempfile
import uuid
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pdfplumber

# Set page configuration
st.set_page_config(
    page_title="WebRAG - Website QA System",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    .stAlert {
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        border-radius: 0.25rem;
    }
    .result-item {
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #dee2e6;
    }
    .answer-container {
        background-color: #f0f7ff;
        border-left: 4px solid #4a6fa5;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 0 4px 4px 0;
    }
    .index-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# Create directories for storing data
@st.cache_resource
def initialize_directories():
    # Create folders for storing indices and PDFs
    indices_folder = 'indices'
    pdf_folder = 'pdfs'
    os.makedirs(indices_folder, exist_ok=True)
    os.makedirs(pdf_folder, exist_ok=True)
    return indices_folder, pdf_folder

indices_folder, pdf_folder = initialize_directories()

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = embedder.get_sentence_embedding_dimension()
    return embedder, embedding_dim

embedder, embedding_dim = load_embedding_model()

# Initialize OpenAI client
def initialize_openai_client():
    # Try to load from environment variable, then from session state
    api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
    
    if api_key:
        return OpenAI(api_key=api_key)
    return None

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = None
if 'current_index_name' not in st.session_state:
    st.session_state.current_index_name = None
if 'crawled_urls' not in st.session_state:
    st.session_state.crawled_urls = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Helper Functions

def crawl_website(start_url, max_pages=50, progress_callback=None):
    visited = set()
    to_visit = [start_url]
    pages = []

    # Get the domain for limiting the crawl
    domain = urlparse(start_url).netloc
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    crawl_count = 0
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            status_text.text(f"Crawling: {url}")
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
            content_type = response.headers.get('Content-Type', '')
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get visible text
            text = soup.get_text(separator=" ", strip=True)
            pages.append({"url": url, "content": text})
            visited.add(url)
            
            # Update progress
            crawl_count += 1
            if progress_callback:
                progress_callback(url)
            progress_bar.progress(min(crawl_count / max_pages, 1.0))
            
            # Find all links on the current page
            for link in soup.find_all("a", href=True):
                href = link.get("href")
                
                if "#" in href:
                    continue
                # Create absolute URL if needed
                next_url = urljoin(url, href)
                # Only consider links within the same domain
                if urlparse(next_url).netloc == domain and next_url not in visited:
                    to_visit.append(next_url)

        except Exception as e:
            status_text.text(f"Error crawling {url}: {e}")
            continue

    progress_bar.empty()
    status_text.empty()
    return pages

def process_pdf(pdf_file, progress_callback=None):
    documents = []
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf.pages):
                status_text.text(f"Processing page {i+1} of {total_pages}")
                text = page.extract_text() or ""
                if text.strip():  # Only add non-empty pages
                    documents.append({
                        "url": f"{pdf_file.name}#page={i+1}",
                        "content": text
                    })
                    if progress_callback:
                        progress_callback(f"PDF: {pdf_file.name} - Page {i+1}")
                
                # Update progress
                progress_bar.progress((i + 1) / total_pages)
                
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    finally:
        # Clean up the temporary file
        os.unlink(pdf_path)
    
    return documents

def create_faiss_index(documents):
    texts = [doc["content"] for doc in documents]
    
    with st.spinner("Generating embeddings... This might take a moment."):
        embeddings = embedder.encode(texts, show_progress_bar=True)
        
    # Convert embeddings to float32 and create index
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_dim)  # using L2 similarity
    index.add(embeddings)
    return index, embeddings

def search_index(query, k=3):
    if st.session_state.index is None:
        return []
    
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = st.session_state.index.search(query_vec, k)
    results = []
    for idx in indices[0]:
        if idx < len(st.session_state.documents):
            results.append(st.session_state.documents[idx])
    return results

def generate_answer(query, retrieved_docs):
    client = initialize_openai_client()
    
    # Return error message if API key is missing
    if client is None:
        return "Error: OpenAI API key not configured. Please enter your API key in the settings tab."
    
    # Concatenate the documents as context
    context = "\n\n".join([f"URL: {doc['url']}\nContent: {doc['content'][:500]}..." for doc in retrieved_docs])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to other models as needed
            messages=[
                {"role": "system", "content": "You are an expert on summarizing web content. Format your answers using Markdown where appropriate: use **bold** for emphasis, headings with #, lists with - or 1., and `code` for technical terms. Include at least one structured element in your response."},
                {"role": "user", "content": f"Answer the following question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        answer = f"Sorry, I could not generate an answer at this time. Error: {str(e)}"
    return answer

def save_index(index_name):
    if st.session_state.index is None or not st.session_state.documents:
        return False, "No index to save. Please crawl a website first."
    
    # Create a unique index name if none provided
    if not index_name:
        index_name = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directory for this index
    index_dir = os.path.join(indices_folder, index_name)
    os.makedirs(index_dir, exist_ok=True)
    
    try:
        # Save the FAISS index
        faiss.write_index(st.session_state.index, os.path.join(index_dir, "index.faiss"))
        
        # Save the document embeddings
        with open(os.path.join(index_dir, "embeddings.npy"), 'wb') as f:
            np.save(f, st.session_state.doc_embeddings)
        
        # Save the documents
        with open(os.path.join(index_dir, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(st.session_state.documents, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "document_count": len(st.session_state.documents),
            "embedding_dimension": embedding_dim,
            "first_url": st.session_state.documents[0]["url"] if st.session_state.documents else "Unknown"
        }
        with open(os.path.join(index_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        st.session_state.current_index_name = index_name
        return True, f"Index saved successfully as '{index_name}'"
        
    except Exception as e:
        return False, f"Error saving index: {str(e)}"

def load_index(index_name):
    index_dir = os.path.join(indices_folder, index_name)
    
    if not os.path.exists(index_dir):
        return False, f"Index '{index_name}' not found."
    
    try:
        # Load the FAISS index
        st.session_state.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        
        # Load the document embeddings
        with open(os.path.join(index_dir, "embeddings.npy"), 'rb') as f:
            st.session_state.doc_embeddings = np.load(f)
        
        # Load the documents
        with open(os.path.join(index_dir, "documents.json"), 'r', encoding='utf-8') as f:
            st.session_state.documents = json.load(f)
        
        st.session_state.current_index_name = index_name
        return True, f"Successfully loaded index '{index_name}' with {len(st.session_state.documents)} documents."
        
    except Exception as e:
        return False, f"Error loading index: {str(e)}"

def get_available_indices():
    indices = []
    
    for index_name in os.listdir(indices_folder):
        index_path = os.path.join(indices_folder, index_name)
        if os.path.isdir(index_path):
            # Try to read metadata
            try:
                with open(os.path.join(index_path, "metadata.json"), 'r') as f:
                    metadata = json.load(f)
                
                indices.append({
                    "name": index_name,
                    "created_at": metadata.get("created_at", "Unknown"),
                    "document_count": metadata.get("document_count", 0),
                    "first_url": metadata.get("first_url", "Unknown")
                })
            except:
                # If metadata not available, add minimal info
                indices.append({
                    "name": index_name,
                    "created_at": "Unknown",
                    "document_count": "Unknown",
                    "first_url": "Unknown"
                })
    
    # Sort by name (could be changed to sort by date if preferred)
    indices.sort(key=lambda x: x["name"])
    return indices

# Main App UI

st.title("WebRAG - Website Question Answering System")
st.write("Crawl websites or upload PDFs, then ask questions about the content.")

# Current Index Display
if st.session_state.current_index_name:
    st.sidebar.success(f"Current Index: {st.session_state.current_index_name}")
else:
    st.sidebar.info("No index loaded")

# API Key Input
with st.sidebar.expander("OpenAI API Settings"):
    api_key = st.text_input("OpenAI API Key", 
                           value=st.session_state.get("openai_api_key", ""), 
                           type="password",
                           help="Required for generating answers using GPT models")
    if api_key:
        st.session_state.openai_api_key = api_key
        if st.button("Test API Key"):
            client = initialize_openai_client()
            if client:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    st.success("API key is valid!")
                except Exception as e:
                    st.error(f"API key error: {str(e)}")
            else:
                st.error("Please enter a valid API key")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Crawl Website", "Upload PDF", "Ask Questions", "Manage Indices"])

# Tab 1: Crawl Website
with tab1:
    st.header("Crawl and Index a Website")
    
    with st.form("crawl_form"):
        website_url = st.text_input("Website URL", placeholder="https://example.com")
        max_pages = st.slider("Maximum Pages to Crawl", 1, 200, 50)
        index_name = st.text_input("Save Index As (optional)", placeholder="my-website-index")
        
        submit_crawl = st.form_submit_button("Start Crawling")
    
    if submit_crawl and website_url:
        st.session_state.crawled_urls = []
        
        def update_crawl_status(url):
            st.session_state.crawled_urls.append(url)
        
        st.subheader("Crawl Progress")
        
        # Crawl the website
        st.session_state.documents = crawl_website(website_url, max_pages=max_pages, progress_callback=update_crawl_status)
        
        if not st.session_state.documents:
            st.error("No pages were crawled. Please try a different URL.")
        else:
            st.success(f"Crawled {len(st.session_state.documents)} pages from: {website_url}")
            
            # Create FAISS index for the crawled documents
            st.session_state.index, st.session_state.doc_embeddings = create_faiss_index(st.session_state.documents)
            
            # Save the index if a name was provided
            if index_name:
                success, message = save_index(index_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Show crawled URLs
    if st.session_state.crawled_urls:
        with st.expander(f"Crawled URLs ({len(st.session_state.crawled_urls)})"):
            for url in st.session_state.crawled_urls:
                st.write(url)

# Tab 2: Upload PDF
with tab2:
    st.header("Upload and Process PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        pdf_index_name = st.text_input("Save PDF Index As (optional)", placeholder="my-pdf-index")
        
        if st.button("Process PDF"):
            st.session_state.crawled_urls = []
            
            def update_pdf_status(page):
                st.session_state.crawled_urls.append(page)
            
            st.subheader("Processing Progress")
            
            # Process the PDF
            st.session_state.documents = process_pdf(uploaded_file, progress_callback=update_pdf_status)
            
            if not st.session_state.documents:
                st.error("Could not extract text from the PDF. The file might be empty or protected.")
            else:
                st.success(f"Processed {len(st.session_state.documents)} pages from PDF: {uploaded_file.name}")
                
                # Create FAISS index for the extracted text
                st.session_state.index, st.session_state.doc_embeddings = create_faiss_index(st.session_state.documents)
                
                # Save the index if a name was provided
                if pdf_index_name:
                    success, message = save_index(pdf_index_name)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # Show processed pages
    if st.session_state.get('crawled_urls') and uploaded_file:
        with st.expander(f"Processed Pages ({len(st.session_state.crawled_urls)})"):
            for page in st.session_state.crawled_urls:
                st.write(page)

# Tab 3: Ask Questions
with tab3:
    st.header("Ask Questions")
    
    if st.session_state.index is None:
        st.warning("Please crawl a website, upload a PDF, or load an index first.")
    else:
        query = st.text_input("Ask a question", placeholder="What is...?")
        generate = st.checkbox("Generate AI answer", value=True, help="Requires OpenAI API key")
        
        if st.button("Ask Question") and query:
            # Retrieve similar documents using the FAISS index
            retrieved_docs = search_index(query, k=3)
            
            # Generate answer if requested
            if generate:
                answer = generate_answer(query, retrieved_docs)
                st.markdown("### Answer:")
                st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)
            
            # Display retrieved documents
            st.markdown("### Retrieved Documents:")
            
            if not retrieved_docs:
                st.info("No relevant documents found.")
            
            for doc in retrieved_docs:
                with st.expander(f"URL: {doc['url']}"):
                    snippet = doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"]
                    st.text(snippet)

# Tab 4: Manage Indices
with tab4:
    st.header("Manage Saved Indices")
    
    if st.button("Refresh Indices"):
        st.rerun()
    
    indices = get_available_indices()
    
    if not indices:
        st.info("No saved indices found.")
    else:
        for idx in indices:
            col1, col2 = st.columns([3, 1])
            with col1:
                expander_label = f"{idx['name']}"
                if idx['name'] == st.session_state.current_index_name:
                    expander_label += " (Current)"
                
                with st.expander(expander_label):
                    st.write(f"Created: {idx['created_at']}")
                    st.write(f"Documents: {idx['document_count']}")
                    st.write(f"Source: {idx['first_url']}")
            
            with col2:
                if st.button("Load", key=f"load_{idx['name']}"):
                    success, message = load_index(idx['name'])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                
                if idx['name'] != st.session_state.current_index_name:
                    if st.button("Delete", key=f"delete_{idx['name']}"):
                        if st.session_state.get(f"confirm_delete_{idx['name']}", False):
                            try:
                                import shutil
                                index_dir = os.path.join(indices_folder, idx['name'])
                                shutil.rmtree(index_dir)
                                st.success(f"Index '{idx['name']}' deleted successfully")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting index: {str(e)}")
                            finally:
                                st.session_state[f"confirm_delete_{idx['name']}"] = False
                        else:
                            st.session_state[f"confirm_delete_{idx['name']}"] = True
                            st.warning(f"Click Delete again to confirm removing '{idx['name']}'")
