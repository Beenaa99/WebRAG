import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Any
import json
import os

class RAGBackend:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the RAG backend with the specified embedding model."""
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.doc_embeddings = None
        # Default chunking settings
        self.chunk_size = 500
        self.chunk_overlap = 200

    def set_chunking_params(self, chunk_size: int, chunk_overlap: int):
        """
        Set the chunking parameters.
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Split a document into chunks using RecursiveCharacterTextSplitter.
        Args:
            document: Dict containing 'url' and 'content' keys
        Returns:
            List of document chunks with the same structure
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Split the text into chunks
        text_chunks = text_splitter.split_text(document["content"])

        # Handle cases where no chunks are returned
        if not text_chunks:
            text_chunks = [document["content"]]

        # Create new documents for each chunk
        chunked_docs = []
        for i, chunk in enumerate(text_chunks):
            chunked_doc = {
                "url": f"{document['url']}#chunk={i+1}",
                "content": chunk
            }
            chunked_docs.append(chunked_doc)

        return chunked_docs

    def create_index(self, documents: List[Dict[str, str]], show_progress: bool = False) -> Tuple[Any, np.ndarray]:
        """
        Create a FAISS index from a list of documents.
        Args:
            documents: List of documents to index
            show_progress: Whether to show progress bar during encoding
        Returns:
            Tuple of (faiss index, document embeddings)
        """
        # First chunk all documents
        chunked_documents = []
        for doc in documents:
            chunked_documents.extend(self.chunk_document(doc))
        
        # Store the chunked documents
        self.documents = chunked_documents
        
        # Get text content for embedding
        texts = [doc["content"] for doc in chunked_documents]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=show_progress)
        
        # Convert embeddings to float32 and create index
        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)
        
        # Store index and embeddings
        self.index = index
        self.doc_embeddings = embeddings
        
        return index, embeddings

    def search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Search the index for documents similar to the query.
        Args:
            query: Search query
            k: Number of results to return
        Returns:
            List of k most similar documents
        """
        if self.index is None:
            return []
        
        # Encode query and search
        query_vec = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k)
        
        # Get the corresponding documents
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def save_index(self, save_dir: str, index_name: str) -> Tuple[bool, str]:
        """
        Save the current index and related data to disk.
        Args:
            save_dir: Directory to save the index in
            index_name: Name of the index
        Returns:
            Tuple of (success boolean, message string)
        """
        if self.index is None or not self.documents:
            return False, "No index to save"
        
        # Create directory for this index
        index_dir = os.path.join(save_dir, index_name)
        os.makedirs(index_dir, exist_ok=True)
        
        try:
            # Save the FAISS index
            faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
            
            # Save the document embeddings
            with open(os.path.join(index_dir, "embeddings.npy"), 'wb') as f:
                np.save(f, self.doc_embeddings)
            
            # Save the documents
            with open(os.path.join(index_dir, "documents.json"), 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            return True, f"Index saved successfully as '{index_name}'"
            
        except Exception as e:
            return False, f"Error saving index: {str(e)}"

    def load_index(self, index_dir: str, index_name: str) -> Tuple[bool, str]:
        """
        Load an index and related data from disk.
        Args:
            index_dir: Directory containing the index
            index_name: Name of the index
        Returns:
            Tuple of (success boolean, message string)
        """
        full_path = os.path.join(index_dir, index_name)
        
        if not os.path.exists(full_path):
            return False, f"Index '{index_name}' not found"
        
        try:
            # Load the FAISS index
            self.index = faiss.read_index(os.path.join(full_path, "index.faiss"))
            
            # Load the document embeddings
            with open(os.path.join(full_path, "embeddings.npy"), 'rb') as f:
                self.doc_embeddings = np.load(f)
            
            # Load the documents
            with open(os.path.join(full_path, "documents.json"), 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            return True, f"Successfully loaded index '{index_name}' with {len(self.documents)} documents"
            
        except Exception as e:
            return False, f"Error loading index: {str(e)}"