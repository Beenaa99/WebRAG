import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Any, Optional, Union
import json
import os
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from datetime import datetime


class RAGBackend:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', openai_client=None):
        """
        Initialize the RAG backend with the specified embedding model.
        
        Args:
            model_name: Local embedding model name (used as fallback)
            openai_client: OpenAI client for API calls
        """
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.openai_client = openai_client
        self.index = None
        self.bm25_index = None
        self.documents = []
        self.doc_embeddings = None
        self.chunks = []
        self.chunk_sources = []
        self.use_openai = openai_client is not None
        
        # Default chunking settings
        self.chunk_size = 500
        self.chunk_overlap = 200
        self.chunking_method = "semantic"  # Default to semantic chunking
        
        # Fixed hybrid search settings
        self.bm25_weight = 0.1
        self.semantic_weight = 0.9

    def set_openai_client(self, client):
        """Set the OpenAI client for embeddings and generation."""
        self.openai_client = client
        self.use_openai = client is not None

    def set_chunking_params(self, chunk_size: int, chunk_overlap: int, method: str = "recursive"):
        """
        Set the chunking parameters.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            method: Chunking method ("recursive" or "semantic")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = method

    def set_hybrid_weights(self, bm25_weight: float, semantic_weight: float):
        """
        Set weights for hybrid retrieval.
        
        Args:
            bm25_weight: Weight for BM25 lexical search
            semantic_weight: Weight for semantic search
        """
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Split a document into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            document: Dict containing 'url' and 'content' keys
            
        Returns:
            List of document chunks with the same structure
        """
        # We'll use semantic chunking only if explicitly required and OpenAI client is available
        if self.chunking_method == "semantic" and self.openai_client:
            try:
                from chonkie import SemanticChunker
                chunker = SemanticChunker(return_type="texts")
                text_chunks = chunker(document["content"])
                
                # Handle cases where no chunks are returned
                if not text_chunks:
                    text_chunks = [document["content"]]
            except Exception as e:
                print(f"Semantic chunking failed: {e}. Falling back to recursive chunking.")
                return self._recursive_chunk_document(document)
        else:
            # Use recursive chunking
            return self._recursive_chunk_document(document)
            
        # Create new documents for each chunk
        chunked_docs = []
        for i, chunk in enumerate(text_chunks):
            chunked_doc = {
                "url": f"{document['url']}#chunk={i+1}",
                "content": chunk
            }
            chunked_docs.append(chunked_doc)

        return chunked_docs
        
    def _recursive_chunk_document(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Split a document using RecursiveCharacterTextSplitter.
        
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

    def get_openai_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Uses OpenAI API to generate embeddings for given texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not set. Call set_openai_client() first.")
            
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",  # OpenAI Embedding Model
            input=texts
        )
        
        return np.array([embedding.embedding for embedding in response.data])

    def create_index(self, documents: List[Dict[str, str]], show_progress: bool = False) -> Tuple[Any, np.ndarray]:
        """
        Create FAISS and BM25 indices from a list of documents.
        
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
        
        # Handle empty document case
        if not chunked_documents:
            print("Warning: No documents to index.")
            return None, None
        
        # Get text content for embedding and BM25
        # Get text content for embedding and BM25
        texts = [doc["content"] for doc in chunked_documents]
        self.chunks = texts
        self.chunk_sources = [doc["url"] for doc in chunked_documents]
        
        # Create BM25 index
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in texts]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        
        # Generate embeddings
        if self.use_openai and self.openai_client:
            # Use OpenAI for embeddings
            try:
                embeddings = self.get_openai_embedding(texts)
                self.embedding_dim = embeddings.shape[1]  # Update dimension based on OpenAI
            except Exception as e:
                print(f"OpenAI embedding failed: {e}. Falling back to local model.")
                embeddings = self.embedder.encode(texts, show_progress_bar=show_progress)
        else:
            # Use local model
            embeddings = self.embedder.encode(texts, show_progress_bar=show_progress)
        
        # Convert embeddings to float32 and create index
        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)
        
        # Store index and embeddings
        self.index = index
        self.doc_embeddings = embeddings
        
        return index, embeddings

    def search(self, query: str, k: int = 6, use_hybrid: bool = True) -> List[Dict[str, str]]:
        """
        Search the index for documents similar to the query.
        
        Args:
            query: Search query
            k: Number of results to return
            use_hybrid: Whether to use hybrid retrieval
            
        Returns:
            List of k most similar documents
        """
        if self.index is None:
            return []
        
        # Handle case where there are no documents
        if len(self.documents) == 0:
            return []
        
        # Get query embedding - ensure dimensions match the index
        if self.use_openai and self.openai_client:
            try:
                query_vec = self.get_openai_embedding([query])
                # Check and fix dimensions if needed
                if query_vec.shape[1] != self.embedding_dim:
                    print(f"Warning: OpenAI embedding dimension ({query_vec.shape[1]}) doesn't match index ({self.embedding_dim}). Using local model instead.")
                    query_vec = self.embedder.encode([query]).astype("float32")
            except Exception as e:
                print(f"OpenAI embedding failed: {e}. Falling back to local model.")
                query_vec = self.embedder.encode([query]).astype("float32")
        else:
            query_vec = self.embedder.encode([query]).astype("float32")
        
        # Standard semantic search (if not using hybrid or if hybrid isn't available)
        if not use_hybrid or self.bm25_index is None:
            # Search FAISS index
            distances, indices = self.index.search(query_vec, min(k, len(self.documents)))
            
            # Get the corresponding documents
            results = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            return results
        
        # If using hybrid search
        # Lexical search with BM25
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
            
        # Semantic search - use safe k value
        k_search = min(k * 2, len(self.chunks))
        if k_search <= 0:
            # If no documents, just return empty list
            return []
            
        # Get distances from FAISS
        distances, indices = self.index.search(query_vec, k_search)
        
        # Convert distances to similarity scores
        max_dist = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        semantic_scores = np.zeros(len(self.chunks))
        
        # Set semantic scores for found items
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(semantic_scores):
                semantic_scores[idx] = 1.0 - (distances[0][i] / max_dist)
        
        # Combine scores with weights
        combined_scores = self.bm25_weight * bm25_scores + self.semantic_weight * semantic_scores
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        # Return documents
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def hybrid_retrieval_safe(self, query: str, k: int = 3) -> List[int]:
        """
        Performs hybrid retrieval using both BM25 and FAISS (safer implementation).
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document indices
        """
        if self.bm25_index is None or self.index is None:
            raise ValueError("Both BM25 and FAISS indices must be created first")
        
        # Ensure we have documents
        if not self.chunks:
            return []
            
        # Lexical search with BM25
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0,1] range
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
            
        # Semantic search with FAISS
        query_vec = None
        if self.use_openai and self.openai_client:
            # Use OpenAI for query embedding
            try:
                query_vec = self.get_openai_embedding([query])
            except Exception as e:
                print(f"OpenAI embedding failed: {e}. Falling back to local model.")
                query_vec = self.embedder.encode([query]).astype("float32")
        else:
            query_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check: ensure we don't request more results than we have documents
        k_search = min(k * 3, len(self.chunks))
        if k_search <= 0:
            return []
            
        # Get distances from FAISS
        distances, indices = self.index.search(query_vec, k_search)
        
        # Convert distances to similarity scores (closer = more similar)
        semantic_scores = np.zeros(len(self.chunks))
        
        # Safety check for empty results
        if len(distances[0]) == 0:
            # Just use BM25 if semantic search fails
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            return top_indices.tolist()
            
        max_dist = np.max(distances[0])
        if max_dist == 0:
            max_dist = 1.0  # Avoid division by zero
            
        # Convert indices to a list to handle different index types
        idx_list = indices[0].tolist()
        for i, idx in enumerate(idx_list):
            if 0 <= idx < len(semantic_scores):  # Make sure index is valid
                semantic_scores[idx] = 1.0 - (distances[0][i] / max_dist)
        
        # Combine scores using weighted sum with fixed weights
        combined_scores = self.bm25_weight * bm25_scores + self.semantic_weight * semantic_scores
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[::-1][:k]
        
        return top_indices.tolist()

    def generate_answer_with_openai(self, query: str, retrieved_chunks: List[str], scores: np.ndarray) -> str:
        """
        Generates an answer using OpenAI API based on retrieved document chunks.
        
        Args:
            query: User question
            retrieved_chunks: List of retrieved document chunks
            scores: Relevance scores for each chunk
            
        Returns:
            Generated answer
        """
        if not self.openai_client:
            return "OpenAI client not set. Cannot generate answer."
            
        # If no chunks retrieved or all have low scores, return fallback message
        if not retrieved_chunks or (len(scores) > 0 and max(scores) < 0.1):
            return "No relevant information found in the document."
            
        # Combine retrieved chunks as context
        context = "\n\n".join(retrieved_chunks)
        
        # Create prompt
        prompt = f"""You are an AI assistant that must strictly answer questions based on the provided context.
        Your task is to extract and synthesize relevant information from the retrieved document chunks.
        If there is relevant information in the document, summarize it accurately.
        
        If no relevant information is found, respond with:
        'No relevant information found in the document.'
        
        Context:
        {context}
        
        Question: {query}
        """
        
        # Generate response using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # OpenAI Model
                messages=[
                    {"role": "system", "content": "You are an AI assistant that only answers based on given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

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
                
            # Save BM25 index
            if self.bm25_index:
                import pickle
                with open(os.path.join(index_dir, "bm25_index.pkl"), 'wb') as f:
                    pickle.dump(self.bm25_index, f)
            
            # Save chunking method and parameters
            metadata = {
                "created_at": datetime.now().isoformat(),
                "document_count": len(self.documents),
                "first_url": self.documents[0]["url"] if self.documents else "Unknown",
                "chunking_method": self.chunking_method,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "hybrid_weights": {
                    "bm25_weight": self.bm25_weight,
                    "semantic_weight": self.semantic_weight
                }
            }
            with open(os.path.join(index_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
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
                
            # Get chunks for BM25 and hybrid search
            self.chunks = [doc["content"] for doc in self.documents]
            self.chunk_sources = [doc["url"] for doc in self.documents]
                
            # Try to load BM25 index
            import pickle
            bm25_path = os.path.join(full_path, "bm25_index.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
            else:
                # Recreate BM25 index if not found
                tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.chunks]
                self.bm25_index = BM25Okapi(tokenized_chunks)
                
            # Try to load metadata
            metadata_path = os.path.join(full_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # Restore chunking parameters if available
                self.chunking_method = metadata.get("chunking_method", "recursive")
                self.chunk_size = metadata.get("chunk_size", 500)
                self.chunk_overlap = metadata.get("chunk_overlap", 200)
                
                # Restore hybrid weights if available
                hybrid_weights = metadata.get("hybrid_weights", {})
                self.bm25_weight = hybrid_weights.get("bm25_weight", 0.1)
                self.semantic_weight = hybrid_weights.get("semantic_weight", 0.9)
            
            return True, f"Successfully loaded index '{index_name}' with {len(self.documents)} documents"
            
        except Exception as e:
            return False, f"Error loading index: {str(e)}"