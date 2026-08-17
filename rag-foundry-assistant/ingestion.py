"""
ingestion.py - Document Loading, Chunking, Embedding, and Ingestion Pipeline

This module handles:
1. Scanning documents directory (.txt and .md files).
2. Splitting document text into overlapping chunks (~500 chars with ~50 char overlap).
3. Utilizing Microsoft Foundry Local SDK (`foundry-local-sdk`) embedding models to compute vector embeddings.
4. Persisting chunks, metadata, and vector embeddings into SQLite (`rag_knowledge.db`).
"""

import os
import glob
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable

from database import (
    DEFAULT_DB_PATH,
    init_db,
    clear_database,
    insert_chunks_batch,
    get_indexed_files,
    get_chunk_count
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default Embedding Model Alias in Foundry Local Catalog
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding-0.6b"


def derive_topic_label(file_name: str) -> str:
    """
    Turn a filename like 'uefa_şampiyonlar_ligi.txt' into a readable topic
    label like 'uefa şampiyonlar ligi', used to give each chunk's embedding
    document-level context so semantically similar files (e.g. 'süper_lig.txt'
    vs 'uefa_şampiyonlar_ligi.txt') don't get confused with each other during
    vector search.
    
    Args:
        file_name (str): Source file name.
        
    Returns:
        str: Human-readable topic label derived from the filename.
    """
    base = os.path.splitext(file_name)[0]
    return base.replace("_", " ").replace("-", " ").strip()


def build_embedding_input(file_name: str, chunk_text: str) -> str:
    """
    Build the text that actually gets embedded for a chunk: the raw chunk
    content plus a short document-context prefix. This is NOT what gets
    stored/displayed as the chunk's content (that stays the clean original
    text) — it's only used to compute a more topically-disambiguated vector,
    since near-duplicate topics (e.g. multiple football competitions) can
    otherwise get confused with each other in embedding space.
    
    Args:
        file_name (str): Source file name.
        chunk_text (str): Original chunk text.
        
    Returns:
        str: Text to pass to the embedding model.
    """
    topic = derive_topic_label(file_name)
    return f"Konu: {topic}\n\n{chunk_text}"


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Splits a raw text string into chunks of approximately `chunk_size` characters,
    with an `overlap` character count between adjacent chunks.
    Attempts to respect paragraph and sentence boundaries where possible.
    
    Args:
        text (str): Raw document text to split.
        chunk_size (int): Max target character length per chunk (default ~500).
        overlap (int): Character overlap between successive chunks (default ~50).
        
    Returns:
        List[str]: List of text chunks.
    """
    text = text.strip()
    if not text:
        return []
        
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If we are not at the end of the text, try to find a natural break (newline or period)
        if end < text_length:
            # Look for paragraph break near end
            break_pos = text.rfind("\n\n", start + chunk_size // 2, end)
            if break_pos == -1:
                # Look for line break
                break_pos = text.rfind("\n", start + chunk_size // 2, end)
            if break_pos == -1:
                # Look for sentence period
                break_pos = text.rfind(". ", start + chunk_size // 2, end)
            if break_pos != -1:
                end = break_pos + 1
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move start pointer forward, accounting for overlap
        start_next = end - overlap
        if start_next <= start:
            start_next = start + chunk_size // 2
            
        # Snap forward to the next word boundary so the following chunk
        # never starts mid-word (e.g. "...olarak" split into "o" + "larak").
        if 0 < start_next < text_length and not text[start_next - 1].isspace():
            next_space = text.find(" ", start_next)
            if next_space != -1 and next_space - start_next < chunk_size // 4:
                start_next = next_space + 1
                
        start = start_next
        
    return chunks


class LocalEmbeddingProvider:
    """
    Wrapper for Microsoft Foundry Local SDK embedding models.
    Supports initializing, downloading, loading, and generating embeddings via Foundry Local.
    Includes a deterministic offline fallback (hash vector) if local SDK native engine is unavailable.
    """
    
    def __init__(self, model_alias: str = DEFAULT_EMBEDDING_MODEL):
        self.model_alias = model_alias
        self._manager = None
        self._embedding_client = None
        self._initialized = False
        self._fallback_mode = False
        
    def initialize(self) -> bool:
        """
        Initialize the Foundry Local Manager and prepare the embedding client.
        
        Returns:
            bool: True if Foundry Local embedding client initialized successfully.
        """
        if self._initialized:
            return True
            
        try:
            logger.info(f"Initializing Foundry Local SDK for embedding model '{self.model_alias}'...")
            from foundry_local_sdk import FoundryLocalManager, Configuration
            
            if FoundryLocalManager.instance is None:
                config = Configuration(app_name="rag_foundry_assistant")
                FoundryLocalManager.initialize(config)
            self._manager = FoundryLocalManager.instance

            
            model = self._manager.catalog.get_model(self.model_alias)
            if model is None:
                logger.warning(f"Model alias '{self.model_alias}' not found in Foundry Local catalog. Falling back to default list.")
                models = self._manager.catalog.list_models()
                # Find any embedding model
                for m in models:
                    if "embed" in m.alias.lower():
                        model = m
                        self.model_alias = m.alias
                        break
                        
            if model is not None:
                if not model.is_cached:
                    logger.info(f"Downloading model '{self.model_alias}' via Foundry Local SDK...")
                    model.download()
                if not model.is_loaded:
                    logger.info(f"Loading model '{self.model_alias}' into memory...")
                    model.load()
                    
                self._embedding_client = model.get_embedding_client()
                self._initialized = True
                logger.info(f"Foundry Local embedding model '{self.model_alias}' ready.")
                return True
            else:
                raise RuntimeError("No embedding model could be resolved in Foundry Local catalog.")
                
        except Exception as e:
            logger.warning(f"Foundry Local SDK embedding initialization warning: {e}")
            logger.warning("Switching to offline fallback vector generator for testing/offline mode.")
            self._fallback_mode = True
            self._initialized = True
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single string.
        
        Args:
            text (str): Input text string.
            
        Returns:
            List[float]: Embedding vector float array.
        """
        if not self._initialized:
            self.initialize()
            
        if not self._fallback_mode and self._embedding_client is not None:
            try:
                response = self._embedding_client.generate_embedding(text)
                if response and hasattr(response, 'data') and len(response.data) > 0:
                    return list(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error calling Foundry Local embedding client: {e}. Using fallback vector.")
                
        # Deterministic offline fallback embedding vector (384 dim normalized pseudo-random float vector based on text hash)
        return self._generate_fallback_vector(text)

    def _generate_fallback_vector(self, text: str, dim: int = 384) -> List[float]:
        """Generate a deterministic normalized vector based on character hash for fallback mode."""
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def load_documents_from_dir(docs_dir: str) -> List[Tuple[str, str]]:
    """
    Load all .txt and .md document files from the target directory.
    
    Args:
        docs_dir (str): Path to documents directory.
        
    Returns:
        List[Tuple[str, str]]: List of (filename, file_content) tuples.
    """
    if not os.path.exists(docs_dir):
        logger.error(f"Documents directory '{docs_dir}' does not exist.")
        return []
        
    supported_extensions = ("*.txt", "*.md")
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(docs_dir, ext)))
        
    documents = []
    for file_path in sorted(files):
        try:
            file_name = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                documents.append((file_name, content))
                logger.info(f"Loaded document '{file_name}' ({len(content)} characters).")
        except Exception as e:
            logger.error(f"Failed to read file '{file_path}': {e}")
            
    return documents


def ingest_documents(
    docs_dir: str = "documents",
    db_path: str = DEFAULT_DB_PATH,
    model_alias: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 500,
    overlap: int = 50,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Full document ingestion pipeline:
    1. Read all .txt and .md files from `docs_dir`.
    2. Split document texts into chunks.
    3. Generate vector embeddings using Foundry Local SDK.
    4. Store metadata, text chunks, and vectors into SQLite DB.
    
    Args:
        docs_dir (str): Folder containing text documents.
        db_path (str): Path to SQLite database.
        model_alias (str): Foundry Local embedding model alias.
        chunk_size (int): Character length per chunk.
        overlap (int): Character overlap.
        progress_callback (Optional[Callable]): Optional callback for UI progress updates.
        
    Returns:
        Dict[str, Any]: Summary dictionary with file count, chunk count, and execution status.
    """
    if progress_callback:
        progress_callback("Clearing old database records...", 0.05)
        
    clear_database(db_path)
    init_db(db_path)
    
    documents = load_documents_from_dir(docs_dir)
    if not documents:
        msg = f"No valid .txt or .md files found in '{docs_dir}'."
        logger.warning(msg)
        if progress_callback:
            progress_callback(msg, 1.0)
        return {
            "status": "warning",
            "message": msg,
            "file_count": 0,
            "chunk_count": 0
        }
        
    if progress_callback:
        progress_callback("Initializing embedding provider...", 0.15)
        
    embedding_provider = LocalEmbeddingProvider(model_alias=model_alias)
    embedding_provider.initialize()
    
    all_chunks_to_insert = []
    total_docs = len(documents)
    
    for doc_idx, (file_name, content) in enumerate(documents):
        chunks = split_text_into_chunks(content, chunk_size=chunk_size, overlap=overlap)
        logger.info(f"Splitting '{file_name}' into {len(chunks)} chunks...")
        
        for chunk_id, chunk_text in enumerate(chunks):
            if progress_callback:
                progress = 0.20 + 0.70 * ((doc_idx + (chunk_id / max(len(chunks), 1))) / total_docs)
                progress_callback(f"Embedding chunk {chunk_id+1}/{len(chunks)} of '{file_name}'...", min(progress, 0.90))
                
            vector = embedding_provider.generate_embedding(
                build_embedding_input(file_name, chunk_text)
            )
            
            all_chunks_to_insert.append({
                "file_name": file_name,
                "chunk_id": chunk_id,
                "content": chunk_text,
                "embedding": vector
            })
            
    if progress_callback:
        progress_callback("Saving vector embeddings into SQLite database...", 0.95)
        
    inserted_count = insert_chunks_batch(all_chunks_to_insert, db_path=db_path)
    
    summary = {
        "status": "success",
        "message": f"Successfully ingested {len(documents)} files ({inserted_count} chunks) into SQLite.",
        "file_count": len(documents),
        "chunk_count": inserted_count,
        "files": [doc[0] for doc in documents]
    }
    
    if progress_callback:
        progress_callback(summary["message"], 1.0)
        
    logger.info(summary["message"])
    return summary


if __name__ == "__main__":
    print("Testing document ingestion pipeline...")
    # Create target documents dir if needed
    os.makedirs("documents", exist_ok=True)
    res = ingest_documents(docs_dir="documents", db_path="test_ingest.db")
    print("Ingestion summary:", res)
    import gc
    gc.collect()
    try:
        if os.path.exists("test_ingest.db"):
            os.remove("test_ingest.db")
    except Exception as e:
        print(f"Cleanup note: {e}")
    print("Ingestion module self-test complete.")