"""
database.py - SQLite Database Management for Local RAG Knowledge Assistant

This module provides functions for initializing the SQLite database (`rag_knowledge.db`),
storing document chunks alongside their vector embeddings, fetching stored records for vector search,
and performing database maintenance tasks completely offline.
"""

import sqlite3
import json
import os
from typing import List, Dict, Any, Tuple, Optional


DEFAULT_DB_PATH = "rag_knowledge.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Establish and return a connection to the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Returns:
        sqlite3.Connection: Database connection object.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize the SQLite database and create the `documents_chunks` table if it does not exist.
    
    Schema:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - file_name: TEXT NOT NULL
        - chunk_id: INTEGER NOT NULL
        - content: TEXT NOT NULL
        - embedding: TEXT NOT NULL (JSON-serialized float array)
    
    Args:
        db_path (str): Path to the SQLite database file.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            );
        """)
        # Index on file_name for fast querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_name ON documents_chunks(file_name);
        """)
        conn.commit()


def insert_chunk(
    file_name: str,
    chunk_id: int,
    content: str,
    embedding_vector: List[float],
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Insert a single document chunk and its embedding vector into the database.
    
    Args:
        file_name (str): Name of the source document file.
        chunk_id (int): Zero-indexed chunk position within the document.
        content (str): Text content of the chunk.
        embedding_vector (List[float]): Numerical embedding vector array.
        db_path (str): Path to the SQLite database file.
        
    Returns:
        int: ID of the inserted row.
    """
    init_db(db_path)
    serialized_embedding = json.dumps(embedding_vector)
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents_chunks (file_name, chunk_id, content, embedding)
            VALUES (?, ?, ?, ?);
        """, (file_name, chunk_id, content, serialized_embedding))
        conn.commit()
        return cursor.lastrowid


def insert_chunks_batch(
    chunks: List[Dict[str, Any]],
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Insert a list of document chunk dicts into the database in a single transaction.
    
    Each dict in `chunks` should contain:
        - 'file_name': str
        - 'chunk_id': int
        - 'content': str
        - 'embedding': List[float]
        
    Args:
        chunks (List[Dict[str, Any]]): List of chunk records.
        db_path (str): Path to the SQLite database file.
        
    Returns:
        int: Number of rows inserted.
    """
    if not chunks:
        return 0
        
    init_db(db_path)
    records = [
        (
            item['file_name'],
            item['chunk_id'],
            item['content'],
            json.dumps(item['embedding'])
        )
        for item in chunks
    ]
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO documents_chunks (file_name, chunk_id, content, embedding)
            VALUES (?, ?, ?, ?);
        """, records)
        conn.commit()
        return cursor.rowcount


def fetch_all_chunks(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Fetch all stored document chunks and deserialize their vector embeddings.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Returns:
        List[Dict[str, Any]]: List of dictionary objects containing:
            - 'id': int
            - 'file_name': str
            - 'chunk_id': int
            - 'content': str
            - 'embedding': List[float]
    """
    if not os.path.exists(db_path):
        return []
        
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_name, chunk_id, content, embedding FROM documents_chunks;")
        rows = cursor.fetchall()
        
    results = []
    for row in rows:
        try:
            vector = json.loads(row['embedding'])
        except Exception:
            vector = []
            
        results.append({
            'id': row['id'],
            'file_name': row['file_name'],
            'chunk_id': row['chunk_id'],
            'content': row['content'],
            'embedding': vector
        })
        
    return results


def clear_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Delete all stored chunks from the database table.
    
    Args:
        db_path (str): Path to the SQLite database file.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents_chunks;")
        conn.commit()


def get_indexed_files(db_path: str = DEFAULT_DB_PATH) -> List[str]:
    """
    Get a list of distinct source file names currently indexed in the database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Returns:
        List[str]: List of unique file names.
    """
    if not os.path.exists(db_path):
        return []
        
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT file_name FROM documents_chunks ORDER BY file_name;")
        rows = cursor.fetchall()
        
    return [row['file_name'] for row in rows]


def get_chunk_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Get total count of document chunks currently in the database.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Returns:
        int: Total number of records.
    """
    if not os.path.exists(db_path):
        return 0
        
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM documents_chunks;")
        row = cursor.fetchone()
        
    return row['cnt'] if row else 0


if __name__ == "__main__":
    # Self-test script when executing database.py directly
    print("Testing database.py...")
    test_db = "test_rag.db"
    init_db(test_db)
    
    # Test single insert
    insert_id = insert_chunk("test.txt", 0, "Sample chunk content", [0.1, 0.2, 0.3], test_db)
    print(f"Inserted test chunk with ID: {insert_id}")
    
    # Test fetch
    chunks = fetch_all_chunks(test_db)
    print(f"Fetched {len(chunks)} chunks from database.")
    print("Files indexed:", get_indexed_files(test_db))
    
    # Clean up test db safely
    import gc
    gc.collect()
    try:
        if os.path.exists(test_db):
            os.remove(test_db)
    except Exception as e:
        print(f"Cleanup note: {e}")
    print("Database module self-test complete.")

