"""
cli.py - Command Line Interface for Local RAG Knowledge Assistant

Terminal-based interactive CLI alternative to the Streamlit web app.
Supports document ingestion, interactive question answering, cosine similarity score inspection,
and source citation displays directly in your shell.
"""

import sys
import os
import time

from database import DEFAULT_DB_PATH, get_indexed_files, get_chunk_count
from ingestion import ingest_documents, DEFAULT_EMBEDDING_MODEL
from rag_engine import generate_rag_response, DEFAULT_LLM_MODEL


def print_banner():
    """Print ASCII art banner and project header."""
    print("=" * 70)
    print("      🤖 LOCAL RAG KNOWLEDGE ASSISTANT (MICROSOFT FOUNDRY LOCAL)")
    print("=" * 70)
    print("  Stack: Python 3.10+ | foundry-local-sdk | SQLite3 | NumPy")
    print("  Constraint: 100% Offline execution with source citations")
    print("=" * 70)
    print()


def print_status():
    """Print current SQLite database indexing status."""
    files = get_indexed_files(DEFAULT_DB_PATH)
    count = get_chunk_count(DEFAULT_DB_PATH)
    print(f"📊 [Database Status]: {count} chunks indexed from {len(files)} file(s).")
    if files:
        print("   Indexed files: " + ", ".join(f"'{f}'" for f in files))
    else:
        print("   ⚠️  No documents indexed yet. Please run Ingestion first (Option 1).")
    print()


def handle_ingestion():
    """Handle document ingestion option from CLI."""
    print("\n--- 📥 Starting Document Ingestion Pipeline ---")
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        print(f"Created '{docs_dir}' directory. Please add text files and retry.")
        return
        
    start_time = time.time()
    
    def progress_callback(msg: str, pct: float):
        percent = int(pct * 100)
        print(f"[{percent:3d}%] {msg}")
        
    res = ingest_documents(
        docs_dir=docs_dir,
        db_path=DEFAULT_DB_PATH,
        model_alias=DEFAULT_EMBEDDING_MODEL,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Ingestion completed in {elapsed:.2f} seconds.")
    print(f"   Status: {res['message']}\n")


def handle_chat_loop():
    """Interactive question answering loop."""
    print("\n--- 💬 Interactive Q&A Session ---")
    print("Type your question and press Enter. (Type 'back' or 'exit' to return to main menu)\n")
    
    while True:
        try:
            user_input = input("\n❓ [Question]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "back", "q"):
                break
                
            print("\n🔍 Retrieving top matches & invoking Foundry Local LLM...")
            start_time = time.time()
            
            result = generate_rag_response(
                user_query=user_input,
                k=3,
                db_path=DEFAULT_DB_PATH,
                llm_alias=DEFAULT_LLM_MODEL,
                embedding_alias=DEFAULT_EMBEDDING_MODEL
            )
            
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("💡 [Answer]:")
            print(result["answer"])
            print("=" * 60)
            
            # Print Retrieved Context Chunks
            retrieved = result["retrieved_chunks"]
            if retrieved:
                print(f"\n📚 [Retrieved Source Contexts ({len(retrieved)} chunks, query time: {elapsed:.2f}s)]:")
                for idx, chunk in enumerate(retrieved, 1):
                    print(f"\n   --- Passage {idx} ---")
                    print(f"   📄 File: {chunk['file_name']} (Chunk {chunk['chunk_id']})")
                    print(f"   🎯 Cosine Similarity Score: {chunk['score']:.4f}")
                    snippet = chunk['content'].strip().replace("\n", " ")
                    if len(snippet) > 150:
                        snippet = snippet[:150] + "..."
                    print(f"   📝 Text Snippet: \"{snippet}\"")
            else:
                print("\n⚠️ No context passages were retrieved.")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")


def main():
    """CLI Main Menu Loop."""
    print_banner()
    
    while True:
        print_status()
        print("MAIN MENU:")
        print("  [1] Process & Ingest Documents ('documents/' -> SQLite)")
        print("  [2] Ask a Question (Interactive Q&A)")
        print("  [3] Exit CLI")
        print()
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == "1":
            handle_ingestion()
        elif choice == "2":
            handle_chat_loop()
        elif choice in ("3", "exit", "quit", "q"):
            print("\nExiting Local RAG Knowledge Assistant. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please select 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
