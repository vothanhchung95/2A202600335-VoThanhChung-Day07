"""
Test script for Step 10: Chunking Strategy Benchmark

This script helps you:
1. Load documents from data/
2. Chunk them with different strategies
3. Embed and store in EmbeddingStore
4. Test retrieval with sample queries
5. Compare strategy performance

Run: python test_benchmark.py
"""

import sys
sys.path.insert(0, r'E:\aithucchien - vinuni\Day07\Day-07-Lab-Data-Foundations')

from src import (
    Document, EmbeddingStore, KnowledgeBaseAgent,
    FixedSizeChunker, SentenceChunker, RecursiveChunker,
    ChunkingStrategyComparator, LocalEmbedder
)

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================
# CONFIGURATION - Change these to test different setups
# ============================================
FIXED_CHUNK_SIZE = 325  # For FixedSizeChunker
FIXED_OVERLAP = 50  # For FixedSizeChunker overlap (must be < chunk_size)
RECURSIVE_CHUNK_SIZE = 250  # For RecursiveChunker
MAX_SENTENCES = 3  # For SentenceChunker
TOP_K = 3  # Number of results to retrieve

# Sample queries to test retrieval (customize for your domain)
TEST_QUERIES = [
    "Tôi gặp tai nạn trong chuyến xe thì phải làm gì?",
    "Thông tin tài xế không giống trên app thì sao?",
    "Làm sao để đặt xe trên ứng dụng XanhSM?",
    "Tôi để quên đồ trên xe thì làm sao?",
    "Tài xế yêu cầu đi ngoài app có nên không?"
]


def load_documents():
    """Load documents from data/ folder."""
    import os

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    docs = []

    print(f"Looking for documents in: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")

    # Find all .txt and .md files in data directory
    import glob
    data_files = glob.glob(os.path.join(data_dir, '*.txt')) + glob.glob(os.path.join(data_dir, '*.md'))

    # Skip .gitkeep and chunking experiment report
    skip_files = ['.gitkeep', 'chunking_experiment_report.md']

    for filepath in data_files:
        filename = os.path.basename(filepath)
        if any(skip in filename for skip in skip_files):
            continue

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create metadata based on filename
            if 'electric_car' in filename.lower():
                metadata = {'type': 'faq', 'topic': 'electric_car', 'audience': 'drivers'}
            elif 'motor' in filename.lower():
                metadata = {'type': 'faq', 'topic': 'electric_motor', 'audience': 'drivers'}
            elif 'restaurant' in filename.lower():
                metadata = {'type': 'faq', 'topic': 'restaurant', 'audience': 'merchants'}
            elif 'user' in filename.lower():
                metadata = {'type': 'faq', 'topic': 'user_app', 'audience': 'customers'}
            else:
                metadata = {'type': 'faq', 'topic': 'general'}

            doc_id = filename.replace('.', '_').replace(' ', '_').replace('-', '_')
            docs.append(Document(id=doc_id, content=content, metadata=metadata))
            print(f"Loaded: {filename} ({len(content)} chars) - {metadata}")

    return docs


def chunk_documents(docs, chunker):
    """Chunk all documents using specified chunker."""
    chunked_docs = []

    for doc in docs:
        chunks = chunker.chunk(doc.content)
        print(f"  {doc.id}: {len(chunks)} chunks")

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.id}_chunk_{i}"
            # Inherit metadata from parent document
            chunk_metadata = {
                **doc.metadata,
                'parent_doc': doc.id,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            chunked_docs.append(Document(
                id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            ))

    return chunked_docs


def test_strategy(name, chunker, docs, queries):
    """Test a chunking strategy: chunk, embed, store, retrieve."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Step 1: Chunk documents
    print("\n1. Chunking documents...")
    chunked_docs = chunk_documents(docs, chunker)
    total_chunks = len(chunked_docs)
    avg_length = sum(len(d.content) for d in chunked_docs) / total_chunks if total_chunks else 0
    print(f"   Total chunks: {total_chunks}")
    print(f"   Avg chunk length: {avg_length:.1f} chars")

    # Step 2: Create store and add documents
    print("\n2. Embedding and storing...")
    embedder = LocalEmbedder()  # Uses deterministic mock embeddings
    store = EmbeddingStore(
        collection_name=f"test_{name.lower().replace(' ', '_')}",
        embedding_fn=embedder
    )
    store.add_documents(chunked_docs)
    print(f"   Stored {store.get_collection_size()} chunks")

    # Step 3: Test retrieval
    print("\n3. Testing retrieval...")
    results_summary = []

    for query in queries:
        retrieved = store.search(query, top_k=TOP_K)
        top_score = retrieved[0]['score'] if retrieved else 0

        print(f"\n   Query: '{query}'")
        print(f"   Top score: {top_score:.4f}")
        if retrieved:
            content = retrieved[0]['content'][:100]
            try:
                print(f"   Top result: {content}...")
            except UnicodeEncodeError:
                print(f"   Top result: [Vietnamese content, length={len(retrieved[0]['content'])} chars]")

        results_summary.append({
            'query': query,
            'top_score': top_score,
            'results_count': len(retrieved)
        })

    # Summary
    avg_score = sum(r['top_score'] for r in results_summary) / len(results_summary)
    print(f"\n4. Summary for {name}:")
    print(f"   - Chunks: {total_chunks}")
    print(f"   - Avg chunk length: {avg_length:.1f}")
    print(f"   - Avg top retrieval score: {avg_score:.4f}")

    return {
        'name': name,
        'chunk_count': total_chunks,
        'avg_length': avg_length,
        'avg_score': avg_score,
        'store': store
    }


def compare_strategies():
    """Run comparison of all three chunking strategies."""
    print("Loading documents...")
    docs = load_documents()

    if not docs:
        print("No documents found! Check data/ folder.")
        return

    print(f"\nLoaded {len(docs)} documents")

    # Define chunkers to test
    chunkers = [
        ('FixedSizeChunker', FixedSizeChunker(chunk_size=FIXED_CHUNK_SIZE, overlap=FIXED_OVERLAP)),
        ('SentenceChunker', SentenceChunker(max_sentences_per_chunk=MAX_SENTENCES)),
        ('RecursiveChunker', RecursiveChunker(chunk_size=RECURSIVE_CHUNK_SIZE)),
    ]

    results = []

    for name, chunker in chunkers:
        result = test_strategy(name, chunker, docs, TEST_QUERIES)
        results.append(result)

    # Final comparison table
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Strategy':<20} {'Chunks':<10} {'Avg Length':<15} {'Avg Score':<15}")
    print("-"*70)

    for r in results:
        print(f"{r['name']:<20} {r['chunk_count']:<10} {r['avg_length']:<15.1f} {r['avg_score']:<15.4f}")

    print("\n" + "="*70)
    print("ANALYSIS:")
    print("- Lower chunk count + higher avg score = better retrieval quality")
    print("- Consider context preservation (sentences vs arbitrary cuts)")
    print("="*70)


def test_chunking_comparator():
    """Test the built-in ChunkingStrategyComparator."""
    print("\n" + "="*70)
    print("Testing ChunkingStrategyComparator")
    print("="*70)

    # Load one document for comparison
    docs = load_documents()
    if docs:
        sample_text = docs[0].content[:1000]  # First 1000 chars

        comparator = ChunkingStrategyComparator()
        result = comparator.compare(sample_text, chunk_size=FIXED_CHUNK_SIZE)

        print(f"\nResults for: {docs[0].id} (first 1000 chars)")
        print(f"{'Strategy':<20} {'Count':<10} {'Avg Length':<15}")
        print("-"*45)

        for name, stats in result.items():
            print(f"{name:<20} {stats['count']:<10} {stats['avg_length']:<15.1f}")


if __name__ == "__main__":
    print("STEP 10: CHUNKING STRATEGY BENCHMARK")
    print("====================================\n")

    # Run the comparison
    compare_strategies()

    # Also test the comparator
    test_chunking_comparator()

    print("\n[OK] Benchmark complete!")
    print("\nNext steps:")
    print("1. Review the comparison table above")
    print("2. Choose your preferred strategy based on:")
    print("   - Retrieval scores (higher is better)")
    print("   - Context preservation (sentences vs arbitrary cuts)")
    print("   - Chunk count efficiency")
    print("3. Fill in your choice in report/REPORT.md Section 3")
