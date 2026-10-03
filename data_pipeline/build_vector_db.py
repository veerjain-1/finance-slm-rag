import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Mocking gathered investment strategies from reports/Kaggle.
# Exposed at module level (not just a local var inside build_faiss_index)
# so evaluation/eval.py can build the same index in-memory for retrieval
# evaluation against a known ground truth, without duplicating this list.
RAW_DOCUMENTS = [
    "The '60/40 Portfolio' strategy involves investing 60% in equities and 40% in fixed-income assets. This provides a balance of growth and safety.",
    "Value investing requires buying securities that appear underpriced by some form of fundamental analysis.",
    "Momentum investing involves buying securities that have shown an upward price trend and short-selling those with a downward trend.",
    "The Barbell Strategy in bond investing focuses on short-term and long-term bonds, avoiding intermediate-term bonds to balance liquidity and high yield."
]


def build_document_chunks(raw_documents=None):
    """
    Chunks the raw investment-strategy documents and tags each chunk with
    its source document's index (via chunk.metadata['source_doc_index']),
    so downstream consumers (e.g. retrieval evaluation) can trace a chunk
    back to its ground-truth source document.
    """
    if raw_documents is None:
        raw_documents = RAW_DOCUMENTS

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )

    chunks = []
    for doc_index, doc_text in enumerate(raw_documents):
        doc_chunks = text_splitter.create_documents([doc_text])
        for chunk in doc_chunks:
            chunk.metadata["source_doc_index"] = doc_index
        chunks.extend(doc_chunks)

    return chunks


def build_faiss_index(output_dir="data/faiss_index"):
    """
    Builds a FAISS vector database from investment strategy documents
    for the RAG pipeline.
    """
    print("📚 Loading investment strategy documents...")

    print("✂️ Chunking documents...")
    chunks = build_document_chunks(RAW_DOCUMENTS)

    print("🧠 Generating embeddings using sentence-transformers...")
    # Using a lightweight, fast embedding model suitable for local pipelines
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🗄️ Building FAISS vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    vector_store.save_local(output_dir)
    print(f"✅ FAISS index saved to {output_dir}")

    return vector_store


if __name__ == "__main__":
    build_faiss_index()
