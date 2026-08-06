import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def build_faiss_index(output_dir="data/faiss_index"):
    """
    Builds a FAISS vector database from investment strategy documents
    for the RAG pipeline.
    """
    print("📚 Loading investment strategy documents...")
    
    # Mocking gathered investment strategies from reports/Kaggle
    raw_documents = [
        "The '60/40 Portfolio' strategy involves investing 60% in equities and 40% in fixed-income assets. This provides a balance of growth and safety.",
        "Value investing requires buying securities that appear underpriced by some form of fundamental analysis.",
        "Momentum investing involves buying securities that have shown an upward price trend and short-selling those with a downward trend.",
        "The Barbell Strategy in bond investing focuses on short-term and long-term bonds, avoiding intermediate-term bonds to balance liquidity and high yield."
    ]
    
    print("✂️ Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=20
    )
    chunks = text_splitter.create_documents(raw_documents)
    
    print("🧠 Generating embeddings using sentence-transformers...")
    # Using a lightweight, fast embedding model suitable for local pipelines
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("🗄️ Building FAISS vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    vector_store.save_local(output_dir)
    print(f"✅ FAISS index saved to {output_dir}")

if __name__ == "__main__":
    build_faiss_index()
