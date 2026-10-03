import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Allow importing the sibling data_pipeline package when this script is run
# directly (python evaluation/eval.py) rather than as part of an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_pipeline.build_vector_db import RAW_DOCUMENTS, build_document_chunks

# Held-out question -> ground-truth source-document index pairs, matched
# against the same `raw_documents` list used in data_pipeline/build_vector_db.py.
# Each question is a paraphrase of its ground-truth doc so Recall@K actually
# measures semantic retrieval instead of exact string overlap.
RETRIEVAL_TEST_SET = [
    {
        "question": "What investment approach splits a portfolio 60% stocks and 40% bonds?",
        "ground_truth_doc_index": 0,  # "60/40 Portfolio" strategy
    },
    {
        "question": "What strategy looks for stocks trading below their fundamental worth?",
        "ground_truth_doc_index": 1,  # Value investing
    },
    {
        "question": "What strategy buys assets that are trending up and shorts ones trending down?",
        "ground_truth_doc_index": 2,  # Momentum investing
    },
    {
        "question": "What bond strategy avoids intermediate-term bonds in favor of short and long term?",
        "ground_truth_doc_index": 3,  # Barbell Strategy
    },
]

def calculate_perplexity(model, tokenizer, texts):
    """
    Computes Perplexity (PPL) using cross-entropy loss.
    Lower is better.
    """
    print("📈 Calculating Perplexity...")
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 512
    stride = 512
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    if not nlls:
        return float('inf')
        
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def evaluate_bleu(predictions, references):
    """
    Computes BLEU score for generation accuracy.
    """
    print("📏 Calculating BLEU score...")
    bleu = evaluate.load("sacrebleu")
    results = bleu.compute(predictions=predictions, references=references)
    return results['score']

def evaluate_retrieval_accuracy(k=2, test_set=None):
    """
    Computes real Recall@K for the FAISS index built in
    data_pipeline/build_vector_db.py: builds the same index in-memory,
    queries it with a held-out set of (question, ground-truth-doc-index)
    pairs, and measures the fraction of queries whose ground-truth source
    document appears among the top-K retrieved chunks.

    @param k - number of nearest-neighbor chunks to retrieve per query.
    @param test_set - list of {"question": str, "ground_truth_doc_index": int}
                       dicts; defaults to RETRIEVAL_TEST_SET.
    @returns float - Recall@K in [0.0, 1.0].
    """
    print(f"🎯 Calculating Retrieval Accuracy (Recall@{k})...")

    if test_set is None:
        test_set = RETRIEVAL_TEST_SET

    if not test_set:
        return 0.0

    chunks = build_document_chunks(RAW_DOCUMENTS)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    hits = 0
    for case in test_set:
        results = vector_store.similarity_search(case["question"], k=k)
        retrieved_doc_indices = {
            result.metadata.get("source_doc_index") for result in results
        }
        if case["ground_truth_doc_index"] in retrieved_doc_indices:
            hits += 1

    recall_at_k = hits / len(test_set)
    print(f"   => {hits}/{len(test_set)} queries retrieved their ground-truth document")
    return recall_at_k

def run_evaluation():
    print("🧪 Starting SLM Evaluation Pipeline...")
    # Mock data for demonstration
    test_texts = [
        "Dollar Cost Averaging reduces volatility impact.",
        "A 60/40 portfolio balances equities and fixed income."
    ]
    predictions = ["A 60/40 portfolio balances equities and bonds."]
    references = [["A 60/40 portfolio balances equities and fixed income."]]
    
    # 1. Retrieval Accuracy
    recall = evaluate_retrieval_accuracy(k=2)
    print(f"   => Recall@2: {recall:.2f}")
    
    # 2. BLEU Score
    try:
        bleu_score = evaluate_bleu(predictions, references)
        print(f"   => BLEU Score: {bleu_score:.2f}")
    except Exception as e:
        print(f"   => BLEU Score calculation skipped (requires network for metric load): {e}")

    # 3. Perplexity
    print("⚠️ Skipping Perplexity calculation to avoid loading heavy model in demo...")
    # ppl = calculate_perplexity(model, tokenizer, test_texts)
    # print(f"   => Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    run_evaluation()
