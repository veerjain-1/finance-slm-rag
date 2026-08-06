import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

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

def evaluate_retrieval_accuracy():
    """
    Simulates Mean Reciprocal Rank (MRR) or Recall@K for the FAISS index.
    """
    print("🎯 Calculating Retrieval Accuracy (Recall@K)...")
    # In a real scenario, we'd query the FAISS index with a test set of questions 
    # and check if the ground-truth document ID is in the top-K results.
    mock_recall_at_2 = 0.85
    return mock_recall_at_2

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
    recall = evaluate_retrieval_accuracy()
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
