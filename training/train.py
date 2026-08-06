import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

def train_slm(
    dataset_path=os.path.join(os.path.dirname(__file__), "../data/finance_sft_dataset"),
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir=os.path.join(os.path.dirname(__file__), "../slm_finance_model")
):
    """
    Supervised Fine-Tuning (SFT) pipeline using PyTorch and Hugging Face.
    """
    print(f"🚀 Initializing Real SLM Training Pipeline using {model_name}")
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    model.to(device)
    
    # 2. Load and tokenize Dataset
    print(f"📂 Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path {dataset_path} does not exist.")
        return
        
    dataset = load_from_disk(dataset_path)
    
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
        
    print("✂️ Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 3. Define Training Arguments
    # Running genuinely on the machine using MPS/CUDA. Batch size is 1 to prevent OOM on typical laptops.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        logging_dir='./logs',
        logging_steps=5,
        learning_rate=2e-5,
        fp16=(device == "cuda"),
        report_to="none" # Disable wandb/mlflow for local test
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # 5. Train
    print("🔥 Starting ACTUAL PyTorch training loop (Forward/Backward passes)...")
    trainer.train() 
    
    # 6. Save final model
    print(f"💾 Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Training complete.")

if __name__ == "__main__":
    train_slm()
