import os
from datasets import load_dataset

def prepare_huggingface_dataset(output_dir="data"):
    """
    Downloads a real finance dataset from Hugging Face and saves it locally.
    We are using 'gbharti/finance-alpaca' which contains over 68k finance instruction pairs.
    For local testing, we will take a small subset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("📥 Downloading 'gbharti/finance-alpaca' from Hugging Face...")
    # Load the dataset
    dataset = load_dataset("gbharti/finance-alpaca", split="train")
    
    # Take a small subset (e.g., 50 samples) for local demo purposes to avoid multi-hour training
    subset = dataset.select(range(50))
    print(f"📊 Selected {len(subset)} samples for local training demo.")
    
    def format_prompt(example):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        prompt = f"### Human: {instruction}\n"
        if input_text:
            prompt += f"Context: {input_text}\n"
        prompt += f"### Assistant: {output}"
        
        return {"text": prompt}
    
    print("⚙️ Formatting dataset for Causal LM...")
    formatted_dataset = subset.map(format_prompt)
    
    # Save to disk
    dataset_path = os.path.join(output_dir, "finance_sft_dataset")
    formatted_dataset.save_to_disk(dataset_path)
    print(f"✅ Real Hugging Face Dataset saved to {dataset_path}")
    
    return formatted_dataset

if __name__ == "__main__":
    prepare_huggingface_dataset()
