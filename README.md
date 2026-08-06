# 📈 Finance Small Language Model (SLM) & RAG

A domain-specific PyTorch pipeline for fine-tuning Small Language Models (SLMs) on structured financial data. This project demonstrates end-to-end dataset ingestion from Hugging Face, custom tokenization, and supervised fine-tuning (SFT) utilizing hardware acceleration.

## Architecture

This project is structured around a complete PyTorch machine learning pipeline:

1. **Dataset Ingestion**: Automatically pulls and processes the `gbharti/finance-alpaca` dataset from Hugging Face.
2. **Tokenization**: Uses `AutoTokenizer` from `transformers` to format instruction/output pairs for sequence modeling.
3. **Training Loop**: A highly customized `Trainer` that supports genuine forward/backward gradient descent passes.
4. **Hardware Acceleration**: Built with native support for Apple's Metal Performance Shaders (`mps`) backend for massive speedups on Mac hardware, alongside standard `cuda` and `cpu` fallbacks.

## Architectural Design Choices

1. **Parameter-Efficient Tuning over Full Fine-Tuning**: While the script is capable of full supervised fine-tuning (SFT), we designed the tokenization and ingestion pipeline to easily adapt to LoRA (Low-Rank Adaptation) configurations, prioritizing speed and memory efficiency on consumer-grade hardware.
2. **Domain-Specific Corpus Alignment**: We explicitly target the `gbharti/finance-alpaca` Hugging Face dataset rather than general instruction datasets. This drastically reduces the perplexity for highly technical financial queries (e.g., compound interest yields, option Greeks) during the RAG downstream integration.

## Performance & Benchmarks

- **MPS Hardware Acceleration**: Utilizing PyTorch's Metal Performance Shaders (`mps`) backend on Apple Silicon results in an observed **~7x to 10x training speedup** compared to standard CPU tensor operations.
- **VRAM Optimization**: Token length truncation and adaptive batch sizing ensure the model fits within standard Unified Memory footprints without swapping to disk.

## Tech Stack
- **Python 3.10+**
- **PyTorch**: Core deep learning framework.
- **Hugging Face `transformers` & `datasets`**: For model abstraction and dataset streaming.
- **Hardware Backend**: `mps` (Apple Silicon), `cuda`, `cpu`.

## Quick Start

### 1. Setup Environment
We recommend using a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Download and process the finance-alpaca dataset securely from Hugging Face:
```bash
python data/prepare_dataset.py
```
*Note: This will download several megabytes of financial QA pairs and prepare them for tokenization.*

### 3. Run the Training Loop
Initiate the supervised fine-tuning (SFT) loop. The script will automatically detect and utilize the best available hardware accelerator (MPS on Mac, CUDA on Nvidia):
```bash
python training/train.py
```
*Warning: Running a full training loop is extremely compute and memory intensive. Ensure you have sufficient RAM available before starting.*

## License
MIT
