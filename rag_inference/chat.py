import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class FinanceChatbotRAG:
    def __init__(
        self, 
        model_path="../training/slm_finance_model",
        vector_db_path="../data_pipeline/data/faiss_index",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_history=3,
        demo_mode=False
    ):
        self.demo_mode = demo_mode
        self.conversation_history = []
        self.max_history = max_history
        print("🧠 Initializing SLM...")
        # Determine if we should load fine-tuned model or fallback to base
        load_path = model_path if os.path.exists(model_path) else base_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        
        # Hugging Face Pipeline
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,
            temperature=0.3,
            device_map="auto"
        )
        
        print("🗄️ Loading FAISS Vector Database...")
        # We must use the same embeddings model used during indexing
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists(vector_db_path):
            self.vector_store = FAISS.load_local(
                vector_db_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None
            print("⚠️ Warning: Vector DB not found. Operating without RAG.")

    def retrieve_context(self, query: str, k=2):
        if not self.vector_store:
            return ""
        
        docs = self.vector_store.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        return context

    def generate_response(self, query: str):
        print(f"\n💬 Query: {query}")
        
        # 1. Retrieve
        context = self.retrieve_context(query)
        print(f"🔍 Retrieved Context:\n{context}\n" if context else "🔍 No context retrieved.\n")
        
        # Format history
        history_str = ""
        if self.conversation_history:
            history_str = "### Chat History:\n"
            for turn in self.conversation_history:
                history_str += f"Human: {turn['human']}\nAssistant: {turn['ai']}\n"
            history_str += "\n"

        # 2. Build Prompt
        prompt = f"""### System:
You are a highly capable financial advisor AI. Use the provided context and chat history to answer the user's question accurately. If the context does not contain the answer, rely on your general knowledge.

{history_str}### Context:
{context}

### Human: {query}
### Assistant:"""

        # 3. Generate
        # In a real environment, this invokes the LLM pipeline
        print("🤖 Generating response...")
        
        # For demonstration to avoid downloading heavy weights if base model is used
        if self.model.config.model_type == "llama" and self.demo_mode: 
            # We skip heavy inference in demo mode
            response_text = "[Simulated LLM Output based on context]: " + context
        else:
            outputs = self.llm_pipeline(prompt)
            response_text = outputs[0]['generated_text'].split("### Assistant:")[-1].strip()
            
        # Save to history
        self.conversation_history.append({"human": query, "ai": response_text})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

        print(f"✅ Response:\n{response_text}\n")
        return response_text


if __name__ == "__main__":
    chatbot = FinanceChatbotRAG()
    
    # Test complex finance queries
    chatbot.generate_response("What is the 60/40 portfolio strategy?")
    chatbot.generate_response("Explain momentum investing.")
